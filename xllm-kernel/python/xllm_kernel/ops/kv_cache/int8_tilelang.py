# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""A3 vector-core writer for contiguous two-head GQA decode graphs."""

from functools import cache
from typing import TYPE_CHECKING

import tilelang
import torch
from tilelang import language as T

from xllm_kernel.ops.kv_cache import WRITE_INT8_KV_CONTRACT
from xllm_kernel.ops.kv_cache.int8_npu import _write_int8_kv as _triton_write
from xllm_kernel.platform import PlatformEnum, current_platform
from xllm_kernel.registry import register_kernel

if TYPE_CHECKING:
    from tilelang.jit import JITKernel


@cache
def _prepare_writer(page: int) -> "JITKernel":
    from xllm_kernel.ops._tilelang import VECTOR_CORES

    heads, dim = 2, 256
    tokens = T.symbolic("tokens")
    blocks = T.symbolic("blocks")
    groups = T.min(VECTOR_CORES // 2, T.ceildiv(tokens, 2))

    @T.prim_func
    def write(
        key: T.Tensor((tokens, heads, dim), "bfloat16"),
        value: T.Tensor((tokens, heads, dim), "bfloat16"),
        slots: T.Tensor((tokens,), "int32"),
        kp: T.Tensor((blocks, heads, page, dim), "int8"),
        vp: T.Tensor((blocks, heads, page, dim), "int8"),
        ks: T.Tensor((blocks, heads, page), "float32"),
        vs: T.Tensor((blocks, heads, page), "float32"),
        divisor: T.float32,
    ):
        with T.Kernel(groups, is_npu=True) as (core, vector), T.Scope("V"):
            source = T.alloc_shared((heads, dim), "bfloat16")
            data = T.alloc_shared((heads, dim), "float32")
            magnitude = T.alloc_shared((heads, dim), "float32")
            scale = T.alloc_shared((heads, 1), "float32")
            safe = T.alloc_shared((heads, 1), "float32")
            denominator = T.alloc_shared((heads, dim), "float32")
            rounded = T.alloc_shared((heads, dim), "int32")
            fp16 = T.alloc_shared((heads, dim), "float16")
            quantized = T.alloc_shared((heads, dim), "int8")
            # MTE3 requires 32-byte alignment even for a single FP32 scale.
            dma_scale = T.alloc_shared((8,), "float32")
            for group in T.serial(T.ceildiv(tokens, VECTOR_CORES)):
                token = group * VECTOR_CORES + core * 2 + vector
                if token < tokens:
                    slot = slots[token]
                    if slot >= 0 and slot < blocks * page:
                        for component in T.serial(2):
                            if component == 0:
                                T.copy(key[token, :, :], source)
                            else:
                                T.copy(value[token, :, :], source)
                            T.tile.cast(data, source, "CAST_NONE", heads * dim)
                            T.tile.abs(magnitude, data)
                            T.reduce_max(magnitude, scale, dim=-1)
                            for head in T.serial(heads):
                                # A runtime scalar division preserves IEEE rounding of peak/127.
                                s = scale[head, 0] / divisor
                                scale[head, 0] = s
                                safe[head, 0] = T.if_then_else(s > 0.0, s, T.float32(1.0))
                            T.tile.broadcast(denominator, safe)
                            T.tile.div(data, data, denominator)
                            T.tile.cast(rounded, data, "CAST_RINT", heads * dim)
                            # The A3 S32-to-F16 cast also applies a dequantization register.
                            T.tile.cast(data, rounded, "CAST_NONE", heads * dim)
                            T.tile.cast(fp16, data, "CAST_RINT", heads * dim)
                            T.tile.cast(quantized, fp16, "CAST_RINT", heads * dim)
                            for head in T.serial(heads):
                                dma_scale[0] = scale[head, 0]
                                if component == 0:
                                    T.copy(quantized[head, :], kp[slot // page, head, slot % page, 0])
                                    T.copy(dma_scale[0:1], ks[slot // page, head, slot % page])
                                else:
                                    T.copy(quantized[head, :], vp[slot // page, head, slot % page, 0])
                                    T.copy(dma_scale[0:1], vs[slot // page, head, slot % page])

    return tilelang.compile(write, pass_configs={"tl.ascend_auto_sync": True, "tl.ascend_memory_planning": True})


if current_platform.get_ascend_soc_generation() == "a3":

    @register_kernel(
        name="npu.tilelang.write_int8_kv",
        platform=PlatformEnum.NPU,
        execution_modes=("aclgraph",),
        input_domain="A3 GQA with contiguous BF16 B32/B48/B64 H2 D256 graph tiles and general strided Triton writes.",
        solution="tilelang",
        binding="python",
        priority=1,
        contract=WRITE_INT8_KV_CONTRACT,
    )
    def _write_int8_kv(
        key: torch.Tensor,
        value: torch.Tensor,
        slots: torch.Tensor,
        key_pages: torch.Tensor,
        value_pages: torch.Tensor,
        key_scales: torch.Tensor,
        value_scales: torch.Tensor,
    ) -> None:
        args = (key, value, slots, key_pages, value_pages, key_scales, value_scales)
        if (
            key.shape in ((32, 2, 256), (48, 2, 256), (64, 2, 256))
            and key.dtype == value.dtype == torch.bfloat16
            and slots.dtype == torch.int32
            and key.is_contiguous()
            and value.is_contiguous()
            and slots.is_contiguous()
            and key_pages.numel() <= 2**31 - 1
            and torch.npu.is_current_stream_capturing()
        ):
            _prepare_writer(key_pages.size(2))(*args, 127.0)
        else:
            _triton_write(*args)
