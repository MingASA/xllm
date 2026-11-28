# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""A3 vector-core writer for BF16 GQA decode graphs."""

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
def _prepare_writer(page: int, heads: int, dim: int, key_stride: int, value_stride: int) -> "JITKernel":
    from xllm_kernel.ops._tilelang import VECTOR_CORES

    tile_heads = 2 if heads % 2 == 0 else 1
    head_groups = heads // tile_heads
    tokens = T.symbolic("tokens")
    blocks = T.symbolic("blocks")
    key_rows = T.symbolic("key_rows")
    value_rows = T.symbolic("value_rows")
    packed = key_stride == value_stride == heads
    k_shape = (tokens, heads, dim) if packed else (key_rows, dim)
    v_shape = (tokens, heads, dim) if packed else (value_rows, dim)
    tiles = tokens * head_groups
    groups = T.min(VECTOR_CORES // 2, T.ceildiv(tiles, 2))

    @T.prim_func
    def write(
        key: T.Tensor(k_shape, "bfloat16"),
        value: T.Tensor(v_shape, "bfloat16"),
        slots: T.Tensor((tokens,), "int32"),
        kp: T.Tensor((blocks, heads, page, dim), "int8"),
        vp: T.Tensor((blocks, heads, page, dim), "int8"),
        ks: T.Tensor((blocks, heads, page), "float32"),
        vs: T.Tensor((blocks, heads, page), "float32"),
        divisor: T.float32,
    ):
        with T.Kernel(groups, is_npu=True) as (core, vector), T.Scope("V"):
            source = T.alloc_shared((tile_heads, dim), "bfloat16")
            data = T.alloc_shared((tile_heads, dim), "float32")
            magnitude = T.alloc_shared((tile_heads, dim), "float32")
            scale = T.alloc_shared((tile_heads, 1), "float32")
            safe = T.alloc_shared((tile_heads, 1), "float32")
            denominator = T.alloc_shared((tile_heads, dim), "float32")
            rounded = T.alloc_shared((tile_heads, dim), "int32")
            fp16 = T.alloc_shared((tile_heads, dim), "float16")
            quantized = T.alloc_shared((tile_heads, dim), "int8")
            # MTE3 requires 32-byte alignment even for a single FP32 scale.
            dma_scale = T.alloc_shared((8,), "float32")
            for group in T.serial(T.ceildiv(tiles, VECTOR_CORES)):
                tile = group * VECTOR_CORES + core * 2 + vector
                if tile < tiles:
                    token = tile // head_groups
                    first_head = tile % head_groups * tile_heads
                    slot = slots[token]
                    if slot >= 0 and slot < blocks * page:
                        for component in T.serial(2):
                            if packed:
                                if component == 0:
                                    T.copy(key[token, first_head : first_head + tile_heads, :], source)
                                else:
                                    T.copy(value[token, first_head : first_head + tile_heads, :], source)
                            elif component == 0:
                                row = token * key_stride + first_head
                                T.copy(key[row : row + tile_heads, :], source)
                            else:
                                row = token * value_stride + first_head
                                T.copy(value[row : row + tile_heads, :], source)
                            T.tile.cast(data, source, "CAST_NONE", tile_heads * dim)
                            T.tile.abs(magnitude, data)
                            T.reduce_max(magnitude, scale, dim=-1)
                            for head in T.serial(tile_heads):
                                # A runtime scalar division preserves IEEE rounding of peak/127.
                                s = scale[head, 0] / divisor
                                scale[head, 0] = s
                                safe[head, 0] = T.if_then_else(s > 0.0, s, T.float32(1.0))
                            T.tile.broadcast(denominator, safe)
                            T.tile.div(data, data, denominator)
                            T.tile.cast(rounded, data, "CAST_RINT", tile_heads * dim)
                            # The A3 S32-to-F16 cast also applies a dequantization register.
                            T.tile.cast(data, rounded, "CAST_NONE", tile_heads * dim)
                            T.tile.cast(fp16, data, "CAST_RINT", tile_heads * dim)
                            T.tile.cast(quantized, fp16, "CAST_RINT", tile_heads * dim)
                            for head in T.serial(tile_heads):
                                dma_scale[0] = scale[head, 0]
                                if component == 0:
                                    T.copy(quantized[head, :], kp[slot // page, first_head + head, slot % page, 0])
                                    T.copy(dma_scale[0:1], ks[slot // page, first_head + head, slot % page])
                                else:
                                    T.copy(quantized[head, :], vp[slot // page, first_head + head, slot % page, 0])
                                    T.copy(dma_scale[0:1], vs[slot // page, first_head + head, slot % page])

    return tilelang.compile(write, pass_configs={"tl.ascend_auto_sync": True, "tl.ascend_memory_planning": True})


if current_platform.get_ascend_soc_generation() == "a3":

    @register_kernel(
        name="npu.tilelang.write_int8_kv",
        platform=PlatformEnum.NPU,
        execution_modes=("aclgraph",),
        input_domain="A3 GQA graph writes with packed head rows and a general Triton fallback.",
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
            torch.npu.is_current_stream_capturing()
            and key.size(0) > 0
            and key.size(1) > 0
            and 32 <= key.size(2) <= 512
            and key.size(2) % 32 == 0
            and key.dtype == value.dtype == torch.bfloat16
            and slots.dtype == torch.int32
            and key.stride()[1:] == value.stride()[1:] == (key.size(2), 1)
            and min(key.stride(0), value.stride(0)) >= key.size(1) * key.size(2)
            and key.stride(0) % key.size(2) == value.stride(0) % key.size(2) == 0
            and max(key.stride(0), value.stride(0)) * key.size(0) < 2**31
            and slots.is_contiguous()
            and key_pages.numel() <= 2**31 - 1
        ):
            tokens, heads, dim = key.shape
            k_stride, v_stride = key.stride(0) // dim, value.stride(0) // dim
            if k_stride != heads or v_stride != heads:
                # The binding requires contiguous descriptors. View the physical span
                # of strided QKV projections without copying their logical rows.
                key = key.as_strided(((tokens - 1) * k_stride + heads, dim), (dim, 1))
                value = value.as_strided(((tokens - 1) * v_stride + heads, dim), (dim, 1))
            _prepare_writer(key_pages.size(2), heads, dim, k_stride, v_stride)(
                key, value, slots, key_pages, value_pages, key_scales, value_scales, 127.0
            )
        else:
            _triton_write(*args)
