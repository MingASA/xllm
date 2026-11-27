# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The xLLM Authors.
# Adapted from vLLM contributors' _reshape_cache_per_token_head in
# vllm/v1/attention/ops/triton_reshape_and_cache_flash.py.
"""Dynamic GQA K/V quantization into head-major Ascend FIA pages."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cann import libdevice

from xllm_kernel.ops.kv_cache import WRITE_INT8_KV_CONTRACT
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import register_kernel


@triton.jit
def _write_int8_head(
    key,
    value,
    slots,
    key_pages,
    value_pages,
    key_scales,
    value_scales,
    key_stride_t: tl.constexpr,
    key_stride_h: tl.constexpr,
    key_stride_d: tl.constexpr,
    value_stride_t: tl.constexpr,
    value_stride_h: tl.constexpr,
    value_stride_d: tl.constexpr,
    slot_stride: tl.constexpr,
    heads: tl.constexpr,
    dim: tl.constexpr,
    page_size: tl.constexpr,
    capacity: tl.constexpr,
    channels: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(slots + token * slot_stride).to(tl.int64)
    if slot >= 0 and slot < capacity:
        lane = tl.arange(0, channels)
        k = tl.load(key + token * key_stride_t + head * key_stride_h + lane * key_stride_d, lane < dim, 0).to(
            tl.float32
        )
        v = tl.load(value + token * value_stride_t + head * value_stride_h + lane * value_stride_d, lane < dim, 0).to(
            tl.float32
        )
        k_scale = tl.max(tl.abs(k), 0) / 127.0
        v_scale = tl.max(tl.abs(v), 0) / 127.0
        k_int8 = libdevice.rint(k / tl.where(k_scale > 0, k_scale, 1.0)).to(tl.int8)
        v_int8 = libdevice.rint(v / tl.where(v_scale > 0, v_scale, 1.0)).to(tl.int8)
        # One scalar address per program keeps Ascend loads and stores contiguous.
        row = slot + ((slot // page_size) * (heads - 1) + head) * page_size
        tl.store(key_pages + row * dim + lane, k_int8, lane < dim)
        tl.store(value_pages + row * dim + lane, v_int8, lane < dim)
        tl.store(key_scales + row, k_scale)
        tl.store(value_scales + row, v_scale)


@register_kernel(
    name="npu.triton.write_int8_kv",
    platform=PlatformEnum.NPU,
    execution_modes=("eager", "aclgraph"),
    input_domain="Strided floating-point [token, head, dim] K/V and contiguous head-major FIA pages on Ascend 910C.",
    solution="triton",
    binding="python",
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
    tokens, heads, dim = key.shape
    if tokens:
        _write_int8_head[(tokens, heads)](
            key,
            value,
            slots,
            key_pages,
            value_pages,
            key_scales,
            value_scales,
            *key.stride(),
            *value.stride(),
            slots.stride(0),
            heads,
            dim,
            key_pages.size(2),
            key_pages.size(0) * key_pages.size(2),
            triton.next_power_of_2(dim),
        )
