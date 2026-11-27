# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Eager GQA with paged INT8 K/V and per-token, per-head dequantization scales."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from xllm_kernel.ops.kv_cache import reshape_paged_cache
from xllm_kernel.ops.quantization import dynamic_quant

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.npu_paged_attention import SPARSE_MODE_RIGHT_DOWN_CAUSAL, NpuPagedAttentionBackend

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention


class NpuInt8AttentionBackend(NpuPagedAttentionBackend):
    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        super().bind_kv_caches(kv_caches)
        if not 32 <= self.page_size <= 512 or self.page_size % 32:
            raise ValueError("INT8 paged FIA requires a page size divisible by 32 and at most 512")
        for cache in kv_caches:
            if cache.key is None:
                continue
            expected = (self.num_kv_blocks, self.page_size, self.num_kv_heads, self.head_dim)
            if any(
                page is None or page.dtype != torch.int8 or page.shape != expected for page in (cache.key, cache.value)
            ):
                raise ValueError("INT8 GQA requires matching signed INT8 K/V pages")
            scale_shape = (self.num_kv_blocks, self.num_kv_heads, self.page_size)
            if any(
                scale is None or scale.dtype != torch.float32 or scale.shape != scale_shape
                for scale in (cache.key_scale, cache.value_scale)
            ):
                raise ValueError("INT8 GQA requires FP32 scale pages shaped [block, head, token]")

    def execute(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer: Attention) -> torch.Tensor:
        metadata = self._metadata
        assert metadata is not None
        cache = self._kv_caches[layer.layer_id]
        query = q.view(-1, self.num_heads, self.head_dim).contiguous()
        key = k.view(-1, self.num_kv_heads, self.head_dim)
        value = v.view_as(key)
        slots = self._full_slot_mapping(metadata)
        key_int8, key_scale = dynamic_quant(key.contiguous())
        value_int8, value_scale = dynamic_quant(value.contiguous())
        reshape_paged_cache(slots, key_int8, value_int8, cache.key, cache.value)
        live = torch.where(slots >= 0)[0]
        pages, offsets = slots[live] // self.page_size, slots[live] % self.page_size
        cache.key_scale[pages, :, offsets] = key_scale[live]
        cache.value_scale[pages, :, offsets] = value_scale[live]

        if metadata.is_prefill:
            # The whole prompt is present in model precision before the first decode.
            ends = self._cumulative_seq_lens(metadata, query.shape[0])
            output, _ = self._fused_attention(
                query,
                key.contiguous(),
                value.contiguous(),
                input_layout="TND",
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                scale=self.scale,
                atten_mask=self._get_causal_mask(),
                sparse_mode=SPARSE_MODE_RIGHT_DOWN_CAUSAL,
                actual_seq_lengths=ends,
                actual_seq_lengths_kv=ends,
            )
        else:
            output, _ = self._fused_attention(
                query.unsqueeze(2),
                cache.key.flatten(2),
                cache.value.flatten(2),
                input_layout="BNSD",
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                scale=self.scale,
                block_size=self.page_size,
                block_table=self._block_table_i32,
                actual_seq_lengths_kv=self._actual_seq_kv,
                key_antiquant_scale=cache.key_scale,
                value_antiquant_scale=cache.value_scale,
                key_antiquant_mode=5,
                value_antiquant_mode=5,
            )
        return output.reshape(q.shape[0], self.num_heads * self.head_dim)
