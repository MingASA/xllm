# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Eager GQA with paged INT8 K/V and per-token, per-head dequantization scales."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from xllm_kernel.ops.kv_cache import write_int8_kv

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
                page is None
                or page.dtype != torch.int8
                or page.shape != expected
                or not page.transpose(1, 2).is_contiguous()
                for page in (cache.key, cache.value)
            ):
                raise ValueError("INT8 GQA requires head-major signed INT8 K/V pages")
            scale_shape = (self.num_kv_blocks, self.num_kv_heads, self.page_size)
            if any(
                scale is None or scale.dtype != torch.float32 or scale.shape != scale_shape or not scale.is_contiguous()
                for scale in (cache.key_scale, cache.value_scale)
            ):
                raise ValueError("INT8 GQA requires contiguous FP32 scale pages shaped [block, head, token]")

    def execute(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer: Attention) -> torch.Tensor:
        metadata = self._metadata
        assert metadata is not None
        cache = self._kv_caches[layer.layer_id]
        query = q.view(-1, self.num_heads, self.head_dim).contiguous()
        key = k.view(-1, self.num_kv_heads, self.head_dim)
        value = v.view_as(key)
        key_pages = cache.key.transpose(1, 2)
        value_pages = cache.value.transpose(1, 2)
        write_int8_kv(
            key, value, self._full_slot_mapping(metadata), key_pages, value_pages, cache.key_scale, cache.value_scale
        )

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
                key_pages,
                value_pages,
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
