# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Ascend GQA with paged INT8 K/V and per-token, per-head dequantization scales."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch_npu
from xllm_kernel.ops.kv_cache import reshape_paged_cache, write_int8_kv
from xllm_kernel.ops.quantization import dynamic_quant

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.npu_paged_attention import SPARSE_MODE_RIGHT_DOWN_CAUSAL, NpuPagedAttentionBackend
from xllm.python.model_executor.forward_context import borrow_acl_graph_tensor, get_forward_context_or_none

if TYPE_CHECKING:
    from xllm.python.attention.backend import AttentionMetadata
    from xllm.python.layers.attention import Attention


class NpuInt8AttentionBackend(NpuPagedAttentionBackend):
    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        super().bind_kv_caches(kv_caches)
        self._bulk_indices: tuple[torch.Tensor | None, torch.Tensor] | None = None
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

    def prepare(self, metadata: AttentionMetadata, *, graph_mode: bool = False) -> None:
        super().prepare(metadata, graph_mode=graph_mode)
        self._bulk_indices = None
        slots = self._full_slot_mapping(metadata)
        # At 4096 token/head rows CANN's bulk quantization overtakes the scalar scatter.
        if metadata.is_prefill and slots.numel() * self.num_kv_heads >= 4096:
            live = torch.where(slots >= 0)[0]
            if live.numel() == slots.numel():
                live = None
            else:
                slots = slots.index_select(0, live)
            heads = torch.arange(self.num_kv_heads, device=slots.device, dtype=slots.dtype)
            first = slots + (slots // self.page_size) * ((self.num_kv_heads - 1) * self.page_size)
            self._bulk_indices = live, (first[:, None] + heads * self.page_size).reshape(-1).int()

    def _fia_options(self, cache: LayerCache) -> dict:
        return dict(
            input_layout="BNSD",
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            scale=self.scale,
            block_size=self.page_size,
            key_antiquant_scale=cache.key_scale,
            value_antiquant_scale=cache.value_scale,
            key_antiquant_mode=5,
            value_antiquant_mode=5,
        )

    def _allocate_graph_workspace(
        self, graph_batch_size: int, block_table: torch.Tensor, *, actual_seq_kv: list[int] | torch.Tensor | None = None
    ) -> torch.Tensor:
        # All full-attention layers have the same page and scale descriptors.
        cache = next(cache for cache in self._kv_caches if cache.key is not None)
        query = torch.empty(graph_batch_size, self.num_heads, 1, self.head_dim, device=self.device, dtype=self.dtype)
        return self._fused_attention.workspace(
            query=query,
            key=cache.key.transpose(1, 2),
            value=cache.value.transpose(1, 2),
            block_table=block_table,
            actual_seq_lengths_kv=self._actual_seq_kv if actual_seq_kv is None else actual_seq_kv,
            **self._fia_options(cache),
        )

    def execute(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer: Attention) -> torch.Tensor:
        metadata = self._metadata
        assert metadata is not None
        cache = self._kv_caches[layer.layer_id]
        query = q.view(-1, self.num_heads, self.head_dim).contiguous()
        key = k.view(-1, self.num_kv_heads, self.head_dim)
        value = v.view_as(key)
        key_pages = cache.key.transpose(1, 2)
        value_pages = cache.value.transpose(1, 2)
        if self._bulk_indices is None:
            write_int8_kv(
                key,
                value,
                self._full_slot_mapping(metadata),
                key_pages,
                value_pages,
                cache.key_scale,
                cache.value_scale,
            )
        else:
            live, indices = self._bulk_indices
            key, value = key.contiguous(), value.contiguous()
            if indices.numel():
                ki, ks = dynamic_quant(key if live is None else key.index_select(0, live))
                vi, vs = dynamic_quant(value if live is None else value.index_select(0, live))
                reshape_paged_cache(
                    indices,
                    ki.view(-1, 1, self.head_dim),
                    vi.view(-1, 1, self.head_dim),
                    key_pages.view(-1, self.page_size, 1, self.head_dim),
                    value_pages.view(-1, self.page_size, 1, self.head_dim),
                )
                torch_npu.npu_scatter_nd_update_(cache.key_scale.view(-1), indices[:, None], ks.flatten())
                torch_npu.npu_scatter_nd_update_(cache.value_scale.view(-1), indices[:, None], vs.flatten())

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
            query = query.unsqueeze(2)
            options = self._fia_options(cache)
            context = get_forward_context_or_none()
            if context is None or context.execution_state is None:
                output, _ = self._fused_attention(
                    query,
                    key_pages,
                    value_pages,
                    block_table=self._block_table_i32,
                    actual_seq_lengths_kv=self._actual_seq_kv,
                    **options,
                )
            else:
                state = self._paged_graph_state
                if state is None:
                    raise RuntimeError("INT8 decode graph requires prepared FIA workspace")
                output = state.output.view_as(query) if state.output is not None else torch.empty_like(query)
                capture = context.acl_graph
                borrow = capture is not None and state.output is None
                options.update(
                    block_table=state.block_table,
                    actual_seq_lengths_kv=state.kv,
                    workspace=state.workspace,
                    out=[borrow_acl_graph_tensor(output) if borrow else output, state.lse],
                )
                if capture is None:
                    self._fused_attention.out(query, key_pages, value_pages, **options)
                else:
                    capture.capture_task(
                        self._fused_attention.out,
                        borrow_acl_graph_tensor(query) if borrow else query,
                        key_pages,
                        value_pages,
                        **options,
                    )
        return output.reshape(q.shape[0], self.num_heads * self.head_dim)
