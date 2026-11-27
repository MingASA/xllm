# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Explicit SM90 XQA decode over resident pages and stable graph metadata."""

from __future__ import annotations

import torch
from xllm_kernel.ops.kv_cache import convert_xqa_pages

from xllm.python.attention.backend import AttentionMetadata

_XQA_PAGE_SIZES = (16, 32, 64, 128)


class XQADecodePlan:
    """One prepared bucket; no BF16 KV copy, host sync, or planner inside replay."""

    def __init__(self, batch_size: int, page_size: int, maximum: int, device: torch.device) -> None:
        if batch_size <= 0:
            raise ValueError("XQA requires a positive batch size")
        if page_size not in _XQA_PAGE_SIZES or maximum <= 0:
            raise ValueError("XQA requires a supported page size and a positive maximum context")
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != (9, 0):
            raise ValueError("This explicit production XQA profile is validated only for SM90")
        self._page_size, self._maximum = page_size, maximum
        self._batch_size = batch_size
        self._table = torch.empty(
            (batch_size, (maximum + page_size - 1) // page_size), dtype=torch.int32, device=device
        )
        self._lengths = torch.empty(batch_size, dtype=torch.uint32, device=device)
        self._metadata_addresses: tuple[int, int, int] | None = None

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def prepare(self, metadata: AttentionMetadata, *, graph_mode: bool) -> None:
        lengths = metadata.kv_seq_lens_host
        if (
            lengths is None
            or lengths.device.type != "cpu"
            or lengths.dtype != torch.int32
            or lengths.shape != (self._batch_size,)
        ):
            raise ValueError("XQA requires one int32 host KV length per sequence")
        if lengths.max().item() > self._maximum:
            raise ValueError("XQA live context exceeds the declared model context")

        inputs = (metadata.paged_kv_indptr, metadata.paged_kv_indices, metadata.paged_kv_last_page_len)
        if any(
            tensor.device != self._table.device
            or tensor.dtype != torch.int32
            or tensor.ndim != 1
            or not tensor.is_contiguous()
            for tensor in inputs
        ):
            raise ValueError("XQA paged input must be a contiguous 1D int32 vector on the planned CUDA device")
        if inputs[0].numel() != self._batch_size + 1 or inputs[2].numel() != self._batch_size:
            raise ValueError("XQA paging shapes differ from the planned batch")
        addresses = tuple(tensor.data_ptr() for tensor in inputs)
        if graph_mode and self._metadata_addresses is not None and addresses != self._metadata_addresses:
            raise RuntimeError("XQA graph paging input addresses changed")
        self._metadata_addresses = addresses
        convert_xqa_pages(*inputs, self._table, self._lengths, self._page_size)

    def run(
        self,
        query: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor],
        workspace: torch.Tensor,
        softmax_scale: float,
        key_scale: float,
        value_scale: float,
    ) -> torch.Tensor:
        from flashinfer.decode import xqa_batch_decode_with_kv_cache

        if self._metadata_addresses is None:
            raise RuntimeError("XQA decode must be prepared before execution")
        # Immutable cache geometry/device/dtype are validated once by the
        # native backend's constructor/bind boundary, rather than every layer.
        # The SM90 wrapper forwards Q as a raw packed pointer without Q strides.
        # Fused QKV projections can leave token-stride gaps; materialize inside
        # capture rather than allowing reads into the adjacent K/V projections.
        query = query.contiguous()
        return xqa_batch_decode_with_kv_cache(
            query,
            cache,
            workspace,
            self._table,
            self._lengths,
            self._table.size(1) * self._page_size,
            bmm1_scale=softmax_scale * key_scale,
            bmm2_scale=value_scale,
            kv_layout="NHD",
            enable_pdl=True,
        )
