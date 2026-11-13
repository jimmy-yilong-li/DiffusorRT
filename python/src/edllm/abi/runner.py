"""TypedBackendRunner — drive the existing backend from a ForwardRequest alone.

This is the sufficiency proof for the contract, and it is proven against the
REAL MLXBackendRunner (see tests), not a fake. If a field is missing from
ForwardRequest, the bit-identity test fails here while it is still cheap to fix.

The logits array is registered as a BUFFER handle on the response rather than
copied: v0 fixes the contract, and the ownership field already encodes what
Phase 1 needs for zero-copy transfer.

Model config is MODEL-LIFETIME state: the forward needs hidden_size /
attention_heads / rms_norm_eps / … and a per-step request must not carry them
(the C ABI mirror of that mistake is DiffusorRTForwardRequest growing thirty
fields). The runner therefore takes model_config at CONSTRUCTION — the same
moment a native engine binds a loaded package — and the request supplies only
per-step data. MaskGeometry crosses as a struct and is translated here to the
{"prompt_len", "block_size"} mapping the backend validates strictly.
"""
from __future__ import annotations

from typing import Any

from .buffers import describe_array
from .enums import AttentionMaskKind
from .forward import AbiStatus, ForwardRequest, ForwardResponse
from .handles import NULL_HANDLE, HandleKind, HandleRegistry
from .plan_codec import decode_plan

_MASK_KIND_NAMES = {
    int(AttentionMaskKind.BIDIRECTIONAL): "bidirectional",
    int(AttentionMaskKind.D2F_HYBRID): "d2f_hybrid",
    int(AttentionMaskKind.FULL): "full",
}


class TypedBackendRunner:
    __slots__ = ("_registry", "_inner", "_model_config")

    def __init__(self, *, registry: HandleRegistry, inner: Any | None = None,
                 model_config: dict | None = None) -> None:
        self._registry = registry
        self._inner = inner
        self._model_config = model_config

    def _resolve_inner(self):
        if self._inner is None:
            from ..backends.backend_runner import MLXBackendRunner

            self._inner = MLXBackendRunner()
        return self._inner

    def _optional(self, token: int, kind: HandleKind):
        if token == NULL_HANDLE:
            return None
        return self._registry.resolve(token, kind=kind)

    def execute_window_typed(self, request: ForwardRequest) -> ForwardResponse:
        weights = self._registry.resolve(
            request.weights_handle, kind=HandleKind.WEIGHTS)
        graph = self._optional(request.graph_handle, HandleKind.GRAPH)
        kv_cache = self._optional(request.kv_cache_handle, HandleKind.KV_CACHE_STATE)
        state_cache = self._optional(
            request.state_cache_handle, HandleKind.STATE_CACHE_MANAGER)
        kernels = self._optional(request.kernel_table_handle, HandleKind.KERNEL_TABLE)
        if self._model_config is None:
            raise ValueError(
                "TypedBackendRunner has no model_config; model-lifetime config "
                "is bound at construction, not carried per request"
            )

        mask_kind = _MASK_KIND_NAMES[int(request.session.attention_mask_kind)]
        model_config = dict(self._model_config)
        model_config["attention_mask_kind"] = mask_kind
        model_config["mlx_diffusion_block_size"] = int(request.session.block_size)
        # Presence is meaningful: a supplied geometry is translated and passed
        # through; the backend itself rejects it for non-d2f_hybrid kinds (no
        # silently-dead arguments). Absent stays absent.
        mask_geometry = None
        if request.mask_geometry is not None:
            mask_geometry = {
                "prompt_len": int(request.mask_geometry.prompt_len),
                "block_size": int(request.mask_geometry.block_size),
            }
        result = self._resolve_inner().execute_window(
            plan=decode_plan(request.plan),
            weights=weights,
            model_config=model_config,
            token_ids=list(request.token_ids),
            attention_mask_kind=mask_kind,
            kv_cache_state=kv_cache,
            graph=graph,
            state_cache_manager=state_cache,
            masked_positions=(
                None if request.masked_positions is None
                else list(request.masked_positions)),
            mask_geometry=mask_geometry,
            kernels=kernels,
        )
        response = ForwardResponse(
            status=AbiStatus.OK,
            logits=describe_array(result.logits, registry=self._registry),
            positions=tuple(result.positions),
            forwarded_rows=(
                None if result.forwarded_rows is None
                else int(result.forwarded_rows)),
            error_code=0,
        )
        response.validate_against(
            request, vocab_size=int(model_config["vocab_size"]))
        return response
