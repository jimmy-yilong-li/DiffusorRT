"""The per-step request/response pair.

Carries per-step DATA only: session config holds decode shape, engine config
holds numerics, and policy tuning never crosses at all (policy_handle). Every
opaque object execute_window takes has its own kinded handle, because
kv_cache_state and state_cache_manager are different objects with different
lifetimes and collapsing them loses that.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .buffers import TensorDesc
from .handles import NULL_HANDLE
from .lifetimes import SessionDecodeConfig, UNSET
from .plan_codec import PlanV1


class AbiStatus(IntEnum):
    OK = 0
    VERSION_MISMATCH = 1
    INVALID_ARGUMENT = 2
    UNSUPPORTED = 3
    INTERNAL_ERROR = 4


@dataclass(frozen=True, slots=True)
class MaskGeometry:
    """Per-request mask geometry — consumed only by the d2f_hybrid mask kind."""
    prompt_len: int
    block_size: int
    mask_token_id: int


@dataclass(frozen=True, slots=True)
class ForwardRequest:
    abi_version: tuple[int, int, int]
    session: SessionDecodeConfig
    plan: PlanV1
    weights_handle: int
    graph_handle: int
    kv_cache_handle: int
    state_cache_handle: int
    kernel_table_handle: int
    policy_handle: int
    token_ids: tuple[int, ...]
    masked_positions: tuple[int, ...] | None
    mask_geometry: MaskGeometry | None

    def __post_init__(self) -> None:
        from . import ABI_VERSION

        if tuple(self.abi_version) != ABI_VERSION:
            raise ValueError(
                f"abi_version {tuple(self.abi_version)} != runtime {ABI_VERSION}"
            )
        if self.weights_handle == NULL_HANDLE:
            raise ValueError("weights_handle must not be NULL_HANDLE")
        limit = len(self.token_ids)
        window_end = self.plan.window_end
        if window_end != UNSET:
            if not (0 <= window_end <= limit):
                raise ValueError(
                    f"window_end {window_end} outside token_ids boundary "
                    f"[0, {limit}]"
                )
            scope_limit = window_end
        else:
            scope_limit = limit
        if self.masked_positions is not None:
            for position in self.masked_positions:
                if not (0 <= position < limit):
                    raise ValueError(
                        f"masked_positions entry {position} outside token_ids "
                        f"[0, {limit})"
                    )
        plan_tracks = (
            ("active_positions", self.plan.active_positions),
            ("dirty_positions", self.plan.dirty_positions),
            ("remask_positions", self.plan.remask_positions),
            ("frozen_positions", self.plan.frozen_positions),
            ("dependency_risk_positions",
             self.plan.dependency_risk_positions),
            ("cache_action_positions", self.plan.cache_action_positions),
            ("reuse_source_positions", self.plan.reuse_source_positions),
        )
        for field, positions in plan_tracks:
            for position in positions:
                if not (0 <= position < limit):
                    raise ValueError(
                        f"{field} entry {position} outside token_ids canvas "
                        f"[0, {limit})"
                    )
        # forward_scope ORDER is caller-defined: the runtime honours whatever
        # order it carries (tests/runtime/test_execute_window_parity.py:252
        # exercises (6, 2, 8)), so the ABI checks membership, not monotonicity.
        for position in self.plan.forward_scope:
            if not (0 <= position < scope_limit):
                boundary = (
                    f"window_end [0, {scope_limit})"
                    if window_end != UNSET
                    else f"token_ids [0, {scope_limit})"
                )
                raise ValueError(
                    f"forward_scope position {position} outside {boundary}"
                )


@dataclass(frozen=True, slots=True)
class ForwardResponse:
    status: int
    logits: TensorDesc
    positions: tuple[int, ...]
    forwarded_rows: int | None
    error_code: int

    def __post_init__(self) -> None:
        if self.status != AbiStatus.OK:
            return
        if len(self.logits.shape) != 2:
            raise ValueError(
                f"logits must be 2-D [rows, vocab]; got shape "
                f"{self.logits.shape}"
            )
        rows = self.logits.shape[0]
        if rows != len(self.positions):
            raise ValueError(
                f"logits rows {rows} != positions {len(self.positions)}; the "
                "row-to-position map must be total"
            )

    def validate_against(self, request: ForwardRequest, *, vocab_size: int) -> None:
        """The cross-check __post_init__ cannot do: an OK response must agree
        with the request that produced it — positions identical to the request's
        forward_scope in order, and logits width equal to the model's vocab."""
        if self.status != AbiStatus.OK:
            return
        expected = tuple(request.plan.forward_scope)
        if tuple(self.positions) != expected:
            raise ValueError(
                f"positions {tuple(self.positions)} != request forward_scope "
                f"{expected}; the row-to-position map must be identical, in order"
            )
        if self.logits.shape[1] != int(vocab_size):
            raise ValueError(
                f"logits width {self.logits.shape[1]} != vocab_size "
                f"{int(vocab_size)}"
            )
