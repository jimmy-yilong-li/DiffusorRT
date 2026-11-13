"""Exact-D2F scheduling over the native K/V owner.

This adapter owns scheduling metadata only: prompt identity, frozen-block order,
the current read lease, and the capture transactions opened for one forward.
It never retains a committed K/V tensor.  Persistent payloads, publication,
eviction, leases, byte accounting, and release all belong to
``NativeKVManager`` and its device backend.

K2-3 removed the old Python-owned exact-D2F strategy. Keeping the ownership
switch separate from the forward math was the K2-2 boundary: the forward still
asks for cached parts and hands over freshly projected slices at the same
points as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .kv_backend import ContiguousKV
from .kv_manager import (
    KVCapturePlan,
    KVCaptureRequest,
    KVIdentity,
    KVPageState,
    KVTensorSpec,
    Lease,
    ReleaseOutcome,
)
from .kv_slices import ExactKVSliceMode
from .mlx_state_cache import D2FForwardPlan


@dataclass(frozen=True, slots=True)
class NativeD2FIdentityContext:
    """Decode-constant identity fields supplied by the engine.

    Span tokens, freeze state, and state epoch are added by the strategy at the
    moment a prompt or frozen block is captured.  ``cache_salt`` is deliberately
    generation-local: K2-3 removed the old cross-request/session owners without
    admitting reuse on the native identity. Any future reuse requires a separate
    gate on this identity rather than a fallback owner.
    """

    package_digest: str
    weights_digest: str
    tokenizer_identity: str
    graph_digest: str
    attention_mask_kind: str
    model_geometry: tuple[tuple[str, int], ...]
    layer_set: tuple[int, ...]
    dtype: str
    quantization_identity: str
    policy_version: str
    cache_salt: str
    privacy_scope: str = "session"


@dataclass(slots=True)
class _Capture:
    transaction: str
    plan: KVCapturePlan
    identity: KVIdentity
    label: str
    block_index: int | None
    produced: dict[int, tuple[Any, Any]] = field(default_factory=dict)


class NativeD2FExactKVCacheStrategy:
    """D2F scheduling facade whose only persistent K/V owner is native core."""

    def __init__(
        self,
        *,
        manager: Any,
        session: str,
        identity_context: NativeD2FIdentityContext,
        prompt_len: int,
        block_size: int,
        window_mode: str,
        num_layers: int,
        tensor_spec: KVTensorSpec,
        slice_mode: ExactKVSliceMode = ExactKVSliceMode.OFF,
    ) -> None:
        if prompt_len < 0:
            raise ValueError("prompt_len must be >= 0")
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if window_mode not in {"single_block", "tail"}:
            raise ValueError("window_mode must be 'single_block' or 'tail'")
        if slice_mode is not ExactKVSliceMode.OFF:
            raise ValueError(
                "the native K/V owner currently admits exact_kv_slice_mode='off' "
                "only; legacy paged slice ownership is not retained as a fallback"
            )
        if tuple(identity_context.layer_set) != tuple(range(num_layers)):
            raise ValueError(
                "the native exact-D2F identity layer set must cover every layer"
            )
        if tensor_spec.dtype != identity_context.dtype:
            raise ValueError("the tensor spec and native identity disagree on dtype")
        if tensor_spec.quantization_identity != identity_context.quantization_identity:
            raise ValueError(
                "the tensor spec and native identity disagree on quantization"
            )

        self._manager = manager
        self._session = session
        self._identity_context = identity_context
        self._prompt_len = int(prompt_len)
        self._block_size = int(block_size)
        self._window_mode = window_mode
        self._num_layers = int(num_layers)
        self._tensor_spec = tensor_spec
        self._slice_mode = slice_mode

        self._session_open = False
        self._prompt_signature: tuple[int, ...] | None = None
        self._prompt_identity: KVIdentity | None = None
        self._block_identities: dict[int, KVIdentity] = {}
        self._populated = False
        self._captured_count = 0
        self._pending_blocks: list[int] = []
        self._mutable_state_epoch = 0

        self._captures: dict[str, _Capture] = {}
        self._read_lease: Lease | None = None
        self._read_parts: dict[int, tuple[ContiguousKV, ...]] = {}
        # Handles that a backend refused before the native manager adopted
        # them.  They stay reachable until a retry hands them back; counting a
        # cleanup failure without retaining its payload would be fictional debt.
        self._orphaned_payloads: list[Any] = []
        self._forward_open = False

        self._full_forward_count = 0
        self._active_only_count = 0
        self._kv_proj_rows_skipped = 0
        self._tx_committed_blocks = 0
        self._tx_publish_failures = 0

    @property
    def prompt_len(self) -> int:
        return self._prompt_len

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def window_mode(self) -> str:
        return self._window_mode

    @property
    def slice_mode(self) -> ExactKVSliceMode:
        return self._slice_mode

    @property
    def frozen_block_count(self) -> int:
        return self._captured_count + len(self._pending_blocks)

    @property
    def mutable_state_epoch(self) -> int:
        return self._mutable_state_epoch

    def is_populated(self) -> bool:
        return self._populated

    def freeze_block(self, block_index: int) -> None:
        block_index = int(block_index)
        if block_index < self.frozen_block_count:
            return
        expected = self.frozen_block_count
        if block_index != expected:
            raise ValueError(
                f"d2f blocks must freeze in order: expected {expected}, got "
                f"{block_index}"
            )
        self._pending_blocks.append(block_index)

    def observe_commit(self, *, position: int, token_id: int) -> None:
        del position, token_id
        self._mutable_state_epoch += 1

    def _prefix_of(self, token_ids: list[int]) -> tuple[int, ...]:
        return tuple(int(token) for token in token_ids[: self._prompt_len])

    def _active_end(self, active_start: int, seq_len: int) -> int:
        if self._window_mode == "tail":
            return seq_len
        return min(active_start + self._block_size, seq_len)

    def _identity(
        self,
        token_ids: list[int],
        *,
        start: int,
        end: int,
        freeze_state: str,
        state_epoch: int,
    ) -> KVIdentity:
        context = self._identity_context
        return KVIdentity(
            package_digest=context.package_digest,
            weights_digest=context.weights_digest,
            tokenizer_identity=context.tokenizer_identity,
            graph_digest=context.graph_digest,
            attention_mask_kind=context.attention_mask_kind,
            model_geometry=context.model_geometry,
            layer_set=context.layer_set,
            dtype=context.dtype,
            quantization_identity=context.quantization_identity,
            token_ids=tuple(int(token) for token in token_ids[start:end]),
            position_start=start,
            position_end=end,
            freeze_state=freeze_state,
            state_epoch=state_epoch,
            policy_version=context.policy_version,
            cache_salt=context.cache_salt,
            privacy_scope=context.privacy_scope,
        )

    def _ensure_session(self, token_ids: list[int]) -> None:
        if self._session_open:
            return
        if not token_ids:
            raise ValueError("an exact-D2F session must contain at least one token")
        identity = self._identity(
            token_ids,
            start=0,
            end=len(token_ids),
            freeze_state="session",
            state_epoch=0,
        )
        self._manager.open_session(self._session, identity, reserve_bytes=0)
        self._session_open = True

    def _capture_plan(
        self,
        prepared: dict[str, _Capture],
        token_ids: list[int],
        *,
        label: str,
        start: int,
        end: int,
        committed_state: KVPageState,
        block_index: int | None,
    ) -> _Capture:
        identity = self._identity(
            token_ids,
            start=start,
            end=end,
            freeze_state=(
                "prompt" if block_index is None else f"frozen_block:{block_index}"
            ),
            state_epoch=(0 if block_index is None else self._mutable_state_epoch),
        )
        plan = KVCapturePlan(
            requests=tuple(
                KVCaptureRequest(layer, start, end)
                for layer in range(self._num_layers)
            ),
            expected_layers=tuple(range(self._num_layers)),
            spec=self._tensor_spec,
            committed_state=committed_state,
        )
        # Publish the strategy-side record BEFORE native admission.  A dict
        # insertion after begin_transaction can allocate and fail, which would
        # leave a real reservation with no transaction id this facade could
        # abort.  Replacing the already-present string field after admission is
        # allocation-free bookkeeping over an object we can already name.
        capture = _Capture(
            transaction="",
            plan=plan,
            identity=identity,
            label=label,
            block_index=block_index,
        )
        prepared[label] = capture
        try:
            capture.transaction = self._manager.begin_transaction(
                self._session, identity, plan, plan.total_bytes
            )
        except BaseException:
            prepared.pop(label, None)
            raise
        return capture

    def _prepare_captures(
        self,
        token_ids: list[int],
        *,
        include_prompt: bool,
        pending: list[int],
    ) -> None:
        prepared: dict[str, _Capture] = {}
        try:
            if include_prompt and self._prompt_len > 0:
                self._capture_plan(
                    prepared,
                    token_ids,
                    label="prompt",
                    start=0,
                    end=self._prompt_len,
                    committed_state=KVPageState.COMMITTED_PROMPT,
                    block_index=None,
                )
            for block_index in pending:
                start = self._prompt_len + block_index * self._block_size
                end = min(start + self._block_size, len(token_ids))
                if end <= start:
                    raise ValueError(
                        f"pending block {block_index} has no rows in a sequence "
                        f"of length {len(token_ids)}"
                    )
                label = f"block:{block_index}"
                self._capture_plan(
                    prepared,
                    token_ids,
                    label=label,
                    start=start,
                    end=end,
                    committed_state=KVPageState.COMMITTED_FROZEN_BLOCK,
                    block_index=block_index,
                )
        except BaseException as primary:
            unresolved: dict[str, _Capture] = {}
            for label, capture in reversed(tuple(prepared.items())):
                try:
                    self._manager.abort(capture.transaction)
                except Exception as cleanup:
                    unresolved[label] = capture
                    primary.add_note(
                        f"aborting prepared K/V transaction "
                        f"{capture.transaction!r} failed: {cleanup}"
                    )
            # An abort failure remains named so a checked close can retry it.
            # Dropping these records would leave native reservations reachable
            # only through manager shutdown and falsely report a clean facade.
            self._captures = unresolved
            raise
        self._captures = prepared

    def _full_plan(
        self,
        token_ids: list[int],
        *,
        signature: tuple[int, ...],
        preserve_frozen: bool,
    ) -> D2FForwardPlan:
        frozen_count = self.frozen_block_count if preserve_frozen else 0
        self._captured_count = 0
        self._pending_blocks = list(range(frozen_count))
        self._prompt_identity = None
        self._block_identities = {}
        self._populated = self._prompt_len == 0
        self._prompt_signature = signature
        pending = list(self._pending_blocks)
        self._prepare_captures(
            token_ids, include_prompt=self._prompt_len > 0, pending=pending
        )
        self._full_forward_count += 1
        self._forward_open = True
        active_block = self.frozen_block_count
        active_start = self._prompt_len + active_block * self._block_size
        return D2FForwardPlan(
            is_full_forward=True,
            prompt_len=self._prompt_len,
            block_size=self._block_size,
            seq_len=len(token_ids),
            cached_len=0,
            boundary=0,
            active_block=active_block,
            active_start=active_start,
            active_end=self._active_end(active_start, len(token_ids)),
            pending_blocks=pending,
            kv_len=len(token_ids),
        )

    def begin_forward(self, *, token_ids: list[int]) -> D2FForwardPlan:
        if self._forward_open:
            raise RuntimeError("the previous exact-D2F forward is still open")
        if self._captures or self._read_lease is not None or self._orphaned_payloads:
            raise RuntimeError(
                "the previous exact-D2F forward has unresolved cleanup debt"
            )
        seq_len = len(token_ids)
        if seq_len <= self._prompt_len:
            raise ValueError(
                "d2f exact KV cache requires a generated region: got "
                f"seq_len={seq_len} <= prompt_len={self._prompt_len}"
            )
        self._ensure_session(token_ids)
        signature = self._prefix_of(token_ids)
        prompt_changed = (
            self._prompt_signature is not None
            and signature != self._prompt_signature
        )
        if not self._populated or prompt_changed:
            return self._full_plan(
                token_ids,
                signature=signature,
                preserve_frozen=not prompt_changed,
            )

        pending = list(self._pending_blocks)
        cached_len = self._prompt_len + self._captured_count * self._block_size
        active_block = self.frozen_block_count
        active_start = self._prompt_len + active_block * self._block_size
        if active_start > seq_len or (active_start == seq_len and not pending):
            raise ValueError(
                "every generated block is frozen; there is no active block to "
                f"decode (seq_len={seq_len}, frozen={self.frozen_block_count})"
            )

        candidates: list[tuple[KVIdentity, int, int]] = []
        if self._prompt_len:
            if self._prompt_identity is None:
                raise RuntimeError("the native prompt identity is missing")
            candidates.append((self._prompt_identity, 0, self._prompt_len))
        for block_index in range(self._captured_count):
            identity = self._block_identities.get(block_index)
            if identity is None:
                raise RuntimeError(
                    f"the native identity for frozen block {block_index} is missing"
                )
            start = self._prompt_len + block_index * self._block_size
            candidates.append((identity, start, start + self._block_size))

        read_plan, lease = self._manager.resolve_read_plan(
            self._session,
            tuple(range(self._num_layers)),
            cached_len,
            tuple(candidates),
        )
        # The lease becomes facade-owned the instant the manager returns it.
        # Recording it only after materialisation/transaction preparation made
        # both refusal paths lose the sole retry handle when release itself
        # failed -- cleanup debt reduced to an exception with no object behind
        # it, the exact defect K2-1 corrected for commit receipts.
        self._read_lease = lease
        expected_reused = self._num_layers * cached_len
        if (
            read_plan.reused_rows != expected_reused
            or read_plan.request_local_rows != 0
        ):
            self._release_read()
            return self._full_plan(
                token_ids, signature=signature, preserve_frozen=True
            )
        try:
            read_parts = getattr(self._manager.backend, "read_parts", None)
            if read_parts is None:
                # Compatibility with installed providers predating borrowed
                # read parts. Ownership and exact results are identical; those
                # providers still assemble the cached prefix themselves.
                self._read_parts = {
                    layer: (part,) if part.rows else ()
                    for layer, part in self._manager.backend.materialize(read_plan).items()
                }
            else:
                self._read_parts = read_parts(read_plan)
            self._prepare_captures(
                token_ids, include_prompt=False, pending=pending
            )
        except BaseException as primary:
            try:
                self._release_read()
            except BaseException as cleanup:
                primary.add_note(f"releasing native K/V read lease failed: {cleanup}")
            raise

        self._active_only_count += 1
        self._kv_proj_rows_skipped += cached_len
        self._forward_open = True
        active_end = self._active_end(active_start, seq_len)
        boundary = cached_len if pending else max(0, cached_len - 1)
        return D2FForwardPlan(
            is_full_forward=False,
            prompt_len=self._prompt_len,
            block_size=self._block_size,
            seq_len=seq_len,
            cached_len=cached_len,
            boundary=boundary,
            active_block=active_block,
            active_start=active_start,
            active_end=active_end,
            pending_blocks=pending,
            kv_len=active_end,
        )

    def _validated_layer(self, layer_index: int) -> int:
        if type(layer_index) is not int:
            raise ValueError(
                "exact K/V layer_index must be a non-boolean int; got "
                f"{layer_index!r}"
            )
        layer = layer_index
        if layer not in range(self._num_layers):
            raise ValueError(f"layer {layer} is outside [0, {self._num_layers})")
        return layer

    def _record(
        self, label: str, *, layer_index: int, key: Any, value: Any
    ) -> None:
        capture = self._captures.get(label)
        if capture is None:
            raise ValueError(f"no admitted native capture exists for {label}")
        layer = self._validated_layer(layer_index)
        if layer in capture.produced:
            raise ValueError(f"{label} layer {layer} was captured twice")
        capture.produced[layer] = (key, value)

    def capture_prompt_layer(
        self, *, layer_index: int, key: Any, value: Any
    ) -> None:
        self._record("prompt", layer_index=layer_index, key=key, value=value)

    def capture_block_layer(
        self,
        *,
        block_index: int,
        layer_index: int,
        key: Any,
        value: Any,
    ) -> None:
        self._record(
            f"block:{int(block_index)}",
            layer_index=layer_index,
            key=key,
            value=value,
        )

    def cached_kv_parts(self, *, layer_index: int) -> tuple[list[Any], list[Any]]:
        layer = self._validated_layer(layer_index)
        cached_rows = self._prompt_len + self._captured_count * self._block_size
        # A prompt-free decode before its first block freezes has a legal
        # zero-row read plan. Expose an empty cache, never a None placeholder.
        if cached_rows == 0:
            return [], []
        parts = self._read_parts.get(layer)
        if parts is None:
            raise KeyError(f"the native read plan has no read parts for layer {layer}")
        rows = sum(part.rows for part in parts)
        if rows != cached_rows:
            raise ValueError(
                f"native layer {layer} exposes {rows} rows, expected "
                f"{cached_rows}"
            )
        return [part.key for part in parts], [part.value for part in parts]

    def _release_read(self) -> None:
        lease = self._read_lease
        self._read_parts = {}
        if lease is not None:
            self._manager.release(lease)
            self._read_lease = None

    def _abort_captures(self) -> None:
        failures: list[BaseException] = []
        for label, capture in reversed(tuple(self._captures.items())):
            try:
                if self._manager.transaction_is_open(capture.transaction):
                    self._manager.abort(capture.transaction)
                self._captures.pop(label, None)
            except BaseException as exc:  # cleanup must remain visible
                failures.append(exc)
        if failures:
            raise RuntimeError(
                f"failed to abort {len(failures)} native K/V transaction(s)"
            ) from failures[0]

    def _release_unstaged(self, fragments: tuple[Any, ...]) -> None:
        payloads = [
            payload
            for fragment in fragments
            for payload in (fragment.slice_.key, fragment.slice_.value)
        ]
        if not payloads:
            return
        try:
            requested = sum(
                int(self._manager.backend.physical_bytes(payload))
                for payload in payloads
            )
            outcome = self._manager.backend.free(*payloads)
        except BaseException:
            self._orphaned_payloads.extend(payloads)
            raise
        if (
            not isinstance(outcome, ReleaseOutcome)
            or not outcome.conserves(requested)
            or not outcome.is_atomic(requested)
        ):
            self._orphaned_payloads.extend(payloads)
            raise RuntimeError(
                "the native K/V backend returned an invalid release outcome "
                "while rolling back unstaged captures"
            )
        if outcome.failed:
            self._orphaned_payloads.extend(payloads)
            raise RuntimeError(
                "the native K/V backend failed to release unstaged captures"
            )
        # Deferred handles are now owned by the backend's sweep debt. Retaining
        # a second retry list here would eventually offer a handle that sweep
        # has already made stale.

    def _retry_orphaned_payloads(self) -> None:
        if not self._orphaned_payloads:
            return
        payloads = tuple(self._orphaned_payloads)
        requested = sum(
            int(self._manager.backend.physical_bytes(payload))
            for payload in payloads
        )
        outcome = self._manager.backend.free(*payloads)
        if (
            not isinstance(outcome, ReleaseOutcome)
            or not outcome.conserves(requested)
            or not outcome.is_atomic(requested)
            or outcome.failed
        ):
            raise RuntimeError(
                "the native K/V backend still cannot release orphaned captures"
            )
        self._orphaned_payloads = []

    def promote_pending(self) -> tuple[int, ...]:
        if not self._forward_open:
            raise RuntimeError("no exact-D2F forward is open")
        committed: list[int] = []
        try:
            for capture in tuple(self._captures.values()):
                expected = set(range(self._num_layers))
                if set(capture.produced) != expected:
                    raise RuntimeError(
                        f"{capture.label} captured layers "
                        f"{sorted(capture.produced)}, expected {sorted(expected)}"
                    )
                fragments = self._manager.backend.capture(
                    self._manager.capture_permit(capture.transaction),
                    capture.produced,
                )
                # Once adopted, raw arrays must not be retained here: keeping
                # them made a later manager-driven free look externally held
                # and converted ordinary publication into cleanup debt.
                capture.produced = {}
                staged = 0
                try:
                    for fragment in fragments:
                        self._manager.stage(capture.transaction, fragment)
                        staged += 1
                except BaseException as primary:
                    try:
                        self._release_unstaged(tuple(fragments[staged:]))
                    except BaseException as cleanup:
                        primary.add_note(
                            f"releasing unstaged native K/V captures failed: {cleanup}"
                        )
                    raise
                world = self._manager.commit(capture.transaction)
                if (
                    world.identity != capture.identity
                    or world.layers != tuple(range(self._num_layers))
                    or world.span != capture.plan.span
                    or world.nbytes != capture.plan.total_bytes
                ):
                    raise RuntimeError(
                        "the native manager committed a world that disagrees "
                        "with the admitted exact-D2F capture"
                    )
                if capture.block_index is None:
                    self._prompt_identity = capture.identity
                    self._populated = True
                else:
                    block_index = capture.block_index
                    if not self._pending_blocks or self._pending_blocks[0] != block_index:
                        raise RuntimeError(
                            f"native frozen block {block_index} is not the first "
                            f"pending block {self._pending_blocks}"
                        )
                    self._pending_blocks.pop(0)
                    self._captured_count = block_index + 1
                    self._block_identities[block_index] = capture.identity
                    self._tx_committed_blocks += 1
                    committed.append(block_index)
                self._captures.pop(capture.label, None)
            if self._prompt_len == 0:
                self._populated = True
            self._captures = {}
            self._release_read()
            self._forward_open = False
            return tuple(committed)
        except BaseException as primary:
            self._tx_publish_failures += 1
            try:
                self._abort_captures()
            except BaseException as cleanup:
                primary.add_note(f"aborting native K/V captures failed: {cleanup}")
            try:
                self._release_read()
            except BaseException as cleanup:
                primary.add_note(f"releasing native K/V read lease failed: {cleanup}")
            self._forward_open = False
            raise

    def abort_forward(self) -> None:
        if not self._forward_open and not self._captures and self._read_lease is None:
            return
        primary: BaseException | None = None
        try:
            self._abort_captures()
        except BaseException as exc:
            primary = exc
        try:
            self._release_read()
        except BaseException as exc:
            if primary is None:
                primary = exc
            else:
                primary.add_note(f"releasing native K/V read lease failed: {exc}")
        self._forward_open = False
        if primary is not None:
            raise primary

    def close_decode(self) -> None:
        if not self._session_open:
            self._retry_orphaned_payloads()
            return
        primary: BaseException | None = None
        try:
            self.abort_forward()
        except BaseException as exc:
            primary = exc
        try:
            self._retry_orphaned_payloads()
        except BaseException as exc:
            if primary is None:
                primary = exc
            else:
                primary.add_note(f"retrying orphaned K/V releases failed: {exc}")
        try:
            self._manager.close_session(self._session)
        except BaseException as exc:
            if primary is None:
                primary = exc
            else:
                primary.add_note(f"closing native K/V session failed: {exc}")
        else:
            self._session_open = False
            self._captures = {}
            self._read_lease = None
            self._read_parts = {}
        if primary is not None:
            raise primary

    def manager_metrics(self) -> dict[str, int]:
        return self._manager.metrics()

    def backend_metrics(self) -> dict[str, int]:
        return dict(self._manager.backend.movement())

    def manager_trace(self) -> dict[str, Any]:
        return self._manager.trace_document()

    def metrics(self) -> dict[str, int | bool | str]:
        return {
            "mlx_d2f_exact_kv_cache_enabled": True,
            "mlx_d2f_exact_kv_populated": self.is_populated(),
            "mlx_d2f_exact_kv_prompt_len": self._prompt_len,
            "mlx_d2f_exact_kv_block_size": self._block_size,
            "mlx_d2f_exact_kv_full_forward_count": self._full_forward_count,
            "mlx_d2f_exact_kv_active_only_count": self._active_only_count,
            "mlx_kv_proj_rows_skipped": self._kv_proj_rows_skipped,
            "mlx_d2f_exact_kv_frozen_blocks": self._captured_count,
            "mlx_d2f_exact_kv_pending_blocks": len(self._pending_blocks),
            "mlx_d2f_exact_kv_freeze_tx_committed_blocks": (
                self._tx_committed_blocks
            ),
            "mlx_d2f_exact_kv_freeze_tx_publish_failures": (
                self._tx_publish_failures
            ),
            "kv_owner": "native.core",
        }


__all__ = ["NativeD2FExactKVCacheStrategy", "NativeD2FIdentityContext"]
