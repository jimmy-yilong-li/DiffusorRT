"""The private native K/V owner, acquired through one fail-closed seam.

This module is public framework glue, not a second K/V implementation. It owns
no cache policy and stores no tensor: every lifecycle operation goes to the
compiled manager, while the selected device backend owns the K/V buffers.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from .kv_backend import KVDeviceBackend, open_backend
from .kv_manager import (
    AdmissionRequired,
    CapturePermit,
    CommittedWorld,
    KVCacheManager,
    KVCapturePlan,
    KVFragment,
    KVIdentity,
    KVOperation,
    KVPageState,
    KVReadPlan,
    Lease,
    TRACE_SCHEMA_VERSION,
)
from .kv_slices import KVSlice
from .kv_trace import parse_wire_detail


class NativeKVUnavailable(RuntimeError):
    """The private runtime or requested device backend cannot be acquired."""


_REQUIRED_NATIVE_KV_METHODS = (
    "kv_manager_create",
    "kv_manager_shutdown",
    "kv_manager_destroy",
    "kv_open_session",
    "kv_close_session",
    "kv_begin_transaction",
    "kv_capture_permit",
    "kv_permit_claim",
    "kv_stage",
    "kv_commit",
    "kv_commit_ack",
    "kv_abort",
    "kv_transaction_state",
    "kv_session_is_open",
    "kv_resolve_read_plan",
    "kv_release",
    "kv_metrics",
    "kv_trace",
)


def _validate_native_core(core: Any) -> None:
    """Refuse an older or partial private extension at acquisition time."""

    missing = tuple(
        name
        for name in _REQUIRED_NATIVE_KV_METHODS
        if not callable(getattr(core, name, None))
    )
    if missing:
        raise NativeKVUnavailable(
            "the private native K/V extension is incomplete; missing "
            + ", ".join(missing)
        )


def _typed_detail(encoded: str) -> dict[str, Any]:
    """Turn the native trace wire string back into ``KVOperation`` values.

    The raw C trace deliberately keeps ``detail`` as a string in schema v3.
    ``KVCacheManager.trace()``, however, promises the typed operation contract
    consumed by ``validate_trace``. Keeping the whole wire value under a
    ``native_detail`` key made that method look typed while dropping every
    transaction, layer, byte count and boolean the validator needs.
    """
    return parse_wire_detail(encoded)


def _identity_dict(identity: KVIdentity) -> dict[str, Any]:
    return {
        "package_digest": identity.package_digest,
        "weights_digest": identity.weights_digest,
        "tokenizer_identity": identity.tokenizer_identity,
        "graph_digest": identity.graph_digest,
        "attention_mask_kind": identity.attention_mask_kind,
        "model_geometry": ";".join(
            f"{name}={value}" for name, value in identity.model_geometry
        ),
        "layer_set": ",".join(str(layer) for layer in identity.layer_set),
        "dtype": identity.dtype,
        "quantization_identity": identity.quantization_identity,
        "token_ids": ",".join(str(token) for token in identity.token_ids),
        "position_start": identity.position_start,
        "position_end": identity.position_end,
        "freeze_state": identity.freeze_state,
        "state_epoch": identity.state_epoch,
        "policy_version": identity.policy_version,
        "cache_salt": identity.cache_salt,
        "privacy_scope": identity.privacy_scope,
    }


def _plan_dict(plan: KVCapturePlan) -> dict[str, Any]:
    start, end = plan.span
    committed = {
        KVPageState.COMMITTED_PROMPT: "prompt",
        KVPageState.COMMITTED_FROZEN_BLOCK: "frozen_block",
    }.get(plan.committed_state)
    if committed is None:
        raise ValueError(
            f"{plan.committed_state.value} is not a committed page state"
        )
    return {
        "layers": list(plan.expected_layers),
        "start": start,
        "end": end,
        "dtype": plan.spec.dtype,
        "quantization_identity": plan.spec.quantization_identity,
        "kv_heads": plan.spec.kv_heads,
        "head_dim": plan.spec.head_dim,
        "committed_state": committed,
    }


class _SessionState(Enum):
    """A session is OPEN, or CLOSING until its idempotent close succeeds.

    THE FACADE HAD NO SESSION STATE, and that is why closing one could not be
    made honest. Closing cascades: the owner aborts every transaction, releases
    every lease and drops the arena, and the binding still holds a receipt for
    each committed-but-unacknowledged commit. Native `CloseSession` cannot be
    made allocation-atomic without also making `AbortLocked` and `ReleaseLocked`
    allocation-free -- they build detail strings -- so "either the whole close
    happened or none of it" is not a promise this stack can keep.

    What it CAN keep is this: from the first call the session refuses new work,
    and repeated closes drive the remaining cleanup to completion. A failure
    part-way leaves the session CLOSING with exactly the work that is left
    still visible in the metrics, which is a state a caller can act on.

    CLOSING IS NOT PERMANENT, and an earlier comment here said it was. The
    frozen corpus makes `session_reopened_after_close` legal -- "one session
    name may be used again once its lifetime has ended" -- and the state is
    dropped when the close finally succeeds, so the name is free again. What is
    forbidden is reopening BEFORE that: until the owner has released the
    session, the name still refers to something that exists.
    """

    OPEN = "open"
    CLOSING = "closing"
    # An open call and its authoritative follow-up query both failed.  Calling
    # this OPEN would admit work against a session the owner may never have
    # created; dropping it could strand one the owner did create.
    NEEDS_RECONCILIATION = "needs_reconciliation"


@dataclass(slots=True)
class _Transaction:
    """One transaction, and the state it is actually in.

    ONE TABLE WITH A STATE, not two tables and a migration between them. The
    previous shape kept open transactions in one dict and
    committed-awaiting-acknowledgement ones in another, and moving a record
    across meant a pop plus an insert -- which is two failure modes for one
    transition:

      * the move happened BEFORE the native call, so an ordinary commit refusal
        (a layer never staged) left the owner's transaction open and the facade
        calling it committed. Measured: owner_open=True, stage=KeyError, and a
        phantom pending receipt that survived abort.
      * the insert can raise. Measured with a refusing dict: MemoryError with
        the native commit never attempted, the owner still open, and the record
        gone from both tables.

    Mutating the state fields on a record that already exists allocates nothing
    and cannot half-happen, so the transition is atomic by construction. The
    dataclass is no longer frozen for exactly that reason: `replace()` would
    allocate, which is the failure this design removes. If both an operation and
    its authoritative follow-up query fail, `needs_reconciliation` keeps the
    record actionable while refusing to guess whether it is open or committed.
    """

    session: str
    identity: KVIdentity | None
    plan: KVCapturePlan
    total_bytes: int
    # True once the OWNER has committed and the typed result has not yet been
    # acknowledged. §17.3 makes such a transaction closed, so nothing may stage
    # into it or take a permit for it; only a commit retry may name it.
    awaiting_ack: bool = False
    # A native operation raised and the authoritative follow-up query also
    # failed. Keeping the previous ``awaiting_ack`` value is unsafe: the owner
    # may already have committed and retained a receipt. Until a later query
    # resolves the state, no operation may treat this record as open or discard
    # it during cleanup.
    needs_reconciliation: bool = False


class NativeCapturePermit(CapturePermit):
    """A typed plan plus the compiled manager's single-use permit."""

    __slots__ = ("_native_capsule",)

    def __init__(
        self,
        transaction: str,
        plan: KVCapturePlan,
        issuer: "NativeKVManager",
        native_capsule: Any,
    ) -> None:
        super().__init__(transaction, plan, issuer)
        self._native_capsule = native_capsule

    def claim(self) -> KVCapturePlan:
        self._issuer._claim_capture(self._transaction, self._native_capsule)
        self._claimed = True
        return self._plan


@dataclass(frozen=True, slots=True)
class NativeLease(Lease):
    """A public lease value that keeps the native RAII lease alive."""

    _native_capsule: Any = None
    _issuer: Any = field(default=None, compare=False, repr=False)


class NativeKVManager(KVCacheManager):
    """A typed Python facade over the single C++ production owner."""

    def __init__(
        self,
        core: Any,
        backend: KVDeviceBackend,
        *,
        session_reserved_bytes: int = -1,
        shared_cache_budget_bytes: int = -1,
        staging_headroom_bytes: int = -1,
    ) -> None:
        self._core = core
        self._backend = backend
        self._native = core.kv_manager_create(
            session_reserved_bytes,
            shared_cache_budget_bytes,
            staging_headroom_bytes,
            backend,
        )
        self._transactions: dict[str, _Transaction] = {}
        self._sessions: dict[str, _SessionState] = {}
        self._closed = False
        self._destroyed = False
        # Whether a shutdown has completed WITHOUT reporting cleanup debt. A
        # shutdown that raised leaves this false so the cleanup can be retried.
        self._shutdown_reported = False
        # Persist evidence of boundary disagreement across a successful retry.
        # The detecting operation still raises, but sampled metrics must not
        # erase the only signal merely because reconciliation later succeeds.
        self._boundary_state_mismatches = 0
        self._reconciliation_failures = 0
        # ONE LOCK OVER THE NATIVE CALL AND THE BOOKKEEPING IT IMPLIES.
        #
        # Every mutating method used to do its native call and then update
        # `_transactions` with nothing between them but Python's own
        # scheduling. `close_session` made that fatal: it snapshotted the
        # surviving transactions, called native, then REPLACED the whole dict.
        # A `begin_transaction` that completed in between was admitted by the
        # owner and then erased from the facade by the write-back.
        #
        # Reproduced deterministically by interleaving the two: native reported
        # the transaction open and the facade reported it closed -- a
        # reservation the facade can no longer abort, commit or even name.
        #
        # RLock, not Lock: a native call can re-enter Python through the
        # release trampoline on this same thread. That trampoline calls the
        # BACKEND rather than this manager, so it does not need the lock, but
        # re-entrancy costs nothing and removes a whole class of surprise.
        self._lock = threading.RLock()

    @property
    def backend(self) -> KVDeviceBackend:
        return self._backend

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("the native K/V manager is closed")

    def open_session(
        self, session: str, identity: KVIdentity, reserve_bytes: int = 0
    ) -> None:
        with self._lock:
            self._require_open()
            if self._sessions.get(session) is _SessionState.CLOSING:
                raise ValueError(
                    f"session {session!r} is closing; the name is free again "
                    "once that close succeeds, not before"
                )
            if self._sessions.get(session) is _SessionState.NEEDS_RECONCILIATION:
                self._reconcile_session(session)
            # THE MIRROR IS PREPARED FIRST, so nothing has to be allocated
            # after the owner acts. Writing it afterwards meant an allocation
            # failure left the owner holding a session and its budget with the
            # caller told the open had failed -- measured, owner_sessions={'s'}
            # against facade_sessions={}. Rolling back is a `del`, which
            # allocates nothing and so cannot fail in turn.
            self._sessions[session] = _SessionState.OPEN
            try:
                self._core.kv_open_session(
                    self._native, session, _identity_dict(identity),
                    reserve_bytes
                )
            except BaseException as primary:
                # ASKED, NOT ASSUMED. A raised call does not say whether the
                # owner acted before it: rolling the mirror back unconditionally
                # loses a session the owner really opened, and keeping it
                # unconditionally invents one it refused. Both were measured by
                # the failure sweep, one per direction.
                try:
                    owner_has_it = self._core.kv_session_is_open(
                        self._native, session)
                except Exception as cleanup:             # noqa: BLE001
                    # THE MIRROR IS KEPT WHEN THE TRUTH CANNOT BE HAD. The two
                    # wrong guesses are not equally bad: a session recorded
                    # that the owner never opened costs one idempotent close,
                    # while one dropped that the owner DID open leaks its
                    # budget for the life of the manager. Same asymmetry the
                    # gate's worktree pruning settled on.
                    owner_has_it = True
                    self._sessions[session] = (
                        _SessionState.NEEDS_RECONCILIATION
                    )
                    self._reconciliation_failures += 1
                    primary.add_note(
                        f"asking the owner about session {session!r} failed: "
                        f"{cleanup}; the mirror is kept so the session can "
                        "still be closed"
                    )
                if not owner_has_it:
                    del self._sessions[session]
                raise

    def _reconcile_session(self, session: str) -> bool:
        """Resolve an uncertain open without guessing in either direction."""

        owner_has_it = bool(
            self._core.kv_session_is_open(self._native, session)
        )
        if owner_has_it:
            self._sessions[session] = _SessionState.OPEN
        else:
            self._sessions.pop(session, None)
        return owner_has_it

    def close_session(self, session: str) -> None:
        with self._lock:
            self._require_open()
            # Prepare facade bookkeeping before the native owner releases
            # anything. Building this set after a successful native close could
            # raise and leave Python claiming transactions the real owner had
            # already aborted.
            doomed = [
                name
                for name, transaction in self._transactions.items()
                if transaction.session == session
            ]
            # CLOSING FROM THE FIRST CALL, until this close succeeds. Acking the
            # receipts before the native close was not failure-atomic: a close
            # that then failed left the session's fate unknown, the records
            # retained, the receipts already unreplayable, and the metric
            # claiming nothing was pending. Marking the state first is what
            # makes a partial close a state rather than a mystery.
            self._sessions[session] = _SessionState.CLOSING

            # THE OWNER FIRST. Its close is idempotent (§13), so a retry after
            # a partial one is safe, and until it succeeds the receipts are
            # left alone -- still replayable, still counted.
            self._core.kv_close_session(self._native, session)

            # ONLY THEN the binding's receipts, which are not this side's to
            # forget: dropping the record made `pending_commit_receipts` report
            # zero while the C list still held a world id. Each record is
            # discharged and dropped one at a time, so a failure part-way
            # leaves the REST still counted and a repeat close finishes them.
            for name in doomed:
                record = self._transactions.get(name)
                if record is None:
                    continue
                if record.needs_reconciliation:
                    self._reconcile_transaction(name)
                    record = self._transactions.get(name)
                    if record is None:
                        continue
                if record.awaiting_ack:
                    try:
                        self._core.kv_commit_ack(self._native, name)
                    except BaseException as primary:
                        self._reconcile_after(primary, name)
                        raise
                    record.awaiting_ack = False
                self._transactions.pop(name, None)
            self._sessions.pop(session, None)

    def begin_transaction(
        self,
        session: str,
        identity: KVIdentity | None,
        plan: KVCapturePlan,
        total_bytes: int,
    ) -> str:
        with self._lock:
            return self._begin_transaction_locked(
                session, identity, plan, total_bytes)

    def _begin_transaction_locked(
        self,
        session: str,
        identity: KVIdentity | None,
        plan: KVCapturePlan,
        total_bytes: int,
    ) -> str:
        self._require_open()
        if self._sessions.get(session) is _SessionState.NEEDS_RECONCILIATION:
            if not self._reconcile_session(session):
                raise KeyError(f"session {session!r} is not open")
        if self._sessions.get(session) is _SessionState.CLOSING:
            # A session refuses new work from the moment its close begins, even
            # if that close has not finished. Admitting one now would be a
            # reservation the close has already decided to release.
            raise ValueError(f"session {session!r} is closing")
        name = str(self._core.kv_begin_transaction(
            self._native,
            session,
            None if identity is None else _identity_dict(identity),
            _plan_dict(plan),
            total_bytes,
        ))
        try:
            self._transactions[name] = _Transaction(
                session=session,
                identity=identity,
                plan=plan,
                total_bytes=total_bytes,
            )
        except BaseException as primary:
            # Native admission happened first. If the facade cannot record the
            # transaction (notably MemoryError), it must not strand staging and
            # reservation in the real owner. Preserve the primary exception;
            # cleanup failure is important context, not a replacement verdict.
            try:
                self._core.kv_abort(self._native, name)
            except Exception as cleanup:
                primary.add_note(f"native transaction rollback failed: {cleanup}")
            raise
        return name

    def transaction_is_open(self, transaction: str) -> bool:
        """§17.3's answer, from the owner rather than from a mirror of it.

        This is an authoritative boundary query, not a pure dictionary
        predicate: it may raise when the boundary is unavailable, and it
        reconciles stale facade bookkeeping as a side effect.

        The facade's own bookkeeping cannot answer this. A native commit ERASES
        the transaction, so a record that survives in `_transactions` -- which
        is exactly what a commit whose typed result failed used to leave --
        reports an open reservation that no longer exists. The protocol makes
        "committed" false in the same sentence as "unknown" and "aborted", so
        answering from the mirror was answering a different question.
        """
        with self._lock:
            if self._closed:
                return False
            # ONE QUESTION, ONE ANSWER. `kv_transaction_state` already
            # distinguishes open from committed-awaiting-acknowledgement from
            # absent, and a second query answering a coarser version of the
            # same thing is a surface that can disagree with itself.
            return self._reconcile_transaction(transaction) == "open"

    def capture_permit(self, transaction: str) -> NativeCapturePermit:
        with self._lock:
            self._require_open()
            record = self._transactions.get(transaction)
            if record is not None and record.needs_reconciliation:
                self._reconcile_transaction(transaction)
                record = self._transactions.get(transaction)
            if record is not None and record.awaiting_ack:
                # Committed and awaiting acknowledgement: closed, so it has no
                # capture left to permit.
                record = None
            if record is None:
                # Ask the native owner so the exception vocabulary remains one.
                self._core.kv_capture_permit(self._native, transaction)
                raise AssertionError("the native manager accepted an unknown transaction")
            capsule = self._core.kv_capture_permit(self._native, transaction)
            return NativeCapturePermit(transaction, record.plan, self, capsule)

    def _claim_capture(self, transaction: str, native_capsule: Any = None) -> None:
        """Consume the compiled owner's transaction-scoped capture claim."""
        with self._lock:
            self._require_open()
            record = self._transactions.get(transaction)
            if record is not None and record.needs_reconciliation:
                self._reconcile_transaction(transaction)
                record = self._transactions.get(transaction)
            if record is None or record.awaiting_ack:
                raise AdmissionRequired(
                    f"the capture for transaction {transaction} is no longer open"
                )
            if native_capsule is None:
                native_capsule = self._core.kv_capture_permit(
                    self._native, transaction
                )
            self._core.kv_permit_claim(native_capsule)

    def stage(self, transaction: str, fragment: KVFragment) -> None:
        with self._lock:
            self._require_open()
            if fragment.state is not KVPageState.TRANSACTION_STAGING:
                raise ValueError(
                    f"stage requires transaction_staging, got {fragment.state.value}"
                )
            record = self._transactions.get(transaction)
            if record is not None and record.needs_reconciliation:
                self._reconcile_transaction(transaction)
                record = self._transactions.get(transaction)
            if record is None or record.awaiting_ack:
                # `awaiting_ack` means the OWNER has already committed and
                # erased it. §17.3 makes such a transaction closed, so staging
                # into it is the same request as staging into an unknown one --
                # and reaching native would ask the owner about a reservation
                # that no longer exists.
                raise KeyError(f"transaction {transaction!r} is not open")
            self._core.kv_stage(
                self._native,
                transaction,
                fragment.layer,
                fragment.slice_.start,
                fragment.slice_.end,
                record.plan.spec.dtype,
                record.plan.spec.kv_heads,
                record.plan.spec.head_dim,
                fragment.slice_.key,
                fragment.slice_.value,
            )

    def commit(self, transaction: str) -> CommittedWorld:
        with self._lock:
            self._require_open()
            record = self._transactions.get(transaction)
            if record is not None and record.needs_reconciliation:
                self._reconcile_transaction(transaction)
                record = self._transactions.get(transaction)
            if record is None:
                raise KeyError(f"transaction {transaction!r} is not open")
            # THE OWNER DECIDES FIRST. A commit can be REFUSED for ordinary
            # reasons -- a layer never staged -- and the owner then keeps the
            # transaction open so the caller can finish it. Flipping this
            # side's state before the call turned every such refusal into a
            # transaction the facade believed committed: measured as
            # owner_open=True with stage raising KeyError, and a phantom
            # pending receipt that survived abort.
            #
            # A retry after a typed-result failure re-enters here with
            # `awaiting_ack` already true; the owner has erased the
            # transaction, so `kv_commit` replays the extension's receipt and
            # returns the same world id rather than publishing a second world.
            try:
                world_id = str(self._core.kv_commit(self._native, transaction))
            except BaseException as primary:
                # THE BOUNDARY IS ASKED, never inferred from the exception. The
                # extension deliberately keeps the receipt when it fails to
                # hand back the id AFTER a successful native commit, and from
                # here that is indistinguishable from a refusal -- measured as
                # owner_open=False with a live receipt, awaiting_ack unset,
                # `pending_commit_receipts` reporting 0, and close_session
                # never acknowledging it.
                self._reconcile_after(primary, transaction)
                raise
            # AN ASSIGNMENT TO A FIELD THAT ALREADY EXISTS: no allocation, so
            # this cannot half-happen the way a pop-then-insert across two
            # tables could.
            record.awaiting_ack = True
            try:
                start, end = record.plan.span
                result = CommittedWorld(
                    world_id=world_id,
                    transaction_id=transaction,
                    session=record.session,
                    identity=record.identity,
                    layers=tuple(sorted(record.plan.expected_layers)),
                    span=(start, end),
                    nbytes=record.total_bytes,
                )
            except BaseException:
                # The record stays, with `awaiting_ack` set: the owner HAS
                # committed and a retry must be able to collect the same world.
                raise
            # ACK ONLY AFTER the whole typed result exists. If Python
            # allocation fails anywhere above, the extension retains the native
            # receipt and a retry returns the same world id without publishing
            # a second world.
            try:
                self._core.kv_commit_ack(self._native, transaction)
            except BaseException as primary:
                # The receipt may already be discharged; if it is, this record
                # must stop claiming one.
                self._reconcile_after(primary, transaction)
                raise
            self._transactions.pop(transaction, None)
            return result

    def _reconcile_after(self, primary: BaseException, transaction: str) -> None:
        """Reconcile, and never let the reconciliation become the verdict.

        The boundary query can fail too, and then a caller was told the QUERY
        failed rather than why its operation did -- measured: a `ValueError`
        from abort surfaced as a `RuntimeError` about the state lookup. The
        same file already answers this for lease rollback with `add_note`; this
        follows it.
        """
        record = self._transactions.get(transaction)
        if record is not None:
            # Mark uncertainty BEFORE asking. If the query fails, later calls
            # must not reuse the stale pre-operation state or discard a receipt
            # the boundary may already hold.
            record.needs_reconciliation = True
        try:
            self._reconcile_transaction(transaction)
        except Exception as cleanup:                     # noqa: BLE001
            self._reconciliation_failures += 1
            primary.add_note(
                f"reconciling {transaction!r} with the boundary failed: "
                f"{cleanup}; the record remains blocked until reconciliation"
            )

    def _reconcile_transaction(self, transaction: str) -> str:
        """Make the record agree with the boundary after a call that raised.

        Every native call that can change the owner's state can also fail after
        changing it, and from here the two are indistinguishable. Rather than
        guess from the exception, the record is rebuilt from the one place that
        knows: `open` keeps it, `receipt_pending` marks it committed and
        awaiting acknowledgement, and `absent` drops it.
        """
        state = self._core.kv_transaction_state(self._native, transaction)
        if state not in {"open", "receipt_pending", "absent"}:
            raise RuntimeError(
                f"native transaction state {state!r} is outside the private "
                "contract"
            )
        record = self._transactions.get(transaction)
        if record is None:
            return state
        record.needs_reconciliation = False
        if state == "open":
            record.awaiting_ack = False
        elif state == "receipt_pending":
            record.awaiting_ack = True
        else:
            self._transactions.pop(transaction, None)
        return state

    def abort(self, transaction: str) -> None:
        with self._lock:
            self._require_open()
            record = self._transactions.get(transaction)
            if record is not None and record.needs_reconciliation:
                self._reconcile_transaction(transaction)
            try:
                self._core.kv_abort(self._native, transaction)
            except BaseException as primary:
                # The owner may have aborted it and then failed on the way out.
                self._reconcile_after(primary, transaction)
                raise
            self._transactions.pop(transaction, None)

    def resolve_read_plan(
        self,
        session: str,
        layers: Iterable[int],
        total_rows: int,
        candidates: Iterable[tuple[KVIdentity, int, int]] = (),
    ) -> tuple[KVReadPlan, Lease]:
        with self._lock:
            self._require_open()
            if self._sessions.get(session) is _SessionState.NEEDS_RECONCILIATION:
                if not self._reconcile_session(session):
                    raise KeyError(f"session {session!r} is not open")
            declared_layers = tuple(int(layer) for layer in layers)
            offered = tuple(candidates)
            identities: list[KVIdentity] = []
            for identity, start, end in offered:
                if (start, end) != identity.span:
                    raise ValueError(
                        f"candidate span {(start, end)} disagrees with "
                        f"identity span {identity.span}"
                    )
                identities.append(identity)
            (native_lease, reused_rows, pages, native_lease_id,
             native_worlds) = self._core.kv_resolve_read_plan(
                self._native,
                session,
                declared_layers,
                total_rows,
                [_identity_dict(identity) for identity in identities],
            )
            try:
                state_names = {
                    0: KVPageState.REQUEST_PROMPT,
                    1: KVPageState.ACTIVE_WINDOW,
                    2: KVPageState.TRANSACTION_STAGING,
                    3: KVPageState.COMMITTED_PROMPT,
                    4: KVPageState.COMMITTED_FROZEN_BLOCK,
                }
                lease_id = str(native_lease_id)
                worlds = tuple(str(world) for world in native_worlds)
                if not lease_id or len(set(worlds)) != len(worlds):
                    raise RuntimeError(
                        "native manager returned an empty lease id or duplicate "
                        "worlds in one lease"
                    )
                filed: dict[int, list[KVFragment]] = {
                    layer: [] for layer in declared_layers
                }
                for page in pages:
                    state = state_names.get(page["state"])
                    if state is None:
                        raise RuntimeError(
                            "native manager returned unknown page state "
                            f"{page['state']}"
                        )
                    candidate_index = int(page["candidate_index"])
                    world_id = str(page["world_id"])
                    identity = None
                    if state.shareable:
                        if not 0 <= candidate_index < len(identities):
                            raise RuntimeError(
                                "shared native page names candidate index "
                                f"{candidate_index} outside {len(identities)} offers"
                            )
                        if not world_id or world_id not in worlds:
                            raise RuntimeError(
                                "shared native page names a world outside its lease"
                            )
                        identity = identities[candidate_index]
                    elif candidate_index != -1 or world_id:
                        raise RuntimeError(
                            "request-local native page carried shared provenance"
                        )
                    filed[page["layer"]].append(
                        KVFragment(
                            layer=page["layer"],
                            slice_=KVSlice(
                                page["start"],
                                page["end"],
                                page["key"],
                                page["value"],
                            ),
                            state=state,
                            identity=identity,
                        )
                    )
                plan = KVReadPlan(
                    layers=declared_layers,
                    by_layer=tuple(
                        (layer, tuple(filed[layer])) for layer in declared_layers
                    ),
                    total_rows=total_rows,
                )
                # The native plan is authoritative. This assertion catches a
                # facade that accidentally rebuilds a different plan from the same
                # pages.
                if plan.reused_rows != int(reused_rows):
                    raise RuntimeError(
                        f"native manager reported {reused_rows} reused rows and "
                        f"the typed plan contains {plan.reused_rows}"
                    )
                lease = NativeLease(
                    lease_id=lease_id,
                    session=session,
                    worlds=worlds,
                    _native_capsule=native_lease,
                    _issuer=self,
                )
                return plan, lease
            except BaseException as primary:
                # PyCapsule finalization is eventual and traceback frames retain
                # locals. A facade refusal must give the native pin back NOW, not
                # when the exception object is eventually collected.
                try:
                    self._core.kv_release(native_lease)
                except Exception as cleanup:
                    primary.add_note(f"native read-lease rollback failed: {cleanup}")
                raise

    def release(self, lease: Lease) -> None:
        with self._lock:
            self._require_open()
            if not isinstance(lease, NativeLease):
                raise TypeError("this manager can release only a NativeLease")
            if lease._issuer is not self:
                raise ValueError("this lease was issued by a different manager")
            self._core.kv_release(lease._native_capsule)

    def metrics(self) -> dict[str, int]:
        with self._lock:
            self._require_open()
            reported = {
                str(name): int(value)
                for name, value in json.loads(
                    self._core.kv_metrics(self._native)
                ).items()
            }
            # A COMMITTED-BUT-UNACKNOWLEDGED COMMIT IS NOT AN OPEN TRANSACTION,
            # and it is not bounded by them either: the native commit erases the
            # transaction, so the receipt outlives it and is discharged only by
            # ACK or shutdown. Nothing counted them, so a review could describe
            # the backlog as "bounded by open transactions" -- which is exactly
            # the number that has already gone to zero. It is counted here.
            for name in tuple(self._transactions):
                record = self._transactions.get(name)
                if record is not None and record.needs_reconciliation:
                    self._reconcile_transaction(name)
            pending = [
                name for name, record in self._transactions.items()
                if record.awaiting_ack
            ]
            # CROSS-CHECKED AGAINST THE BOUNDARY, not just counted here. This
            # number exists because a review could otherwise describe a backlog
            # by reading a mirror; a mirror that has drifted would describe it
            # just as confidently.
            for name, record in self._transactions.items():
                actual = self._core.kv_transaction_state(self._native, name)
                expected = "receipt_pending" if record.awaiting_ack else "open"
                if actual != expected:
                    # The guard is also a state transition: after detecting
                    # drift, cleanup must re-query instead of later trusting
                    # the same stale value that triggered this refusal.
                    record.needs_reconciliation = True
                    self._boundary_state_mismatches += 1
                    if record.awaiting_ack:
                        raise RuntimeError(
                            f"facade counts a pending receipt for {name!r} that "
                            f"the boundary does not hold (reported {actual!r})"
                        )
                    raise RuntimeError(
                        f"facade records {name!r} as {expected} while the "
                        f"boundary reports {actual!r}"
                    )
            reported["pending_commit_receipts"] = len(pending)
            reported["facade_boundary_state_mismatches"] = (
                self._boundary_state_mismatches
            )
            reported["facade_reconciliation_failures"] = (
                self._reconciliation_failures
            )
            return reported

    def trace(self) -> tuple[KVOperation, ...]:
        with self._lock:
            self._require_open()
            document = self._trace_document_locked()
            operations = document["operations"]
            return tuple(
                KVOperation.of(
                    str(item["op"]),
                    str(item["session"]),
                    **_typed_detail(str(item["detail"])),
                )
                for item in operations
            )

    def _trace_document_locked(self) -> dict[str, Any]:
        """Validate and return the native wire document while ``_lock`` is held."""
        document = json.loads(self._core.kv_trace(self._native))
        if type(document) is not dict or set(document) != {
            "schema_version", "operations"
        }:
            raise RuntimeError(
                "native trace must contain exactly schema_version and operations"
            )
        version = document["schema_version"]
        if type(version) is not int or version != TRACE_SCHEMA_VERSION:
            raise RuntimeError(
                f"native trace schema version {version!r} is not "
                f"{TRACE_SCHEMA_VERSION}"
            )
        operations = document["operations"]
        if type(operations) is not list:
            raise RuntimeError("native trace operations must be a list")
        for index, item in enumerate(operations):
            if type(item) is not dict or set(item) != {"op", "session", "detail"}:
                raise RuntimeError(
                    f"native trace operation {index} has an unknown shape"
                )
            if any(
                type(item[name]) is not str
                for name in ("op", "session", "detail")
            ):
                raise RuntimeError(
                    f"native trace operation {index} fields must be strings"
                )
        return document

    def trace_document(self) -> dict[str, Any]:
        """The validated native wire trace for product/debug evidence.

        ``trace()`` exposes the typed manager protocol.  The product result must
        carry the native document itself -- schema version plus string-valued
        operation fields -- so it cannot accidentally invent a second serializer
        that drops an identity, byte count, or escaping rule.
        """
        with self._lock:
            self._require_open()
            return self._trace_document_locked()

    def shutdown(self) -> None:
        """Close the owner, and be closed whatever it reports.

        A CHECKED SHUTDOWN RAISES ON CLEANUP DEBT, and by then the owner has
        already closed -- the binding shuts the manager down and clears its
        receipts before it inspects the debt. It does NOT destroy: the handle
        survives, and `close()` is the entry point that ends it. Setting this side's state only on
        success therefore left a manager that looked open and was not:
        measured, `shutdown_error=RuntimeError`, `facade_closed=False`,
        `owner_open=False`, with the transaction still listed here. K2-2 makes
        `Engine.close()` drive this, so a half-state here is a product defect.

        Shutdown is the explicit exception to a transaction record's "retain
        until reconciled" rule: the native operation authoritatively destroys
        all transactions and receipts, leaving nothing to query or acknowledge.

        The state is set FIRST and the failure still propagates: a caller that
        must know cleanup failed still learns it, and what it holds afterwards
        is consistent. `_shutdown_reported` keeps the retry honest -- a second
        call re-drives the owner's cleanup rather than reporting a success this
        one never had.
        """
        with self._lock:
            if self._shutdown_reported:
                return
            self._closed = True
            self._transactions.clear()
            self._sessions.clear()
            self._core.kv_manager_shutdown(self._native)
            self._shutdown_reported = True

    def close(self) -> None:
        with self._lock:
            if self._destroyed:
                return
            # CLOSED FIRST, DESTROYED ONLY ON SUCCESS. The binding keeps the
            # handle live and returns a failure when destroy hits cleanup debt,
            # precisely so the caller can retry -- and setting `_destroyed`
            # before the call marked a live native handle destroyed for good.
            # Measured: destroy_calls stayed at 1 across two closes.
            #
            # `_closed` may go first because a caller must not keep using a
            # manager that is being torn down either way; only the RETRY hinges
            # on `_destroyed`.
            self._closed = True
            self._transactions.clear()
            self._sessions.clear()
            self._core.kv_manager_destroy(self._native)
            self._destroyed = True

    def __enter__(self) -> "NativeKVManager":
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def open_manager(
    *,
    device: str = "mlx",
    session_reserved_bytes: int = -1,
    shared_cache_budget_bytes: int = -1,
    staging_headroom_bytes: int = -1,
) -> NativeKVManager:
    """Open the native owner or refuse; never return the Python oracle."""

    try:
        from diffusorrt_runtime import _core
    except (ImportError, OSError) as exc:
        raise NativeKVUnavailable(
            "the private native K/V runtime is not installed"
        ) from exc
    _validate_native_core(_core)

    try:
        if device == "mlx":
            import edllm_apple  # noqa: F401  # registers the private provider
        backend = open_backend(device)
    except (ImportError, OSError, KeyError, RuntimeError, TypeError) as exc:
        raise NativeKVUnavailable(
            f"the private {device!r} K/V device backend is unavailable"
        ) from exc
    try:
        return NativeKVManager(
            _core,
            backend,
            session_reserved_bytes=session_reserved_bytes,
            shared_cache_budget_bytes=shared_cache_budget_bytes,
            staging_headroom_bytes=staging_headroom_bytes,
        )
    except Exception as exc:
        raise NativeKVUnavailable(
            "the private native K/V manager could not be created"
        ) from exc


__all__ = [
    "NativeCapturePermit",
    "NativeKVManager",
    "NativeKVUnavailable",
    "NativeLease",
    "open_manager",
]
