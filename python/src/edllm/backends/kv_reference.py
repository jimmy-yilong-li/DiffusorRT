"""The Python K/V oracle. Never a production owner (§17.8).

`docs/product/kv-cache-manager-v1.md` §3 and `plan.md` make the Python
implementation the executable reference oracle and the private C++ core the
single production owner. This module is that oracle: it holds no production
K/V, is not wired into `mlx_dream`, replaces no existing product cache, and is
not optimised. Its output is expectations, traces and differential verdicts.

It implements §17's frozen semantics exactly, because the C++ manager will be
compared against it step by step. Where a rule below looks pedantic, it is
because the rejected first attempt was measured breaking it:

  §17.1  a page is layer AND span. Two layers over the same rows collapsed.
  §17.2  the whole staging reservation is proven at `begin`, manager-globally.
         Per-transaction checking let two transactions holding 512 bytes each
         pass a 300-byte headroom.
  §17.3  the identity is declared at `begin`; there is no free-form publish.
         Publication against a transaction that never existed succeeded.
  §17.4  commit is atomic and returns the only token a reader may name.
  §17.5  reuse is leased, and success, failure and close all release. A
         consumer closed with the refcount still at 1.
  §17.6  identifiers are monotonic and never reused. Two consecutive
         transactions were both `s1:t1`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Iterable

from .kv_manager import (
    AdmissionRequired,
    CapturePermit,
    CommittedWorld,
    KVCapturePlan,
    KVFragment,
    KVIdentity,
    KVOperation,
    KVPageState,
    KVReadPlan,
    Lease,
    PayloadReleaser,
    ReleaseOutcome,
    SESSION_FIXED_FIELDS,
)

__all__ = [
    "UnknownPayloadBytes", "AdmissionRefused", "StaleHandle", "KVBudgets",
    "ReferenceKVCacheManager", "payload_bytes",
]


class UnknownPayloadBytes(ValueError):
    """A payload's size could not be determined (§8).

    Raised rather than defaulted: a zero charge does not fail, it quietly makes
    every later budget decision wrong and surfaces elsewhere as an over-budget
    process.
    """


class AdmissionRefused(Exception):
    """A reservation did not fit after eviction was attempted (§8 step 3)."""


class StaleHandle(ValueError):
    """An identifier that named something which no longer exists (§17.6).

    A stale handle resolves to a refusal, never to another object -- which is
    only possible because identifiers are never reused.
    """


def _trace_refusals(operation: str, owner: str):
    """Record failed public manager operations in one place (§17.7).

    Recording beside every individual ``raise`` is the copied-rule shape that
    already made release and eviction drift. This boundary wrapper makes a new
    refusal observable by default.
    """
    def decorate(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            try:
                return method(self, *args, **kwargs)
            except (AdmissionRefused, StaleHandle, UnknownPayloadBytes,
                    ValueError, KeyError) as caught:
                if owner == "session":
                    session = str(args[0] if args else kwargs.get("session", ""))
                elif owner == "lease":
                    lease = args[0] if args else kwargs.get("lease")
                    session = getattr(lease, "session", "")
                elif owner == "transaction":
                    transaction = str(
                        args[0] if args else kwargs.get("transaction", "")
                    )
                    record = self._transactions.get(transaction)
                    session = record.session if record is not None else ""
                else:  # pragma: no cover - fixed at decorator construction
                    raise AssertionError(
                        f"unknown refusal owner {owner!r}"
                    ) from caught
                if isinstance(caught, AdmissionRefused):
                    status = "admission_refused"
                elif isinstance(caught, StaleHandle):
                    status = "stale_handle"
                elif isinstance(caught, UnknownPayloadBytes):
                    status = "unknown_bytes"
                else:
                    status = "invalid_argument"
                self._record(
                    "refusal", session, operation=operation, status=status,
                )
                raise
        return wrapped
    return decorate


def payload_bytes(value: Any) -> int:
    """Real bytes, or a refusal. Never a guess (§8)."""
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int) and nbytes >= 0:
        return nbytes
    shape = getattr(value, "shape", None)
    itemsize = getattr(value, "itemsize", None)
    if shape is not None and isinstance(itemsize, int):
        total = itemsize
        for dimension in shape:
            total *= int(dimension)
        return total
    raise UnknownPayloadBytes(
        f"cannot determine the byte size of {type(value).__name__}; §8 forbids "
        "charging unknown bytes as zero"
    )


def fragment_bytes(fragment: KVFragment) -> int:
    """Key plus value. Both are charged: they are two tensors."""
    return payload_bytes(fragment.slice_.key) + payload_bytes(fragment.slice_.value)


@dataclass(frozen=True, slots=True)
class KVBudgets:
    """The three budgets of §8, each manager-global (§17.2).

    `None` is unbounded for that region and is a decision, not an absent value.
    """

    session_reserved_bytes: int | None = None
    shared_cache_budget_bytes: int | None = None
    staging_headroom_bytes: int | None = None


@dataclass
class _World:
    """One committed, immutable world: every layer of one span."""

    world: CommittedWorld
    fragments: dict[int, KVFragment]
    refs: int = 0
    sequence: int = 0

    @property
    def pinned(self) -> bool:
        return self.refs > 0


@dataclass
class _Session:
    identity: KVIdentity
    # GRANTED at open, and CONSUMED by request-local commits. Collapsing them
    # charged a session twice: it reserved 128 bytes, then committing a
    # 128-byte world asked the budget for another 128 and was refused inside
    # its own reservation.
    reserved: int = 0
    consumed: int = 0
    arena: dict[tuple[int, int, int], KVFragment] = field(default_factory=dict)
    leases: set[str] = field(default_factory=set)


@dataclass
class _Transaction:
    session: str
    identity: KVIdentity | None
    plan: KVCapturePlan
    reserved: int          # proven at begin, never increased (§17.2)
    consumed: int = 0
    capture_claimed: bool = False
    staged: dict[int, KVFragment] = field(default_factory=dict)


class ReferenceKVCacheManager:
    """The oracle. Implements `KVCacheManager` at §17's semantics."""

    def __init__(self, budgets: KVBudgets | None = None,
                 releaser: PayloadReleaser | None = None) -> None:
        """`releaser` is called with every payload this manager drops.

        WHY A CALLBACK, AND NOT "EVICTION RETURNS THE PAYLOADS". §10 gives a
        backend allocation and free PRIMITIVES and explicitly denies it
        "independent identity, eviction, or publication policy". That is exactly
        the split a callback expresses: the manager decides WHEN a page stops
        being reachable, the backend performs HOW its bytes go back, and neither
        one takes the other's job. Handing evicted payloads back to the caller
        would put "did you remember to free this" on every call site -- and
        eviction happens INSIDE admission, so the caller would have to unwind it
        from a return value it did not ask for. That is fail-open, and a leak
        nobody notices is precisely what this closes.

        The manager imports no backend and never inspects a payload: the
        callback is opaque to it, exactly as the payloads are.

        WITHOUT ONE, drops are COUNTED rather than silent. A manager holding
        numpy arrays needs no releaser and that is the honest configuration; a
        manager holding device handles without one is a leak, and it cannot tell
        which it is -- so it reports how many payloads it dropped with nobody to
        release them, and a wired configuration asserts that number is zero.
        """
        self._budgets = budgets or KVBudgets()
        self._releaser = releaser
        # The last cleanup failure, for a caller that wants to know WHY rather
        # than only how many. Not a metric: metrics are numbers.
        self._last_cleanup_detail = ""
        # Fragments a release could not complete, kept so a sweep has something
        # to retry. A count with no queue behind it is not debt.
        self._debt: list[KVFragment] = []
        # A shutdown manager stays shut: it has released its memory, and
        # accepting work against it would be accepting work against nothing.
        self._closed = False
        self._sessions: dict[str, _Session] = {}
        self._transactions: dict[str, _Transaction] = {}
        self._worlds: dict[str, _World] = {}          # world_id -> world
        self._index: dict[KVIdentity, str] = {}       # §4.6: owns no payload
        self._leases: dict[str, Lease] = {}
        # Every lease id ever issued. Without it "already released" and "never
        # issued" are indistinguishable, and the prefix check accepted a lease
        # this manager never handed out.
        self._issued_leases: set[str] = set()
        self._trace: list[KVOperation] = []
        self._next_id = 0
        self._clock = 0
        self._counters = {
            "identity_hits": 0, "identity_misses": 0, "applied_reuses": 0,
            "worlds_evicted": 0, "admission_refusals": 0,
            "transaction_commits": 0, "transaction_aborts": 0,
            "unknown_byte_refusals": 0, "leases_taken": 0,
            "worlds_superseded": 0,
            # §10's other half, now that something drives it: how many
            # FRAGMENTS this manager handed back, and how many it dropped with
            # nobody to hand them to.
            #
            # COUNTED IN FRAGMENTS, not payloads, because that is the unit both
            # implementations have. A `KVSlice` carries a key and a value; the
            # C++ `Fragment` carries ONE opaque handle. Counting payloads made
            # the two report 2 and 1 for the same eviction -- the differential
            # caught it -- and neither number was wrong, they were answers to
            # different questions. A fragment is §17.1's page, which both models
            # agree on.
            "fragments_released": 0, "fragments_dropped_unreleased": 0,
            # DEFERRED and FAILED are different facts: the backend understood
            # and could not free yet, versus the call did not complete. Both
            # become debt and only one is a fault.
            "fragments_release_deferred": 0, "cleanup_failures": 0,
            "request_local_replacements": 0, "sweeps_that_reclaimed": 0,
            "fragments_owed": 0,
            "release_freed_bytes": 0, "release_deferred_bytes": 0,
            "release_retired_bytes": 0,
            "leases_released": 0, "stale_handle_refusals": 0,
        }

    # -- identifiers (§17.6) ------------------------------------------------

    def _drop(self, *fragments: KVFragment) -> None:
        """Hand every payload in `fragments` back, or record that nobody could.

        Called from EVERY site where a page stops being reachable: eviction,
        supersede, the last release of a retired world, session close, and a
        transaction abort. Before this the manager dropped its reference and the
        device kept the memory -- the backend could free, and nothing ever asked
        it to.
        """
        dropped = [fragment for fragment in fragments
                   if fragment.slice_.key is not None
                   or fragment.slice_.value is not None]
        if not dropped:
            return
        if self._releaser is None:
            self._counters["fragments_dropped_unreleased"] += len(dropped)
            return
        # ONE CALL PER FRAGMENT, which is also what the C++ manager does: it
        # walks `pending_releases_` and calls the releaser once per intent. This
        # side used to hand EVERY dropped fragment's payloads to a single
        # `free()`, so the two implementations were asking their backends
        # different questions and only the C++ one could attribute an answer to
        # a fragment.
        #
        # It matters beyond symmetry. The manager's debt unit is the fragment;
        # it cannot subdivide one. An outcome that freed PART of a batch could
        # only be retried by re-offering the whole batch, which asks the backend
        # to free what it already freed. Per fragment, "what was deferred" and
        # "what is owed" are the same thing.
        for fragment in dropped:
            self._release_one(fragment)

    def _release_one(self, fragment: KVFragment) -> None:
        """Hand one fragment's payloads over, and account for every byte."""
        payloads = [payload for payload in (fragment.slice_.key,
                                            fragment.slice_.value)
                    if payload is not None]
        # NOTHING FROM A RELEASER ESCAPES THIS METHOD. §9: a post-commit cleanup
        # failure records DEBT; it does not roll back a durable commit or expose
        # a partial one. Measured before this: a releaser that raised came out
        # of `abort` with the transaction already erased, the payloads
        # unreleased and nothing recorded -- exactly the partial state the
        # clause forbids.
        requested = fragment_bytes(fragment)
        try:
            outcome = self._releaser.free(*payloads)
        except Exception as caught:                # noqa: BLE001 - recorded
            # THE PAYLOADS ARE KEPT, not just counted. Debt used to be a number
            # with nothing behind it: the reference was dropped on the way out,
            # so a later sweep had nothing to retry and "cleanup debt" named an
            # obligation the manager could no longer discharge.
            self._owe([fragment], f"{type(caught).__name__}: {caught}")
            return
        # A RELEASER THAT REPORTS NOTHING IS NOT A RELEASER THAT FREED. The old
        # code ignored the return entirely, so one that freed zero bytes and
        # deferred everything still counted a release.
        if not isinstance(outcome, ReleaseOutcome):
            self._owe([fragment], f"releaser returned {type(outcome).__name__}, "
                                  "not a ReleaseOutcome")
            return
        if not outcome.conserves(requested):
            # EVERY BYTE, ONCE. A ceiling let a backend under-report and the
            # remainder left the ledger and the retry queue together: measured
            # at requested=256, `freed=1, deferred=0` produced
            # fragments_released=1, owed=0, deferred_bytes=0, cleanup_failures=0.
            self._owe([fragment],
                      f"releaser accounted freed={outcome.freed_bytes} "
                      f"deferred={outcome.deferred_bytes} of a {requested}-byte "
                      "release, which does not partition it")
            return
        if not outcome.is_atomic(requested):
            # WHAT GETS RETRIED IS ALL OF IT OR NONE OF IT. Conservation alone
            # still permits deferring half a fragment, and the manager cannot
            # own half a fragment: retrying it would hand back payloads the
            # backend already released, and a backend that refuses a handle it
            # no longer holds turns that into a debt nothing can pay.
            self._owe([fragment],
                      f"releaser deferred {outcome.deferred_bytes} of a "
                      f"{requested}-byte release; what is retried is all of it "
                      "or none of it")
            return
        if outcome.failed:
            self._owe([fragment], outcome.detail)
            return
        if outcome.deferred_bytes:
            # DEFERRED KEEPS THE PAYLOADS TOO. Measured: a releaser answering
            # `deferred=256` produced release_deferred_bytes=256 and
            # fragments_owed=0 -- the same "debt with nothing behind it" that
            # was fixed for a FAILED release and left standing for a deferred
            # one, which is the ordinary case rather than the exceptional one.
            self._defer([fragment])
            return
        self._counters["fragments_released"] += 1
        self._counters["release_freed_bytes"] += outcome.freed_bytes
        # RETIRED IS NOT FREED AND NOT OWED. The ownership went; the device has
        # not confirmed the bytes came back. Recorded so the difference is
        # visible rather than folded into a number that would claim more than
        # was observed.
        self._counters["release_retired_bytes"] += outcome.retired_bytes

    def _defer(self, fragments: list[KVFragment]) -> None:
        """Keep what a backend understood and could not free YET (§9).

        Debt without a fault: `_owe` counts a cleanup failure, and a deferral is
        explicitly not one -- "deferred means the backend understood and could
        not free yet; failed means the call did not complete".
        """
        self._debt.extend(fragments)
        self._counters["fragments_release_deferred"] += len(fragments)
        self._counters["release_deferred_bytes"] += sum(
            fragment_bytes(fragment) for fragment in fragments)

    def _owe(self, fragments: list[KVFragment], detail: str) -> None:
        """Keep what could not be released, so a later sweep can retry it."""
        self._debt.extend(fragments)
        self._counters["cleanup_failures"] += 1
        self._counters["fragments_release_deferred"] += len(fragments)
        self._counters["release_deferred_bytes"] += sum(
            fragment_bytes(fragment) for fragment in fragments)
        self._last_cleanup_detail = detail

    def _retry_debt(self) -> None:
        """Re-offer everything a previous release could not complete.

        The queue is taken first, so a retry that fails again re-queues through
        `_drop` rather than looping on the same list.
        """
        if not self._debt or self._releaser is None:
            return
        owed, self._debt = self._debt, []
        # The bytes were already counted as deferred when they were owed;
        # counting them again on every retry would inflate the debt forever.
        self._counters["release_deferred_bytes"] -= sum(
            fragment_bytes(fragment) for fragment in owed)
        self._counters["fragments_release_deferred"] -= len(owed)
        self._drop(*owed)

    def _sweep(self) -> None:
        """Ask the releaser to retry its deferred work.

        DRIVEN FROM HERE, at the three points this manager owns: after an
        eviction pass, after a session closes, and at shutdown. Before this
        `sweep` had no caller outside two backend unit tests -- the first
        release was wired and the debt behind it was left to whoever remembered.

        The fourth point the contract wants, after a forward's synchronisation,
        belongs to the runtime loop, which does not exist yet. Saying so beats
        pretending this covers it.
        """
        if self._releaser is None:
            return
        self._retry_debt()
        try:
            reclaimed = self._releaser.sweep()
        except Exception as caught:                # noqa: BLE001 - recorded
            self._counters["cleanup_failures"] += 1
            self._last_cleanup_detail = f"sweep: {type(caught).__name__}: {caught}"
            return
        if isinstance(reclaimed, int) and reclaimed > 0:
            self._counters["release_freed_bytes"] += reclaimed
            self._counters["sweeps_that_reclaimed"] += 1

    def shutdown(self) -> None:
        """Close every session, release everything owned, and stay closed.

        ONE WHOLE SEMANTIC, CHOSEN. The first version cleared the worlds and
        left the sessions and leases standing: measured, `sessions=1 leases=1
        worlds=0 pinned=0` afterwards, and `begin_transaction` still succeeded
        -- a manager that had released its memory and would happily accept more
        work against it. The two coherent alternatives were "refuse unless
        quiescent" and "close everything and stay closed"; this is the second,
        because a shutdown that can be refused is one a caller has to retry in a
        loop it does not have.

        Closing each session first is what makes it whole: `close_session`
        aborts that session's transactions and releases its leases, so no world
        is dropped out from under a live pin.

        EXPLICIT, not a destructor. Python collects when it chooses, and a
        `__del__` reaching into a device backend during interpreter teardown is
        a hazard rather than a safety net.

        Idempotent, and permanent: a closed manager refuses new work.
        """
        # CLOSED BEFORE THE SNAPSHOT, not after the teardown. The flag was set
        # last, so between listing the sessions and finishing with them a
        # concurrent `open_session` was still admitted -- it would miss the
        # closing list and survive on a manager that is permanently shut. The
        # C++ manager has a lock to make this atomic and had the same ordering
        # bug; this side is single-owner but must state the same rule, or the
        # two disagree about what "shut down" means. `close_session` does not
        # consult the flag, so the teardown below still runs.
        self._closed = True
        for session in list(self._sessions):
            self.close_session(session)
        for world in list(self._worlds.values()):
            self._drop(*world.fragments.values())
        self._worlds.clear()
        self._index.clear()
        self._sweep()

    def _mint(self, kind: str) -> str:
        """Monotonic, never reused, never derived from a container's size.

        The rejected prototype built transaction ids from the count of OPEN
        transactions, so two consecutive transactions in one session were both
        `s1:t1` -- a stale id silently naming a live object.
        """
        self._next_id += 1
        return f"{kind}-{self._next_id}"

    # -- sessions -----------------------------------------------------------

    def _require_open(self) -> None:
        if self._closed:
            raise ValueError(
                "this manager has been shut down; its payloads are released "
                "and it accepts no further work")

    @_trace_refusals("open_session", "session")
    def open_session(self, session: str, identity: KVIdentity,
                     reserve_bytes: int = 0) -> None:
        self._require_open()
        if session in self._sessions:
            raise ValueError(f"session {session!r} is already open")
        self._admit_session(reserve_bytes)
        self._sessions[session] = _Session(identity=identity,
                                           reserved=reserve_bytes)
        self._record("open_session", session, reserve_bytes=reserve_bytes)

    def close_session(self, session: str) -> None:
        """Releases staging, leases and arena. Idempotent (§13, §17.5)."""
        record = self._sessions.get(session)
        if record is None:
            return
        for name in [n for n, t in self._transactions.items()
                     if t.session == session]:
            self.abort(name)
        for lease_id in sorted(record.leases):
            lease = self._leases.get(lease_id)
            if lease is not None:
                self._release_lease(lease)
        # §7 step 8: closing releases request-local ownership, and now that
        # includes the bytes. The arena held every request-local world for this
        # session's lifetime; dropping the dict used to be the end of it.
        self._drop(*record.arena.values())
        self._sessions.pop(session, None)
        self._record("close_session", session)
        self._sweep()

    # -- reads and leases (§17.5) -------------------------------------------

    @_trace_refusals("resolve_read_plan", "session")
    def resolve_read_plan(
        self,
        session: str,
        layers: Iterable[int],
        total_rows: int,
        candidates: Iterable[tuple[KVIdentity, int, int]] = (),
    ) -> tuple[KVReadPlan, Lease]:
        """Resolve per layer, pin under a lease, and release every pin on
        failure so a partially pinned plan is never observable."""
        record = self._require_session(session)
        declared = tuple(layers)
        pinned: list[str] = []
        try:
            claimed: dict[tuple[int, int], _World] = {}
            considered: set = set()
            for identity, start, end in candidates:
                # The same candidate twice names ONE world. Counting it twice
                # reported reuse that did not happen, and pinned the world
                # twice -- so a single release left it pinned and permanently
                # unevictable.
                key = (identity, start, end)
                if key in considered:
                    continue
                considered.add(key)
                # THE LOOKUP COMES FIRST. §9 counts an identity hit separately
                # from an applied reuse, and a second identity over the same
                # rows genuinely DOES resolve to a resident world -- so it is a
                # hit. Arbitrating before looking up under-reported hits, and
                # the strict differential caught C++ saying 2 where this said 1.
                world = self._lookup(identity)
                if world is None:
                    continue
                if world.world.span != (start, end):
                    continue
                if set(world.world.layers) != set(declared):
                    # A world covering a different layer set cannot serve this
                    # plan: some layer would be left without an owner.
                    continue
                # ONE world per span. Both are hits; the plan consumes one.
                # Pinning both reported reuse that never happened and left a pin
                # no release could clear.
                if any(existing == start for existing, _ in claimed):
                    continue
                world.refs += 1
                pinned.append(world.world.world_id)
                self._counters["applied_reuses"] += 1
                claimed[(start, end)] = world

            by_layer: list[tuple[int, tuple[KVFragment, ...]]] = []
            for layer in declared:
                fragments: list[KVFragment] = []
                cursor = 0
                while cursor < total_rows:
                    world = next(
                        (w for (s, _e), w in sorted(claimed.items()) if s == cursor),
                        None,
                    )
                    if world is not None:
                        fragments.append(world.fragments[layer])
                        cursor = world.world.span[1]
                        continue
                    local = record.arena.get((layer, cursor, total_rows))
                    if local is None:
                        local = next(
                            (f for (lyr, s, _e), f in sorted(record.arena.items())
                             if lyr == layer and s == cursor),
                            None,
                        )
                    if local is None:
                        raise ValueError(
                            f"layer {layer} row {cursor} has no owner: neither a "
                            "reusable world nor this session's arena covers it"
                        )
                    fragments.append(local)
                    cursor = local.slice_.end
                by_layer.append((layer, tuple(fragments)))

            plan = KVReadPlan(layers=declared, by_layer=tuple(by_layer),
                              total_rows=total_rows)
        except Exception:
            # §17.5 failure path. Every pin this call took is released before
            # the failure returns.
            for world_id in pinned:
                self._worlds[world_id].refs -= 1
            # AND THE COUNTER COMES BACK. A refused plan applied nothing, so
            # leaving `applied_reuses` raised reported reuse that never served a
            # row -- §9's reuse rate is what a policy reads to decide whether
            # caching is paying for itself. `identity_hits` deliberately stays:
            # the lookup did hit, and §9 counts a hit separately from a reuse.
            # BOTH implementations had this, so the differential agreed and saw
            # nothing; that is what per-implementation tests are for.
            self._counters["applied_reuses"] -= len(pinned)
            raise

        lease = Lease(lease_id=self._mint("lease"), session=session,
                      worlds=tuple(pinned))
        self._leases[lease.lease_id] = lease
        self._issued_leases.add(lease.lease_id)
        record.leases.add(lease.lease_id)
        self._counters["leases_taken"] += 1
        self._record(
            "resolve_read_plan", session, lease=lease.lease_id,
            layers=len(declared), reused_rows=plan.reused_rows,
            request_local_rows=plan.request_local_rows,
        )
        return plan, lease

    @_trace_refusals("release_lease", "lease")
    def release(self, lease: Lease) -> None:
        """Idempotent. A lease that was never issued is a stale handle.

        VERIFIED AS A WHOLE. Only `lease_id` used to be read, so a lease built
        from a real id with `session="forged"` and `worlds=("bogus",)` released
        the pin -- measured, `pinned_worlds` 1 -> 0. A struct that publishes
        fields it does not check is worse than one that publishes none: a caller
        can reasonably believe they mean something.
        """
        held = self._leases.get(lease.lease_id)
        if held is not None and (lease.session != held.session
                                 or tuple(lease.worlds) != tuple(held.worlds)):
            self._counters["stale_handle_refusals"] += 1
            raise StaleHandle(
                f"{lease.lease_id!r} names session {lease.session!r} over "
                f"{tuple(lease.worlds)}, and this manager issued it to "
                f"{held.session!r} over {tuple(held.worlds)}"
            )
        if held is None:
            # Idempotent for a lease this manager ISSUED; a refusal for one it
            # never issued. The check used to be a name prefix (§17.6).
            if lease.lease_id in self._issued_leases:
                return
            self._counters["stale_handle_refusals"] += 1
            raise StaleHandle(f"{lease.lease_id!r} was never issued")
        self._release_lease(held)

    def _release_lease(self, lease: Lease) -> None:
        for world_id in lease.worlds:
            world = self._worlds.get(world_id)
            if world is None:
                continue
            if world.refs > 0:
                world.refs -= 1
            # A RETIRED world is freed by its last release, not by the next
            # memory pressure. Supersede keeps a pinned predecessor alive for
            # its lease, and that world is already out of the index -- nothing
            # can reach it again. Measured before this: shared_worlds 2 ->
            # release -> 2, still holding its bytes. With a real payload that
            # is a deferred free.
            if world.refs == 0 and self._retired(world):
                self._drop(*world.fragments.values())
                self._worlds.pop(world_id, None)
                self._counters["worlds_superseded"] += 1
        self._leases.pop(lease.lease_id, None)
        session = self._sessions.get(lease.session)
        if session is not None:
            session.leases.discard(lease.lease_id)
        self._counters["leases_released"] += 1
        self._record("release_lease", lease.session, lease=lease.lease_id)

    def _retired(self, world: _World) -> bool:
        """Out of the index, or superseded by a newer world under the same key:
        either way no lookup can reach this one again."""
        identity = world.world.identity
        if identity is None:
            return False
        return self._index.get(identity) != world.world.world_id

    def _lookup(self, identity: KVIdentity) -> _World | None:
        world_id = self._index.get(identity)
        world = self._worlds.get(world_id) if world_id else None
        if world is None:
            self._counters["identity_misses"] += 1
            if world_id:                     # evicted: drop the dead entry (§4.6)
                self._index.pop(identity, None)
            return None
        self._counters["identity_hits"] += 1
        self._clock += 1
        world.sequence = self._clock
        return world

    # -- transactions (§17.2, §17.3, §17.4) ---------------------------------

    def _require_identity_layers(self, identity: KVIdentity | None,
                                 plan: KVCapturePlan) -> None:
        """§6 binds the LAYER SET, so an identity cannot key another one.

        Without this, one identity committed over layer 0 and then over layer 1
        left a single world -- by the key they were the same -- and the layer-0
        rows became permanently unreachable while still resident.
        """
        if identity is None:
            return
        declared = tuple(sorted(plan.expected_layers))
        if tuple(identity.layer_set) != declared:
            raise ValueError(
                f"identity layer set {tuple(identity.layer_set)} does not match "
                f"the captured layers {declared}"
            )

    def _require_session_identity(self, record: _Session,
                                  identity: KVIdentity | None) -> None:
        """§4.2: a session is one decode run against ONE model for ONE tenant.

        The session's identity was stored by both implementations and read by
        neither. Measured: a session opened `privacy_scope="private-A"`
        committed a transaction claiming `private-B`.

        Only the model- and tenant-identifying fields are fixed; token ids,
        span, freeze state, epoch, salt and policy version vary per request by
        design.
        """
        if identity is None:
            return
        differing = [name for name in SESSION_FIXED_FIELDS
                     if getattr(record.identity, name) != getattr(identity, name)]
        if differing:
            raise ValueError(
                "a transaction cannot replace what its session fixed: "
                + ", ".join(
                    f"{name} {getattr(record.identity, name)!r} -> "
                    f"{getattr(identity, name)!r}" for name in differing)
            )

    def _require_tensor_spec(self, identity: KVIdentity | None,
                             plan: KVCapturePlan,
                             session_identity: KVIdentity) -> None:
        """§6/§7 step 5: an identity-keyed world describes its tensors, HERE.

        The spec used to be a parameter the device backend's caller supplied,
        which proved only that the tensors matched what that caller said.
        Measured: an identity naming bfloat16 and `int8-weight-only` accepted
        float32 tensors under a spec claiming quantization `totally-wrong`, and
        this manager committed it.

        A request-local transaction may omit the spec: it is never keyed, never
        offered for reuse, and has no identity to contradict.
        """
        if plan.spec is None:
            # EVERY transaction, not only identity-keyed ones. A request-local
            # world is read by the same forward and charged against the same
            # budget, and an audit committed a (2,1) int8 payload through one by
            # simply omitting the spec.
            raise ValueError(
                "a capture plan must declare its tensor spec; §7 step 5 "
                "validates shape and dtype and there is nothing here to "
                "validate against"
            )
        if identity is None:
            # A REQUEST-LOCAL WORLD IS STILL THIS SESSION'S. It has no identity
            # to contradict, so an earlier version returned here -- and a
            # session fixed at float32/none/2x4 accepted a request-local
            # transaction producing float16/other/1x8 into the same arena.
            # Measured. The session's own identity is what it must agree with.
            plan.spec.agrees_with(session_identity)
            return
        plan.spec.agrees_with(identity)

    def _require_identity_span(self, identity: KVIdentity | None,
                               plan: KVCapturePlan) -> None:
        """§6 binds the position span into the identity, so an identity naming
        rows [0,4) cannot key a world covering [0,2). That mismatch committed
        and then served 2 rows as an exact hit for a 4-row key."""
        if identity is None:
            return
        if identity.span != plan.span:
            raise ValueError(
                f"identity span {identity.span} does not match the captured "
                f"span {plan.span}"
            )

    @_trace_refusals("begin_transaction", "session")
    def begin_transaction(self, session: str, identity: KVIdentity | None,
                          plan: KVCapturePlan, total_bytes: int) -> str:
        """The WHOLE reservation, proven manager-globally, before opening.

        `identity=None` opens a REQUEST-LOCAL transaction: its world lands in
        the session's own arena, is never indexed for reuse and is never
        evictable while the session lives (§5, §4.3). An identity opens a
        shareable one.

        The distinction was missing until a test asked for it. Without it the
        session arena was unreachable -- every commit went to the shared pool --
        so §5's "request prompt under construction" and "active-window" states
        had no way to exist, and a decode's own K/V could only be modelled as
        something offered for reuse.
        """
        self._require_open()
        open_session = self._require_session(session)
        if total_bytes < 0:
            raise ValueError("total_bytes must be non-negative")
        # DERIVED, NOT TRUSTED. `total_bytes` used to be whatever the caller
        # said: a two-layer plan needing 512 bytes opened with `total_bytes=1`
        # and the refusal arrived at `stage`, after the caller had allocated the
        # real device tensors. Admission before allocation was a calling
        # convention, not an invariant. The plan states its own size now, so the
        # argument is a CLAIM the manager checks rather than a number it obeys.
        self._require_identity_span(identity, plan)
        self._require_identity_layers(identity, plan)
        self._require_session_identity(open_session, identity)
        self._require_tensor_spec(identity, plan, open_session.identity)
        # AFTER the spec check, which is what makes the size derivable at all.
        required = plan.total_bytes
        if total_bytes != required:
            raise ValueError(
                f"this plan produces {required} bytes and the caller reserved "
                f"{total_bytes}; §17.2 proves the whole reservation up front"
            )
        self._admit_staging(total_bytes)
        # Transaction IDs are opaque and never reused. The session already
        # lives in the transaction record; embedding a caller-controlled name
        # made the native C representation unbounded for no semantic gain.
        name = self._mint("txn")
        self._transactions[name] = _Transaction(
            session=session, identity=identity, plan=plan, reserved=total_bytes)
        self._record("begin_transaction", session, transaction=name,
                     expected_layers=tuple(plan.expected_layers),
                     reserved_bytes=total_bytes)
        return name

    def transaction_is_open(self, transaction: str) -> bool:
        """§17.6: unknown, committed and aborted are all "not open"."""
        return transaction in self._transactions

    @_trace_refusals("capture_permit", "transaction")
    def capture_permit(self, transaction: str) -> CapturePermit:
        """Mint the proof that this transaction was admitted (§7 step 2).

        Refused for a transaction this manager does not have open, so a permit
        cannot be obtained by asking nicely after the fact.
        """
        self._require_open()
        if transaction not in self._transactions:
            raise StaleHandle(f"unknown transaction {transaction}")
        return CapturePermit(transaction, self._transactions[transaction].plan,
                             self)

    def _claim_capture(self, transaction: str) -> None:
        """Consume the transaction-owned admission exactly once."""
        self._require_open()
        record = self._transactions.get(transaction)
        if record is None:
            self._counters["stale_handle_refusals"] += 1
            raise AdmissionRequired(
                f"the capture for transaction {transaction} is no longer open"
            )
        if record.capture_claimed:
            raise AdmissionRequired(
                f"the capture for transaction {transaction} was already "
                "claimed; one admitted transaction is one capture"
            )
        record.capture_claimed = True

    @_trace_refusals("stage", "transaction")
    def stage(self, transaction: str, fragment: KVFragment) -> None:
        """Consumes reservation already proven; never asks for more (§17.2)."""
        record = self._require_transaction(transaction)
        if not record.capture_claimed:
            raise AdmissionRequired(
                f"transaction {transaction} has not claimed its capture "
                "permit; staging an allocated payload before admission proof "
                "is forbidden"
            )
        # §5: staging is ONE state, and the manager assigns the committed one.
        # This was accepted-and-ignored on both sides -- commit overwrote
        # whatever arrived -- so an ACTIVE_WINDOW fragment staged and committed
        # here while C++ took PageState(999). A provider handing over a fragment
        # in the wrong state is a handover bug, and overwriting it hides that.
        if fragment.state is not KVPageState.TRANSACTION_STAGING:
            raise ValueError(
                f"a staged fragment must be in {KVPageState.TRANSACTION_STAGING.value}, "
                f"not {fragment.state.value}"
            )
        layer = fragment.layer
        if layer not in set(record.plan.expected_layers):
            raise ValueError(
                f"layer {layer} is not in the transaction's declared set "
                f"{sorted(record.plan.expected_layers)}"
            )
        if layer in record.staged:
            raise ValueError(f"layer {layer} staged twice")
        # §7 step 5 validates "layer, span, shape, dtype". Checking the plan's
        # spec against the IDENTITY proved only that two declarations agreed;
        # the tensor that actually arrives is the thing the identity will key.
        # Measured before this: a spec declaring (2,2,4) float32 committed a
        # (2,9,9) payload.
        rows = fragment.slice_.end - fragment.slice_.start
        record.plan.spec.validate(fragment.slice_.key, rows,
                                  what=f"layer {layer} key")
        record.plan.spec.validate(fragment.slice_.value, rows,
                                  what=f"layer {layer} value")
        try:
            nbytes = fragment_bytes(fragment)
        except UnknownPayloadBytes:
            self._counters["unknown_byte_refusals"] += 1
            raise
        if record.consumed + nbytes > record.reserved:
            self._counters["admission_refusals"] += 1
            raise AdmissionRefused(
                f"staging layer {layer} would consume "
                f"{record.consumed + nbytes} bytes against the "
                f"{record.reserved} proven at begin; §17.2 forbids asking for "
                "more during staging"
            )
        record.consumed += nbytes
        record.staged[layer] = fragment
        self._record("stage", record.session, transaction=transaction, layer=layer)

    @_trace_refusals("commit", "transaction")
    def commit(self, transaction: str) -> CommittedWorld:
        """Atomic across validation, reservation, visibility and indexing."""
        record = self._require_transaction(transaction)
        if not record.capture_claimed:
            raise AdmissionRequired(
                f"transaction {transaction} has not claimed its capture "
                "permit; commit cannot publish it"
            )
        expected = set(record.plan.expected_layers)
        missing = sorted(expected - set(record.staged))
        if missing:
            raise ValueError(
                f"committing without layers {missing}; no partial layer set "
                "becomes visible"
            )
        span = record.plan.span
        for layer, fragment in record.staged.items():
            if (fragment.slice_.start, fragment.slice_.end) != span:
                raise ValueError(
                    f"layer {layer} staged span "
                    f"({fragment.slice_.start}, {fragment.slice_.end}) against "
                    f"the transaction's {span}"
                )
        nbytes = sum(fragment_bytes(f) for f in record.staged.values())
        shareable = record.identity is not None
        if shareable:
            # Reservation BEFORE anything becomes visible. A refusal here leaves
            # the transaction open and nothing published.
            self._admit_shared(nbytes, record.session)
        else:
            # A REQUEST-LOCAL world lands in the session arena, which is what
            # `session_reserved_bytes` bounds (§8). This used to add to the
            # session's reserved total without proving it fit: measured, a
            # session opened with reserve 0 against a budget of 0 committed its
            # world and refused nothing. THE C++ MANAGER HAD THE SAME DEFECT,
            # so the differential agreed -- two implementations agreeing proves
            # they share an assumption, not that the assumption is right.
            # REPLACEMENT IS CHARGED BY THE DELTA, and the slot it replaces is
            # released. A request-local world is keyed (layer, start, end), and
            # the same slot is legitimately re-captured every forward -- that is
            # what an active window IS. Silently overwriting it did two wrong
            # things at once: the old payload became unreachable without ever
            # being released, and a same-sized replacement was charged to the
            # session a second time. Measured: two identical commits left
            # arena=1 with reserved 128 then 256.
            #
            # REFUSING the second commit was the other coherent choice and is
            # the wrong one here: it would refuse the ordinary d2f pattern.
            owner = self._sessions[record.session]
            replaced = [owner.arena[(layer, span[0], span[1])]
                        for layer in record.staged
                        if (layer, span[0], span[1]) in owner.arena]
            self._consume_session(owner, nbytes - sum(
                fragment_bytes(fragment) for fragment in replaced))

        world_id = self._mint("world")
        committed = CommittedWorld(
            world_id=world_id, transaction_id=transaction, session=record.session,
            identity=record.identity,
            layers=tuple(sorted(record.plan.expected_layers)),
            span=span, nbytes=nbytes,
        )
        self._clock += 1
        state = (record.plan.committed_state if shareable
                 else KVPageState.ACTIVE_WINDOW)
        published = {
            layer: KVFragment(layer=layer, slice_=fragment.slice_, state=state,
                              identity=record.identity if shareable else None)
            for layer, fragment in record.staged.items()
        }
        session = self._sessions[record.session]
        if shareable:
            self._worlds[world_id] = _World(world=committed, fragments=published,
                                            sequence=self._clock)
            # The identity is exact (§6): the same key means the same rows of
            # the same weights, so a second commit supersedes the first. An
            # UNPINNED predecessor is dropped rather than left resident and
            # unreachable; a pinned one stays until its lease releases, which
            # the ownership-guarded pop above now makes safe.
            previous = self._worlds.get(self._index.get(record.identity))
            if previous is not None and not previous.pinned:
                self._drop(*previous.fragments.values())
                self._worlds.pop(previous.world.world_id, None)
                self._counters["worlds_superseded"] += 1
            self._index[record.identity] = world_id
        else:
            # Request-local: the session's own arena. Never indexed, never
            # evictable while the session lives (§4.3).
            # ATOMIC: the new fragments go in, and only then does the old set
            # go to the releaser. Releasing first would leave a window in which
            # the slot names memory that is already gone.
            superseded = [session.arena[(layer, span[0], span[1])]
                          for layer in published
                          if (layer, span[0], span[1]) in session.arena]
            for layer, fragment in published.items():
                session.arena[(layer, span[0], span[1])] = fragment
            if superseded:
                self._counters["request_local_replacements"] += len(superseded)
                self._drop(*superseded)
            # The bytes were accounted by `_consume_session` above, charged as
            # the delta against what these slots already held.
        del self._transactions[transaction]
        self._counters["transaction_commits"] += 1
        self._record("commit", record.session, transaction=transaction,
                     world=world_id, layers=len(published), bytes=nbytes,
                     shareable=shareable)
        if shareable:
            # Two entries in order from ONE atomic step: the ordering §7
            # requires is a property of the code, not a rule a caller can break.
            self._record("publish_identity", record.session,
                         transaction=transaction, world=world_id,
                         identity=record.identity.short)
        return committed

    @_trace_refusals("abort", "transaction")
    def abort(self, transaction: str) -> None:
        record = self._transactions.pop(transaction, None)
        if record is None:
            self._counters["stale_handle_refusals"] += 1
            raise StaleHandle(f"transaction {transaction!r} is not open")
        # A capture that never became a world still allocated. §7 step 7 says a
        # failed transaction releases staging; the payloads are the staging.
        self._drop(*record.staged.values())
        record.staged.clear()
        record.consumed = 0
        record.reserved = 0
        self._counters["transaction_aborts"] += 1
        self._record("abort", record.session, transaction=transaction)

    # -- capacity (§8, §17.2) -----------------------------------------------

    def _admit_session(self, nbytes: int) -> None:
        if nbytes < 0:
            # A negative reservation subtracts from the budget, so it always
            # "fits".
            self._counters["admission_refusals"] += 1
            raise AdmissionRefused("a reservation cannot be negative")
        budget = self._budgets.session_reserved_bytes
        if budget is None:
            return
        current = sum(s.reserved for s in self._sessions.values())
        if current + nbytes > budget:
            self._counters["admission_refusals"] += 1
            raise AdmissionRefused(
                f"session reservation of {nbytes} does not fit the "
                f"{budget}-byte session budget with {current} already reserved"
            )

    def _consume_session(self, session: _Session, nbytes: int) -> None:
        """Charge a request-local world against this session's own grant."""
        if nbytes < 0:
            self._counters["admission_refusals"] += 1
            raise AdmissionRefused("a consumption cannot be negative")
        after = session.consumed + nbytes
        if after <= session.reserved:
            session.consumed = after      # inside the grant already held
            return
        # Only the excess beyond the grant asks the global budget.
        self._admit_session(after - session.reserved)
        session.reserved = after
        session.consumed = after

    def _admit_staging(self, nbytes: int) -> None:
        """MANAGER-GLOBAL, across every open transaction (§17.2)."""
        budget = self._budgets.staging_headroom_bytes
        if budget is None:
            return
        current = sum(t.reserved for t in self._transactions.values())
        if current + nbytes > budget:
            self._counters["admission_refusals"] += 1
            raise AdmissionRefused(
                f"staging {nbytes} bytes does not fit the {budget}-byte global "
                f"headroom with {current} already reserved across "
                f"{len(self._transactions)} open transaction(s)"
            )

    def _admit_shared(self, nbytes: int, session: str) -> None:
        budget = self._budgets.shared_cache_budget_bytes
        if budget is None:
            return
        if self._shared_bytes() + nbytes <= budget:
            return
        self._evict_until(budget - nbytes, session)
        if self._shared_bytes() + nbytes <= budget:
            return
        self._counters["admission_refusals"] += 1
        raise AdmissionRefused(
            f"{nbytes} bytes do not fit the shared budget of {budget} even "
            f"after eviction ({self._shared_bytes()} resident and pinned)"
        )

    def _evict_until(self, target: int, session: str) -> None:
        """LRU among UNPINNED worlds only (§9).

        A pinned world is skipped, not evicted last. When everything is pinned
        the correct outcome is a refusal, not evicting something live.
        """
        candidates = sorted(
            (world for world in self._worlds.values() if not world.pinned),
            key=lambda world: world.sequence,
        )
        for world in candidates:
            if self._shared_bytes() <= target:
                # BREAK, NOT RETURN. The sweep below is the point of this pass
                # ending, and an early `return` walked straight past it -- so
                # the eviction sweep was unreachable in the ordinary case, which
                # is the only case. Its test caught that.
                break
            evicted_bytes = world.world.nbytes
            evicted_id = world.world.world_id
            self._drop(*world.fragments.values())
            self._worlds.pop(world.world.world_id, None)
            # ONLY IF IT STILL NAMES THIS WORLD. Committing one identity twice
            # left two worlds sharing a key; evicting the older one erased the
            # entry pointing at the NEWER, still-resident world, which then
            # answered every lookup as a miss while occupying its bytes.
            # Measured identically here and in C++ -- shared_worlds=2,
            # identity_hits=0, identity_misses=1, plan refused with "no owner"
            # -- so the differential compared two wrong answers and agreed.
            if self._index.get(world.world.identity) == world.world.world_id:
                self._index.pop(world.world.identity, None)
            self._counters["worlds_evicted"] += 1
            self._record(
                "evict", session, bytes=evicted_bytes, world=evicted_id,
                shared_resident_total=self._shared_bytes(),
            )
        # After the pass, before admission retries: what eviction deferred may
        # be reclaimable now, and the retry is the moment it matters.
        self._sweep()

    def _shared_bytes(self) -> int:
        return sum(world.world.nbytes for world in self._worlds.values())

    # -- metrics and trace --------------------------------------------------

    def metrics(self) -> dict[str, int]:
        snapshot = dict(self._counters)
        # THE QUEUE, not a running tally. "Cleanup debt" meant a number with
        # nothing behind it until releases became retryable; this is how many
        # fragments are actually still owed and still dischargeable.
        snapshot["fragments_owed"] = len(self._debt)
        snapshot.update(
            session_reserved_bytes=sum(s.reserved for s in self._sessions.values()),
            staging_reserved_bytes=sum(t.reserved for t in self._transactions.values()),
            shared_resident_bytes=self._shared_bytes(),
            shared_worlds=len(self._worlds),
            pinned_worlds=sum(1 for w in self._worlds.values() if w.pinned),
            live_sessions=len(self._sessions),
            open_transactions=len(self._transactions),
            open_leases=len(self._leases),
        )
        return snapshot

    def trace(self) -> tuple[KVOperation, ...]:
        return tuple(self._trace)

    def _record(self, op: str, session: str, **detail: Any) -> None:
        self._trace.append(KVOperation.of(op, session, **detail))

    # -- internals ----------------------------------------------------------

    def _require_session(self, session: str) -> _Session:
        record = self._sessions.get(session)
        if record is None:
            raise StaleHandle(f"session {session!r} is not open")
        return record

    def _require_transaction(self, transaction: str) -> _Transaction:
        record = self._transactions.get(transaction)
        if record is None:
            self._counters["stale_handle_refusals"] += 1
            raise StaleHandle(f"transaction {transaction!r} is not open")
        return record
