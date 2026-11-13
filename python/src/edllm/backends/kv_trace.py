"""K0: the maintained operation-trace corpus and the rules a trace must obey.

`plan.md` §4 makes this half of K0. The corpus exists because two managers can
produce identical logits and completely different ownership behaviour: one that
reuses nothing and one that reuses everything both return the right tokens. Only
the operation sequence distinguishes them, so the sequence is what gets compared
-- Python reference against Python reference across a refactor, and later
against C++ (`kv-cache-manager-v1.md` §15, §16.6).

WHY THIS SHIPS BEFORE ANY MANAGER EXISTS. `validate_trace` enforces the ordering
rules of §7 and §13, and those are properties of the CONTRACT, not of an
implementation: a failed transaction publishes nothing whichever manager failed
it. Writing them now means the first implementation is measured against the
specification rather than against its own first output -- which is the trap a
recorded-only corpus sets, since whatever the implementation did on day one
becomes the definition of correct.

The corpus file itself is therefore seeded with sequences that are LEGAL and
sequences that are ILLEGAL, and the illegal ones carry the clause they violate.
A corpus of only-good traces cannot tell a validator that works from one that
returns None.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .kv_manager import TRACE_SCHEMA_VERSION, KVOperation

__all__ = [
    "TraceViolation",
    "validate_trace",
    "load_corpus",
    "dump_trace",
    "load_trace",
    "load_wire_trace",
    "parse_wire_detail",
]

# The operations a trace may contain, matching the KVCacheManager protocol.
RELEASE_LEASE = "release_lease"
OBSERVED_LEAK = "observed_leak"
OPEN_SESSION = "open_session"
RESOLVE_READ_PLAN = "resolve_read_plan"
BEGIN_TRANSACTION = "begin_transaction"
STAGE = "stage"
COMMIT = "commit"
PUBLISH_IDENTITY = "publish_identity"
ABORT = "abort"
CLOSE_SESSION = "close_session"
REFUSAL = "refusal"
EVICT = "evict"

KNOWN_OPS = frozenset({
    OPEN_SESSION, RESOLVE_READ_PLAN, RELEASE_LEASE, BEGIN_TRANSACTION, STAGE,
    COMMIT, PUBLISH_IDENTITY, ABORT, CLOSE_SESSION, OBSERVED_LEAK,
    REFUSAL, EVICT,
})

_INTEGER_WIRE_DETAIL_FIELDS = frozenset({
    "bytes",
    "layer",
    "layers",
    "request_local_rows",
    "reserve_bytes",
    "reserved_bytes",
    "reused_rows",
    "shared_resident_total",
})


class TraceViolation(AssertionError):
    """A trace broke an ordering rule the contract states.

    An AssertionError rather than a ValueError: a violation means an
    implementation did something the specification forbids, which is a failed
    check, not a malformed input.
    """


def _unescape_wire_detail(value: str) -> str:
    """Inverse of native ``Detail`` escaping, in the producer's fixed order."""
    for token, character in (("%3B", ";"), ("%3D", "="), ("%25", "%")):
        value = value.replace(token, character)
    return value


def _escape_wire_detail(value: str) -> str:
    """Reproduce the native encoder so the decoder can reject aliases."""
    return value.replace("%", "%25").replace(";", "%3B").replace("=", "%3D")


def parse_wire_detail(encoded: str) -> dict[str, Any]:
    """Decode schema-v3 native detail without inventing a second parser."""
    parsed: dict[str, Any] = {}
    if not encoded:
        return parsed
    for encoded_field in encoded.split(";"):
        key, separator, value = encoded_field.partition("=")
        if not separator:
            raise RuntimeError(
                f"native trace detail has no '=': {encoded_field!r}"
            )
        encoded_key, encoded_value = key, value
        key = _unescape_wire_detail(encoded_key)
        value = _unescape_wire_detail(encoded_value)
        if (
            _escape_wire_detail(key) != encoded_key
            or _escape_wire_detail(value) != encoded_value
        ):
            raise RuntimeError(
                f"native trace detail is not canonically escaped: {encoded_field!r}"
            )
        if key in parsed:
            raise RuntimeError(f"native trace detail repeats {key!r}")
        if key == "expected_layers":
            try:
                wire_items = value.split(",")
                layers = tuple(int(item) for item in wire_items)
            except ValueError as exc:
                raise RuntimeError(
                    f"native trace has invalid expected_layers={value!r}"
                ) from exc
            if any(
                str(layer) != item
                for layer, item in zip(layers, wire_items, strict=True)
            ):
                raise RuntimeError(
                    f"native trace has non-canonical expected_layers={value!r}"
                )
            parsed[key] = layers
        elif key == "shareable":
            if value not in {"True", "False"}:
                raise RuntimeError(
                    f"native trace has invalid shareable={value!r}"
                )
            parsed[key] = value == "True"
        elif key in _INTEGER_WIRE_DETAIL_FIELDS:
            try:
                number = int(value)
            except ValueError as exc:
                raise RuntimeError(
                    f"native trace has non-integer {key}={value!r}"
                ) from exc
            if str(number) != value:
                raise RuntimeError(
                    f"native trace has non-canonical {key}={value!r}"
                )
            parsed[key] = number
        else:
            parsed[key] = value
    return parsed


def load_wire_trace(serialised: object) -> tuple[KVOperation, ...]:
    """Load the exact schema-v3 document returned by the native C boundary."""
    if type(serialised) is not dict or set(serialised) != {
        "schema_version",
        "operations",
    }:
        raise ValueError("native trace must have schema_version and operations")
    if serialised["schema_version"] != TRACE_SCHEMA_VERSION:
        raise ValueError(
            f"native trace schema version {serialised['schema_version']!r} is "
            f"not {TRACE_SCHEMA_VERSION}"
        )
    operations = serialised["operations"]
    if type(operations) is not list:
        raise ValueError("native trace operations must be a list")
    loaded: list[KVOperation] = []
    for entry in operations:
        if type(entry) is not dict or set(entry) != {"op", "session", "detail"}:
            raise ValueError(f"native trace entry has an invalid shape: {entry!r}")
        if any(type(entry[name]) is not str for name in entry):
            raise ValueError(f"native trace entry fields must be strings: {entry!r}")
        loaded.append(
            KVOperation.of(
                entry["op"],
                entry["session"],
                **parse_wire_detail(entry["detail"]),
            )
        )
    return tuple(loaded)


def _detail(op: KVOperation) -> dict:
    return dict(op.detail)


def validate_trace(operations: tuple[KVOperation, ...]) -> None:
    """Enforce every ordering rule §7 and §13 state. Raises on the first break.

    Each rule below cites the clause it comes from, because a rule without a
    clause is a preference, and the next person cannot tell which is which.
    """
    live_sessions: set[str] = set()
    closed_sessions: set[str] = set()
    # transaction -> (session, expected layer set, staged layers, resolved)
    open_transactions: dict[str, dict] = {}
    committed: dict[str, str] = {}   # transaction -> owning session
    published: set[str] = set()
    # §17.5 / §17.6: the gate used to SKIP resolve and release entirely, so a
    # lease could be opened and never closed, or a lease nobody issued could be
    # released, and the trace still validated. A gate that skips an operation
    # cannot enforce anything about it.
    open_leases: dict[str, str] = {}     # lease -> owning session
    issued_leases: set[str] = set()
    # A shareable commit that never publishes leaves pages committed and
    # unreachable -- §7's ordering says publication FOLLOWS commit, not that it
    # is optional.
    awaiting_publish: dict[str, str] = {}

    for index, op in enumerate(operations):
        where = f"operation {index} ({op.op})"
        if op.op not in KNOWN_OPS:
            raise TraceViolation(f"{where}: unknown operation")
        detail = _detail(op)

        # A refusal describes an attempted operation that changed no ownership.
        # It may name a stale or never-opened session, so it is checked before
        # the live-session rule. The status is part of the differential truth:
        # INVALID_ARGUMENT and ADMISSION_REFUSED are different contracts.
        if op.op == REFUSAL:
            for name in ("operation", "status"):
                if not isinstance(detail.get(name), str) or not detail[name]:
                    raise TraceViolation(
                        f"{where}: refusal has no named {name}"
                    )
            continue

        if op.op == OPEN_SESSION:
            if op.session in live_sessions:
                raise TraceViolation(
                    f"{where}: session {op.session!r} is already open; one "
                    "session is one decode-run lifetime (§4.2)"
                )
            live_sessions.add(op.session)
            closed_sessions.discard(op.session)
            continue

        # §4.2: every other operation belongs to a live session. A staging or
        # publish that outlives its session is precisely the leaked ownership
        # this milestone exists to end.
        if op.session not in live_sessions:
            reason = ("was closed" if op.session in closed_sessions
                      else "was never opened")
            raise TraceViolation(
                f"{where}: session {op.session!r} {reason} (§4.2)"
            )

        if op.op == BEGIN_TRANSACTION:
            transaction = detail.get("transaction")
            if transaction is None:
                raise TraceViolation(f"{where}: no transaction named")
            if transaction in open_transactions:
                raise TraceViolation(
                    f"{where}: transaction {transaction!r} is already open"
                )
            layers = detail.get("expected_layers")
            if not isinstance(layers, tuple) or not layers:
                raise TraceViolation(
                    f"{where}: a transaction must declare its expected layer "
                    "set up front, or completeness is decided at commit by "
                    "whatever arrived (§4.7)"
                )
            open_transactions[transaction] = {
                "session": op.session, "expected": set(layers), "staged": set(),
            }
            continue

        if op.op in {STAGE, COMMIT, ABORT}:
            transaction = detail.get("transaction")
            record = open_transactions.get(transaction)
            if record is None:
                state = ("already finished" if transaction in committed
                         else "was never opened")
                raise TraceViolation(
                    f"{where}: transaction {transaction!r} {state}"
                )
            if record["session"] != op.session:
                raise TraceViolation(
                    f"{where}: transaction {transaction!r} belongs to session "
                    f"{record['session']!r}, not {op.session!r}"
                )

        if op.op == STAGE:
            layer = detail.get("layer")
            if layer not in record["expected"]:
                raise TraceViolation(
                    f"{where}: layer {layer} is not in the transaction's "
                    f"declared set {sorted(record['expected'])} (§7 step 5)"
                )
            if layer in record["staged"]:
                raise TraceViolation(
                    f"{where}: layer {layer} staged twice"
                )
            record["staged"].add(layer)
            continue

        if op.op == COMMIT:
            # §4.7 / §7 step 5: the COMPLETE expected layer set, or nothing
            # becomes visible. A short commit is a failed transaction, not a
            # smaller world.
            missing = sorted(record["expected"] - record["staged"])
            if missing:
                raise TraceViolation(
                    f"{where}: committing without layers {missing}; no partial "
                    "layer set becomes visible (§4.7)"
                )
            committed[transaction] = op.session
            if detail.get("shareable") is True:
                awaiting_publish[transaction] = op.session
            del open_transactions[transaction]
            continue

        if op.op == ABORT:
            # §7 step 7 / §13: a failed transaction publishes nothing.
            del open_transactions[transaction]
            continue

        if op.op == PUBLISH_IDENTITY:
            transaction = detail.get("transaction")
            if transaction in committed and committed[transaction] != op.session:
                raise TraceViolation(
                    f"{where}: transaction {transaction!r} committed in session "
                    f"{committed[transaction]!r} is being published from "
                    f"{op.session!r}; a committed transaction keeps its owning "
                    "session (§17.7)"
                )
            if transaction not in committed:
                raise TraceViolation(
                    f"{where}: identity published for transaction "
                    f"{transaction!r} which has not committed; publication "
                    "order is strict -- physical commit precedes identity "
                    "visibility (§7)"
                )
            if transaction in published:
                raise TraceViolation(
                    f"{where}: identity for {transaction!r} published twice"
                )
            if not detail.get("identity"):
                raise TraceViolation(
                    f"{where}: no identity named; an entry without one is a "
                    "refused store, not a weak hit (§6)"
                )
            published.add(transaction)
            awaiting_publish.pop(transaction, None)
            continue

        if op.op == OBSERVED_LEAK:
            # An observer reporting that its subject leaked. Not a manager
            # operation, and never legal in a trace that claims a clean run.
            raise TraceViolation(
                f"{where}: the observed sequence leaked "
                f"{dict(detail)}; §7 step 8 requires close to release every "
                "request-local and staging reference"
            )
        if op.op == EVICT:
            if not isinstance(detail.get("bytes"), int) or detail["bytes"] < 0:
                raise TraceViolation(f"{where}: eviction has no byte count")
            if not detail.get("world"):
                raise TraceViolation(f"{where}: eviction names no world")
            continue
        if op.op == RESOLVE_READ_PLAN:
            lease = detail.get("lease")
            if lease is None:
                raise TraceViolation(
                    f"{where}: a read plan names no lease; §17.5 makes every "
                    "reuse a leased claim with a release path"
                )
            if lease in issued_leases:
                raise TraceViolation(
                    f"{where}: lease {lease!r} was already issued; §17.6 "
                    "forbids reusing an identifier"
                )
            issued_leases.add(lease)
            open_leases[lease] = op.session
            continue

        if op.op == RELEASE_LEASE:
            lease = detail.get("lease")
            if lease not in issued_leases:
                raise TraceViolation(
                    f"{where}: lease {lease!r} was never issued; a stale handle "
                    "resolves to a refusal, never to another object (§17.6)"
                )
            if lease not in open_leases:
                raise TraceViolation(
                    f"{where}: lease {lease!r} was already released; one lease "
                    "has one release (§17.5)"
                )
            if open_leases[lease] != op.session:
                raise TraceViolation(
                    f"{where}: lease {lease!r} belongs to session "
                    f"{open_leases[lease]!r}, not {op.session!r}"
                )
            open_leases.pop(lease, None)
            continue

        if op.op == CLOSE_SESSION:
            # §7 step 8 / §4.2: closing releases every request-local and
            # staging reference. A transaction still open at close is exactly
            # the leak, so it is a violation rather than an implicit abort --
            # an implicit abort would let a manager leak and still look clean.
            stranded = [name for name, rec in open_transactions.items()
                        if rec["session"] == op.session]
            if stranded:
                raise TraceViolation(
                    f"{where}: session closed with transaction(s) "
                    f"{sorted(stranded)} still open; closing must release every "
                    "request-local and staging reference (§4.2, §7 step 8)"
                )
            held = sorted(name for name, owner in open_leases.items()
                          if owner == op.session)
            if held:
                raise TraceViolation(
                    f"{where}: session closed holding lease(s) {held}; a lease "
                    "never released is a page that can never be evicted "
                    "(§17.5)"
                )
            unpublished = sorted(name for name, owner in awaiting_publish.items()
                                 if owner == op.session)
            if unpublished:
                raise TraceViolation(
                    f"{where}: session closed with shareable transaction(s) "
                    f"{unpublished} committed but never published; §7's "
                    "ordering makes publication follow commit, not optional"
                )
            live_sessions.discard(op.session)
            closed_sessions.add(op.session)
            continue

    if open_transactions:
        raise TraceViolation(
            f"trace ends with transaction(s) {sorted(open_transactions)} still "
            "open"
        )
    if live_sessions:
        raise TraceViolation(
            f"trace ends with session(s) {sorted(live_sessions)} still open; "
            "§13 requires destruction to leave no request-local residency"
        )


def _to_json(value):
    """Recursive. The first version converted only the OUTER level, so a nested
    tuple did not merely compare unequal after a round trip -- it RAISED on
    reload, because the inner list is not a value `KVOperation` accepts. The
    round-trip test used a flat tuple and never saw it."""
    if isinstance(value, tuple):
        return [_to_json(item) for item in value]
    return value


def _from_json(value):
    if isinstance(value, list):
        return tuple(_from_json(item) for item in value)
    return value


def dump_trace(operations: tuple[KVOperation, ...]) -> dict:
    """Versioned canonical JSON form (§17.7)."""
    return {
        "version": TRACE_SCHEMA_VERSION,
        "operations": [
            {"op": op.op, "session": op.session,
             "detail": {key: _to_json(value) for key, value in op.detail}}
            for op in operations
        ],
    }


def load_trace(serialised: dict) -> tuple[KVOperation, ...]:
    """Inverse of `dump_trace`, recursively and fail-closed on version."""
    if not isinstance(serialised, dict):
        raise ValueError("a trace must be a versioned object, not a bare list")
    version = serialised.get("version")
    if version != TRACE_SCHEMA_VERSION:
        raise ValueError(
            f"trace schema version {version!r} is not {TRACE_SCHEMA_VERSION}; "
            "an unknown trace version is refused rather than parsed"
        )
    entries = serialised.get("operations")
    if not isinstance(entries, list):
        raise ValueError("a trace must carry an operations list")
    restored = []
    for entry in entries:
        detail = {key: _from_json(value)
                  for key, value in entry.get("detail", {}).items()}
        restored.append(KVOperation.of(entry["op"], entry["session"], **detail))
    return tuple(restored)


def load_corpus(path: Path) -> list[dict]:
    """The maintained corpus: named cases, each legal or illegal-with-a-clause.

    A corpus of only-legal traces cannot distinguish a validator that works from
    one that returns None, so the illegal cases are part of the artifact rather
    than part of a test file.
    """
    data = json.loads(Path(path).read_text())
    version = data.get("version")
    if not isinstance(version, int):
        raise ValueError("the corpus states no integer schema version")
    if version != TRACE_SCHEMA_VERSION:
        # Refused, not best-effort parsed: a differential oracle that guesses at
        # an older shape compares two things it cannot name (§17.7).
        raise ValueError(
            f"corpus schema version {version!r} is not {TRACE_SCHEMA_VERSION}; "
            "an unknown trace version is refused rather than parsed"
        )
    cases = data["cases"]
    for case in cases:
        if not case.get("name"):
            raise ValueError("every corpus case must be named")
        if case.get("legal") is None:
            raise ValueError(f"case {case['name']!r} does not say if it is legal")
        if not case["legal"] and not case.get("violates"):
            raise ValueError(
                f"illegal case {case['name']!r} must name the clause it "
                "violates, or it is an assertion without a reason"
            )
        if not case["legal"] and not case.get("expect"):
            # Asserting only THAT a trace was refused is not enough, and this
            # was measured: with the cross-session-ownership rule disabled and
            # with the close-releases-staging rule disabled, both traces were
            # still refused -- by a LATER rule, for an unrelated reason -- and
            # the suite stayed green. `expect` is the phrase the refusal must
            # contain, so a rule that stops working is visible even when some
            # other rule happens to reject the same trace.
            raise ValueError(
                f"illegal case {case['name']!r} must state the refusal it "
                "expects; 'something raised' cannot tell a working rule from "
                "a later rule refusing the same trace for another reason"
            )
        # The maintained corpus itself carries one schema version. Normalise
        # its compact per-case lists into the same object `dump_trace` emits so
        # every consumer crosses the version check rather than bypassing it.
        if isinstance(case.get("trace"), list):
            case["trace"] = {
                "version": version,
                "operations": case["trace"],
            }
    return cases
