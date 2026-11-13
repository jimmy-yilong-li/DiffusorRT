"""WindowExecutionPlan <-> the all-integer ABI form.

The plan carries five position tracks plus a per-position cache action; all of
it crosses. Parallel arrays (positions, values) encode the cache-action map
because a C ABI has no dict — the two arrays share a length, given explicitly as
n_cache_entries so no consumer has to infer it.
"""
from __future__ import annotations

from dataclasses import dataclass
import operator

from .enums import CACHE_ACTION_BY_NAME, REUSE_SOURCE_BY_NAME, encode
from .lifetimes import UNSET

_REUSE_APPROX_ABI_VALUE = int(CACHE_ACTION_BY_NAME["reuse_approx"])
_REUSE_APPROX_ERROR = (
    "REUSE_APPROX is unsupported by ABI v1; approximate reuse is "
    "experimental Python-only"
)
_INT32_MAX = (1 << 31) - 1
_INT32_MIN = -(1 << 31)
_INT64_MAX = (1 << 63) - 1
_INT64_MIN = -(1 << 63)


@dataclass(frozen=True, slots=True)
class PlanV1:
    step: int
    window_end: int
    active_positions: tuple[int, ...]
    dirty_positions: tuple[int, ...]
    remask_positions: tuple[int, ...]
    frozen_positions: tuple[int, ...]
    dependency_risk_positions: tuple[int, ...]
    forward_scope: tuple[int, ...]
    n_cache_entries: int
    cache_action_positions: tuple[int, ...]
    cache_action_values: tuple[int, ...]
    n_reuse_entries: int
    reuse_source_positions: tuple[int, ...]
    reuse_source_values: tuple[int, ...]


def _validate_exact_reuse_semantics(*, reuse_source, cache_action,
                                    forward_scope) -> None:
    """Exact provenance is meaningful only for an exact-reused query row."""
    scope = set(forward_scope)
    for position, tag in reuse_source.items():
        if position not in scope:
            raise ValueError(
                f"reuse_source position {position} ({tag!r}) is not in "
                "forward_scope; exact provenance is valid only for forwarded "
                "positions"
            )
        action = cache_action.get(position)
        if action is None:
            raise ValueError(
                f"reuse_source position {position} ({tag!r}) has no "
                "cache_action; exact provenance requires REUSE_EXACT"
            )
        if getattr(action, "value", action) != "reuse_exact":
            raise ValueError(
                f"reuse_source position {position} ({tag!r}) has cache_action "
                f"{action!r}, not REUSE_EXACT"
            )


def _require_integer(value, *, field: str) -> int:
    """Return an exact Python integer; never truncate a wire value."""
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer, not {value!r}")
    try:
        return operator.index(value)
    except TypeError:
        raise ValueError(f"{field} must be an integer, not {value!r}") from None


def _normalize_positions(name: str, positions) -> tuple[int, ...]:
    normalized = tuple(
        _require_integer(position, field=f"{name} position")
        for position in positions
    )
    negative = [position for position in normalized if position < 0]
    if negative:
        raise ValueError(
            f"{name} contains negative positions {negative}; positions are "
            "absolute canvas indices and cannot be negative"
        )
    too_large = [position for position in normalized if position > _INT32_MAX]
    if too_large:
        raise ValueError(
            f"{name} contains positions {too_large} outside the signed int32 "
            "wire range"
        )
    return normalized


def _normalize_wire_scalar(value, *, field: str, bits: int) -> int:
    normalized = _require_integer(value, field=field)
    minimum, maximum = (
        (_INT32_MIN, _INT32_MAX) if bits == 32 else (_INT64_MIN, _INT64_MAX)
    )
    if normalized < minimum or normalized > maximum:
        raise ValueError(
            f"{field} {normalized} is outside the signed int{bits} wire range"
        )
    return normalized


def _validate_context_free(*, step, window_end, position_tracks):
    """Reject everything decidable WITHOUT a canvas length.

    B7 required semantic validation symmetrically on encode and decode. The
    provenance half landed; the RANGE half did not -- the codec accepted
    negative positions, a negative ``window_end``, and a ``forward_scope``
    position at or beyond an explicit ``window_end``. ``ForwardRequest`` rejects
    those, so the Python path was safe, but ``PlanV1`` is a frozen wire type an
    independent decoder reads on its own, where no request wraps it.

    Deliberately NOT adding a canvas length to ``PlanV1``:
    ``ForwardRequest.n_token_ids`` already carries it, and a second copy would
    create a new way for two fields to disagree. So the split is:

    * context-FREE (here): negative positions, ``window_end < -1``, negative
      ``step``, and ``forward_scope`` beyond an explicit ``window_end``;
    * context-DEPENDENT (``ForwardRequest``): ``position < n_token_ids`` and
      ``window_end <= n_token_ids``.

    A ``PlanV1`` alone therefore carries only context-free validity. Executing
    one requires validating the whole ``ForwardRequest`` first, before any
    handle, cache, or backend side effect.
    """
    normalized_step = _normalize_wire_scalar(step, field="step", bits=64)
    normalized_window_end = _normalize_wire_scalar(
        window_end, field="window_end", bits=64)
    if normalized_step < 0:
        raise ValueError(
            f"step {normalized_step} is negative; steps are monotonic counters")
    if normalized_window_end < UNSET:
        raise ValueError(
            f"window_end {normalized_window_end} is below the UNSET sentinel "
            f"({UNSET}); "
            "a window boundary is either unset or a non-negative canvas index"
        )
    normalized_tracks = {
        name: _normalize_positions(name, positions)
        for name, positions in position_tracks
    }
    forward_scope = normalized_tracks["forward_scope"]
    if normalized_window_end != UNSET:
        beyond = [
            position for position in forward_scope
            if position >= normalized_window_end
        ]
        if beyond:
            raise ValueError(
                f"forward_scope positions {beyond} are at or beyond "
                f"window_end={normalized_window_end}; only forward_scope is "
                "window-bounded, "
                "and a position outside the declared window cannot be forwarded"
            )
    return normalized_step, normalized_window_end, normalized_tracks


def encode_plan(plan) -> PlanV1:
    raw_cache_entries = tuple(plan.cache_action.items())
    raw_reuse_source = getattr(plan, "reuse_source", None) or {}
    raw_reuse_entries = tuple(raw_reuse_source.items())
    step, window_end, tracks = _validate_context_free(
        step=plan.step,
        window_end=UNSET if plan.window_end is None else plan.window_end,
        position_tracks=(
            ("active_positions", plan.active_positions),
            ("dirty_positions", plan.dirty_positions),
            ("remask_positions", plan.remask_positions),
            ("frozen_positions", plan.frozen_positions),
            ("dependency_risk_positions", plan.dependency_risk_positions),
            ("forward_scope", plan.forward_scope),
            ("reuse_source_positions", (
                position for position, _ in raw_reuse_entries)),
            ("cache_action_positions", (
                position for position, _ in raw_cache_entries)),
        ),
    )
    cache_action = dict(zip(
        tracks["cache_action_positions"],
        (action for _, action in raw_cache_entries),
    strict=True))
    reuse_source = dict(zip(
        tracks["reuse_source_positions"],
        (tag for _, tag in raw_reuse_entries),
    strict=True))
    missing = [
        position for position in tracks["forward_scope"]
        if position not in cache_action
    ]
    if missing:
        raise ValueError(
            f"cache_action is missing forward_scope positions {missing}; an "
            "under-specified plan would force the engine to guess"
        )
    entries = sorted(cache_action.items())
    for position, action in entries:
        if getattr(action, "value", action) == "reuse_approx":
            raise ValueError(
                f"cache_action position {position}: {_REUSE_APPROX_ERROR}")
    reuse_entries = sorted(reuse_source.items())
    _validate_exact_reuse_semantics(
        reuse_source=reuse_source,
        cache_action=cache_action,
        forward_scope=tracks["forward_scope"],
    )
    cache_action_values = tuple(
        _normalize_wire_scalar(
            CACHE_ACTION_BY_NAME[action.value],
            field="cache_action_values entry",
            bits=32,
        )
        for _, action in entries
    )
    reuse_source_values = tuple(
        _normalize_wire_scalar(
            encode(REUSE_SOURCE_BY_NAME, tag, "reuse_source"),
            field="reuse_source_values entry",
            bits=32,
        )
        for _, tag in reuse_entries
    )
    return PlanV1(
        step=step,
        window_end=window_end,
        active_positions=tracks["active_positions"],
        dirty_positions=tracks["dirty_positions"],
        remask_positions=tracks["remask_positions"],
        frozen_positions=tracks["frozen_positions"],
        dependency_risk_positions=tracks["dependency_risk_positions"],
        forward_scope=tracks["forward_scope"],
        n_cache_entries=len(entries),
        cache_action_positions=tuple(position for position, _ in entries),
        cache_action_values=cache_action_values,
        n_reuse_entries=len(reuse_entries),
        reuse_source_positions=tuple(position for position, _ in reuse_entries),
        reuse_source_values=reuse_source_values,
    )


def _validated_parallel_arrays(*, count: int, positions, values,
                               group: str, count_field: str):
    """B2: the wire form must be internally consistent before it is trusted.

    zip() silently truncates to the shorter array and dict() silently lets a
    later duplicate win; both turn a corrupt wire form into a plausible-looking
    plan. Validate the count against BOTH arrays and reject duplicates.
    """
    normalized_count = _normalize_wire_scalar(
        count, field=count_field, bits=64)
    if normalized_count < 0:
        raise ValueError(f"{count_field} {normalized_count} cannot be negative")
    normalized_values = tuple(
        _normalize_wire_scalar(
            value, field=f"{group}_values entry", bits=32)
        for value in values
    )
    if normalized_count != len(positions) or normalized_count != len(normalized_values):
        raise ValueError(
            f"{group}: {count_field}={normalized_count} but positions has "
            f"{len(positions)} entries and values has {len(normalized_values)}; the wire "
            "form is internally inconsistent"
        )
    if len(set(positions)) != len(positions):
        raise ValueError(
            f"{group}: duplicate positions in the wire form; a position may "
            "not carry two values"
        )
    return zip(positions, normalized_values, strict=True)


def decode_plan(encoded: PlanV1):
    from ..backends.window_plan import CacheAction, WindowExecutionPlan

    # Symmetric with encode_plan: an independent wire decoder must reject the
    # same context-free errors. Untrusted bytes reach here, so this side is the
    # one that actually matters for a native implementation.
    step, window_end, tracks = _validate_context_free(
        step=encoded.step,
        window_end=encoded.window_end,
        position_tracks=(
            ("active_positions", encoded.active_positions),
            ("dirty_positions", encoded.dirty_positions),
            ("remask_positions", encoded.remask_positions),
            ("frozen_positions", encoded.frozen_positions),
            ("dependency_risk_positions", encoded.dependency_risk_positions),
            ("forward_scope", encoded.forward_scope),
            ("cache_action_positions", encoded.cache_action_positions),
            ("reuse_source_positions", encoded.reuse_source_positions),
        ),
    )

    by_abi_value = {
        int(CACHE_ACTION_BY_NAME[action.value]): action for action in CacheAction
    }
    cache_action = {}
    for position, value in _validated_parallel_arrays(
            count=encoded.n_cache_entries,
            positions=tracks["cache_action_positions"],
            values=encoded.cache_action_values,
            group="cache_action", count_field="n_cache_entries"):
        if value == _REUSE_APPROX_ABI_VALUE:
            raise ValueError(
                f"cache_action position {position}: {_REUSE_APPROX_ERROR}")
        try:
            cache_action[position] = by_abi_value[value]
        except KeyError:
            raise ValueError(
                f"cache_action value {value} has no ABI decoding; the wire "
                "form is not one this version can read"
            ) from None
    by_reuse_value = {int(v): name for name, v in REUSE_SOURCE_BY_NAME.items()}
    reuse_source = {}
    for position, value in _validated_parallel_arrays(
            count=encoded.n_reuse_entries,
            positions=tracks["reuse_source_positions"],
            values=encoded.reuse_source_values,
            group="reuse_source", count_field="n_reuse_entries"):
        try:
            reuse_source[position] = by_reuse_value[value]
        except KeyError:
            raise ValueError(
                f"reuse_source value {value} has no ABI decoding; the wire form "
                "is not one this version can read"
            ) from None
    _validate_exact_reuse_semantics(
        reuse_source=reuse_source,
        cache_action=cache_action,
        forward_scope=tracks["forward_scope"],
    )
    return WindowExecutionPlan(
        step=step,
        active_positions=tracks["active_positions"],
        dirty_positions=tracks["dirty_positions"],
        remask_positions=tracks["remask_positions"],
        frozen_positions=tracks["frozen_positions"],
        dependency_risk_positions=tracks["dependency_risk_positions"],
        cache_action=cache_action,
        forward_scope=tracks["forward_scope"],
        window_end=None if window_end == UNSET else window_end,
        reuse_source=reuse_source,
    )
