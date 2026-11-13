"""CorrectionPolicy — thin bridge between error-fixing detectors and WindowExecutionPlan.

MLX-free, import-light (mirrors ``window_plan.py``).  This module is the C1 bridge
(WS3, Active Window Runtime v1 milestone): it maps existing ``ErrorMaskDetector`` /
``RemaskPolicy`` output from ``diffusion_runtime`` into the ``dirty_positions`` /
``remask_positions`` classification consumed by ``WindowExecutionPlan``.

BRIDGE ONLY — no detector logic lives here.  All detector implementations
(``LowConfidenceDetector``, ``RepetitionDetector``, ``KLDriftDetector``,
``CompositeDetector``, ``ConfWeightedRemask``) remain in ``diffusion_runtime.py``
and are IMPORTED here, never reimplemented.

Public names
------------
CorrectionPlan
    Frozen dataclass: ``dirty_positions``, ``remask_positions``, ``reasons``.
flags_to_window_corrections(flagged, remask_selected) -> CorrectionPlan
    Pure mapping from detector output to a CorrectionPlan.
CorrectionPolicy
    Holds a detector + remask_policy (both default to NoOp / OFF).
    ``corrections_for(*, state, commit_history, newly_unmasked_count=0, **signals)``
    runs the detector + remask_policy and returns a CorrectionPlan.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# CorrectionPlan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrectionPlan:
    """The classification output of a single correction pass.

    ``remask_positions`` — positions the RemaskPolicy selected for re-masking
        (a flagged committed position that should be re-opened for correction).
    ``dirty_positions`` — flagged positions NOT yet selected for remask
        (detected as suspicious but not acted on this round).

    Invariant: ``dirty_positions`` and ``remask_positions`` are DISJOINT.

    ``reasons`` is an optional provenance map: position → detector-reason string.
    Kept minimal; callers may ignore it.
    """

    dirty_positions: tuple[int, ...] = ()
    remask_positions: tuple[int, ...] = ()
    reasons: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        dirty = set(self.dirty_positions)
        remask = set(self.remask_positions)
        overlap = dirty & remask
        if overlap:
            raise ValueError(
                f"CorrectionPlan: dirty_positions and remask_positions must be "
                f"disjoint; overlapping positions: {sorted(overlap)}"
            )


# ---------------------------------------------------------------------------
# Pure mapping: detector output → CorrectionPlan
# ---------------------------------------------------------------------------


def flags_to_window_corrections(
    flagged: list,
    remask_selected: list,
) -> CorrectionPlan:
    """Map detector + remask-policy output to a CorrectionPlan.

    Parameters
    ----------
    flagged:
        List of ``FlaggedPosition`` objects returned by an
        ``ErrorMaskDetector.flag(...)`` call.
    remask_selected:
        List of int positions returned by a ``RemaskPolicy.select_remask(...)``
        call (a subset of the positions in ``flagged``).

    Returns
    -------
    CorrectionPlan
        ``remask_positions`` = the positions the policy selected (as ``tuple[int, ...]``).
        ``dirty_positions`` = flagged positions NOT in ``remask_selected``
            (detected but not acted on this round).
        ``reasons`` = position → reason string, covering all flagged positions.

    The two output sets are guaranteed DISJOINT by construction (a position is
    either selected for remask OR left as dirty, never both).
    """
    remask_set: set[int] = set(int(p) for p in remask_selected)
    reasons: dict[int, str] = {}
    dirty_list: list[int] = []
    remask_list: list[int] = []

    for fp in flagged:
        pos = int(fp.position)
        reasons[pos] = str(fp.reason)
        if pos in remask_set:
            remask_list.append(pos)
        else:
            dirty_list.append(pos)

    return CorrectionPlan(
        dirty_positions=tuple(sorted(dirty_list)),
        remask_positions=tuple(sorted(remask_list)),
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# CorrectionPolicy — holds a detector + remask policy (default = NoOp / OFF)
# ---------------------------------------------------------------------------


@dataclass
class CorrectionPolicy:
    """Bridge: runs a detector + remask policy and returns a ``CorrectionPlan``.

    Holds an ``ErrorMaskDetector`` and a ``RemaskPolicy`` (both imported from
    ``diffusion_runtime``; never reimplemented here).  The defaults are the
    NoOp instances (OFF), so the default policy is behavior-neutral:
    ``corrections_for(...)`` always returns an empty ``CorrectionPlan``.

    Parameters
    ----------
    detector:
        Any ``ErrorMaskDetector``-compatible object.  Default:
        ``diffusion_runtime.NoOpErrorMaskDetector()``.
    remask_policy:
        Any ``RemaskPolicy``-compatible object.  Default:
        ``diffusion_runtime.NoOpRemaskPolicy()``.
    """

    detector: Any = field(default=None)
    remask_policy: Any = field(default=None)

    def __post_init__(self) -> None:
        # Import here so the module stays importable without diffusion_runtime
        # being touched at parse time (mirrors the lazy-import pattern in the
        # rest of the codebase).  The NoOp defaults are created lazily on first
        # use when the caller passes None.
        if self.detector is None:
            from edllm.backends.diffusion_runtime import NoOpErrorMaskDetector
            object.__setattr__(self, "detector", NoOpErrorMaskDetector())
        if self.remask_policy is None:
            from edllm.backends.diffusion_runtime import NoOpRemaskPolicy
            object.__setattr__(self, "remask_policy", NoOpRemaskPolicy())

    def corrections_for(
        self,
        *,
        state: Any,
        commit_history: Any,
        newly_unmasked_count: int = 0,
        **signals: Any,
    ) -> CorrectionPlan:
        """Run the detector + remask policy and return a ``CorrectionPlan``.

        Parameters
        ----------
        state:
            Duck-typed canvas state (``MaskState`` or compatible stub).
        commit_history:
            The result of ``DecodeEventLog.commit_history()`` — a dict mapping
            position → ``Commit`` (with ``.token`` and ``.score`` fields).
        newly_unmasked_count:
            The count of positions newly committed in the preceding decode pass.
            Passed to ``RemaskPolicy.select_remask`` as ``newly_unmasked_count``
            (the RemeDi monotone-progress invariant uses this to cap re-masking).
            Default ``0`` → the policy caps re-masking at 0 (safe, no mutations).
        **signals:
            Extra keyword signals forwarded to ``detector.flag(...)`` (e.g.
            ``dist_history`` for ``KLDriftDetector``).  Detectors that don't need
            them ignore the extras.

        Returns
        -------
        CorrectionPlan
            Empty (both tuples ``()``) when the default NoOp detector/policy is
            in use.  Non-empty when a real detector flags committed positions.
        """
        flagged: list = self.detector.flag(
            state=state,
            commit_history=commit_history,
            **signals,
        )
        remask_selected: list = self.remask_policy.select_remask(
            flagged=flagged,
            state=state,
            newly_unmasked_count=newly_unmasked_count,
        )
        return flags_to_window_corrections(flagged, remask_selected)
