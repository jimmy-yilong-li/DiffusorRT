"""Diffusion Runtime Layer (seed): model-agnostic masked-diffusion decode loop.

This module is the first cut of a future *Diffusion Runtime Layer*. It owns the
decode loop that used to live inside ``mlx_dream.MLXDreamEngine.generate_entropy``
and exposes it as three first-class, model-agnostic citizens:

* :class:`MaskState` -- the masked sequence as a managed object. It holds the
  working ids, the prompt length, and the mask / EOS token ids; it answers
  "which generated positions are still masked?", commits a token at a position,
  and finalizes the sequence (first-EOS truncation, remaining-mask accounting,
  finish reason).
* :class:`UnmaskPolicy` (+ :class:`TopConfidence` / :class:`NegEntropy`) -- the
  *scoring* half: per masked position, how confident the model is (higher = more
  confident), on the RAW softmax (no temperature/top_p sharpening). Lifted
  verbatim from the legacy ``_confidence`` helpers.
* :class:`CommitPolicy` (+ entropy-count / threshold / block-threshold impls) --
  the *aggression* half: which of the scored masked positions to commit THIS
  forward. The block-threshold impl carries the active-block pointer as its own
  state.
* :func:`build_decode_strategy` constructs the named strategy for host and
  device callers. Each :class:`DecodingStrategy` assembles its unmask and commit
  policies and carries ``decode_loop_kind`` / ``entropy_used`` labels. The
  generic :func:`run_decode_loop` executes those policies:
  forward -> score -> select -> commit.
* :class:`ErrorMaskDetector` / :class:`RemaskPolicy` / :class:`QualityRecoveryLoop`
  -- the self-correction interfaces (Phase F3). In F1 they ship with **no-op
  defaults (OFF)**: the recovery loop with recovery disabled runs exactly one
  decode pass, so today's behaviour is unchanged.
* :class:`DiffusionIterationManager` -- runs the loop. It owns the
  :class:`MaskState`, calls the backend *forward callable*, applies the
  :class:`DecodingStrategy` each step, counts only real forwards, enforces the
  ``steps`` cap, and produces the stop reason.

The module is intentionally **mlx-free**: it operates on numpy logits plus a
forward callable supplied by the backend (the callable owns the MLX forward and
the Dream/MDLM logit shift). ``import numpy`` stays lazy inside the helpers so
importing this module never pulls numpy/mlx at module-import time, matching the
rest of the backends package.
"""

from __future__ import annotations

import contextlib
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


# A forward callable maps the current working ids to the backend forward output.
# In the stub tests this is already the (numpy) selection logits; in the real
# mlx_dream path it is the raw mlx logits that the injected ``select_transform``
# later shifts + converts to numpy. The manager only ever sees what
# ``select_transform`` returns, so the MLX forward + shift live behind these
# callables and this module stays mlx-free.
ForwardFn = Callable[[list[int]], Any]

# Turns one forward output into the numpy selection logits the strategy slices.
SelectTransform = Callable[[Any], Any]

# Negative-entropy / raw-max-prob confidence policy: rows -> (confidence, ids).
ConfidenceFn = Callable[[Any], tuple]

# Fires once per committed (position, token_id) so a backend KV cache can react.
CommitObserver = Callable[[int, int], None]

# A zero-arg factory returning a context manager that times a loop section.
PhaseFactory = Callable[[], "contextlib.AbstractContextManager"]


@contextlib.contextmanager
def _null_phase():
    yield


def _identity_select(raw: Any) -> Any:
    import numpy as np

    return np.asarray(raw, dtype=np.float32)


# ---------------------------------------------------------------------------
# Confidence policies (the per-position commit *signal*)
# ---------------------------------------------------------------------------
#
# These are the small internal helpers each strategy combines. They are moved
# verbatim from the legacy mlx_dream decode helpers so behavior is byte-exact.


def _entropy_confidence(
    logits,
    *,
    temperature: float,
    top_p: float | None,
) -> tuple:
    """Negative-entropy confidence + greedy-argmax token per row.

    The confidence is the (negative) Shannon entropy of the temperature/top_p
    adjusted softmax; higher (closer to zero) means a more peaked, more
    confident distribution. The committed token is the argmax of that adjusted
    distribution.
    """
    import numpy as np

    adjusted = np.asarray(logits, dtype=np.float64)
    if temperature > 0:
        adjusted = adjusted / float(temperature)
    if top_p is not None and top_p < 1.0:
        adjusted = _top_p_filter(adjusted, top_p=float(top_p))
    probs = _softmax(adjusted)
    sampled = np.argmax(probs, axis=-1).astype(np.int64)
    log_probs = np.log(probs + 1e-10)
    confidence = np.sum(probs * log_probs, axis=-1)
    return confidence.astype(np.float32), sampled.tolist()


_SPLIT_MIN_ELEMENTS = 1 << 20
_SCORING_WORKERS = min(8, os.cpu_count() or 1)
_scoring_executor = None
_scoring_lock = threading.Lock()


def _scoring_pool():
    """One small shared pool; NumPy releases the GIL inside these vector loops."""
    global _scoring_executor
    with _scoring_lock:  # Two first callers must not each build a pool.
        if _scoring_executor is None:
            import concurrent.futures

            _scoring_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=_SCORING_WORKERS, thread_name_prefix="diffusorrt-score",
            )
        return _scoring_executor


def _raw_max_prob_confidence(logits) -> tuple:
    """RAW softmax top-1 probability + argmax token per row.

    No temperature scaling and no top_p filtering: this is the true,
    unsharpened confidence used by the threshold commit policies.

    A window's frontier is up to a canvas of rows by a full vocabulary, and
    each row is scored independently. Such an array is split into blocks of
    rows scored in parallel by the same per-row operations, so every value is
    bit-identical to scoring it whole (measured 163 ms -> 44 ms per 128-row
    step of a 151,936-token vocabulary).
    """
    import numpy as np

    # Only row-major arrays: a block of rows of any other layout can be summed in
    # another order than the whole array, which changes the last bits.
    if isinstance(logits, np.ndarray) and logits.ndim == 2 and logits.shape[0] > 1 \
            and logits.size >= _SPLIT_MIN_ELEMENTS and logits.flags.c_contiguous:
        bounds = np.array_split(np.arange(logits.shape[0]), min(_SCORING_WORKERS, logits.shape[0]))
        try:
            parts = list(_scoring_pool().map(
                lambda rows: _raw_max_prob_rows(logits[rows[0]:rows[-1] + 1]),
                [rows for rows in bounds if len(rows)],
            ))
        except RuntimeError:
            # The pool refuses work once the interpreter is exiting; finish here instead.
            return _raw_max_prob_rows(logits)
        return (np.concatenate([part[0] for part in parts]),
                [token for part in parts for token in part[1]])
    return _raw_max_prob_rows(logits)


def _raw_max_prob_rows(logits) -> tuple:
    """The per-row scoring itself; see ``_raw_max_prob_confidence``.

    The float64 softmax, unchanged. A float32 version was 2.5x faster, but its
    error is not only in the last bits: it can make two distinct confidences an
    exact tie and so change which rows commit (review R7, 2026-11-12). The
    cuda_native fixed-quota routes score on the device and no longer pass here.
    """
    return _raw_max_prob_rows_exact(logits)


def _raw_max_prob_rows_exact(logits) -> tuple:
    """The reference per-row scoring: a float64 softmax, then its maximum and argmax."""
    import numpy as np

    # Keep the existing float64 softmax operations and reduction order, but
    # reuse one owned buffer instead of materializing shifted, exponential and
    # normalized vocabulary-sized arrays. Never overwrite the caller's logits.
    probs = np.array(logits, dtype=np.float64, copy=True)
    probs -= np.max(probs, axis=-1, keepdims=True)
    np.exp(probs, out=probs)
    probs /= np.sum(probs, axis=-1, keepdims=True)
    max_prob = probs.max(axis=-1)
    argmax_ids = probs.argmax(axis=-1).tolist()
    return max_prob, argmax_ids


# Temperature sampling, as the model cards' own samplers do it (dllm-hub Qwen3-0.6B MDLM and
# BD3): at T > 0 a position's token is argmax(logit / T + g) with Gumbel noise g = -log(-log u),
# which is the cards' exp(logits) / (-log u)^T. It is ranked as argmax(logit + T * g), the same
# draw scaled by T, which no temperature overflows. Positions are still ranked by the RAW
# probability of the sampled token. u is a counter-based function of (seed, forward, position,
# token): the CUDA candidate kernel draws the same u, so a seed reproduces a reply on either
# route. The noise is float32 on both and the score it joins is double; the two float32 logs
# may differ in the last bit, which moves a draw only at a near tie.
_SAMPLING_STEP = 0x9E3779B97F4A7C15
_MASK64 = (1 << 64) - 1


def _mix64(value):
    """splitmix64's finalizer, on numpy uint64 (scalars or arrays)."""
    import numpy as np

    value = value ^ (value >> np.uint64(30))
    value = value * np.uint64(0xBF58476D1CE4E5B9)
    value = value ^ (value >> np.uint64(27))
    value = value * np.uint64(0x94D049BB133111EB)
    return value ^ (value >> np.uint64(31))


def sampling_noise(seed: int, step: int, position: int, vocab: int):
    """Gumbel noise for every token of one row, float32, from the device kernel's u."""
    import numpy as np

    golden = np.uint64(_SAMPLING_STEP)
    with np.errstate(over="ignore"):
        key = _mix64(np.uint64(int(seed) & _MASK64) + golden)
        key = _mix64(key + np.uint64((int(step) + 1) & _MASK64) * golden)
        key = _mix64(key + np.uint64((int(position) + 1) & _MASK64) * golden)
        tokens = np.arange(1, vocab + 1, dtype=np.uint64)
        bits = _mix64(key + tokens * golden)
    # 23 bits and a half step: u in (0, 1) exactly representable as float32.
    uniform = ((bits >> np.uint64(41)).astype(np.float32) + np.float32(0.5)) * np.float32(2.0 ** -23)
    return -np.log(-np.log(uniform))


def _sampled_max_prob_confidence(logits, *, temperature: float, seed: int, step: int,
                                 positions) -> tuple:
    """Each row's sampled token and the raw probability of it (see ``sampling_noise``)."""
    import numpy as np

    rows = np.asarray(logits, dtype=np.float32)
    if rows.ndim != 2 or len(positions) != rows.shape[0]:
        raise ValueError("sampling needs one position per logits row")
    scores = np.empty(rows.shape[0], dtype=np.float64)
    tokens: list[int] = []
    for row, position in enumerate(positions):
        values = rows[row].astype(np.float64)
        noise = sampling_noise(seed, step, int(position), rows.shape[1]).astype(np.float64)
        # The device rounds the product and the sum separately too (no fused multiply-add).
        token = int(np.argmax(values + float(temperature) * noise))  # lowest among equals
        top = values.max()
        weights = np.exp(values - top)
        scores[row] = weights[token] / weights.sum()
        tokens.append(token)
    return scores, tokens


def _exclude_mask_token(logits, *, mask_token_id: int):
    """Remove the canvas sentinel from the committed-token distribution.

    A commit that writes ``mask_token_id`` back into a masked row has changed
    no state. The row-based progress floors would nevertheless count it as a
    commit and can repeat it until the step budget expires. Exclude the
    sentinel before both confidence scoring and token selection so every
    selected row really leaves the masked frontier. Out-of-vocabulary
    sentinels used by pure policy tests need no exclusion because the model
    cannot emit them.
    """
    import numpy as np

    values = np.asarray(logits, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(
            "decode frontier logits must be rank 2; got "
            f"shape={values.shape!r}"
        )
    token = int(mask_token_id)
    if token < 0 or token >= values.shape[1]:
        return values
    if values.shape[1] < 2:
        raise ValueError("the mask token is the only token in the model vocabulary")
    filtered = values.copy()
    filtered[:, token] = -np.inf
    return filtered


# ---------------------------------------------------------------------------
# Commit policies (which rows to commit this step)
# ---------------------------------------------------------------------------


def _topk_count_choices(confidence, *, transfer_count: int) -> list:
    """Top-``transfer_count`` rows by descending confidence.

    Mirrors the legacy ``np.argsort(-confidence)[:transfer_count]`` selection
    used by the entropy_count schedule.
    """
    import numpy as np

    chosen = np.argsort(-np.asarray(confidence))[:transfer_count]
    return [int(index) for index in chosen]


def _threshold_choices(max_prob, *, threshold: float) -> list:
    """Rows whose confidence clears ``threshold`` (floor of one).

    Commit every row at or above the threshold; if none clear it, commit the
    single most-confident row to guarantee progress.
    """
    import numpy as np

    confidence = np.asarray(max_prob, dtype=np.float64)
    chosen = np.flatnonzero(confidence >= float(threshold))
    if chosen.size == 0:
        return [int(np.argmax(confidence))]
    return [int(index) for index in chosen]


def _entropy_budget_choices(entropies, *, gamma: float) -> list:
    """EB-Sampler ascending-entropy budget selection (arXiv 2505.24857).

    Given the per-candidate Shannon entropies ``H_l`` of the masked-position
    distributions, return the row indices (into ``entropies``) of the LARGEST
    ascending-entropy PREFIX ``U`` satisfying ``Σ_{l∈U} H_l − max_{l∈U} H_l ≤
    γ``, always committing at least one (the lowest-entropy position).

    The rule sorts candidates by ascending ``H`` and grows the committed set one
    position at a time. Because the set is the ascending prefix, ``max_{l∈U} H_l``
    is always the LAST (highest-entropy) element added, so the budget term is
    ``(running sum including the new element) − (the new element's H)`` =
    ``running sum of all-but-the-newest``. A prefix of one always has
    ``ΣH − maxH = 0 ≤ γ``, so the floor of one is unconditional (EB never stalls,
    even when γ = 0 or the lowest entropy exceeds γ). When uncertainty is broadly
    distributed across the sorted candidates, the bound trips early and EB backs
    off to a near-singleton commit. This is an uncertainty-profile heuristic; the
    runtime does not currently measure token dependencies here.
    """
    import numpy as np

    H = np.asarray(entropies, dtype=np.float64)
    if H.shape[0] == 0:
        return []  # no candidates -> nothing to commit (unreachable via the loop)
    order = np.argsort(H, kind="stable")  # ascending H; stable so ties keep order
    budget = float(gamma)
    committed = 1  # always commit the lowest-entropy position
    # Grow the prefix while ``Σ(prefix) − max(prefix) ≤ γ``. With the ascending
    # prefix, ``max`` is the newest element, so the test for adding the kth element
    # (0-indexed by ``order``) is ``sum(H[order[:k+1]]) − H[order[k]] ≤ γ`` ==
    # ``sum(H[order[:k]]) ≤ γ`` (the sum of everything ALREADY in the prefix).
    prefix_sum_without_newest = float(H[order[0]])
    for k in range(1, H.shape[0]):
        if prefix_sum_without_newest <= budget:
            committed += 1
            prefix_sum_without_newest += float(H[order[k]])
        else:
            break
    return [int(index) for index in order[:committed]]


def _top_p_filter(logits, *, top_p: float):
    import numpy as np

    sorted_indices = np.argsort(-logits, axis=-1)
    sorted_logits = np.take_along_axis(logits, sorted_indices, axis=-1)
    sorted_probs = _softmax(sorted_logits)
    cumulative = np.cumsum(sorted_probs, axis=-1)
    remove = cumulative > top_p
    remove[..., 0] = False
    filtered_sorted = np.where(remove, -np.inf, sorted_logits)
    filtered = np.full_like(logits, -np.inf)
    np.put_along_axis(filtered, sorted_indices, filtered_sorted, axis=-1)
    return filtered


def _softmax(logits):
    import numpy as np

    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# MaskState -- the masked sequence as a managed object
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MaskState:
    """The masked working sequence as a first-class, managed object.

    Two shapes share one object:

    * **Trailing generation** (the default, byte-identical to the legacy path):
      ``x`` is ``prompt + generated region``, the prompt region is never masked,
      and ``masked_positions`` scans positions ``>= prompt_len`` that still equal
      ``mask_token_id``.
    * **Infill / span-fill** (built via :meth:`for_infill`): the fillable
      positions are an EXPLICIT set anywhere in ``x`` -- typically holes BETWEEN a
      fixed prefix and a fixed suffix. Everything not in that set is fixed context
      (prefix AND suffix), so a fixed token that merely happens to equal the mask
      id is never treated as fillable. When ``fillable_positions`` is set it is the
      sole source of maskability and ``prompt_len`` is ignored for scanning.
    """

    x: list[int]
    prompt_len: int
    mask_token_id: int
    eos_token_id: int
    committed_token_count: int = 0
    # Explicit fillable positions for the infill path. ``None`` selects the
    # legacy trailing-region behavior; a list (kept in ascending order) restricts
    # maskability to exactly those positions, so the fixed suffix is never filled.
    fillable_positions: list[int] | None = None

    @classmethod
    def for_infill(
        cls,
        *,
        x: list[int],
        fillable_positions: list[int],
        mask_token_id: int,
        eos_token_id: int,
    ) -> "MaskState":
        """Build a state for infill over an EXPLICIT set of fillable positions.

        ``x`` is the FULL sequence (prefix + holes + suffix) with the fillable
        positions initialized to ``mask_token_id``; everything else is fixed
        context on BOTH sides. The bidirectional forward fills the holes
        respecting prefix AND suffix. ``prompt_len`` is set to the first fillable
        position purely so any trailing-region consumer stays sane; maskability is
        driven solely by ``fillable_positions``. Duplicates are collapsed (a
        position is either fillable or not) and out-of-range positions fail
        closed — a duplicated hole would otherwise be committed twice in one
        step, and a raw ``IndexError`` later would hide the real cause.
        """
        ordered = sorted({int(position) for position in fillable_positions})
        for position in ordered:
            if position < 0 or position >= len(x):
                raise ValueError(
                    f"fillable position {position} is out of range for a "
                    f"sequence of length {len(x)}"
                )
        prompt_len = ordered[0] if ordered else len(x)
        return cls(
            x=list(x),
            prompt_len=prompt_len,
            mask_token_id=mask_token_id,
            eos_token_id=eos_token_id,
            fillable_positions=ordered,
        )

    def masked_positions(self) -> list[int]:
        """Positions still equal to the mask token that are eligible to fill.

        Infill: the explicit ``fillable_positions`` still equal to the mask id
        (works whether the holes are trailing or in the middle). Trailing
        generation: positions ``>= prompt_len`` still equal to the mask id
        (byte-identical to the legacy whole-sequence scan).
        """
        if self.fillable_positions is not None:
            return [
                position
                for position in self.fillable_positions
                if self.x[position] == self.mask_token_id
            ]
        return [
            position
            for position in range(self.prompt_len, len(self.x))
            if self.x[position] == self.mask_token_id
        ]

    def all_fillable_positions(self) -> list[int]:
        """The FULL ordered set of positions this state may ever fill.

        Unlike :meth:`masked_positions` (which drops already-committed holes),
        this returns the complete fillable set fixed at construction, in ascending
        order: the explicit ``fillable_positions`` for infill, or the whole
        trailing generated region (``range(prompt_len, len(x))``) otherwise. The
        block scheduler groups THESE into left-to-right blocks so a fixed gap or
        suffix never enters a block, while trailing generation stays byte-identical
        (the trailing region is exactly that range).
        """
        if self.fillable_positions is not None:
            return list(self.fillable_positions)
        return list(range(self.prompt_len, len(self.x)))

    def commit(self, *, position: int, token_id: int) -> None:
        """Write ``token_id`` at ``position`` (unmasking it)."""
        self.x[position] = token_id

    def remask(self, *, position: int) -> None:
        """Set ``position`` back to ``mask_token_id`` -- the inverse of commit.

        Re-masking a previously-committed position returns it to the maskable
        frontier so a recovery loop (Phase F3) can re-decode it. It is the exact
        inverse of :meth:`commit`: after ``commit(position=p, ...)`` followed by
        ``remask(position=p)`` the position is masked again and reappears in
        :meth:`masked_positions`. ``committed_token_count`` is the manager's
        running tally of commit *operations* and is intentionally left untouched
        here (re-masking does not "un-count" the forward that produced the
        commit); the metric the recovery loop reports is ``remask_count``.
        """
        self.x[position] = self.mask_token_id

    def filled_span(self) -> dict:
        """Infill-aware finalize: the decoded tokens at the fillable positions.

        Returns the tokens at the originally-fillable positions in ascending
        order (``fill_token_ids``) and the full sequence (``full_token_ids``),
        WITHOUT the trailing-region / first-EOS truncation that
        :meth:`finalize` applies. Infill fills a BOUNDED span, so an EOS produced
        inside the span is kept and the fixed suffix is preserved verbatim. Any
        hole the step cap left masked is kept in place and counted in
        ``remaining_mask_count`` so the caller can decide how to render it.
        """
        positions = self.fillable_positions if self.fillable_positions is not None else []
        fill_token_ids = [self.x[position] for position in positions]
        remaining_mask_count = sum(
            1 for token_id in fill_token_ids if token_id == self.mask_token_id
        )
        return {
            "fill_token_ids": fill_token_ids,
            "full_token_ids": list(self.x),
            "remaining_mask_count": remaining_mask_count,
        }

    def finalize(self) -> dict:
        """Resolve the visible output, EOS evidence, and finish reason.

        The answer ends at the first EOS in the generated region; tokens after
        it are diffusion filler. Only a fully resolved answer prefix earns
        ``finish_reason="stop"``. Without EOS, or with unresolved answer rows
        when the decode budget expires, the finish reason is "length".
        ``remaining_mask_count`` counts unresolved answer rows. Visible output
        ends at the first unresolved row: joining tokens across a hole would
        change the answer instead of returning its completed prefix.
        """
        gen = self.x[self.prompt_len:]
        if self.eos_token_id in gen:
            first_eos = gen.index(self.eos_token_id)
            hit_eos = True
            answer = gen[:first_eos]
            finish_reason = "stop"
        else:
            hit_eos = False
            answer = gen
            finish_reason = "length"
        remaining_mask_count = answer.count(self.mask_token_id)
        if remaining_mask_count:
            finish_reason = "length"
            visible = answer[:answer.index(self.mask_token_id)]
        else:
            visible = answer
        return {
            "visible_token_ids": visible,
            "hit_eos": hit_eos,
            "remaining_mask_count": remaining_mask_count,
            "finish_reason": finish_reason,
        }


# ---------------------------------------------------------------------------
# DiffusionIterationResult -- the manager's structured return value
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DiffusionIterationResult:
    visible_generated_token_ids: list[int]
    iteration_count: int
    forward_count: int
    committed_token_count: int
    hit_eos: bool
    remaining_mask_count: int
    finish_reason: str
    decode_loop_kind: str
    entropy_used: bool
    # Phase F metrics. Neutral when error-fixing is OFF (the F1 default): one
    # decode pass, no re-masking, no corrections.
    #   * committed_per_forward -- the aggression / throughput measure
    #     (committed_token_count / forward_count); 0.0 when no forward ran.
    #   * remask_count          -- positions re-masked by the recovery loop (0 OFF).
    #   * corrected_positions    -- positions whose value changed after recovery (0 OFF).
    #   * recovery_rounds        -- detect->remask->re-decode rounds (1 == the single
    #                               decode pass today's behaviour runs).
    #   * wrong_commit_count     -- positions a detector flagged likely-wrong (0 OFF).
    #   * churned_positions      -- re-masked then re-decoded to the SAME token =
    #                               wasted work (0 OFF).
    #   * quality_delta          -- optional objective before/after signal where measured.
    committed_per_forward: float = 0.0
    remask_count: int = 0
    corrected_positions: int = 0
    recovery_rounds: int = 1
    wrong_commit_count: int = 0
    churned_positions: int = 0
    quality_delta: float | None = None


# ---------------------------------------------------------------------------
# Loop context handed to a strategy
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _LoopContext:
    """Mutable counters + the backend callables shared with the strategy.

    A strategy drives its own loop shape (whole-sequence vs block) through this
    context: it asks the manager to run a real forward (which increments
    ``forward_count``), converts the forward output to selection logits via
    :meth:`select_logits`, and commits via the :class:`MaskState`. ``steps`` is
    the global cap on *real forwards*.

    Optional backend hooks let the mlx_dream orchestrator instrument the loop
    WITHOUT putting MLX or timing in this module: ``commit_observer`` fires per
    committed position (KV-cache invalidation), and the three ``*_phase``
    factories wrap the matching loop sections for stage timing. They default to
    no-ops so the pure-numpy stub path is unaffected.
    """

    state: MaskState
    forward_fn: ForwardFn
    steps: int
    select_transform: SelectTransform = _identity_select
    commit_observer: CommitObserver | None = None
    scheduler_phase: PhaseFactory = _null_phase
    select_phase: PhaseFactory = _null_phase
    commit_phase: PhaseFactory = _null_phase
    # Phase F3 error model. When attached, ``commit_rows`` records a ``Commit``
    # event (position, token, the per-position UnmaskPolicy score, the step) per
    # committed position -- the ``commit_history`` a detector reads. ``None``
    # (the default) records nothing, so the OFF path stays byte-identical.
    event_log: "DecodeEventLog | None" = None
    # When set (KL-drift recovery), the loop records each scored frontier
    # position's raw-softmax distribution per step onto ``event_log.dist_history``
    # (the relative signal KLDriftDetector reads). Off by default -> no extra
    # softmax, byte-identical.
    record_distributions: bool = False
    # MaskCanvas v0 control-plane capture. A DUCK-TYPED observer (this module must
    # NOT import ``mask_canvas`` -- the dependency is one-way to avoid a cycle).
    # ``forward``/``commit_rows`` call it as PURE side-effects, guarded by ``is not
    # None``, so the OFF path (the default) is byte-identical: no observer means no
    # call, no decode decision reads it.
    trace_observer: Any = None
    # Active-Window Runtime (Task A3, WS1). A callable taking one WindowExecutionPlan
    # per decode step, emitted BEFORE the forward. Guarded by ``is not None``, so the
    # OFF path (the default, ``None``) runs ZERO new code in the hot path and the
    # decode output is byte-identical. The dependency is one-way: window_plan.py
    # imports nothing from diffusion_runtime, so NO import cycle is introduced.
    window_plan_sink: Any = None
    # Active-Window Runtime (Task B2, WS2). An injected callable
    # ``(window_plan, token_ids) -> WindowExecutionResult`` that replaces the
    # raw ``forward_fn`` -> ``select_logits`` path with a single routed call.
    # ``result.logits`` from ``execute_window`` already has the logits convention
    # applied (dream_ar_shift or raw), so the routed path MUST NOT call
    # ``select_transform`` / ``select_logits`` again — doing so would double-shift
    # Dream logits (finding 1). Guarded by ``is not None``, so the OFF path (the
    # default, ``None``) is BYTE-IDENTICAL: no new code runs, no accounting change.
    execute_window_fn: Any = None
    # Active-Window Runtime (C2a, WS3). A duck-typed ``CorrectionPolicy`` from
    # ``correction_policy.py``. When set (non-None), ``run_decode_loop`` calls
    # ``ctx.correction_policy.corrections_for(state=..., commit_history=..., ...)``
    # each step and passes the result to ``classify_window(corrections=...)``, so
    # emitted ``WindowExecutionPlan``s reflect dirty/remask positions. The policy
    # wraps the SAME detector + remask_policy as the recovery loop (consistency gate b).
    # ``None`` (the default) → ``classify_window`` receives ``corrections=None`` →
    # dirty/remask always empty → BYTE-IDENTICAL to the pre-C2a path. Guarded by
    # ``is not None``, so the OFF path pays zero extra work.
    correction_policy: Any = None
    # Optional device-resident scoring. Only compact proposals cross this seam;
    # the SAME shared CommitPolicy below still selects the committed rows.
    propose_fn: Any = None
    commit_device_fn: Any = None
    forward_count: int = 0
    committed_token_count: int = 0
    iteration_count: int = 0

    def forward(self, *, upto: int | None = None) -> Any:
        """Run one real forward, counting it. Returns the raw forward output.

        ``upto`` (the D2F-3 window end) truncates the forwarded ids to the
        prefix ``x[:upto]`` — the pipelined block scheduler forwards only the
        appended window, exactly as the reference's growing ``x_t``. ``None``
        (every other strategy) forwards the full working sequence, byte-identical
        to before.
        """
        ids = self.state.x if upto is None else self.state.x[:upto]
        raw = self.forward_fn(list(ids))
        self.forward_count += 1
        if self.trace_observer is not None:  # MaskCanvas v0 (neutral side-effect)
            self.trace_observer.on_forward(scope=upto, forwarded_count=len(ids))
        return raw

    def forward_window(self, window_plan: Any) -> Any:
        """Run one real forward via the execute_window_fn, counting it.

        Mirrors ``forward()``'s accounting: increments ``forward_count`` and
        fires the trace observer. Returns the ``WindowExecutionResult`` from
        ``execute_window_fn(window_plan, token_ids)`` whose ``logits`` are
        already convention-shifted (the parity wrapper applies the shift).

        ``forwarded_count`` is the HONEST computed-row count this forward, sourced
        from ``result.forwarded_rows`` (the per-forward delta of
        ``mlx_timing_active_tokens_total``): ``seq_len`` for a full forward, and
        the reduced compute window ``len(plan.forward_positions())`` when a cache
        routes to an active/block-only path. It is NOT ``len(ids)`` (the full
        window prefix ``x[:window_end]`` handed to the backend), which over-counts
        reduced paths. When ``forwarded_rows`` is ``None`` (an execute_window_fn
        that does not report it), fall back to ``len(ids)``.
        """
        ids = (
            self.state.x[: window_plan.window_end]
            if window_plan.window_end is not None
            else self.state.x
        )
        result = self.execute_window_fn(window_plan, list(ids))
        self.forward_count += 1
        if self.trace_observer is not None:  # MaskCanvas v0 (neutral side-effect)
            # Prefer the honest computed-row count carried by the result; fall
            # back to len(ids) only when the backend did not report it.
            _fwd = getattr(result, "forwarded_rows", None)
            if _fwd is None:
                _fwd = len(ids)
            self.trace_observer.on_forward(
                scope=window_plan.window_end, forwarded_count=_fwd
            )
        return result

    def select_logits(self, raw: Any) -> Any:
        """Convert a raw forward output to numpy selection logits."""
        return self.select_transform(raw)

    def commit_rows(
        self,
        *,
        positions: list[int],
        rows: list[int],
        token_ids: list[int],
        scores: list | None = None,
        step: int = 0,
    ) -> None:
        """Commit the chosen ``rows`` (indices into ``positions``).

        ``scores`` (optional, per-row in ``positions`` order, the UnmaskPolicy
        confidence) and ``step`` thread the commit provenance into the attached
        :class:`DecodeEventLog`. They never affect WHICH rows commit -- the
        accounting is a passive observer, so the OFF path is byte-identical
        whether or not a log/scores are supplied.
        """
        if self.commit_device_fn is not None and rows:
            self.commit_device_fn(
                [positions[int(row)] for row in rows],
                [int(token_ids[int(row)]) for row in rows],
            )
        for row in rows:
            index = int(row)
            position = positions[index]
            token_id = int(token_ids[index])
            self.state.commit(position=position, token_id=token_id)
            if self.commit_observer is not None:
                self.commit_observer(position, token_id)
            if self.event_log is not None:
                score = float(scores[index]) if scores is not None else float("nan")
                self.event_log.record_commit(
                    position=position, token=token_id, score=score, step=int(step)
                )
        self.committed_token_count += len(rows)
        self.iteration_count += 1
        if self.trace_observer is not None:  # MaskCanvas v0 (neutral side-effect)
            self.trace_observer.on_commit(
                positions=positions, rows=rows, token_ids=token_ids,
                scores=scores, step=step,
            )


# ---------------------------------------------------------------------------
# UnmaskPolicy -- the SCORING half (which masked positions, how confident)
# ---------------------------------------------------------------------------
#
# An UnmaskPolicy answers, for the masked positions handed to it this forward,
# "how confident is the model at each one" (higher == more confident) on the RAW
# softmax (no temperature/top_p sharpening -- the lesson from the
# confidence-threshold fix). It also resolves the committed token per position
# (the greedy argmax), since the commit needs it; ``score`` exposes JUST the
# scalar confidences (the Protocol surface), ``score_and_tokens`` returns both.


class UnmaskPolicy(Protocol):
    name: str

    def score(self, *, logits_at_masked, masked_positions) -> list:
        """Per masked position: a confidence scalar (higher == more confident)."""
        ...


@dataclass(slots=True)
class TopConfidence:
    """Max RAW softmax probability per position (Fast-dLLM ``max_x p_theta``).

    At temperature 0 the verbatim ``_raw_max_prob_confidence`` math: the
    committed token is the argmax and its raw probability ranks the position.
    Above 0 the token is sampled as the model cards' samplers do (see
    ``sampling_noise``) and ranked by its raw probability; each call is one
    forward, whose index keys the draw. ``confidence_fn`` supports controlled
    policy experiments. Production uses the shared raw-softmax scorer directly.
    """

    name: str = field(default="top_confidence", init=False)
    confidence_fn: ConfidenceFn | None = None
    temperature: float = 0.0
    seed: int = 0
    _forwards: int = field(default=0, init=False)

    def score_and_tokens(self, *, logits_at_masked, masked_positions) -> tuple:
        if self.confidence_fn is not None:
            return self.confidence_fn(logits_at_masked)
        if self.temperature == 0:
            return _raw_max_prob_confidence(logits_at_masked)
        step, self._forwards = self._forwards, self._forwards + 1
        return _sampled_max_prob_confidence(
            logits_at_masked, temperature=self.temperature, seed=self.seed,
            step=step, positions=masked_positions,
        )

    def score(self, *, logits_at_masked, masked_positions) -> list:
        scores, _ = self.score_and_tokens(
            logits_at_masked=logits_at_masked, masked_positions=masked_positions
        )
        return list(scores)


@dataclass(slots=True)
class NegEntropy:
    """Negative Shannon entropy (-H) of the temperature/top_p adjusted softmax.

    The verbatim ``_entropy_confidence`` math: higher (closer to zero) == a more
    peaked, more confident distribution. The committed token is the argmax of the
    adjusted distribution. ``confidence_fn`` is injectable exactly as for
    :class:`TopConfidence`.
    """

    temperature: float = 1.0
    top_p: float | None = None
    name: str = field(default="neg_entropy", init=False)
    confidence_fn: ConfidenceFn | None = None

    def score_and_tokens(self, *, logits_at_masked, masked_positions) -> tuple:
        if self.confidence_fn is not None:
            return self.confidence_fn(logits_at_masked)
        return _entropy_confidence(
            logits_at_masked, temperature=self.temperature, top_p=self.top_p
        )

    def score(self, *, logits_at_masked, masked_positions) -> list:
        scores, _ = self.score_and_tokens(
            logits_at_masked=logits_at_masked, masked_positions=masked_positions
        )
        return list(scores)


# ---------------------------------------------------------------------------
# CommitPolicy -- the AGGRESSION half (how many / which to commit this forward)
# ---------------------------------------------------------------------------
#
# ``select`` is the Protocol surface from the spec: given the scores of the
# masked positions in the CURRENT frontier, return the indices (into that
# frontier) to commit now. ``plan`` is the loop-shape hook the generic loop calls
# BEFORE each forward: it scopes the frontier (all masked positions, or the
# active block) and signals SKIP (no forward this step, e.g. the entropy schedule
# transferring 0) or STOP (no more work). The active-block pointer lives in the
# block policy's own state.


@dataclass(slots=True)
class StepPlan:
    """A CommitPolicy's pre-forward decision for the generic loop.

    ``frontier`` is the masked positions to score this forward. ``stop`` ends the
    loop (no forward). ``skip`` advances to the next step WITHOUT a forward (the
    entropy schedule's ``transfer_count <= 0`` ``continue``). ``window_end``
    (ADDITIVE, D2F-3) asks the loop to forward only ``x[:window_end]`` — the
    pipelined scheduler's appended window; ``None`` (every other policy) keeps
    the full-sequence forward byte-identical.
    """

    frontier: list
    stop: bool = False
    skip: bool = False
    window_end: int | None = None


class CommitPolicy(Protocol):
    name: str

    def plan(self, *, state: "MaskState", step: int) -> StepPlan:
        """Scope the frontier + decide SKIP/STOP before the forward."""
        ...

    def select(self, *, scores, masked_positions, state: "MaskState", step: int) -> list:
        """Indices into ``masked_positions`` to commit this forward."""
        ...


@dataclass(slots=True)
class EntropyCountCommit:
    """Diffusion-schedule top-k commit (the legacy ``entropy_count`` aggression).

    Frontier = ALL masked positions. The schedule sets
    ``transfer_count = int(N * (1 - s/t))`` (full remaining count on the final
    step) from a ``linspace(1, eps, steps+1)`` timestep grid; ``transfer_count <= 0``
    SKIPs the forward (the legacy ``continue``). ``select`` commits the
    top-``transfer_count`` positions by confidence. ``steps`` (== the forward cap)
    is captured at construction so the timestep grid is byte-identical to the
    legacy ``np.linspace(1.0, eps, steps+1)``.
    """

    eps: float
    steps: int
    name: str = field(default="entropy_count", init=False)
    _transfer_count: int = field(default=0, init=False)

    def plan(self, *, state: "MaskState", step: int) -> StepPlan:
        import numpy as np

        steps = int(self.steps)
        timesteps = np.linspace(1.0, float(self.eps), steps + 1, dtype=np.float32)
        mask_positions = state.masked_positions()
        if not mask_positions:
            return StepPlan(frontier=[], stop=True)
        t = float(timesteps[step])
        s = float(timesteps[step + 1])
        if step < steps - 1:
            transfer_count = int(len(mask_positions) * (1.0 - s / t))
        else:
            transfer_count = len(mask_positions)
        self._transfer_count = transfer_count
        if transfer_count <= 0:
            return StepPlan(frontier=mask_positions, skip=True)
        return StepPlan(frontier=mask_positions)

    def select(self, *, scores, masked_positions, state: "MaskState", step: int) -> list:
        return _topk_count_choices(scores, transfer_count=self._transfer_count)


@dataclass(slots=True)
class ThresholdCommit:
    """Whole-sequence confidence-threshold commit (legacy ``confidence_threshold``).

    Frontier = ALL masked positions, a real forward every step (no skip). ``select``
    commits every masked position whose confidence clears ``tau`` (floor of one --
    the single argmax when none clear, the progress guarantee).
    """

    confidence_threshold: float
    name: str = field(default="threshold", init=False)

    def plan(self, *, state: "MaskState", step: int) -> StepPlan:
        mask_positions = state.masked_positions()
        if not mask_positions:
            return StepPlan(frontier=[], stop=True)
        return StepPlan(frontier=mask_positions)

    def select(self, *, scores, masked_positions, state: "MaskState", step: int) -> list:
        return _threshold_choices(scores, threshold=self.confidence_threshold)


@dataclass(slots=True)
class DilatedCommitPolicy:
    """Block-wise Dilated Unmasking Scheduler (DUS) commit policy.

    Frontier = the CURRENT left-to-right block of ``block_size`` fillable
    positions (via :func:`_block_frontier_plan`, same as
    :class:`BlockThresholdCommit`). Within each block, ``select`` returns the
    next dilated group — **unconditional** (ignores ``scores``). When a block's
    schedule is exhausted, the next call detects the new block and recomputes.

    APPROXIMATE (no confidence gate). Default-OFF; quality-gate mandatory.
    Named a **no-remask variant** (the paper's skip/remask mechanism is not
    implemented here).
    """

    block_size: int
    base: int = 2
    name: str = field(default="dilated_unmask", init=False)
    _block_index: int = field(default=0, init=False)
    _schedule: list = field(default_factory=list, init=False, repr=False)
    _initial_positions: list = field(default=None, init=False, repr=False)
    _round: int = field(default=0, init=False, repr=False)

    def plan(self, *, state: "MaskState", step: int) -> StepPlan:
        plan, self._block_index = _block_frontier_plan(
            state=state, block_index=self._block_index, block_size=self.block_size
        )
        return plan

    def select(self, *, scores, masked_positions, state: "MaskState", step: int) -> list:
        if not masked_positions:
            return []
        # Detect a new block: schedule exhausted or first call.
        if self._round >= len(self._schedule) or self._initial_positions is None:
            actual_size = len(masked_positions)
            from edllm.dilated_schedule import dilated_schedule

            self._schedule = dilated_schedule(actual_size, base=self.base)
            self._initial_positions = list(masked_positions)
            self._round = 0

        if self._round >= len(self._schedule):
            return list(range(len(masked_positions)))

        group = self._schedule[self._round]
        result: list[int] = []
        for p in group:
            canvas_pos = self._initial_positions[p - 1]
            try:
                idx = masked_positions.index(canvas_pos)
                result.append(idx)
            except ValueError:
                pass
        self._round += 1
        return result


def _block_frontier_plan(
    *, state: "MaskState", block_index: int, block_size: int
) -> tuple:
    """The Fast-dLLM left-to-right block-frontier walk (shared by block policies).

    Returns ``(StepPlan, next_block_index)``: the CURRENT block's still-masked
    positions over the LOGICAL fillable index order (so infill never walks into
    the fixed suffix; trailing generation stays byte-identical), having advanced
    the pointer past any already-fully-committed blocks -- exactly as the legacy
    outer ``for block_index`` loop moved on once the inner ``while`` drained a
    block. STOP (empty frontier) when no blocks remain. The active-block pointer
    is the CALLER's state (each block CommitPolicy carries its own ``_block_index``
    and writes back the returned ``next_block_index``), so the frontier scoping is
    shared without coupling the two policies' state -- both ``BlockThresholdCommit``
    and the F2 ``EntropyBudgetCommit`` compose with the prefix/block KV cache the
    same way.
    """
    block_size = max(1, int(block_size))
    fillable = state.all_fillable_positions()
    while block_index < len(fillable):
        block_positions = fillable[block_index : block_index + block_size]
        block_masked_positions = [
            position
            for position in block_positions
            if state.x[position] == state.mask_token_id
        ]
        if block_masked_positions:
            return StepPlan(frontier=block_masked_positions), block_index
        block_index += block_size
    return StepPlan(frontier=[], stop=True), block_index


@dataclass(slots=True)
class BlockThresholdCommit:
    """Fast-dLLM block-sequential threshold commit (legacy ``block_threshold``).

    Frontier = the CURRENT left-to-right block of ``block_size`` LOGICAL fillable
    positions (their index order, NOT a raw absolute range -- so infill never
    walks into the fixed suffix; trailing generation stays byte-identical). The
    active-block pointer is this policy's state: ``plan`` returns the current
    block's still-masked positions and advances to the next block when the current
    one is exhausted; STOP when no blocks remain. ``select`` commits every block
    position clearing ``tau`` (floor of one).
    """

    confidence_threshold: float
    block_size: int
    # D2F-2 exact-cache seam. A sequential block is frozen after its final
    # token was committed and before the next block is forwarded. None keeps
    # the standalone scheduler/cache-off behavior unchanged.
    freeze_observer: Callable[[int], None] | None = None
    name: str = field(default="block_threshold", init=False)
    _block_index: int = field(default=0, init=False)

    def plan(self, *, state: "MaskState", step: int) -> StepPlan:
        plan, next_block_index = _block_frontier_plan(
            state=state, block_index=self._block_index, block_size=self.block_size
        )
        # _block_frontier_plan advances only across fully committed blocks. Tell
        # the exact cache about each such block before the next frontier runs so
        # its final-token K/V is captured on that next forward. Advance our
        # pointer after each successful notification: if an observer fails, a
        # retry resumes at the failed block instead of re-firing earlier ones.
        block_size = max(1, int(self.block_size))
        while self._block_index < next_block_index:
            completed_block = self._block_index // block_size
            if self.freeze_observer is not None:
                self.freeze_observer(completed_block)
            self._block_index += block_size
        return plan

    def select(self, *, scores, masked_positions, state: "MaskState", step: int) -> list:
        return _threshold_choices(scores, threshold=self.confidence_threshold)


@dataclass(frozen=True, slots=True)
class BlockLayout:
    """A generated interval intersected with a fixed attention-block grid.

    ``origin=0`` gives absolute blocks, including a partial first block.
    ``origin=start`` gives prompt-relative blocks. Neither adds padding tokens.
    """

    start: int
    length: int
    block_size: int
    origin: int = 0

    def __post_init__(self) -> None:
        for name in ("start", "length", "origin"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"block layout {name} must be a nonnegative integer")
        if type(self.block_size) is not int or self.block_size <= 0:
            raise ValueError("block layout block_size must be a positive integer")
        if self.origin > self.start:
            raise ValueError("block layout origin must not exceed start")

    @property
    def count(self) -> int:
        if self.length == 0:
            return 0
        first = (self.start - self.origin) // self.block_size
        last = (self.start + self.length - 1 - self.origin) // self.block_size
        return last - first + 1

    def span(self, index: int) -> tuple[int, int]:
        if not 0 <= index < self.count:
            raise IndexError(f"block index {index} outside layout with {self.count} blocks")
        first = (self.start - self.origin) // self.block_size
        grid_start = self.origin + (first + index) * self.block_size
        return (
            max(self.start, grid_start),
            min(self.start + self.length, grid_start + self.block_size),
        )

    @property
    def mask_geometry(self) -> dict[str, int]:
        return {"prompt_len": self.origin, "block_size": self.block_size}


@dataclass(slots=True)
class FixedQuotaCommit:
    """Prompt-relative fixed quotas with full bidirectional context.

    Fillable positions, including bounded infill spans, form logical blocks.
    Each block's initial masks are spread over fixed rounds, largest quotas
    first. EOS is finalized only after denoising; it cannot discard unresolved
    earlier rows. Unlike absolute block-causal decoding, future blocks and the
    fixed suffix remain visible to every forward.
    """

    block_size: int
    requested_steps: int
    generated_token_count: int
    name: str = field(default="fixed_quota", init=False)
    _active_block: int | None = field(default=None, init=False)
    _quota: int = field(default=0, init=False)
    _base: int = field(default=0, init=False)
    _remainder: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.block_size <= 0 or self.requested_steps <= 0:
            raise ValueError("fixed quota requires positive block_size and steps")

    @property
    def block_count(self) -> int:
        return (self.generated_token_count + self.block_size - 1) // self.block_size

    @property
    def steps_per_block(self) -> int:
        if not self.block_count:
            return 0
        return (self.requested_steps + self.block_count - 1) // self.block_count

    @property
    def iteration_limit(self) -> int:
        return self.block_count * self.steps_per_block

    def plan(self, *, state: "MaskState", step: int) -> StepPlan:
        if step >= self.iteration_limit:
            return StepPlan(frontier=[], stop=True)
        block = step // self.steps_per_block
        start = block * self.block_size
        positions = state.all_fillable_positions()[start : start + self.block_size]
        frontier = [position for position in positions if state.x[position] == state.mask_token_id]
        if block != self._active_block:
            self._active_block = block
            self._base, self._remainder = divmod(len(frontier), self.steps_per_block)
        self._quota = self._base + int(step % self.steps_per_block < self._remainder)
        return StepPlan(frontier=frontier, skip=not frontier or self._quota == 0)

    def select(self, *, scores, masked_positions, state: "MaskState", step: int) -> list:
        return _topk_count_choices(scores, transfer_count=self._quota)


@dataclass(slots=True)
class BD3FixedQuotaCommit:
    """BD3LM's absolute-block, fixed-transfer commit policy.

    The upstream schedule partitions the complete token canvas at absolute
    multiples of ``block_size`` and allocates ``ceil(requested_steps /
    ceil(tokens / block_size))`` denoising rounds to each generated block,
    however many absolute blocks a ragged prompt adds. Within a block, its initial mask
    count is distributed as ``base`` tokens per round plus one token in each of
    the first ``remainder`` rounds. The highest raw-softmax confidences commit.

    The package policy may require aligned prompts or allow a ragged first
    block. Both use the same absolute layout and no-cache loop. Rounds whose
    fixed transfer quota is zero are elided: at temperature zero they cannot
    change state or any later logits.
    """

    block_size: int
    requested_steps: int
    generated_token_count: int | None = None
    block_layout: BlockLayout | None = None
    requires_prompt_block_alignment: bool = True
    name: str = field(default="bd3_fixed_quota", init=False)
    _active_block_id: int | None = field(default=None, init=False)
    _round: int = field(default=0, init=False)
    _base: int = field(default=0, init=False)
    _remainder: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if type(self.block_size) is not int or self.block_size <= 0:
            raise ValueError("block_size must be a positive integer")
        if type(self.requested_steps) is not int or self.requested_steps <= 0:
            raise ValueError("requested_steps must be a positive integer")
        if self.generated_token_count is not None and (
            type(self.generated_token_count) is not int
            or self.generated_token_count < 0
        ):
            raise ValueError("generated_token_count must be a nonnegative integer")
        if type(self.requires_prompt_block_alignment) is not bool:
            raise ValueError("requires_prompt_block_alignment must be a boolean")

    @property
    def generated_block_count(self) -> int:
        if self.block_layout is not None:
            return self.block_layout.count
        if self.generated_token_count is None:
            raise RuntimeError("generated_token_count has not been resolved")
        return (
            self.generated_token_count + self.block_size - 1
        ) // self.block_size

    @property
    def steps_per_block(self) -> int:
        if self.generated_block_count == 0:
            return 0
        # author:Opus -- both upstream samplers divide the step budget by
        # ceil(max_new_tokens / block_size). A ragged prompt makes the generated
        # span touch one more absolute block; dividing by that count took rounds
        # away from every block (26 instead of 32 at 128 tokens / 128 steps).
        tokens = (
            self.block_layout.length
            if self.block_layout is not None
            else self.generated_token_count
        )
        schedule_blocks = (tokens + self.block_size - 1) // self.block_size
        return (self.requested_steps + schedule_blocks - 1) // schedule_blocks

    @property
    def iteration_limit(self) -> int:
        return self.generated_block_count * self.steps_per_block

    def plan(self, *, state: "MaskState", step: int) -> StepPlan:
        if state.fillable_positions is not None:
            raise ValueError(
                "fixed-quota block diffusion does not support infill; use "
                "generation or a model with an infill-capable decoding policy"
            )
        generated_token_count = len(state.all_fillable_positions())
        if self.generated_token_count is None:
            self.generated_token_count = generated_token_count
        elif generated_token_count != self.generated_token_count:
            raise ValueError("BD3LM generated-token count changed during decode")
        if self.block_layout is None:
            self.block_layout = BlockLayout(
                start=state.prompt_len,
                length=generated_token_count,
                block_size=self.block_size,
            )
        elif (
            self.block_layout.start != state.prompt_len
            or self.block_layout.length != generated_token_count
            or self.block_layout.block_size != self.block_size
            or self.block_layout.origin != 0
        ):
            raise ValueError("fixed-quota block layout changed during decode")
        if step >= self.iteration_limit:
            return StepPlan(frontier=[], stop=True)
        if self.requires_prompt_block_alignment and state.prompt_len % self.block_size:
            raise ValueError(
                "BD3LM Candidate prompt length must be a multiple of block_size"
            )
        # EOS ends the decode only at a block boundary. The official sampler
        # (dllm/core/samplers/bd3lm.py, ccf17df) checks EOS after denoising a block;
        # commits are ordered by confidence, so EOS can land while earlier rows
        # of its block are still masked. Stopping in that round left those rows
        # masked and finalization dropped them from the answer.
        if (
            step % self.steps_per_block == 0
            and state.eos_token_id in state.x[state.prompt_len :]
        ):
            return StepPlan(frontier=[], stop=True)

        masked = state.masked_positions()
        if not masked:
            return StepPlan(frontier=[], stop=True)
        # Upstream advances after fixed rounds even when a selected argmax is
        # MASK. Residual masks in an exhausted block must not stall later ones.
        block_id = step // self.steps_per_block
        block_start, block_end = self.block_layout.span(block_id)
        frontier = [
            position
            for position in masked
            if block_start <= position < block_end
        ]
        if self._active_block_id != block_id:
            self._active_block_id = block_id
            self._base, self._remainder = divmod(
                len(frontier), self.steps_per_block
            )
        self._round = step % self.steps_per_block
        if not frontier or self._base + int(self._round < self._remainder) == 0:
            return StepPlan(frontier=[], skip=True)
        # author:Opus -- upstream appends one block at a time, so its forward
        # never contains the blocks after this one, and block attention never
        # reads them. Forward the same prefix instead of the whole output cap.
        return StepPlan(frontier=frontier, window_end=block_end)

    def select(
        self,
        *,
        scores,
        masked_positions,
        state: "MaskState",
        step: int,
    ) -> list:
        del state, step
        transfer_count = self._base + int(self._round < self._remainder)
        return _topk_count_choices(scores, transfer_count=transfer_count)


@dataclass(slots=True)
class EntropyBudgetCommit:
    """EB-Sampler adaptive entropy-budget commit (arXiv 2505.24857; Phase F F2).

    The literature-validated aggressive parallel-commit rule: within the active
    block frontier, commit a VARIABLE number of positions under one entropy budget
    ``γ``. Unlike :class:`BlockThresholdCommit` (commit every position over a fixed
    confidence τ, floor of one), EB sizes the commit from the step's entropy
    profile -- many when cumulative uncertainty stays within budget, fewer when it
    does not. It does not consume a dependency graph or prove independence between
    positions.

    Frontier: identical to :class:`BlockThresholdCommit` -- the CURRENT
    left-to-right block of ``block_size`` LOGICAL fillable positions, via the
    shared :func:`_block_frontier_plan` (so EB composes with the prefix/block KV
    cache and never walks into a fixed suffix; the active-block pointer is this
    policy's own state). Selection: the pairing's :class:`NegEntropy` UnmaskPolicy
    supplies ``score = −H`` (the negative Shannon entropy), so EB reconstructs
    ``H = −score`` and applies :func:`_entropy_budget_choices`: sort ascending H,
    commit the largest prefix ``U`` with ``Σ_{l∈U} H − max_{l∈U} H ≤ γ`` (always
    ≥1). One knob ``γ`` (the budget) -- larger γ commits more aggressively.

    NOT byte-exact with ``block_threshold`` by design: it changes commit behaviour
    (more tokens/forward on low-cumulative-uncertainty steps). Gated on coherence
    and a measured aggression/quality tradeoff, not bit-identity.
    """

    gamma: float
    block_size: int = 8
    name: str = field(default="entropy_budget", init=False)
    _block_index: int = field(default=0, init=False)

    def plan(self, *, state: "MaskState", step: int) -> StepPlan:
        plan, self._block_index = _block_frontier_plan(
            state=state, block_index=self._block_index, block_size=self.block_size
        )
        return plan

    def select(self, *, scores, masked_positions, state: "MaskState", step: int) -> list:
        import numpy as np

        # NegEntropy gives score = −H; reconstruct the entropy the rule needs.
        entropies = -np.asarray(scores, dtype=np.float64)
        return _entropy_budget_choices(entropies, gamma=self.gamma)


@dataclass(slots=True)
class _D2FBlock:
    """One generated block in the D2F pipeline window (absolute positions)."""

    start: int
    end: int  # half-open; the last block may be ragged
    activated: bool = False  # the reference's ``is_complete`` (sticky)
    frozen: bool = False
    # Freeze-as-Commit v0: the step at which this block first held no masked
    # position ("fully decoded"). ``None`` until that happens. A block commits
    # (freezes) only once ``revision_window`` further steps have elapsed, so
    # "fully decoded" and "committed frozen" are distinguishable states. At the
    # default window of 0 the two coincide and the freeze sequence is unchanged.
    fully_decoded_step: int | None = None
    # DUS: per-block dilated round tracker. None = schedule not yet computed;
    # an int = current round index into the block's dilated schedule.
    _dus_round: int | None = field(default=None, init=False, repr=False)
    _dus_schedule: list = field(default=None, init=False, repr=False)


@dataclass(slots=True)
class D2FBlockPipelineCommit:
    """D2F pipelined multi-block commit policy (D2F-3; arXiv 2508.09192 Alg. 2).

    Where :class:`BlockThresholdCommit` owns a single block-frontier POINTER,
    this policy owns a sliding block WINDOW: several blocks decode CONCURRENTLY,
    each in one of the reference lifecycle states **pending -> semi-activated ->
    fully-activated -> frozen**. Semantics transcribed from the reference
    decode loop (``D2F-eval/eval_dream.py`` ``_generate_block_single`` +
    ``_update_block_completion_states``; repo
    ``SJTU-DENG-Lab/Discrete-Diffusion-Forcing``):

    * **Append (τ_add = ``add_threshold``):** at the start of each step, at most
      ONE new fully-masked block joins the window, iff the decoded ratio of the
      LAST appended block is ``>= τ_add`` (the prompt counts as ratio 1.0, so
      the first block appends immediately), capacity remains, and no EOS has
      been committed. New blocks are SEMI-ACTIVATED.
    * **Activate (τ_act = ``activate_threshold``):** sweep blocks in order —
      when block b's decoded ratio is ``>= τ_act``, block b+1 becomes
      FULLY-ACTIVATED (sticky). The oldest unfrozen block is always
      fully-activated (its predecessor — prompt or frozen block — is at 1.0).
    * **Commit (τ_conf = ``confidence_threshold``):** per active block on one
      shared forward: a semi-activated block commits only positions whose RAW
      max-prob confidence is STRICTLY ``> τ_conf`` (possibly none); a
      fully-activated block additionally FORCE-commits its single
      most-confident position when none clear — the progress guarantee.
    * **Freeze:** fully-decoded blocks leave the window OLDEST-FIRST only;
      ``freeze_observer`` (when injected) notifies the exact d2f KV cache
      (``freeze_block``), whose K/V capture happens at the NEXT forward (the
      capture-at-next-forward lesson). Stop when no block is active.

    ``plan`` returns the union frontier of every active block's masked
    positions plus ``window_end`` (the end of the last appended block) so the
    loop forwards exactly the window — the reference's growing ``x_t``.
    Reference defaults (``eval_dream.sh``): τ_add 0.1, τ_act 0.95, τ_conf 0.9.
    """

    confidence_threshold: float  # tau_conf (strict >, per the reference)
    block_size: int
    add_threshold: float = 0.1  # tau_add
    activate_threshold: float = 0.95  # tau_act
    # Fires freeze_observer(block_index) as each block freezes, in order; the
    # engine wires this to NativeD2FExactKVCacheStrategy.freeze_block when the
    # exact cache is ON. None (cache OFF) keeps the scheduler self-contained.
    freeze_observer: Callable[[int], None] | None = None
    # Commit 5 -- the WITHIN-window commit RULE (the NFE lever). The window
    # lifecycle (append/activate/freeze) is unchanged; only how many of an active
    # block's masked rows commit per forward changes.
    #   * ``"threshold"`` (default, BYTE-IDENTICAL to today): per-position raw
    #     max-prob ``> τ_conf`` (semi AND fully-activated), with the fully-
    #     activated argmax floor. ``scores`` = raw max-prob (TopConfidence).
    #   * ``"entropy_budget"``: the EB-Sampler JOINT-uncertainty selection
    #     (``Σ_{l∈U} H − max_{l∈U} H ≤ γ``) applied PER active block, so the
    #     commit set adapts to cumulative uncertainty instead of a fixed confidence
    #     bar. It does not measure dependencies. ``scores`` = −H (NegEntropy). The
    #     fully-activated floor is EB's
    #     unconditional prefix-of-one; a semi-activated block SUPPRESSES the lone
    #     forced commit (it commits only a genuine >=2-row admissible cluster),
    #     preserving the reference "semi never forces" invariant. τ_conf is
    #     IGNORED in this mode (the budget γ replaces it).
    commit_policy: str = "threshold"
    gamma: float = 0.0  # EB budget; read only when commit_policy == "entropy_budget"
    # Freeze-as-Commit v0: steps a fully-decoded block waits before it COMMITS
    # (freezes and becomes cacheable). ``0`` (default) = today's behaviour
    # exactly — a block freezes on the next plan() after its last mask is filled,
    # so the revision window is zero-length and a decoded block is never
    # revisable. A positive value defers the commit by that many steps, creating
    # the interval in which a later phase may revise the block. Deferring is the
    # SAFE direction: the exact cache captures a block's K/V only at the forward
    # AFTER it freezes, so an uncommitted block is not cached anywhere and
    # revising it invalidates nothing. (Un-freezing after commit would be the
    # unsafe direction and is deliberately not provided.) v0 ships the boundary
    # only; letting recovery act inside the window is a separate, gated phase.
    revision_window: int = 0
    # A recovery pass creates a fresh policy while the native exact cache keeps
    # its immutable prefix. Seed that prefix explicitly so the new policy does
    # not restart at block zero and issue an empty, truncated forward for a
    # block the owner has already frozen.
    initial_frozen_blocks: int = 0
    name: str = field(default="d2f_block_pipeline", init=False)
    _blocks: list = field(default=None, init=False)
    _appended_count: int = field(default=0, init=False)
    _eos_detected: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if int(self.block_size) < 1:
            raise ValueError(f"block_size must be >= 1, got {self.block_size}")
        if self.commit_policy not in {"threshold", "entropy_budget", "dilated"}:
            raise ValueError(
                "commit_policy must be 'threshold' (default, byte-identical), "
                f"'entropy_budget', or 'dilated', got {self.commit_policy!r}"
            )
        if not isinstance(self.revision_window, int) or isinstance(
            self.revision_window, bool
        ):
            raise ValueError(
                f"revision_window must be an int, got {self.revision_window!r}"
            )
        if self.revision_window < 0:
            raise ValueError(
                f"revision_window must be >= 0, got {self.revision_window}"
            )
        if type(self.initial_frozen_blocks) is not int:
            raise ValueError(
                "initial_frozen_blocks must be a non-boolean int; got "
                f"{self.initial_frozen_blocks!r}"
            )
        if self.initial_frozen_blocks < 0:
            raise ValueError(
                "initial_frozen_blocks must be >= 0, got "
                f"{self.initial_frozen_blocks}"
            )
        if float(self.gamma) < 0.0:
            raise ValueError(
                f"gamma (the EB joint-entropy budget) must be >= 0, got {self.gamma}"
            )
        if not (0.0 < float(self.add_threshold) <= 1.0):
            raise ValueError(
                "add_threshold (tau_add) must be in (0, 1], got "
                f"{self.add_threshold} — a value > 1 could never append past a "
                "fully-decoded block and would stall the pipeline"
            )
        if not (0.0 < float(self.activate_threshold) <= 1.0):
            raise ValueError(
                "activate_threshold (tau_act) must be in (0, 1], got "
                f"{self.activate_threshold} — a value > 1 would break the "
                "oldest-block progress guarantee"
            )
        if float(self.confidence_threshold) < 0.0:
            raise ValueError(
                "confidence_threshold (tau_conf) must be >= 0, got "
                f"{self.confidence_threshold}"
            )

    # -- lifecycle helpers ---------------------------------------------------- #

    def _init_blocks(self, state: "MaskState") -> None:
        fillable = state.all_fillable_positions()
        if fillable and fillable != list(
            range(fillable[0], fillable[0] + len(fillable))
        ):
            raise ValueError(
                "d2f_block_pipeline requires a CONTIGUOUS generated region (the "
                "d2f block partition is absolute-position geometry); got "
                "non-contiguous fillable positions"
            )
        start = fillable[0] if fillable else state.prompt_len
        layout = BlockLayout(
            start=start, length=len(fillable), block_size=int(self.block_size), origin=start
        )
        blocks = [_D2FBlock(*layout.span(index)) for index in range(layout.count)]
        if self.initial_frozen_blocks > len(blocks):
            raise ValueError(
                "initial_frozen_blocks exceeds the generated block count: "
                f"{self.initial_frozen_blocks} > {len(blocks)}"
            )
        for block in blocks[: self.initial_frozen_blocks]:
            if self._masked_in(state, block):
                raise ValueError(
                    "an initial frozen D2F block still contains a mask token"
                )
            block.activated = True
            block.frozen = True
            block.fully_decoded_step = 0
        self._blocks = blocks
        self._appended_count = self.initial_frozen_blocks

    def _masked_in(self, state: "MaskState", block: _D2FBlock) -> list:
        return [
            position
            for position in range(block.start, block.end)
            if state.x[position] == state.mask_token_id
        ]

    def _decoded_ratio(self, state: "MaskState", block: _D2FBlock) -> float:
        total = block.end - block.start
        return (total - len(self._masked_in(state, block))) / total

    # -- CommitPolicy surface --------------------------------------------------#

    @property
    def fully_decoded_block_count(self) -> int:
        """Blocks that hold no masked position (decoded), committed or not.

        Freeze-as-Commit v0: this is the OUTER state. It is >= the committed
        count; the difference is exactly the blocks currently inside their
        revision window.
        """
        if self._blocks is None:
            return 0
        return sum(
            1
            for b in self._blocks[: self._appended_count]
            if b.fully_decoded_step is not None
        )

    @property
    def committed_block_count(self) -> int:
        """Blocks that have COMMITTED (frozen, cacheable, irreversible)."""
        if self._blocks is None:
            return 0
        return sum(1 for b in self._blocks[: self._appended_count] if b.frozen)

    def plan(self, *, state: "MaskState", step: int) -> StepPlan:
        if self._blocks is None:
            self._init_blocks(state)
        blocks = self._blocks

        # 1. Freeze sweep — fully-decoded blocks leave the window OLDEST-FIRST
        # (the reference deactivates a block only when no earlier block is
        # active). Fires after the PREVIOUS step's commits, so the exact cache's
        # capture happens at the next forward, never at the commit step.
        for index, block in enumerate(blocks[: self._appended_count]):
            if block.frozen:
                continue
            if not self._masked_in(state, block):
                # Freeze-as-Commit v0: record when the block became fully decoded,
                # then commit only after ``revision_window`` further steps. At the
                # default window of 0 this collapses to the original behaviour —
                # recorded and committed in the same sweep, same step, same order.
                if block.fully_decoded_step is None:
                    block.fully_decoded_step = step
                if step - block.fully_decoded_step < self.revision_window:
                    # Still inside the revision window: NOT yet committed, so the
                    # exact cache is never told about it and nothing is captured.
                    # Oldest-first ordering is preserved by stopping here — a
                    # later block must not commit before this one.
                    break
                block.frozen = True
                if self.freeze_observer is not None:
                    self.freeze_observer(index)
            else:
                break

        # 2. EOS detection — a committed EOS anywhere in the appended region
        # stops future appends (the reference's ``eos_detected``); active
        # blocks keep decoding to completion.
        if not self._eos_detected and self._appended_count:
            eos = state.eos_token_id
            mask = state.mask_token_id
            region_end = blocks[self._appended_count - 1].end
            start = blocks[0].start
            self._eos_detected = any(
                state.x[position] == eos and state.x[position] != mask
                for position in range(start, region_end)
            )

        # 3. Append — at most ONE new semi-activated block per step, iff the
        # LAST appended block's decoded ratio >= tau_add (prompt counts as 1.0).
        if self._appended_count < len(blocks) and not self._eos_detected:
            last_ratio = (
                1.0
                if self._appended_count == 0
                else self._decoded_ratio(state, blocks[self._appended_count - 1])
            )
            if last_ratio >= float(self.add_threshold):
                self._appended_count += 1

        # 4. Activation sweep — block b+1 fully activates when block b's ratio
        # >= tau_act (sticky; the prompt's ratio is 1.0, so the first appended
        # block — and every oldest unfrozen block — is always fully-activated).
        previous_ratio = 1.0
        for block in blocks[: self._appended_count]:
            if previous_ratio >= float(self.activate_threshold):
                block.activated = True
            previous_ratio = self._decoded_ratio(state, block)

        # 5. Stop / frontier — terminate when no block is active (all appended
        # blocks frozen and nothing appended this step). Otherwise the frontier
        # is the union of every active block's masked positions, and the window
        # is the contiguous prefix [0, end-of-last-appended-block).
        active = [b for b in blocks[: self._appended_count] if not b.frozen]
        if not active:
            return StepPlan(frontier=[], stop=True)
        frontier = [
            position for block in active for position in self._masked_in(state, block)
        ]
        return StepPlan(
            frontier=frontier,
            window_end=blocks[self._appended_count - 1].end,
        )

    def select(self, *, scores, masked_positions, state: "MaskState", step: int) -> list:
        import numpy as np

        if self.commit_policy == "entropy_budget":
            return self._select_entropy_budget(
                scores=scores, masked_positions=masked_positions
            )
        if self.commit_policy == "dilated":
            return self._select_dilated(
                scores=scores, masked_positions=masked_positions
            )
        confidence = np.asarray(scores, dtype=np.float64)
        tau_conf = float(self.confidence_threshold)
        chosen: list[int] = []
        for block in self._blocks[: self._appended_count]:
            if block.frozen:
                continue
            rows = [
                row
                for row, position in enumerate(masked_positions)
                if block.start <= position < block.end
            ]
            if not rows:
                continue
            # The reference commit rule: STRICTLY above tau_conf (semi AND
            # fully-activated); a fully-activated block with no clear position
            # force-commits its single most-confident one (progress guarantee).
            clear = [row for row in rows if confidence[row] > tau_conf]
            if clear:
                chosen.extend(clear)
            elif block.activated:
                chosen.append(rows[int(np.argmax(confidence[rows]))])
        return chosen

    def _select_entropy_budget(self, *, scores, masked_positions) -> list:
        """Commit 5: the EB joint-entropy budget applied PER active block.

        ``scores`` are −H (the paired NegEntropy convention). Within each active
        (unfrozen, appended) block, sort that block's masked rows by ascending
        entropy and take the largest prefix with ``ΣH − maxH ≤ γ``
        (:func:`_entropy_budget_choices`, exactly the standalone EB rule). The
        floor-of-one is honored only for FULLY-activated blocks (the progress
        guarantee, the EB analog of threshold-mode's forced argmax); a
        SEMI-activated block suppresses the lone forced commit -- it commits its
        EB prefix only when the budget admits >= 2 jointly-low-uncertainty rows,
        never a single forced one (the reference "semi never forces" invariant).
        ``τ_conf`` is unused here. Only ACTIVE blocks contribute, so a frozen
        block's rows and any row beyond the appended window are never committed.
        """
        import numpy as np

        entropies = -np.asarray(scores, dtype=np.float64)
        chosen: list[int] = []
        for block in self._blocks[: self._appended_count]:
            if block.frozen:
                continue
            rows = [
                row
                for row, position in enumerate(masked_positions)
                if block.start <= position < block.end
            ]
            if not rows:
                continue
            # EB on THIS block's rows: the helper returns indices into the
            # block-local entropy array; map them back to frontier-row indices.
            block_entropies = entropies[rows]
            local = _entropy_budget_choices(block_entropies, gamma=float(self.gamma))
            block_choice = [rows[i] for i in local]
            if block.activated:
                # Fully-activated: EB's prefix-of-one is the progress floor.
                chosen.extend(block_choice)
            elif len(block_choice) >= 2:
                # Semi-activated: commit only a genuine multi-row admissible
                # cluster; never the lone forced floor (semi never forces).
                chosen.extend(block_choice)
        return chosen

    def _select_dilated(self, *, scores, masked_positions) -> list:
        """DUS within the D2F pipeline: per-block dilated commit, unconditional.

        For each non-frozen appended block, use the block's own dilated schedule
        (computed on first encounter, tracked via ``_dus_round``) to select
        which rows to commit THIS forward. Scores are IGNORED (DUS is
        unconditional). The fully-activated progress guarantee (floor of one) is
        naturally satisfied: the schedule always commits at least one position
        per round. Semi-activated blocks still participate (unlike threshold
        mode's "clear-or-nothing" for semi, DUS commits its scheduled group
        unconditionally — this is the APPROXIMATE behavior; quality-gated).

        This isolates the scheduling variable: both ``"threshold"`` and
        ``"dilated"`` arms share the same exact block-causal KV cache, active
        window, and pipeline lifecycle; only the WITHIN-block commit selection
        differs.
        """
        from edllm.dilated_schedule import dilated_schedule

        chosen: list[int] = []
        for block in self._blocks[: self._appended_count]:
            if block.frozen:
                continue
            rows = [
                row
                for row, position in enumerate(masked_positions)
                if block.start <= position < block.end
            ]
            if not rows:
                continue
            # Compute or reuse this block's dilated schedule
            if block._dus_round is None:
                block_size = block.end - block.start
                block._dus_schedule = dilated_schedule(block_size)
                block._dus_round = 0
            schedule = block._dus_schedule
            round_idx = block._dus_round
            if round_idx >= len(schedule):
                # All rounds exhausted; commit everything remaining (safety)
                chosen.extend(rows)
                continue
            # Map the schedule's 1-indexed positions to frontier-row indices.
            # The schedule covers positions 1..block_size; block.start + p - 1
            # is the absolute canvas position; find it in masked_positions.
            group = schedule[round_idx]
            for p in group:
                canvas_pos = block.start + p - 1
                for row in rows:
                    if masked_positions[row] == canvas_pos:
                        chosen.append(row)
                        break
            block._dus_round = round_idx + 1
        return chosen


def run_decode_loop(
    ctx: _LoopContext,
    *,
    unmask_policy: Any,
    commit_policy: Any,
    iteration_limit: int | None = None,
    exclude_mask_token: bool = True,
) -> None:
    """One loop for every decode strategy: forward -> score -> select -> commit.

    Each step the CommitPolicy ``plan``s the frontier (and may SKIP or STOP); on a
    PROCEED the loop runs a real forward, the UnmaskPolicy scores the frontier
    (RAW softmax) + resolves the committed tokens, the CommitPolicy ``select``s
    the rows to commit, and the manager commits them. ``ctx.steps`` caps the
    number of loop iterations (== real forwards, since the only non-forward exit
    is SKIP/STOP) -- byte-identical to the per-strategy caps it replaces.

    ``exclude_mask_token`` preserves the runtime's progress guarantee for the
    legacy strategies. BD3 disables it because the fixed upstream sampler ranks
    positions using the complete vocabulary distribution.
    """
    import numpy as np
    from edllm.cancellation import check_cancelled
    from .window_plan import classify_window  # one-way dep; no cycle

    limit = int(ctx.steps) if iteration_limit is None else iteration_limit
    if type(limit) is not int or limit < 0:
        raise ValueError("iteration_limit must be a nonnegative integer")
    for step in range(limit):
        check_cancelled()
        with ctx.scheduler_phase():
            plan = commit_policy.plan(state=ctx.state, step=step)
        if plan.stop:
            break
        if plan.skip:
            continue
        # Active-Window Runtime (Task A3/B1.5/B2, WS1/WS2): compute the
        # WindowExecutionPlan for this step BEFORE the forward so the plan
        # faithfully tracks the commit-policy frontier at the moment of the
        # forward.  We pass ``frontier=plan.frontier`` and
        # ``window_end=plan.window_end`` so ``forward_scope`` matches the
        # commit-policy's actual scope (not all masked positions).  Guarded by
        # ``is not None`` / ``execute_window_fn`` so the OFF path (the default)
        # runs zero new code and is byte-identical.
        window_plan = None
        if ctx.window_plan_sink is not None or ctx.execute_window_fn is not None:
            # C2a: compute corrections from the correction_policy when it is set.
            # Uses the event_log's commit_history as the detector input. The
            # recovery-loop's precise per-pass count is not available here, so
            # ``newly_unmasked_count`` is approximated with the CUMULATIVE
            # committed count: with ConfWeightedRemask that yields cap =
            # committed-1, so per-step plans CAN carry non-empty
            # remask_positions — today nothing in-repo consumes them (the plan
            # is observational; the recovery loop owns real remask decisions).
            # This is behavior-neutral: with correction_policy=None the call is
            # skipped; with a NoOp policy corrections_for() returns an empty
            # CorrectionPlan.
            corrections = None
            if ctx.correction_policy is not None:
                commit_history = (
                    ctx.event_log.commit_history() if ctx.event_log is not None else {}
                )
                dist_history = (
                    ctx.event_log.dist_history if ctx.event_log is not None else {}
                )
                corrections = ctx.correction_policy.corrections_for(
                    state=ctx.state,
                    commit_history=commit_history,
                    newly_unmasked_count=ctx.committed_token_count,
                    dist_history=dist_history,
                )
            window_plan = classify_window(
                ctx.state,
                frontier=plan.frontier,
                window_end=plan.window_end,
                step=step,
                corrections=corrections,
            )
            if ctx.window_plan_sink is not None:
                ctx.window_plan_sink(window_plan)
        frontier = list(plan.frontier)
        # Forward execution has its own backend timers. Keep host selection
        # timing around convention/gather/scoring/choice only, so the two
        # reported phases do not count the same forward interval twice.
        if ctx.propose_fn is None:
            if ctx.execute_window_fn is not None:
                exec_result = ctx.forward_window(window_plan)
            else:
                raw = ctx.forward(upto=plan.window_end)
        check_cancelled()
        with ctx.select_phase():
            if ctx.propose_fn is not None:
                if (not isinstance(unmask_policy, TopConfidence) or
                    unmask_policy.confidence_fn is not None or ctx.record_distributions):
                    raise ValueError("device proposals require raw TopConfidence without distribution tracing")
                scores, token_ids, forwarded = ctx.propose_fn(plan)
                if len(scores) != len(frontier) or len(token_ids) != len(frontier):
                    raise ValueError("device proposal does not match the policy frontier")
                ctx.forward_count += 1
                if ctx.trace_observer is not None:
                    ctx.trace_observer.on_forward(
                        scope=plan.window_end, forwarded_count=forwarded
                    )
            elif ctx.execute_window_fn is not None:
                # B2 routed path: execute_window already applies the logits
                # convention (dream_ar_shift / raw). result.logits rows are in
                # forward_scope == frontier order, so they ARE the frontier logits.
                # MUST NOT call select_transform / select_logits again — that
                # would double-shift Dream (finding 1 pin).
                frontier_logits = np.asarray(exec_result.logits, dtype=np.float32)
            else:
                # Original path: raw forward + convention shift + frontier slice.
                selection = ctx.select_logits(raw)
                # Gather on the originating backend before the host conversion.
                # MLX otherwise materializes every prefix/vocabulary row even
                # though the policy only reads this frontier. Lists retain the
                # reference/fake-forward compatibility path.
                frontier_logits = np.asarray(
                    selection[frontier] if hasattr(selection, "shape")
                    else np.asarray(selection)[frontier], dtype=np.float32,
                )
                # The gathered host rows own their storage. Do not retain the
                # full device logits until the next forward allocates its output.
                del raw, selection
            if ctx.propose_fn is None:
                selection_logits = frontier_logits
                if exclude_mask_token:
                    selection_logits = _exclude_mask_token(
                        frontier_logits, mask_token_id=ctx.state.mask_token_id
                    )
                scores, token_ids = unmask_policy.score_and_tokens(
                    logits_at_masked=selection_logits, masked_positions=frontier
                )
            if ctx.record_distributions and ctx.event_log is not None:
                # The relative KL-drift signal observes the model's unfiltered
                # raw-softmax distribution. The selection-only mask exclusion
                # above must not rewrite diagnostic history. Gated, so OFF pays
                # nothing.
                probs = _softmax(np.asarray(frontier_logits, dtype=np.float64))
                for row, position in enumerate(frontier):
                    ctx.event_log.record_distribution(
                        position=int(position), distribution=probs[row]
                    )
            chosen = commit_policy.select(
                scores=scores,
                masked_positions=frontier,
                state=ctx.state,
                step=step,
            )
        with ctx.commit_phase():
            ctx.commit_rows(
                positions=frontier,
                rows=chosen,
                token_ids=token_ids,
                scores=scores,
                step=step,
            )
        if ctx.propose_fn is None:
            # These arrays can span the entire vocabulary. Their ownership ends
            # with this commit, before the next forward allocates fresh logits.
            del frontier_logits, selection_logits
            if ctx.execute_window_fn is not None:
                del exec_result


# ---------------------------------------------------------------------------
# Strategy construction -- shared by host and device execution
# ---------------------------------------------------------------------------


class DecodingStrategy(Protocol):
    decode_loop_kind: str
    entropy_used: bool

    def run(self, ctx: _LoopContext) -> None:
        """Drive the decode loop for this strategy over ``ctx``."""
        ...


def build_decode_strategy(
    *,
    decode_strategy: str,
    temperature: float = 0.0,
    top_p: float = 1.0,
    eps: float = 1e-8,
    confidence_threshold: float = 0.9,
    block_size: int = 8,
    block_layout: BlockLayout | None = None,
    requires_prompt_block_alignment: bool = True,
    gamma: float = 0.1,
    d2f_add_threshold: float = 0.1,
    d2f_activate_threshold: float = 0.95,
    d2f_commit_policy: str = "threshold",
    d2f_freeze_observer: Callable[[int], None] | None = None,
    d2f_frozen_block_count_fn: Callable[[], int] | None = None,
    seed: int = 0,
) -> DecodingStrategy:
    """Assemble the existing policy for host or device execution.

    Scoring, commit budgets and stop behavior belong to the shared strategies,
    not to the backend adapter. Device adapters still admit only the policies
    their proposal contract implements; this factory adds no device capability.
    Cache lifecycle hooks come from the caller's existing cache owner.
    """
    if decode_strategy == "entropy_count":
        return EntropyCountStrategy(temperature=temperature, top_p=top_p, eps=eps)
    if decode_strategy == "confidence_threshold":
        return ConfidenceThresholdStrategy(confidence_threshold=confidence_threshold)
    if decode_strategy == "block_threshold":
        return BlockThresholdStrategy(
            confidence_threshold=confidence_threshold, block_size=int(block_size),
            freeze_observer=d2f_freeze_observer,
        )
    if decode_strategy == "dilated_unmask":
        return DilatedUnmaskStrategy(block_size=int(block_size))
    if decode_strategy == "entropy_budget":
        # The budget uses raw entropy, independently of token-sampling options.
        return EntropyBudgetStrategy(
            gamma=float(gamma), block_size=int(block_size), temperature=1.0, top_p=1.0,
        )
    if decode_strategy == "d2f_block_pipeline":
        return D2FBlockPipelineStrategy(
            confidence_threshold=confidence_threshold, block_size=int(block_size),
            add_threshold=float(d2f_add_threshold),
            activate_threshold=float(d2f_activate_threshold),
            freeze_observer=d2f_freeze_observer,
            frozen_block_count_fn=d2f_frozen_block_count_fn,
            commit_policy=str(d2f_commit_policy), gamma=float(gamma),
        )
    if decode_strategy == "fixed_quota":
        from ..runtime_config import validate_fixed_quota_request_sampling

        validate_fixed_quota_request_sampling(temperature=temperature, top_p=top_p)
        return FixedQuotaStrategy(block_size=int(block_size), temperature=float(temperature),
                                  seed=int(seed))
    if decode_strategy == "bd3_fixed_quota":
        from ..runtime_config import validate_fixed_quota_request_sampling

        validate_fixed_quota_request_sampling(
            temperature=temperature, top_p=top_p, strategy="bd3_fixed_quota"
        )
        return BD3FixedQuotaStrategy(
            block_size=int(block_size), block_layout=block_layout,
            requires_prompt_block_alignment=requires_prompt_block_alignment,
            temperature=float(temperature), seed=int(seed),
        )
    raise ValueError(f"unknown decode_strategy: {decode_strategy!r}")


@dataclass(slots=True)
class EntropyCountStrategy:
    """Diffusion-schedule entropy decoding (the legacy ``entropy_count`` path).

    UnmaskPolicy: :class:`NegEntropy` (temperature/top_p adjusted). CommitPolicy:
    :class:`EntropyCountCommit` (top-``transfer_count`` by confidence; SKIP when
    ``transfer_count <= 0``).
    """

    temperature: float
    top_p: float | None
    eps: float
    # Optional scorer for controlled policy experiments; production uses the default.
    confidence_fn: ConfidenceFn | None = None
    decode_loop_kind: str = field(default="mlx_dream_masked_diffusion_entropy", init=False)
    entropy_used: bool = field(default=True, init=False)

    def _confidence(self, mask_logits) -> tuple:
        if self.confidence_fn is not None:
            return self.confidence_fn(mask_logits)
        return _entropy_confidence(
            mask_logits, temperature=self.temperature, top_p=self.top_p
        )

    def run(self, ctx: _LoopContext) -> None:
        unmask_policy = NegEntropy(
            temperature=self.temperature,
            top_p=self.top_p,
            confidence_fn=self.confidence_fn,
        )
        commit_policy = EntropyCountCommit(eps=self.eps, steps=int(ctx.steps))
        run_decode_loop(ctx, unmask_policy=unmask_policy, commit_policy=commit_policy)


@dataclass(slots=True)
class ConfidenceThresholdStrategy:
    """Whole-sequence confidence-threshold decoding (legacy ``confidence_threshold``).

    UnmaskPolicy: :class:`TopConfidence` (RAW softmax max-probability, no
    temperature/top_p). CommitPolicy: :class:`ThresholdCommit` (every masked
    position clearing ``tau``; floor of one). Carries the entropy decode-kind
    label + ``entropy_used=True`` exactly as the legacy whole-sequence path did.
    """

    confidence_threshold: float
    # Optional scorer for controlled policy experiments; production uses raw softmax.
    confidence_fn: ConfidenceFn | None = None
    decode_loop_kind: str = field(default="mlx_dream_masked_diffusion_entropy", init=False)
    entropy_used: bool = field(default=True, init=False)

    def _confidence(self, mask_logits) -> tuple:
        if self.confidence_fn is not None:
            return self.confidence_fn(mask_logits)
        return _raw_max_prob_confidence(mask_logits)

    def run(self, ctx: _LoopContext) -> None:
        unmask_policy = TopConfidence(confidence_fn=self.confidence_fn)
        commit_policy = ThresholdCommit(confidence_threshold=self.confidence_threshold)
        run_decode_loop(ctx, unmask_policy=unmask_policy, commit_policy=commit_policy)


@dataclass(slots=True)
class DilatedUnmaskStrategy:
    """Dilated Unmasking Scheduler (DUS, arXiv:2506.19037).

    APPROXIMATE — unconditional commit via a deterministic dilated pattern (no
    confidence gate). Default-OFF; quality-gate mandatory.
    """

    block_size: int
    base: int = 2
    confidence_fn: ConfidenceFn | None = None
    decode_loop_kind: str = field(default="mlx_dream_dilated_unmask", init=False)
    entropy_used: bool = field(default=False, init=False)

    def run(self, ctx: "_LoopContext") -> None:
        unmask_policy = TopConfidence(confidence_fn=self.confidence_fn)
        commit_policy = DilatedCommitPolicy(
            block_size=self.block_size, base=self.base
        )
        run_decode_loop(ctx, unmask_policy=unmask_policy, commit_policy=commit_policy)


@dataclass(slots=True)
class BlockThresholdStrategy:
    """Fast-dLLM block-sequential confidence decoding (legacy ``block_threshold``).

    UnmaskPolicy: :class:`TopConfidence` (RAW softmax max-probability).
    CommitPolicy: :class:`BlockThresholdCommit` (within the current left-to-right
    block of ``block_size`` positions, commit every masked position clearing
    ``tau``; floor of one; later blocks stay masked until reached). Carries the
    distinct block decode-kind label + ``entropy_used=False`` so honesty gates
    stay accurate.
    """

    confidence_threshold: float
    block_size: int
    # Confidence policy: RAW softmax max-probability (injectable, as above).
    confidence_fn: ConfidenceFn | None = None
    # D2F-2: injected by the engine only when the exact single-block cache is
    # active. The strategy remains self-contained when it is None.
    freeze_observer: Callable[[int], None] | None = None
    decode_loop_kind: str = field(default="mlx_dream_block_threshold", init=False)
    entropy_used: bool = field(default=False, init=False)

    def _confidence(self, block_logits) -> tuple:
        if self.confidence_fn is not None:
            return self.confidence_fn(block_logits)
        return _raw_max_prob_confidence(block_logits)

    def run(self, ctx: _LoopContext) -> None:
        unmask_policy = TopConfidence(confidence_fn=self.confidence_fn)
        commit_policy = BlockThresholdCommit(
            confidence_threshold=self.confidence_threshold,
            block_size=int(self.block_size),
            freeze_observer=self.freeze_observer,
        )
        run_decode_loop(ctx, unmask_policy=unmask_policy, commit_policy=commit_policy)


@dataclass(slots=True)
class FixedQuotaStrategy:
    """Fixed-quota scoring and commits through the shared decode loop."""

    block_size: int
    confidence_fn: ConfidenceFn | None = None
    temperature: float = 0.0
    seed: int = 0
    decode_loop_kind: str = field(default="mlx_bidirectional_fixed_quota", init=False)
    entropy_used: bool = field(default=False, init=False)

    def run(self, ctx: _LoopContext) -> None:
        policy = FixedQuotaCommit(
            block_size=self.block_size, requested_steps=ctx.steps,
            generated_token_count=len(ctx.state.all_fillable_positions()),
        )
        run_decode_loop(
            ctx, unmask_policy=TopConfidence(confidence_fn=self.confidence_fn,
                                             temperature=self.temperature, seed=self.seed),
            commit_policy=policy, iteration_limit=policy.iteration_limit,
            exclude_mask_token=False,
        )


@dataclass(slots=True)
class BD3FixedQuotaStrategy:
    """Deterministic fixed-quota decoding over absolute attention blocks.

    It reuses the common loop and raw-max-prob scorer. The only new behavior is
    BD3LM's fixed per-round transfer count and absolute block partition.
    """

    block_size: int
    confidence_fn: ConfidenceFn | None = None
    block_layout: BlockLayout | None = None
    requires_prompt_block_alignment: bool = True
    temperature: float = 0.0
    seed: int = 0
    decode_loop_kind: str = field(
        default="mlx_qwen3_bd3_fixed_quota", init=False
    )
    entropy_used: bool = field(default=False, init=False)

    def run(self, ctx: _LoopContext) -> None:
        generated_token_count = len(ctx.state.all_fillable_positions())
        layout = self.block_layout or BlockLayout(
            start=ctx.state.prompt_len,
            length=generated_token_count,
            block_size=int(self.block_size),
        )
        commit_policy = BD3FixedQuotaCommit(
            block_size=int(self.block_size),
            requested_steps=int(ctx.steps),
            generated_token_count=generated_token_count,
            block_layout=layout,
            requires_prompt_block_alignment=self.requires_prompt_block_alignment,
        )
        run_decode_loop(
            ctx,
            unmask_policy=TopConfidence(confidence_fn=self.confidence_fn,
                                        temperature=self.temperature, seed=self.seed),
            commit_policy=commit_policy,
            iteration_limit=commit_policy.iteration_limit,
            # The fixed upstream BD3 sampler scores and selects over the full
            # vocabulary, including the mask-token column.
            exclude_mask_token=False,
        )


@dataclass(slots=True)
class EntropyBudgetStrategy:
    """EB-Sampler adaptive entropy-budget decoding (Phase F F2; opt-in).

    UnmaskPolicy: :class:`NegEntropy` (the −H confidence, so the commit gets the
    entropy the budget rule needs). CommitPolicy: :class:`EntropyBudgetCommit`
    (within the current left-to-right block, commit the largest ascending-entropy
    prefix with ``ΣH − maxH ≤ γ``; floor of one). Carries the entropy decode-kind
    label + ``entropy_used=True`` (it genuinely decodes on entropy). NOT byte-exact
    with ``block_threshold`` by design -- it is the measured aggression upgrade.

    ``confidence_fn`` supports controlled policy experiments. Production uses
    the shared entropy scorer directly.
    """

    gamma: float
    block_size: int = 8
    temperature: float = 1.0
    top_p: float | None = None
    confidence_fn: ConfidenceFn | None = None
    decode_loop_kind: str = field(default="mlx_dream_entropy_budget", init=False)
    entropy_used: bool = field(default=True, init=False)

    def run(self, ctx: _LoopContext) -> None:
        unmask_policy = NegEntropy(
            temperature=self.temperature,
            top_p=self.top_p,
            confidence_fn=self.confidence_fn,
        )
        commit_policy = EntropyBudgetCommit(
            gamma=float(self.gamma), block_size=int(self.block_size)
        )
        run_decode_loop(ctx, unmask_policy=unmask_policy, commit_policy=commit_policy)


@dataclass(slots=True)
class D2FBlockPipelineStrategy:
    """D2F pipelined multi-block decoding (D2F-3; opt-in, arXiv 2508.09192).

    UnmaskPolicy: :class:`TopConfidence` (RAW softmax max-probability — the
    reference's ``initial_confidence`` at temperature 0). CommitPolicy:
    :class:`D2FBlockPipelineCommit` (the sliding block window: τ_add append /
    τ_act activate / strict-τ_conf commit with the fully-activated argmax floor
    / oldest-first freeze). The plan's ``window_end`` makes the loop forward
    only the appended window each step — blocks decode CONCURRENTLY in one
    forward, which is where D2F's speedup comes from.

    ``freeze_observer`` is the exact d2f KV cache seam: the engine injects
    ``NativeD2FExactKVCacheStrategy.freeze_block`` when the cache is ON, so a
    frozen block's K/V is captured at the NEXT forward (never at the commit
    step). ``confidence_fn`` is injectable exactly as the other strategies.
    Carries a distinct kind label + ``entropy_used=False`` (raw-confidence
    decode). NOT byte-exact with any sequential strategy by design — gated on
    the reduction (window of one == sequential) and cache-composition tests.
    """

    confidence_threshold: float  # tau_conf
    block_size: int
    add_threshold: float = 0.1  # tau_add (reference eval default)
    activate_threshold: float = 0.95  # tau_act (reference eval default)
    confidence_fn: ConfidenceFn | None = None
    freeze_observer: Callable[[int], None] | None = None
    frozen_block_count_fn: Callable[[], int] | None = None
    # Commit 5 -- the within-window commit RULE. "threshold" (default) keeps the
    # byte-identical τ_conf pipeline with the RAW max-prob UnmaskPolicy;
    # "entropy_budget" swaps the within-block rule for the EB joint-budget and the
    # NegEntropy (−H) UnmaskPolicy (the committed argmax token is identical, so
    # only the commit SET size changes). ``gamma`` is the EB budget.
    commit_policy: str = "threshold"
    gamma: float = 0.0
    decode_loop_kind: str = field(default="mlx_dream_d2f_block_pipeline", init=False)
    # Honest label: the default threshold pipeline decodes on RAW max-prob
    # (entropy_used False); the opt-in entropy_budget rule genuinely decodes on
    # the model's Shannon entropy (−H), so it reports entropy_used True. Computed
    # in __post_init__ so the flag never lies about which signal drove the commit.
    entropy_used: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.commit_policy not in {"threshold", "entropy_budget", "dilated"}:
            raise ValueError(
                "commit_policy must be 'threshold' (default, byte-identical), "
                f"'entropy_budget', or 'dilated', got {self.commit_policy!r}"
            )
        self.entropy_used = self.commit_policy == "entropy_budget"
        # DUS is an APPROXIMATE policy (unconditional commit); give it a distinct
        # decode_loop_kind so downstream metrics can distinguish it from the exact
        # threshold default.
        if self.commit_policy == "dilated":
            self.decode_loop_kind = "mlx_dream_d2f_block_pipeline_dilated"

    def run(self, ctx: _LoopContext) -> None:
        if self.commit_policy == "entropy_budget":
            # EB needs the RAW Shannon entropy (no temperature/top_p sharpening,
            # per the spec) -> NegEntropy at temperature=1.0/top_p=1.0. Its argmax
            # token equals the raw-max-prob argmax, so the committed token is
            # unchanged vs threshold mode; only the commit set differs.
            unmask_policy = NegEntropy(
                temperature=1.0, top_p=1.0, confidence_fn=self.confidence_fn
            )
        else:
            unmask_policy = TopConfidence(confidence_fn=self.confidence_fn)
        commit_policy = D2FBlockPipelineCommit(
            confidence_threshold=self.confidence_threshold,
            block_size=int(self.block_size),
            add_threshold=float(self.add_threshold),
            activate_threshold=float(self.activate_threshold),
            freeze_observer=self.freeze_observer,
            commit_policy=str(self.commit_policy),
            gamma=float(self.gamma),
            initial_frozen_blocks=(
                0
                if self.frozen_block_count_fn is None
                else self.frozen_block_count_fn()
            ),
        )
        run_decode_loop(ctx, unmask_policy=unmask_policy, commit_policy=commit_policy)


# ---------------------------------------------------------------------------
# DecodeEventLog -- the error model's ordered provenance (Phase F3)
# ---------------------------------------------------------------------------
#
# The thesis (DMax arXiv 2604.08302 / IBM Soft-Masked Diffusion): binary
# mask->token is too hard, so error ACCUMULATION is the core failure. The cure is
# to treat commits as REVISABLE and to ACCOUNT for the correction process
# explicitly. This log is that accounting: an ordered list of typed events --
# Commit / Flag / Remask / Recommit / RecoveryRound -- recording the full
# commit->flag->remask->recommit provenance per position.
#
# It is the ``commit_history`` a detector reads (``commit_history()`` returns the
# CURRENT committed token + score per position, recommits superseding the original
# commit). It is mlx-free and a PASSIVE observer: attaching one to the decode loop
# records events but NEVER feeds back into a commit decision, so the error-fixing
# OFF path stays byte-identical.


@dataclass(frozen=True, slots=True)
class Commit:
    """A position was unmasked: ``token`` committed at ``position`` on ``step``.

    ``score`` is the UnmaskPolicy confidence at the commit (raw max-prob for the
    threshold policies, ``-H`` for the entropy ones -- higher == more confident),
    threaded from the loop so detectors can read a committed token's confidence
    without re-forwarding.
    """

    position: int
    token: int
    score: float
    step: int


@dataclass(frozen=True, slots=True)
class Flag:
    """A detector believes the commit at ``position`` is likely wrong.

    ``reason`` names the detector; ``severity`` (higher == more suspect) lets a
    RemaskPolicy rank flags; ``round`` is the recovery round it was raised in.
    """

    position: int
    reason: str
    severity: float
    round: int


@dataclass(frozen=True, slots=True)
class Remask:
    """A flagged commit was re-masked (token -> mask) in ``round``.

    ``prev_token`` is the value that was discarded, so a Recommit can tell whether
    the re-decode changed it (a correction) or reproduced it (churn).
    """

    position: int
    prev_token: int
    round: int


@dataclass(frozen=True, slots=True)
class Recommit:
    """A re-masked position was re-decoded to ``new_token`` in ``round``.

    A *correction* iff ``new_token != prev_token``; otherwise it is *churn* (a
    re-mask + re-decode that reproduced the same token -- wasted work the
    accounting must surface so the recovery loop's value is honest).
    """

    position: int
    new_token: int
    prev_token: int
    round: int

    @property
    def is_correction(self) -> bool:
        return self.new_token != self.prev_token


@dataclass(frozen=True, slots=True)
class RecoveryRound:
    """The summary of one detect -> remask -> re-decode round.

    ``flagged`` / ``remasked`` / ``corrected`` / ``churned`` are the per-round
    counts; ``stop_reason`` records WHY the loop ended after this round (one of
    ``"no_flags"`` / ``"round_budget"`` / ``"no_improvement"`` / ``"no_masks"``)
    or ``""`` if the loop continued.
    """

    round: int
    flagged: int
    remasked: int
    corrected: int
    churned: int
    stop_reason: str = ""


@dataclass(slots=True)
class DecodeEventLog:
    """An ordered, mlx-free log of decode events (the error model's spine).

    Append-only via the typed ``record_*`` helpers; queried by type
    (:meth:`commits` / :meth:`flags` / :meth:`remasks` / :meth:`recommits` /
    :meth:`corrections` / :meth:`recovery_rounds`). :meth:`commit_history` folds
    the Commit + Recommit events into the CURRENT committed (token, score) per
    position -- the view a detector reads.
    """

    events: list = field(default_factory=list)
    # Per-position list of the (raw-softmax) distributions seen across the steps a
    # position appeared in the scored frontier. Recorded by the loop ONLY when
    # ``ctx.record_distributions`` is set (KL-drift recovery), so the OFF / non-KL
    # paths pay nothing. Read by :class:`KLDriftDetector`.
    dist_history: dict = field(default_factory=dict)

    def record_distribution(self, *, position: int, distribution) -> None:
        self.dist_history.setdefault(int(position), []).append(distribution)

    def record_commit(self, *, position: int, token: int, score: float, step: int) -> None:
        self.events.append(
            Commit(position=int(position), token=int(token), score=float(score), step=int(step))
        )

    def record_flag(self, *, position: int, reason: str, severity: float, round: int) -> None:
        self.events.append(
            Flag(position=int(position), reason=str(reason), severity=float(severity), round=int(round))
        )

    def record_remask(self, *, position: int, prev_token: int, round: int) -> None:
        self.events.append(
            Remask(position=int(position), prev_token=int(prev_token), round=int(round))
        )

    def record_recommit(self, *, position: int, new_token: int, prev_token: int, round: int) -> None:
        self.events.append(
            Recommit(
                position=int(position),
                new_token=int(new_token),
                prev_token=int(prev_token),
                round=int(round),
            )
        )

    def record_recovery_round(self, round_summary: RecoveryRound) -> None:
        self.events.append(round_summary)

    def commits(self) -> list:
        return [e for e in self.events if isinstance(e, Commit)]

    def flags(self) -> list:
        return [e for e in self.events if isinstance(e, Flag)]

    def remasks(self) -> list:
        return [e for e in self.events if isinstance(e, Remask)]

    def recommits(self) -> list:
        return [e for e in self.events if isinstance(e, Recommit)]

    def corrections(self) -> list:
        """The subset of recommits that actually changed the token."""
        return [e for e in self.recommits() if e.is_correction]

    def recovery_rounds(self) -> list:
        return [e for e in self.events if isinstance(e, RecoveryRound)]

    def commit_history(self) -> dict:
        """The CURRENT committed token (+ its commit score) at each position.

        Folds the events in order, latest wins for the TOKEN: a Commit sets the
        position's (token, score); a Recommit supersedes the token. For the
        SCORE, the recovery loop appends the re-decode's fresh ``Commit`` (with a
        real score) BEFORE its provenance ``Recommit``, so the Recommit keeps the
        existing entry's score (the fresh one) rather than discarding it -- the
        detector sees the corrected position's real confidence. A Recommit with no
        prior Commit (only the spec-level provenance, never the real loop) carries
        ``NaN`` so a score detector skips it. This is the ``commit_history`` an
        :class:`ErrorMaskDetector` reads.
        """
        history: dict = {}
        for event in self.events:
            if isinstance(event, Commit):
                history[event.position] = event
            elif isinstance(event, Recommit):
                prior = history.get(event.position)
                history[event.position] = Commit(
                    position=event.position,
                    token=event.new_token,
                    score=prior.score if prior is not None else float("nan"),
                    step=prior.step if prior is not None else -1,
                )
        return history


# ---------------------------------------------------------------------------
# Self-correction interfaces (Phase F3)
# ---------------------------------------------------------------------------
#
# The detect -> remask -> re-decode machinery. The interfaces ship with no-op
# defaults (OFF): the recovery loop with recovery disabled runs EXACTLY one decode
# pass, so the default decode is byte-identical to today. F3 fills in the
# training-free detectors (low-psi / inter-step KL-drift / repetition), the
# ReMDM-conf remask, and the bounded monotone-progress loop.


@dataclass(frozen=True, slots=True)
class FlaggedPosition:
    """A committed position a detector believes is likely wrong.

    ``reason`` names the detector; ``severity`` (higher == more suspect) lets a
    :class:`RemaskPolicy` rank flags.
    """

    position: int
    reason: str
    severity: float


class ErrorMaskDetector(Protocol):
    name: str

    def flag(self, *, state: "MaskState", commit_history: Any, **kwargs) -> list:
        """Flag likely-wrong COMMITTED positions (``FlaggedPosition`` list).

        ``commit_history`` is :meth:`DecodeEventLog.commit_history` -- the current
        committed (token, score) per position. ``**kwargs`` carries optional
        signals a detector may need (e.g. ``dist_history`` for KL-drift); a
        detector that doesn't need them ignores the extras, so detectors stay
        composable behind one call.
        """
        ...


class RemaskPolicy(Protocol):
    name: str

    def select_remask(
        self, *, flagged: list, state: "MaskState", newly_unmasked_count: int
    ) -> list:
        """Which flagged positions to actually re-mask this round."""
        ...


@dataclass(slots=True)
class NoOpErrorMaskDetector:
    """The default: flags nothing (error-fixing OFF)."""

    name: str = field(default="noop_detector", init=False)

    def flag(self, *, state: "MaskState", commit_history: Any, **kwargs) -> list:
        return []


@dataclass(slots=True)
class NoOpRemaskPolicy:
    """The default: re-masks nothing (error-fixing OFF)."""

    name: str = field(default="noop_remask", init=False)

    def select_remask(
        self, *, flagged: list, state: "MaskState", newly_unmasked_count: int
    ) -> list:
        return []


# ---------------------------------------------------------------------------
# Training-free ErrorMaskDetectors (Phase F3) -- replace the no-ops; composable
# ---------------------------------------------------------------------------
#
# Each is zero-cost (reads the commit history / the working ids; no extra
# forward) and pluggable. They flag likely-wrong COMMITTED positions; the
# RemaskPolicy decides which flags to act on under the monotone invariant.


@dataclass(slots=True)
class LowConfidenceDetector:
    """Flag commits whose confidence fell below ``floor`` (ReMDM-conf psi).

    Reads the ``commit_history`` (position -> the current :class:`Commit`, whose
    ``score`` is the UnmaskPolicy confidence). Any position committed with
    ``score < floor`` is flagged; severity = ``floor - score`` (higher == more
    suspect), so the RemaskPolicy re-masks the least-confident first. A position
    whose score is ``NaN`` (a recommit not yet re-scored) is skipped -- only freshly
    scored commits carry a confidence signal.
    """

    floor: float = 0.5
    name: str = field(default="low_confidence", init=False)

    def flag(self, *, state: "MaskState", commit_history: Any, **kwargs) -> list:
        import math as _math

        flagged = []
        for position, commit in (commit_history or {}).items():
            position = int(position)
            # Skip positions currently re-masked (in flight) or out of range --
            # only score a standing commit (avoids re-flagging a hole the loop
            # already routed back to the frontier).
            if (
                position < 0
                or position >= len(state.x)
                or state.x[position] == state.mask_token_id
            ):
                continue
            score = float(commit.score)
            if _math.isnan(score):
                continue
            if score < float(self.floor):
                flagged.append(
                    FlaggedPosition(
                        position=int(position),
                        reason="low_confidence",
                        severity=float(self.floor) - score,
                    )
                )
        return flagged


@dataclass(slots=True)
class RepetitionDetector:
    """Flag adjacent / local duplicate n-grams in the committed region.

    Repetition is the signature parallel-decode failure (our tempura bug). Over
    the committed positions in generation order, this flags the SECOND-and-later
    occurrences of a repeated unigram run (``a a a`` -> flag the 2nd, 3rd) and the
    repeated occurrence of a local n-gram up to ``max_ngram`` (``a b a b`` ->
    flag the repeated ``a b``), leaving the first occurrence committed so progress
    is preserved. Severity is uniform (1.0) -- repetition is categorical, not
    graded. Reads only the committed token ids (the ``commit_history`` keys /
    ``state.x``); no model call.
    """

    max_ngram: int = 3
    name: str = field(default="repetition", init=False)

    def flag(self, *, state: "MaskState", commit_history: Any, **kwargs) -> list:
        history = commit_history or {}
        # The committed positions in generation order (ascending position).
        positions = sorted(
            int(position)
            for position in history
            if 0 <= int(position) < len(state.x)
            and state.x[int(position)] != state.mask_token_id
        )
        tokens = [state.x[p] for p in positions]
        flagged_positions: set[int] = set()
        n = len(tokens)
        # Adjacent unigram runs: flag every repeat after the first.
        for i in range(1, n):
            if tokens[i] == tokens[i - 1]:
                flagged_positions.add(positions[i])
        # Local repeated n-grams (length 2..max_ngram): an n-gram immediately
        # followed by the SAME n-gram -> flag the second occurrence's positions.
        for gram in range(2, int(self.max_ngram) + 1):
            for i in range(0, n - 2 * gram + 1):
                first = tokens[i : i + gram]
                second = tokens[i + gram : i + 2 * gram]
                if first == second:
                    for j in range(i + gram, i + 2 * gram):
                        flagged_positions.add(positions[j])
        return [
            FlaggedPosition(position=p, reason="repetition", severity=1.0)
            for p in sorted(flagged_positions)
        ]


@dataclass(slots=True)
class KLDriftDetector:
    """Flag positions whose per-step distribution drifts > ``threshold``.

    A *relative* signal (the spec's hedge against miscalibrated absolute
    confidence): for each position the loop saw at >= 2 steps, compute the KL
    divergence ``KL(p_t || p_{t+1})`` between its last two recorded distributions;
    flag it if the max consecutive-step KL exceeds ``threshold``. Reads
    ``dist_history`` (position -> list of per-step probability vectors), recorded
    by the loop only when KL-drift recovery is enabled (so the OFF / non-KL paths
    pay nothing). Severity = the drift magnitude.
    """

    threshold: float = 1.0
    name: str = field(default="kl_drift", init=False)

    def flag(self, *, state: "MaskState", commit_history: Any, dist_history=None, **kwargs) -> list:
        import numpy as np

        flagged = []
        for position, dists in (dist_history or {}).items():
            position = int(position)
            # Same standing-commit canvas guard as the other detectors: a
            # still-masked (or out-of-range) position was never committed, so
            # flagging it would book a non-existent error (phantom remask +
            # inflated correction counts downstream).
            if (
                position < 0
                or position >= len(state.x)
                or state.x[position] == state.mask_token_id
            ):
                continue
            if len(dists) < 2:
                continue
            drift = 0.0
            for a, b in zip(dists[:-1], dists[1:], strict=True):
                pa = np.asarray(a, dtype=np.float64)
                pb = np.asarray(b, dtype=np.float64)
                kl = float(np.sum(pa * (np.log(pa + 1e-10) - np.log(pb + 1e-10))))
                drift = max(drift, kl)
            if drift > float(self.threshold):
                flagged.append(
                    FlaggedPosition(position=int(position), reason="kl_drift", severity=drift)
                )
        return flagged


@dataclass(slots=True)
class CompositeDetector:
    """Compose several detectors -- the UNION of their flags (highest severity wins).

    Detectors are independent training-free signals (low-confidence OR repetition
    OR KL-drift); a position any of them flags should be revisable. When two
    detectors flag the same position, the higher severity is kept (so the
    RemaskPolicy ranks it by its strongest signal).
    """

    detectors: list
    name: str = field(default="composite", init=False)

    def flag(self, *, state: "MaskState", commit_history: Any, **kwargs) -> list:
        best: dict[int, FlaggedPosition] = {}
        for detector in self.detectors:
            for flag in detector.flag(
                state=state, commit_history=commit_history, **kwargs
            ):
                current = best.get(flag.position)
                if current is None or flag.severity > current.severity:
                    best[flag.position] = flag
        return [best[p] for p in sorted(best)]


# ---------------------------------------------------------------------------
# RemaskPolicy (Phase F3) -- ReMDM-conf + the RemeDi monotone-progress invariant
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ConfWeightedRemask:
    """Re-mask flagged positions preferentially by severity (ReMDM-conf).

    ReMDM-conf re-masks low-confidence tokens preferentially, ``sigma ~
    softmax(-psi)``: higher severity (lower confidence / stronger drift / a
    repeat) -> re-masked first. Routing corrections as token->mask->re-decode is
    more stable than in-place overwrite.

    **The RemeDi monotone-progress invariant (the termination guarantee):** the
    number re-masked is STRICTLY LESS than ``newly_unmasked_count`` this round. So
    every round nets at least one more committed-and-kept position than it
    re-masks; the masked frontier shrinks monotonically and the loop cannot run
    forever. With ``newly_unmasked_count <= 1`` the cap is 0 -> no re-mask (a
    single freshly-unmasked token cannot be revised this round), which is what
    guarantees liveness even under an adversarial flag-everything detector.
    """

    name: str = field(default="conf_weighted_remask", init=False)

    def select_remask(
        self, *, flagged: list, state: "MaskState", newly_unmasked_count: int
    ) -> list:
        if not flagged:
            return []
        # Monotone-progress cap: strictly fewer than were newly unmasked.
        cap = max(0, int(newly_unmasked_count) - 1)
        if cap == 0:
            return []
        # ReMDM-conf preference: highest severity (== softmax(-psi) mass) first.
        ranked = sorted(flagged, key=lambda f: float(f.severity), reverse=True)
        return [int(f.position) for f in ranked[:cap]]


@dataclass(slots=True)
class QualityRecoveryResult:
    """What a recovery loop did on top of the base decode pass.

    The accounting the error model surfaces (DMax: the correction process must be
    explicitly accounted): ``wrong_commit_count`` (positions a detector flagged),
    ``remask_count`` (positions re-masked), ``corrected_positions`` (recommits
    that CHANGED the token -- the payoff), ``churned_positions`` (re-masked then
    re-decoded to the SAME token -- wasted work). ``quality_delta`` is an optional
    objective before/after signal (``None`` when not measured in-loop).
    """

    recovery_rounds: int
    remask_count: int
    corrected_positions: int
    churned_positions: int = 0
    wrong_commit_count: int = 0
    quality_delta: float | None = None


@dataclass(slots=True)
class QualityRecoveryLoop:
    """Bounded detect -> remask -> re-decode orchestration (Phase F3).

    Wraps the generic decode loop. ``enabled=False`` (the default) makes
    :meth:`run` a no-op beyond the single base decode pass the manager already
    ran: ``recovery_rounds=1``, ``remask_count=0``, ``corrected_positions=0`` --
    byte-identical to today.

    With recovery enabled it runs the error-model cycle on top of the base pass::

        repeat:
          flagged   = detector.flag(state, commit_history, dist_history)
          to_remask = remask_policy.select_remask(flagged, state, newly_unmasked)
          for p in to_remask: state.remask(p)            # token -> mask
          decode_pass()                                  # re-decode the frontier
          recommit each remasked p (correction iff token changed)
        until a stop condition.

    **Termination (the RemeDi monotone-progress invariant):** the RemaskPolicy
    re-masks STRICTLY FEWER positions than were newly unmasked the previous round,
    so the masked frontier shrinks each round and the loop cannot run forever --
    even under an adversarial flag-everything detector. The round budget
    (``max_recovery_rounds``) is a second backstop.

    **Stop conditions (any):** no flags, round budget hit, no improvement (a round
    that corrected nothing), or no masks remain. Each terminal reason is recorded
    on the round's :class:`RecoveryRound` event.
    """

    detector: Any = field(default_factory=NoOpErrorMaskDetector)
    remask_policy: Any = field(default_factory=NoOpRemaskPolicy)
    max_recovery_rounds: int = 0
    enabled: bool = False
    # D2F frozen-block safety (C2b-v1 Task 2): when non-None, this callable returns
    # the current set of frozen-block positions (absolute).  The remask step SKIPS
    # any position in that set — frozen blocks are IMMUTABLE.  A flag on a frozen
    # position is still recorded for provenance but NOT acted on.  ``None`` (the
    # default) means "no frozen positions" and leaves every other code path
    # byte-identical to before.
    frozen_positions_fn: Any = field(default=None)  # Callable[[], frozenset] | None
    # C2b-v1 §3/§3a (error-fixing → upper-band invalidation): when non-None, this
    # callable is invoked with the list of remasked active-window positions AFTER
    # the remask and BEFORE the re-decode.  A delayed-KV ``upper_band`` strategy
    # uses it to drop the active-window band cache so the re-decode recomputes
    # FRESH K/V (no stale REUSE_APPROX may resume until recomputed).  ``None`` (the
    # default) leaves every path byte-identical to before.
    on_error_fix_remask: Any = field(default=None)  # Callable[[list[int]], None] | None

    def run(self, ctx: _LoopContext, *, decode_pass: Callable[[], None]) -> QualityRecoveryResult:
        """Run recovery on top of an already-executed base decode pass.

        ``decode_pass`` re-runs the generic decode loop over the (now re-masked)
        frontier. Disabled -> never invoked: the base pass IS the whole decode, so
        this reports the neutral result (byte-identical to today).
        """
        if not self.enabled:
            return QualityRecoveryResult(
                recovery_rounds=1, remask_count=0, corrected_positions=0
            )

        event_log = ctx.event_log
        state = ctx.state
        total_flagged = 0
        total_remasked = 0
        total_corrected = 0
        total_churned = 0
        rounds_run = 0

        # ``newly_unmasked_count`` for round 1 = the positions committed in the
        # base pass (the whole generated region, modulo the step cap). The
        # invariant caps each round's re-masks below this.
        newly_unmasked = sum(
            1
            for position in state.all_fillable_positions()
            if state.x[position] != state.mask_token_id
        )

        for round_index in range(1, int(self.max_recovery_rounds) + 1):
            commit_history = event_log.commit_history() if event_log is not None else {}
            dist_history = event_log.dist_history if event_log is not None else {}
            flagged = self.detector.flag(
                state=state, commit_history=commit_history, dist_history=dist_history
            )
            if not flagged:
                if event_log is not None:
                    event_log.record_recovery_round(
                        RecoveryRound(
                            round=round_index, flagged=0, remasked=0,
                            corrected=0, churned=0, stop_reason="no_flags",
                        )
                    )
                break

            to_remask = self.remask_policy.select_remask(
                flagged=flagged, state=state, newly_unmasked_count=newly_unmasked
            )
            # D2F frozen-block safety: if a frozen-positions oracle is set, strip
            # any frozen-block positions from the remask list.  We STILL record the
            # flag event for provenance (so the caller can audit what was skipped),
            # but the position is NOT remasked and NOT acted on.
            if self.frozen_positions_fn is not None:
                frozen = self.frozen_positions_fn()
                to_remask = [p for p in to_remask if p not in frozen]
            total_flagged += len(flagged)
            if event_log is not None:
                for flag in flagged:
                    event_log.record_flag(
                        position=flag.position, reason=flag.reason,
                        severity=flag.severity, round=round_index,
                    )
            if not to_remask:
                # Flags exist but the monotone cap forbids acting (e.g. only one
                # newly-unmasked position). No progress is possible this round.
                if event_log is not None:
                    event_log.record_recovery_round(
                        RecoveryRound(
                            round=round_index, flagged=len(flagged), remasked=0,
                            corrected=0, churned=0, stop_reason="no_improvement",
                        )
                    )
                break

            # token -> mask for each chosen position (record prev for provenance).
            prev_tokens = {position: state.x[position] for position in to_remask}
            for position in to_remask:
                state.remask(position=position)
                if event_log is not None:
                    event_log.record_remask(
                        position=position, prev_token=prev_tokens[position],
                        round=round_index,
                    )
            total_remasked += len(to_remask)

            # C2b-v1 §3/§3a: an active-window remask perturbs the layer-(l-1)
            # context for the WHOLE active window, so any upper-band stale K/V is
            # now wrong. Drop the active-window band cache BEFORE re-decoding so the
            # re-decode recomputes fresh K/V (no REUSE_APPROX resumes until then).
            if self.on_error_fix_remask is not None:
                self.on_error_fix_remask(to_remask)

            # Re-decode the now-masked frontier (the same strategy, fresh policies).
            decode_pass()

            # Recommit accounting: a correction iff the re-decode produced a
            # DIFFERENT token; otherwise churn (wasted re-mask).
            round_corrected = 0
            round_churned = 0
            for position in to_remask:
                new_token = state.x[position]
                if new_token == state.mask_token_id:
                    # The re-decode left it masked (step cap / no forward): churn,
                    # and there is NO recommit to record — booking one (with the
                    # OLD token) would mark the position committed in
                    # commit_history while the canvas still has it masked.
                    round_churned += 1
                    continue
                if event_log is not None:
                    event_log.record_recommit(
                        position=position, new_token=int(new_token),
                        prev_token=int(prev_tokens[position]), round=round_index,
                    )
                if new_token != prev_tokens[position]:
                    round_corrected += 1
                else:
                    round_churned += 1
            total_corrected += round_corrected
            total_churned += round_churned
            rounds_run += 1

            # The next round's invariant budget = what this re-decode just unmasked.
            newly_unmasked = len(to_remask)

            no_masks = state.masked_positions() == []
            no_improvement = round_corrected == 0
            budget_hit = round_index >= int(self.max_recovery_rounds)
            stop_reason = (
                "no_masks" if no_masks
                else "no_improvement" if no_improvement
                else "round_budget" if budget_hit
                else ""
            )
            if event_log is not None:
                event_log.record_recovery_round(
                    RecoveryRound(
                        round=round_index, flagged=len(flagged),
                        remasked=len(to_remask), corrected=round_corrected,
                        churned=round_churned, stop_reason=stop_reason,
                    )
                )
            if stop_reason:
                break

        return QualityRecoveryResult(
            recovery_rounds=1 + rounds_run,  # the base pass + the recovery rounds
            remask_count=total_remasked,
            corrected_positions=total_corrected,
            churned_positions=total_churned,
            wrong_commit_count=total_flagged,
        )


# ---------------------------------------------------------------------------
# DiffusionIterationManager -- runs the loop
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DiffusionIterationManager:
    """Runs the masked-diffusion decode loop.

    Owns the :class:`MaskState`, calls the backend forward callable, applies the
    :class:`DecodingStrategy` each step, increments ``forward_count`` only on
    real forwards, enforces the ``steps`` cap, and produces the stop reason via
    :meth:`MaskState.finalize`.

    The optional :class:`QualityRecoveryLoop` (``recovery_loop``) wraps the base
    decode pass with the detect -> remask -> re-decode cycle. It defaults to the
    OFF loop, so in F1 the manager runs exactly one decode pass and the Phase F
    metrics (``remask_count`` / ``corrected_positions`` / ``recovery_rounds``)
    are neutral -- byte-identical to today.
    """

    state: MaskState
    strategy: Any
    forward_fn: ForwardFn
    steps: int
    select_transform: SelectTransform = _identity_select
    commit_observer: CommitObserver | None = None
    scheduler_phase: PhaseFactory = _null_phase
    select_phase: PhaseFactory = _null_phase
    commit_phase: PhaseFactory = _null_phase
    recovery_loop: Any = None
    # Phase F3 error model. When attached (or auto-created because the recovery
    # loop is enabled), the loop records the commit->flag->remask->recommit
    # provenance here. ``None`` with recovery OFF records nothing -> byte-identical.
    event_log: "DecodeEventLog | None" = None
    # Record per-step frontier distributions (the KL-drift signal). Gated, so OFF
    # / non-KL recovery pay no extra softmax and stay byte-identical.
    record_distributions: bool = False
    # MaskCanvas v0 control-plane observer, threaded into the ctx below. Duck-typed
    # (no ``mask_canvas`` import here); ``None`` (default) -> byte-identical.
    trace_observer: Any = None
    # Active-Window Runtime (Task A3, WS1). Callable taking one WindowExecutionPlan
    # per step; threaded into the ctx. ``None`` (default) -> byte-identical.
    window_plan_sink: Any = None
    # Active-Window Runtime (Task B2, WS2). An injected callable
    # ``(window_plan, token_ids) -> WindowExecutionResult`` that replaces the
    # raw forward_fn -> select_logits path with execute_window. ``None``
    # (the default) -> byte-identical: the existing forward path is unchanged.
    execute_window_fn: Any = None
    # Active-Window Runtime (C2a, WS3). A duck-typed ``CorrectionPolicy`` that
    # wraps the same detector + remask_policy as the recovery loop. When set,
    # ``run_decode_loop`` calls ``corrections_for(...)`` per step so emitted
    # ``WindowExecutionPlan``s carry dirty/remask (the bridge is live). ``None``
    # (the default) → classify_window receives corrections=None → empty dirty/remask
    # → byte-identical. Only meaningful when ``window_plan_sink`` or
    # ``execute_window_fn`` is set (otherwise no plan is emitted and the call is
    # skipped by the same guard).
    correction_policy: Any = None
    propose_fn: Any = None
    commit_device_fn: Any = None

    def run(self) -> DiffusionIterationResult:
        # The error model needs the commit provenance whenever recovery is enabled
        # (the detector reads it); otherwise honour the caller's explicit log (a
        # caller may want the accounting without recovery) or record nothing.
        # C2a: the correction_policy also reads commit_history, so ensure an event_log
        # exists when it is set (even without the recovery loop enabled). This is a
        # passive log — attaching it does not change the committed stream.
        recovery_loop = self.recovery_loop or QualityRecoveryLoop()
        if self.propose_fn is not None and (
            self.commit_device_fn is None or self.execute_window_fn is not None or
            self.record_distributions or self.correction_policy is not None or
            getattr(recovery_loop, "enabled", False)
        ):
            raise ValueError("device proposals require checked commit and do not support remasking or logits tracing")
        event_log = self.event_log
        if event_log is None and (
            getattr(recovery_loop, "enabled", False) or self.correction_policy is not None
        ):
            event_log = DecodeEventLog()
        ctx = _LoopContext(
            state=self.state,
            forward_fn=self.forward_fn,
            steps=int(self.steps),
            select_transform=self.select_transform,
            commit_observer=self.commit_observer,
            scheduler_phase=self.scheduler_phase,
            select_phase=self.select_phase,
            commit_phase=self.commit_phase,
            event_log=event_log,
            record_distributions=bool(self.record_distributions),
            trace_observer=self.trace_observer,
            window_plan_sink=self.window_plan_sink,
            execute_window_fn=self.execute_window_fn,
            correction_policy=self.correction_policy,
            propose_fn=self.propose_fn,
            commit_device_fn=self.commit_device_fn,
        )
        # Base decode pass (today's whole decode).
        self.strategy.run(ctx)
        # Bounded self-correction (Phase F3). Defaults to the OFF loop: a single
        # decode pass, no re-masking -> neutral metrics, behaviour unchanged. When
        # enabled, ``decode_pass`` re-runs the decode loop over the remasked
        # frontier (the recovery loop owns the unmask/commit policies it needs).
        recovery = recovery_loop.run(ctx, decode_pass=self._make_decode_pass(ctx))
        self.state.committed_token_count = ctx.committed_token_count
        final = self.state.finalize()
        committed_per_forward = (
            ctx.committed_token_count / ctx.forward_count if ctx.forward_count else 0.0
        )
        return DiffusionIterationResult(
            visible_generated_token_ids=final["visible_token_ids"],
            iteration_count=ctx.iteration_count,
            forward_count=ctx.forward_count,
            committed_token_count=ctx.committed_token_count,
            hit_eos=final["hit_eos"],
            remaining_mask_count=final["remaining_mask_count"],
            finish_reason=final["finish_reason"],
            decode_loop_kind=self.strategy.decode_loop_kind,
            entropy_used=self.strategy.entropy_used,
            committed_per_forward=committed_per_forward,
            remask_count=recovery.remask_count,
            corrected_positions=recovery.corrected_positions,
            recovery_rounds=recovery.recovery_rounds,
            wrong_commit_count=recovery.wrong_commit_count,
            churned_positions=recovery.churned_positions,
            quality_delta=recovery.quality_delta,
        )

    def _make_decode_pass(self, ctx: "_LoopContext") -> Callable[[], None]:
        """A re-decode callable the recovery loop runs over the remasked frontier.

        Re-invokes the SAME :class:`DecodingStrategy` over ``ctx`` (whose state now
        has the flagged positions re-masked). Each strategy builds fresh policies
        inside ``run`` (e.g. ``BlockThresholdCommit`` with a reset block pointer),
        so the re-decode walks the still-masked positions -- exactly the re-masked
        frontier, since every other position is committed and skipped. The fresh
        :class:`Commit` events it records refresh the ``commit_history`` the next
        round reads. Recovery OFF never calls this (the base pass is the whole
        decode), so it is inert in that path.
        """
        return lambda: self.strategy.run(ctx)
