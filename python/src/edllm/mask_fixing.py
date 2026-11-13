"""Conservative output-quality guard for parallel-decoded generations.

Parallel decoding occasionally yields obviously broken output: empty text,
numeric loops, or token repetition collapse. :func:`evaluate_generation` is a
pure, deterministic policy that flags such cases so a caller can trigger ONE
conservative retry. It deliberately stays dependency-free (no mlx/model imports)
and never touches the validated engine decode path.

The rules are intentionally narrow to keep false positives low: a normal
sentence must always be accepted.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

#: Minimum number of alphabetic characters a generation must contain.
_MIN_ALPHA_CHARS = 3
#: Token-count threshold above which the repetition-collapse rule applies.
_REPETITION_MIN_TOKENS = 6
#: Fraction of tokens the most-common token may occupy before it is repetition.
_REPETITION_RATIO = 0.40


@dataclass(frozen=True, slots=True)
class MaskFixDecision:
    """Outcome of :func:`evaluate_generation`.

    ``kind`` is ``"accept"`` or ``"retry"``; ``reason`` is a short stable tag
    identifying which rule fired (``"ok"`` when accepted).
    """

    kind: str
    reason: str


def evaluate_generation(text: str) -> MaskFixDecision:
    """Classify a generation as acceptable or worth one conservative retry.

    Rules are evaluated in order and the FIRST match is returned:

    1. empty / whitespace-only -> ``retry("empty")``
    2. fewer than 3 alphabetic characters -> ``retry("low_alpha")``
    3. digit characters outnumber alphabetic characters -> ``retry("numeric_dominant")``
    4. repetition collapse: with >= 6 whitespace tokens, the most-common token
       is >= 40% of them -> ``retry("repetition")``
    5. otherwise -> ``accept("ok")``
    """
    if not text or not text.strip():
        return MaskFixDecision(kind="retry", reason="empty")

    alpha_count = sum(1 for char in text if char.isalpha())
    if alpha_count < _MIN_ALPHA_CHARS:
        return MaskFixDecision(kind="retry", reason="low_alpha")

    digit_count = sum(1 for char in text if char.isdigit())
    if digit_count > alpha_count:
        return MaskFixDecision(kind="retry", reason="numeric_dominant")

    tokens = text.split()
    if len(tokens) >= _REPETITION_MIN_TOKENS:
        _, most_common_count = Counter(tokens).most_common(1)[0]
        if most_common_count >= _REPETITION_RATIO * len(tokens):
            return MaskFixDecision(kind="retry", reason="repetition")

    return MaskFixDecision(kind="accept", reason="ok")
