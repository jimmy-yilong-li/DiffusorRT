from __future__ import annotations

import math
from dataclasses import dataclass

from .executor import BlockLogits


@dataclass(slots=True)
class PolicyDecision:
    kind: str
    entropy: float
    confidence: float


def token_commit_decision(
    token_values: list[float],
    *,
    entropy_threshold: float = 0.9,
    confidence_threshold: float = 0.65,
) -> PolicyDecision:
    probabilities = _softmax(token_values)
    entropy = -sum(prob * math.log(max(prob, 1e-12)) for prob in probabilities)
    confidence = max(probabilities) if probabilities else 1.0
    kind = "commit" if confidence >= confidence_threshold or entropy <= entropy_threshold else "hold"
    return PolicyDecision(kind=kind, entropy=entropy, confidence=confidence)


def baseline_commit_policy(
    block_logits: BlockLogits,
    *,
    entropy_threshold: float = 0.9,
    confidence_threshold: float = 0.65,
) -> PolicyDecision:
    entropies: list[float] = []
    confidences: list[float] = []
    for row in block_logits.values:
        decision = token_commit_decision(
            row,
            entropy_threshold=entropy_threshold,
            confidence_threshold=confidence_threshold,
        )
        entropies.append(decision.entropy)
        confidences.append(decision.confidence)

    mean_entropy = sum(entropies) / len(entropies) if entropies else 0.0
    mean_confidence = sum(confidences) / len(confidences) if confidences else 1.0
    kind = "commit" if mean_confidence >= confidence_threshold or mean_entropy <= entropy_threshold else "hold"
    return PolicyDecision(kind=kind, entropy=mean_entropy, confidence=mean_confidence)


def _softmax(values: list[float]) -> list[float]:
    return softmax_probabilities(values)


def softmax_probabilities(values: list[float]) -> list[float]:
    if not values:
        return []
    if any(math.isnan(value) for value in values):
        raise ValueError("logits must not contain NaN")
    positive_infinity = [value == math.inf for value in values]
    positive_infinity_count = sum(positive_infinity)
    if positive_infinity_count:
        probability = 1.0 / positive_infinity_count
        return [
            probability if is_infinite else 0.0
            for is_infinite in positive_infinity
        ]
    max_value = max(values)
    if max_value == -math.inf:
        return [1.0 / len(values)] * len(values)
    exp_values = [
        math.exp(value - max_value) if value != -math.inf else 0.0
        for value in values
    ]
    total = sum(exp_values)
    return [value / total for value in exp_values]
