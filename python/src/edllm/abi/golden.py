"""Golden vectors — the cross-implementation oracle, plus a harness that
actually drives a candidate.

A vector records what the REFERENCE RUNTIME produced: the control-plane result
(bit-exact), the logits digest and shape (bit-exact within one backend), and the
committed tokens (the cross-backend promise). It identifies its model by package
name + weights digest, never by a process-local handle.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass

from .parity import ParityTier, compare_control_plane, compare_tokens, digest


def _jsonable(value):
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _tupleize(value):
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, dict):
        return {
            int(k) if str(k).lstrip("-").isdigit() else k: _tupleize(v)
            for k, v in value.items()
        }
    return value


@dataclass(frozen=True, slots=True)
class GoldenVector:
    label: str
    source: str                      # "numpy_reference" | "mlx_local"
    model_identity: str              # package name + weights digest
    request_dict: dict
    control_plane: dict
    logits_digest: str
    logits_shape: tuple[int, int]
    committed_tokens: tuple[int, ...]

    def to_dict(self) -> dict:
        return {
            "label": self.label, "source": self.source,
            "model_identity": self.model_identity,
            "request": self.request_dict,
            "control_plane": {k: _jsonable(v) for k, v in self.control_plane.items()},
            "logits_digest": self.logits_digest,
            "logits_shape": list(self.logits_shape),
            "committed_tokens": list(self.committed_tokens),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GoldenVector":
        return cls(
            label=data["label"], source=data["source"],
            model_identity=data["model_identity"],
            request_dict=data["request"],
            control_plane={k: _tupleize(v) for k, v in data["control_plane"].items()},
            logits_digest=data["logits_digest"],
            logits_shape=tuple(int(d) for d in data["logits_shape"]),
            committed_tokens=tuple(int(t) for t in data["committed_tokens"]),
        )


@dataclass(frozen=True, slots=True)
class ReplayOutcome:
    label: str
    passed: bool
    detail: str


def record_vector(*, label: str, source: str, model_identity: str,
                  request_dict: dict, control_plane: dict, logits,
                  committed_tokens) -> GoldenVector:
    shape = tuple(int(d) for d in logits.shape)
    return GoldenVector(
        label=label, source=source, model_identity=model_identity,
        request_dict=request_dict, control_plane=control_plane,
        logits_digest=digest(logits),
        logits_shape=(shape[0], shape[1] if len(shape) > 1 else 0),
        committed_tokens=tuple(int(t) for t in committed_tokens),
    )


def write_corpus(vectors, directory) -> None:
    path = pathlib.Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    for vector in vectors:
        name = vector.label.replace("/", "__") + ".json"
        (path / name).write_text(
            json.dumps(vector.to_dict(), indent=2, sort_keys=True) + "\n")


def load_corpus(directory) -> list[GoldenVector]:
    path = pathlib.Path(directory)
    if not path.is_dir():
        return []
    return [
        GoldenVector.from_dict(json.loads(file.read_text()))
        for file in sorted(path.glob("*.json"))
    ]


def replay(vectors, candidate, *, tier: ParityTier) -> list[ReplayOutcome]:
    """Drive ``candidate`` on every vector and compare at ``tier``.

    ``candidate(vector)`` returns ``(control_plane, logits_or_None, tokens)``.
    Control-plane parity is checked at EVERY tier — it is integer logic, so a
    difference there is a bug regardless of backend.
    """
    outcomes: list[ReplayOutcome] = []
    for vector in vectors:
        control, logits, tokens = candidate(vector)
        result = compare_control_plane(vector.control_plane, control)
        if not result.passed:
            outcomes.append(ReplayOutcome(vector.label, False, result.detail))
            continue
        if tier >= ParityTier.SAME_BACKEND_TENSOR and logits is not None:
            if digest(logits) != vector.logits_digest:
                outcomes.append(ReplayOutcome(
                    vector.label, False,
                    f"logits digest differs (tier {ParityTier(tier).name})"))
                continue
        token_result = compare_tokens(list(vector.committed_tokens), list(tokens))
        outcomes.append(ReplayOutcome(
            vector.label, token_result.passed,
            "ok" if token_result.passed else token_result.detail))
    return outcomes
