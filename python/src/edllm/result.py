from dataclasses import asdict, dataclass, field
from typing import Any


class EmptyGenerationError(RuntimeError):
    """A user-facing reply produced no visible text; no retry is implied."""

    def __init__(self, message: str, *, result: "GenerateResult | None" = None):
        super().__init__(message)
        self.result = result


def require_nonempty_generation(
    text: str, *, result: "GenerateResult | None" = None
) -> None:
    """Keep empty model output out of successful chat turns and responses."""
    if not text.strip():
        raise EmptyGenerationError("model produced no visible answer", result=result)


@dataclass(slots=True)
class DebugBundle:
    # metrics holds JSON-native values: scalars plus nested dict/list/bool (e.g. the
    # canvas_trace summary). The contract is "JSON-serializable", not scalar-only.
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerateResult:
    text: str
    finish_reason: str = "stop"
    debug: DebugBundle | None = None
    elapsed_seconds: float | None = None
    # Tokens up to the end of the reply, from the generated token IDs. Not the
    # filled canvas, which equals max_tokens. None when a backend does not count.
    visible_output_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-native result data, including diagnostics when requested."""
        return asdict(self)


@dataclass(slots=True)
class InfillResult:
    """Result of a dLLM-native infill / span-fill.

    ``filled_text`` is the decoded text at the masked holes between the fixed
    prefix and suffix; ``full_text`` is ``prefix + filled + suffix``. The
    suffix-aware fill is the dLLM-native point: the bidirectional forward filled
    the holes respecting BOTH sides.
    """

    filled_text: str
    full_text: str
    finish_reason: str = "stop"
    debug: DebugBundle | None = None
