"""What the controller tells the window. Plain data; the window draws it."""
from __future__ import annotations

from dataclasses import dataclass

from ..conversation import ChatMessage, ConversationSummary
from .preferences import WINDOW_TEMPERATURE

GREETING = "How can I help you?"
"""Shown as the first AI message of every new chat. Never stored, never sent to the model."""


@dataclass(frozen=True)
class PreferencesShown:
    resources_expanded: bool
    temperature: float = WINDOW_TEMPERATURE
    """What the next reply is sampled at, when the model samples."""


@dataclass(frozen=True)
class ModelLoading:
    spec: str


@dataclass(frozen=True)
class ModelReady:
    spec: str
    model_name: str
    backend: str
    precision: str | None
    gpu_name: str | None
    precisions: tuple[str, ...] = ()
    """The precisions the open package is admitted for on this backend; () when not cuda_native."""
    can_stop: bool = False
    """Whether a running reply can be stopped at a decode step (the plan admits cancellation)."""
    can_sample: bool = False
    """Whether replies are sampled at the window's temperature: cuda_native's MDLM and BD3
    plans. Other models decode greedily, and the control says so."""
    chosen_backend: str | None = None
    """The backend to open with next time: asked for, or decided as cuda_native.

    None when "auto" chose, so the next launch decides again; `backend` still names
    what runs now.
    """


@dataclass(frozen=True)
class ModelFailed:
    reason: str
    next_action: str | None
    stage: str = "open"
    """"open": this model could not be opened. "start": it opened but its runtime
    could not start (its first generation failed), and it was released. "release":
    a model could not be released, so it is still held and no other can open until
    it is."""


@dataclass(frozen=True)
class TrustRequested:
    spec: str
    module: str


@dataclass(frozen=True)
class TrustAnswered:
    """The trust question was answered (or dropped by a close); a reloaded page must not ask again."""

    granted: bool


@dataclass(frozen=True)
class ChatShown:
    session_id: str | None
    """None for a new chat, which is stored only once the user sends a message."""
    messages: tuple[ChatMessage, ...]


@dataclass(frozen=True)
class ChatDeleted:
    """A conversation and its messages were removed from the store."""

    session_id: str


@dataclass(frozen=True)
class RecentChanged:
    conversations: tuple[ConversationSummary, ...]


@dataclass(frozen=True)
class UserMessageShown:
    text: str


@dataclass(frozen=True)
class Generating:
    started: float | None = None
    """When the reply started, as `time.time()`: a reloaded page counts from it."""


@dataclass(frozen=True)
class ReplyForming:
    """The reply as decided so far: text runs and counts of undecided positions, in order."""

    step: int
    steps: int
    segments: tuple
    decided: int
    positions: int
    canvas_decided: int = 0
    canvas_positions: int = 0
    """How much of the whole canvas is decided; only grows, unlike the reply cut at its end."""


@dataclass(frozen=True)
class ReplyStopped:
    """The user stopped the reply. Nothing was stored; ``text`` goes back into the box."""

    text: str


@dataclass(frozen=True)
class ReplyShown:
    text: str
    elapsed_seconds: float | None
    visible_output_tokens: int | None
    session_id: str | None = None
    """The conversation the turn was stored in; a new chat gets its id here."""


@dataclass(frozen=True)
class ReplyFailed:
    reason: str
    details: str | None = None
    """The runtime's own words, shown folded under the reason."""
    text: str | None = None
    """A message that was not stored, to go back into the box (when the box is empty)."""


@dataclass(frozen=True)
class ChatLocked:
    """The open chat cannot take another message: it changed elsewhere and could
    not be read again. Opening a chat or starting a new one ends this."""

    reason: str


@dataclass(frozen=True)
class PickerRequested:
    candidates: tuple
    """Prepared models, each with a model_id."""
    reason: str
    """"no_models", "several", "remembered_missing" or "change"."""


@dataclass(frozen=True)
class Closing:
    pass


@dataclass(frozen=True)
class Closed:
    pass
