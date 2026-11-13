"""What the Mac controller tells a Mac window. Plain data; the window draws it.

author:Opus -- taken over from the window code these windows began as; it is the
Mac windows' own now and nothing outside this package uses it.
"""
from __future__ import annotations

from dataclasses import dataclass

from ...conversation import ChatMessage, ConversationSummary

GREETING = "How can I help you?"
"""Shown as the first AI message of every new chat. Never stored, never sent to the model."""


@dataclass(frozen=True)
class PreferencesShown:
    resources_expanded: bool


@dataclass(frozen=True)
class SettingsShown:
    max_tokens: int | None
    system: str | None
    context_policy: str


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
    """Kept for the wire's shape. A Mac window offers no precision choice; this is ()."""
    cancellable: bool = False
    """Whether a running reply can be stopped at a decode boundary."""
    package_dir: str | None = None


@dataclass(frozen=True)
class ModelFailed:
    reason: str
    next_action: str | None


@dataclass(frozen=True)
class TrustRequested:
    spec: str
    module: str


@dataclass(frozen=True)
class ChatShown:
    session_id: str | None
    """None for a new chat, which is stored only once the user sends a message."""
    messages: tuple[ChatMessage, ...]


@dataclass(frozen=True)
class RecentChanged:
    conversations: tuple[ConversationSummary, ...]


@dataclass(frozen=True)
class UserMessageShown:
    text: str


@dataclass(frozen=True)
class Generating:
    pass


@dataclass(frozen=True)
class ReplyShown:
    text: str
    elapsed_seconds: float | None
    visible_output_tokens: int | None
    context: dict | None = None
    session_id: str | None = None
    """The stored conversation this reply was committed to."""
    finish_reason: str | None = None
    """The engine's word on how the reply ended: "stop" when the model ended it, "length" when
    the output limit or the decode budget ran out first. None when the engine did not say."""


@dataclass(frozen=True)
class ReplyFailed:
    reason: str
    reload_required: bool = False


@dataclass(frozen=True)
class ReplyCancelled:
    """The user stopped the reply. Nothing was committed to the conversation."""


@dataclass(frozen=True)
class Busy:
    """The one worker became busy or free. A window enables its controls from this."""

    busy: bool


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
class CloseFailed:
    reason: str


@dataclass(frozen=True)
class Snapshot:
    events: tuple[object, ...]
    """Current view rebuilt from the controller and its Conversation owner."""


@dataclass(frozen=True)
class Closed:
    pass
