"""Pre-1.0 chat compatibility over the durable conversation transaction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .conversation import ChatMessage, Conversation, ConversationReply

if TYPE_CHECKING:
    from .engine import Engine


class ChatSession:
    """Compatibility adapter returned by :meth:`Engine.chat_session`.

    It owns no renderer or independent history. All state and send semantics
    belong to the wrapped :class:`Conversation`.
    """

    __slots__ = ("_conversation",)

    def __init__(self, *, _conversation: Conversation) -> None:
        if not isinstance(_conversation, Conversation):
            raise TypeError("ChatSession must be created by Engine.chat_session()")
        self._conversation = _conversation

    @property
    def system_prompt(self) -> str | None:
        return self._conversation.system_prompt

    @property
    def turn_count(self) -> int:
        return len(self._conversation.messages())

    def messages(self) -> list[ChatMessage]:
        messages: list[ChatMessage] = []
        if self.system_prompt is not None:
            messages.append(ChatMessage("system", self.system_prompt))
        messages.extend(self._conversation.messages())
        return messages

    def as_messages(self) -> list[dict[str, str]]:
        return [message.as_dict() for message in self.messages()]

    def close(self) -> None:
        self._conversation.close()

    def _send(
        self,
        engine: "Engine",
        message: str,
        *,
        max_tokens: int | None,
        **sampling: Any,
    ) -> ConversationReply:
        if self._conversation._engine is not engine:
            raise ValueError("ChatSession belongs to a different Engine")
        return self._conversation.send(message, max_tokens=max_tokens, **sampling)


__all__ = ["ChatMessage", "ChatSession"]
