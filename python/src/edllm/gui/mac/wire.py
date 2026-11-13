"""Controller events as JSON for a Mac window. author:Opus"""
from __future__ import annotations

from dataclasses import fields
from typing import Any

from . import events as ev


def event_to_json(event: object) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": type(event).__name__}
    for field in fields(event):
        payload[field.name] = getattr(event, field.name)
    if isinstance(event, ev.Snapshot):
        payload["events"] = [event_to_json(item) for item in event.events]
    elif isinstance(event, ev.ModelReady):
        payload["precisions"] = list(event.precisions)
    elif isinstance(event, ev.ChatShown):
        payload["messages"] = [
            {"role": message.role, "content": message.content} for message in event.messages
        ]
    elif isinstance(event, ev.RecentChanged):
        payload["conversations"] = [
            {"session_id": summary.session_id, "title": summary.title,
             "updated_at_ns": summary.updated_at_ns}
            for summary in event.conversations
        ]
    elif isinstance(event, ev.PickerRequested):
        payload["candidates"] = [{"model_id": str(model.model_id)} for model in event.candidates]
    elif isinstance(event, ev.ReplyShown):
        payload["visible_tokens_per_second"] = _per_second(
            event.visible_output_tokens, event.elapsed_seconds
        )
    return payload


def _per_second(tokens: int | None, elapsed: float | None) -> float | None:
    """Visible tokens over the reply's whole elapsed time; None when either is unknown."""
    if tokens is None or not elapsed or elapsed <= 0:
        return None
    return round(tokens / elapsed, 2)
