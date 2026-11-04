"""A tiny synchronous event bus with an ordered history."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Event:
    seq: int
    kind: str
    data: dict = field(default_factory=dict)


class EventBus:
    def __init__(self):
        self._handlers = defaultdict(list)
        self._history: list[Event] = []

    def subscribe(self, kind: str, handler) -> None:
        """Call handler(event) for every later event of this kind ('*' for all kinds)."""
        self._handlers[kind].append(handler)

    def publish(self, kind: str, **data) -> Event:
        event = Event(len(self._history) + 1, kind, dict(data))
        self._history.append(event)
        for handler in self._handlers[kind] + self._handlers["*"]:
            handler(event)
        return event

    def history(self, kind: str | None = None) -> list[Event]:
        return [e for e in self._history if kind is None or e.kind == kind]

    def last(self, kind: str) -> Event | None:
        matches = self.history(kind)
        return matches[-1] if matches else None
