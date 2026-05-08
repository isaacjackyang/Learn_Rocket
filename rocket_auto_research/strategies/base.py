from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from rocket_auto_research.gnc.state import ControlAction, WorldState


@dataclass(slots=True)
class StrategyContext:
    current_target_id: str | None = None
    step_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    def __init__(self, name: str, params: dict[str, Any] | None = None) -> None:
        self.name = name
        self.params = params or {}
        self.context = StrategyContext()

    @abstractmethod
    def act(self, world_state: WorldState) -> ControlAction:
        raise NotImplementedError

