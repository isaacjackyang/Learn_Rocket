from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rocket_auto_research.auto_research.competition_contract import (
    validate_action_payload,
    validate_observation_payload,
)
from rocket_auto_research.gnc.action_formatter import format_action
from rocket_auto_research.gnc.observation_parser import parse_observation
from rocket_auto_research.strategies.registry import build_strategy


class CompetitionAgent:
    def __init__(self, config_path: str | Path) -> None:
        payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
        self.strategy_name = payload["strategy_name"]
        self.observation_schema = str(payload.get("observation_schema", "flat_competition"))
        self.action_schema = str(payload.get("action_schema", "flat_competition"))
        self.strategy = build_strategy(self.strategy_name, payload.get("params", {}))

    def act(self, observation: dict[str, Any]) -> dict[str, float | bool]:
        validate_observation_payload(observation, schema=self.observation_schema)
        world_state = parse_observation(observation, schema=self.observation_schema)
        action = self.strategy.act(world_state)
        payload = format_action(action, schema=self.action_schema)
        validate_action_payload(payload, schema=self.action_schema)
        return payload
