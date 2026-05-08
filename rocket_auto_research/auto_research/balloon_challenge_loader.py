from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_REPO = Path(".external/BalloonPoppingChallenge")


@dataclass(slots=True)
class BalloonChallengeScenario:
    scenario_number: int
    raw: dict[str, Any]
    source_path: Path

    @property
    def environment(self) -> dict[str, Any]:
        return dict(self.raw.get("environment", {}))

    @property
    def simulation(self) -> dict[str, Any]:
        return dict(self.raw.get("simulation", {}))

    @property
    def balloon(self) -> dict[str, Any]:
        return dict(self.raw.get("balloon", {}))

    @property
    def rocket(self) -> dict[str, Any]:
        return dict(self.raw.get("rocket", {}))


def load_balloon_challenge_scenario(
    *,
    repo_root: str | Path | None = None,
    scenario_number: int | None = None,
    scenario_path: str | Path | None = None,
) -> BalloonChallengeScenario | None:
    if scenario_path is not None:
        path = Path(scenario_path).resolve()
    else:
        root = Path(repo_root) if repo_root is not None else DEFAULT_REPO
        number = 1 if scenario_number is None else int(scenario_number)
        path = (root / "BalloonPoppingGymEnv" / "envs" / "scenario_parameters" / f"scenario_{number}_parameters.yaml").resolve()
    if not path.exists():
        return None
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    number = int(payload.get("scenario", {}).get("number", scenario_number or 0))
    return BalloonChallengeScenario(
        scenario_number=number,
        raw=payload,
        source_path=path,
    )
