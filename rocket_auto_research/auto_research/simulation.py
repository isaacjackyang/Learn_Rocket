from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rocket_auto_research.auto_research.experiment_spec import ExperimentSpec


@dataclass(slots=True)
class EpisodeResult:
    seed: int
    score: float
    popped: int
    crashed: bool
    nan_detected: bool
    duration: float
    time_to_launch: float
    time_to_first_pop: float | None
    min_distance_to_any_balloon: float
    target_switch_count: int
    tvc_saturation_ratio: float
    late_launch: bool = False
    near_miss: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SimulationAdapter(ABC):
    @abstractmethod
    def run_episode(self, spec: ExperimentSpec, seed: int) -> EpisodeResult:
        raise NotImplementedError


class MockRocketSimAdapter(SimulationAdapter):
    """Pipeline smoke-test adapter before the real simulator is wired in."""

    STRATEGY_BASE = {
        "baseline_pid": 22.0,
        "greedy_intercept": 28.0,
        "predictive_intercept": 32.0,
        "score_based": 34.0,
        "mpc_light": 38.0,
        "cem_planner": 39.5,
        "rl_policy_wrapper": 31.0,
    }

    def run_episode(self, spec: ExperimentSpec, seed: int) -> EpisodeResult:
        rng = random.Random(f"{spec.experiment_id}:{seed}")
        base = self.STRATEGY_BASE.get(spec.strategy_name, 18.0)
        kp = float(spec.params.get("kp", 1.2))
        kd = float(spec.params.get("kd", 0.15))
        lookahead = float(
            spec.params.get(
                "lookahead_time",
                spec.params.get("max_intercept_time", 1.0),
            )
        )
        throttle = float(spec.params.get("throttle", 0.9))
        switching_penalty = float(spec.params.get("switching_penalty", 0.5))
        angle_weight = float(spec.params.get("target_angle_weight", 0.7))
        launch_wait_time = float(spec.params.get("launch_wait_time", 0.0))

        stability_penalty = max(0.0, kp - 1.8) * 7.5 + max(0.0, throttle - 0.95) * 18.0
        over_damped_penalty = max(0.0, kd - 0.35) * 5.0
        lookahead_bonus = 7.0 - abs(lookahead - 1.1) * 4.0
        selector_bonus = angle_weight * 3.5 - switching_penalty * 2.2
        launch_penalty = max(0.0, launch_wait_time - 0.3) * 20.0
        noise = rng.gauss(0.0, 5.0)

        raw_score = base + lookahead_bonus + selector_bonus - stability_penalty - over_damped_penalty - launch_penalty + noise
        score = max(0.0, raw_score)
        popped = max(0, min(100, int(score // 2 + rng.randint(-2, 2))))
        crashed = raw_score < 10.0 or rng.random() < max(0.0, (kp - 1.7) * 0.08)
        nan_detected = rng.random() < max(0.0, (throttle - 0.98) * 0.15)
        near_miss = not crashed and popped < 18 and rng.random() < 0.45
        duration = rng.uniform(35.0, 60.0)
        time_to_launch = max(0.0, launch_wait_time + rng.uniform(-0.03, 0.08))
        time_to_first_pop = None if popped == 0 else rng.uniform(2.0, 18.0)
        min_distance = max(0.2, rng.uniform(0.1, 12.0) - popped * 0.03)
        target_switch_count = max(0, int(rng.gauss(4.0 + (1.0 - switching_penalty) * 2.0, 2.0)))
        tvc_saturation_ratio = min(1.0, max(0.0, 0.25 + max(0.0, kp - 1.25) * 0.35 + rng.uniform(-0.08, 0.1)))
        late_launch = time_to_launch > 0.35

        return EpisodeResult(
            seed=seed,
            score=round(score, 4),
            popped=popped,
            crashed=crashed,
            nan_detected=nan_detected,
            duration=round(duration, 4),
            time_to_launch=round(time_to_launch, 4),
            time_to_first_pop=round(time_to_first_pop, 4) if time_to_first_pop is not None else None,
            min_distance_to_any_balloon=round(min_distance, 4),
            target_switch_count=target_switch_count,
            tvc_saturation_ratio=round(tvc_saturation_ratio, 4),
            late_launch=late_launch,
            near_miss=near_miss,
            metadata={
                "mock_adapter": True,
                "score_is_proxy": True,
                "balloons_released": 100,
                "apogee_agl_m": max(0.0, 1200.0 - stability_penalty * 10.0),
                "mean_relative_speed_mps": 35.0 + max(0.0, throttle - 0.7) * 25.0,
                "wind_drift_m": 80.0 + abs(noise) * 4.0,
            },
        )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{__import__('json').dumps(payload, indent=2, ensure_ascii=False)}\n", encoding="utf-8")
