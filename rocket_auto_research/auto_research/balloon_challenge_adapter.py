from __future__ import annotations

import math
import sys
import time
from pathlib import Path

from rocket_auto_research.auto_research.external_paths import (
    DEFAULT_CHALLENGE_ACTIVEROCKETPY,
    DEFAULT_CHALLENGE_REPO,
    balloon_challenge_setup_hint,
    resolve_first_existing_path,
)
from rocket_auto_research.auto_research.balloon_challenge_loader import (
    BalloonChallengeScenario,
    load_balloon_challenge_scenario,
)
from rocket_auto_research.auto_research.experiment_spec import ExperimentSpec
from rocket_auto_research.auto_research.simulation import EpisodeResult, SimulationAdapter
from rocket_auto_research.gnc.action_formatter import format_action
from rocket_auto_research.gnc.observation_parser import parse_observation
from rocket_auto_research.strategies.registry import build_strategy


def _status_scalar(raw_value: object) -> int:
    if isinstance(raw_value, (list, tuple)) and raw_value:
        raw_value = raw_value[0]
    elif hasattr(raw_value, "shape"):
        values = getattr(raw_value, "tolist", lambda: raw_value)()
        if isinstance(values, list) and values:
            raw_value = values[0]
    try:
        return int(raw_value)
    except Exception:
        return 0


def _import_balloon_challenge(repo_root: str | Path):
    root = resolve_first_existing_path(repo_root, DEFAULT_CHALLENGE_REPO)
    if root is None:
        raise FileNotFoundError(balloon_challenge_setup_hint())
    package_root = root
    activerocketpy_root = resolve_first_existing_path(root / "ActiveRocketPy", DEFAULT_CHALLENGE_ACTIVEROCKETPY)
    for candidate in (package_root, activerocketpy_root):
        if candidate is not None and candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv  # type: ignore

    return BalloonPoppingEnv


def _clear_balloon_cache(repo_root: str | Path) -> None:
    data_dir = Path(repo_root) / "BalloonPoppingGymEnv" / "envs" / "data"
    for suffix in (".inputs.txt", ".outputs.txt", ".errors.txt"):
        path = data_dir / f"balloon_sim{suffix}"
        if path.exists():
            for attempt in range(5):
                try:
                    path.unlink()
                    break
                except PermissionError:
                    if attempt == 4:
                        break
                    time.sleep(0.2)


class BalloonChallengeSimulationAdapter(SimulationAdapter):
    def __init__(self, repo_root: str | Path = str(DEFAULT_CHALLENGE_REPO), scenario_number: int = 1) -> None:
        self.repo_root = Path(repo_root)
        self.scenario_number = scenario_number

    def run_episode(self, spec: ExperimentSpec, seed: int) -> EpisodeResult:
        repo_root = Path(spec.params.get("challenge_repo_root", self.repo_root))
        scenario_number = int(spec.params.get("challenge_scenario_number", self.scenario_number))
        scenario = load_balloon_challenge_scenario(
            repo_root=repo_root,
            scenario_number=scenario_number,
            scenario_path=spec.params.get("challenge_scenario_path"),
        )
        if scenario is None:
            raise FileNotFoundError(
                f"Balloon challenge scenario not found for repo={repo_root} scenario={scenario_number}. "
                f"{balloon_challenge_setup_hint()}"
            )
        BalloonPoppingEnv = _import_balloon_challenge(repo_root)
        _clear_balloon_cache(repo_root)
        env = BalloonPoppingEnv(render_mode=None, parameters=scenario.raw)
        strategy = build_strategy(spec.strategy_name, dict(spec.params))
        try:
            observation, info = env.reset(seed=seed)
            terminated = False
            truncated = False
            final_reward = 0.0
            launch_time: float | None = None
            first_pop_time: float | None = None
            min_distance_to_any_balloon = float("inf")
            target_switch_count = 0
            last_target_id: str | None = None
            target_history: list[str] = []
            trajectory: list[dict[str, object]] = []
            last_balloons_released = 0

            while not terminated and not truncated:
                world_state = parse_observation(observation, schema="balloon_challenge")
                action_obj = strategy.act(world_state)
                current_target_id = getattr(strategy.context, "current_target_id", None)
                if current_target_id != last_target_id:
                    if last_target_id is not None:
                        target_switch_count += 1
                    last_target_id = current_target_id
                    if current_target_id is not None:
                        target_history.append(current_target_id)

                action = format_action(action_obj, schema="balloon_challenge")
                observation, reward, terminated, truncated, info = env.step(action)
                final_reward = float(reward)

                simulation_time = float(observation.get("simulation_time", 0.0) or 0.0)
                if bool(action.get("launch")) and launch_time is None:
                    launch_time = simulation_time
                popped_count = self._popped_count(observation)
                if popped_count > 0 and first_pop_time is None:
                    first_pop_time = simulation_time

                rocket_states = list(info.get("rocket_states", []))
                rocket_position = (
                    float(rocket_states[0]) if len(rocket_states) > 0 else float("nan"),
                    float(rocket_states[1]) if len(rocket_states) > 1 else float("nan"),
                    float(rocket_states[2]) if len(rocket_states) > 2 else float("nan"),
                )
                balloon_status = observation.get("balloon_status", [])
                balloon_states = observation.get("balloon_states", [])
                released = 0
                for index, status_row in enumerate(balloon_status):
                    status = _status_scalar(status_row)
                    if status >= 1:
                        released += 1
                    if status >= 2:
                        continue
                    if status < 1 or index >= len(balloon_states):
                        continue
                    row = balloon_states[index]
                    dx = float(row[0]) - rocket_position[0]
                    dy = float(row[1]) - rocket_position[1]
                    dz = float(row[2]) - rocket_position[2]
                    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
                    min_distance_to_any_balloon = min(min_distance_to_any_balloon, distance)
                last_balloons_released = released

                if len(trajectory) == 0 or len(trajectory) % 3 == 0:
                    trajectory.append(
                        {
                            "time_s": round(simulation_time, 4),
                            "rocket_position": {"x": rocket_position[0], "y": rocket_position[1], "z": rocket_position[2]},
                            "rocket_velocity": {
                                "x": float(rocket_states[3]) if len(rocket_states) > 3 else 0.0,
                                "y": float(rocket_states[4]) if len(rocket_states) > 4 else 0.0,
                                "z": float(rocket_states[5]) if len(rocket_states) > 5 else 0.0,
                            },
                            "popped": popped_count,
                            "released": released,
                            "target_id": current_target_id,
                        }
                    )

            rocket_states = list(info.get("rocket_states", []))
            final_altitude = float(rocket_states[2]) if len(rocket_states) > 2 else float("nan")
            crashed = bool(terminated and math.isfinite(final_altitude) and final_altitude <= float(scenario.environment.get("elevation", 0.0)) + 1.0)
            if not math.isfinite(min_distance_to_any_balloon):
                min_distance_to_any_balloon = 999999.0

            metadata = {
                "adapter": "balloon_challenge",
                "scenario": f"balloon_challenge_{scenario.scenario_number}",
                "score_mode": "official_reward",
                "score_is_proxy": False,
                "trajectory": trajectory,
                "target_history": target_history,
                "balloons_total": int(scenario.balloon.get("num", 0)),
                "balloons_released": last_balloons_released,
                "balloons_popped": self._popped_count(observation),
                "challenge_scenario_number": scenario.scenario_number,
                "challenge_scenario_path": str(scenario.source_path),
                "objective_note": "Runs the real BalloonPoppingChallenge Gymnasium environment with official observation/action contract.",
            }
            return EpisodeResult(
                seed=seed,
                score=round(final_reward, 4),
                popped=self._popped_count(observation),
                crashed=crashed,
                nan_detected=not math.isfinite(final_reward),
                duration=round(float(observation.get("simulation_time", 0.0) or 0.0), 4),
                time_to_launch=round(launch_time if launch_time is not None else 999.0, 4),
                time_to_first_pop=round(first_pop_time, 4) if first_pop_time is not None else None,
                min_distance_to_any_balloon=round(min_distance_to_any_balloon, 4),
                target_switch_count=target_switch_count,
                tvc_saturation_ratio=0.0,
                late_launch=launch_time is None,
                near_miss=self._popped_count(observation) == 0 and min_distance_to_any_balloon < float(scenario.balloon.get("radius", 1.5)) * 4.0,
                metadata=metadata,
            )
        finally:
            env.close()

    @staticmethod
    def _popped_count(observation: dict[str, object]) -> int:
        count = 0
        for row in observation.get("balloon_status", []):
            status = _status_scalar(row)
            if status >= 2:
                count += 1
        return count
