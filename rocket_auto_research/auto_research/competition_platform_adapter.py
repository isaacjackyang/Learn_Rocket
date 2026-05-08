from __future__ import annotations

import math

from rocket_auto_research.auto_research.competition_platform_api import SimulatedCompetitionEnv
from rocket_auto_research.auto_research.experiment_spec import ExperimentSpec
from rocket_auto_research.auto_research.simulation import EpisodeResult, SimulationAdapter
from rocket_auto_research.gnc.action_formatter import format_action
from rocket_auto_research.gnc.observation_parser import parse_observation
from rocket_auto_research.strategies.registry import build_strategy


class CompetitionPlatformSimulationAdapter(SimulationAdapter):
    def run_episode(self, spec: ExperimentSpec, seed: int) -> EpisodeResult:
        env = SimulatedCompetitionEnv(spec, seed)
        strategy = build_strategy(spec.strategy_name, dict(spec.params))
        observation, _ = env.reset()
        terminated = False
        truncated = False
        last_info: dict[str, object] = {}

        while not terminated and not truncated:
            world_state = parse_observation(observation, schema="balloon_challenge")
            action_obj = strategy.act(world_state)
            env.record_target(strategy.context.current_target_id)
            action = format_action(action_obj, schema="balloon_challenge")
            observation, reward, terminated, truncated, info = env.step(action)
            last_info = dict(info)
            last_info["last_reward"] = reward

        final_score = env.final_score()
        crashed = bool(terminated and env.state.popped == 0 and env.state.position.z <= env.elevation_m + 1.0)
        target_switch_count = int(last_info.get("target_switch_count", 0))
        tvc_saturation_ratio = float(last_info.get("tvc_saturation_ratio", 0.0))
        first_pop_time = env.state.first_pop_time
        min_distance = env.course.min_distance_to_any_balloon
        metadata = {
            "adapter": "competition_platform",
            "scenario": "official_like_step_api",
            "score_mode": "competition_platform_proxy",
            "score_is_proxy": True,
            "wind_drift_m": float(last_info.get("wind_drift_m", 0.0)),
            "mean_relative_speed_mps": float(last_info.get("mean_relative_speed_mps", 0.0)),
            "sensor_jitter_index": float(last_info.get("sensor_jitter_index", 0.0)),
            "trajectory": last_info.get("trajectory", []),
            "target_history": last_info.get("target_history", []),
            "balloons_total": int(last_info.get("balloons_total", len(env.balloons))),
            "balloons_released": int(last_info.get("balloons_released", env.course.release_count)),
            "balloons_popped": env.state.popped,
            "tvc_saturation_threshold": float(last_info.get("tvc_saturation_threshold", 0.72)),
            "target_chatter_threshold": int(last_info.get("target_chatter_threshold", 10)),
            "recommended_launch_floor_s": float(last_info.get("recommended_launch_floor_s", 0.05)),
            "recommended_launch_ceiling_s": float(last_info.get("recommended_launch_ceiling_s", 0.75)),
            "target_altitude_floor_m": float(last_info.get("target_altitude_floor_m", 850.0)),
            "velocity_floor_mps": float(last_info.get("velocity_floor_mps", 28.0)),
            "near_intercept_threshold_m": float(last_info.get("near_intercept_threshold_m", 60.0)),
            "objective_note": (
                "Balloon contract aligned to BalloonPoppingChallenge: simulation_time, balloon_status, balloon_states, "
                "rocket_sensors, and launch_inclination_heading/tvc action fields. Physics remain a simplified surrogate."
            ),
            "apogee_agl_m": max(
                0.0,
                max(
                    (
                        float(snapshot["rocket_position"]["z"]) - env.elevation_m
                        for snapshot in list(last_info.get("trajectory", []))
                    ),
                    default=0.0,
                ),
            ),
        }
        return EpisodeResult(
            seed=seed,
            score=round(final_score, 4),
            popped=env.state.popped,
            crashed=crashed,
            nan_detected=not math.isfinite(final_score),
            duration=round(env.state.time_s, 4),
            time_to_launch=round(env.state.launch_time if env.state.launch_time is not None else 999.0, 4),
            time_to_first_pop=round(first_pop_time, 4) if first_pop_time is not None else None,
            min_distance_to_any_balloon=round(min_distance if math.isfinite(min_distance) else 999999.0, 4),
            target_switch_count=target_switch_count,
            tvc_saturation_ratio=round(tvc_saturation_ratio, 4),
            late_launch=(
                env.state.launch_time is None
                or env.state.launch_time > float(last_info.get("recommended_launch_ceiling_s", 0.75))
            ),
            near_miss=env.course.near_miss_count > max(2, env.state.popped),
            metadata=metadata,
        )
