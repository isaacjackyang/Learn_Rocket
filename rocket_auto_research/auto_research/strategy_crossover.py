from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any

from rocket_auto_research.auto_research.experiment_spec import ExperimentSpec

MODULE_GROUPS = {
    "estimator": {"estimator_mode", "alpha", "beta", "wind_comp_gain"},
    "targeting": {
        "target_selector",
        "target_distance_weight",
        "target_angle_weight",
        "target_height_weight",
        "target_lock_duration",
        "switching_penalty",
    },
    "guidance": {"guidance_mode", "lookahead_time", "desired_speed", "branch_count", "horizon_s"},
    "control": {"controller_mode", "kp", "kd", "max_tvc", "throttle", "failsafe_tilt_rate"},
    "launch": {"launch_wait_time", "stabilize_duration"},
}


@dataclass(slots=True)
class StrategyCrossover:
    def crossover(self, left: ExperimentSpec, right: ExperimentSpec, generation: int, seeds: list[int]) -> ExperimentSpec:
        rng = random.Random(f"crossover:{left.experiment_id}:{right.experiment_id}:{generation}")
        left_params = self._materialize_params(left)
        right_params = self._materialize_params(right)
        merged_params: dict[str, Any] = {}
        ownership: dict[str, str] = {}

        for group_name, keys in MODULE_GROUPS.items():
            donor_name, donor_params = (
                (left.experiment_id, left_params) if rng.random() < 0.5 else (right.experiment_id, right_params)
            )
            ownership[group_name] = donor_name
            for key in keys:
                if key in donor_params:
                    merged_params[key] = copy.deepcopy(donor_params[key])

        for key in sorted(set(left_params) | set(right_params)):
            if key not in merged_params:
                merged_params[key] = copy.deepcopy(left_params.get(key, right_params.get(key)))

        strategy_name = self._choose_strategy_name(merged_params, left.strategy_name, right.strategy_name)
        return ExperimentSpec(
            strategy_name=strategy_name,
            params=merged_params,
            seeds=seeds,
            note=(
                f"crossover:{left.experiment_id}+{right.experiment_id};"
                f"modules:{ownership['estimator'][:4]}/{ownership['targeting'][:4]}/"
                f"{ownership['guidance'][:4]}/{ownership['control'][:4]}"
            ),
            generation=generation,
            parent_ids=[left.experiment_id, right.experiment_id],
        )

    def _materialize_params(self, spec: ExperimentSpec) -> dict[str, Any]:
        params = dict(spec.params)
        defaults = self._legacy_defaults(spec.strategy_name)
        defaults.update(params)
        return defaults

    @staticmethod
    def _legacy_defaults(strategy_name: str) -> dict[str, Any]:
        if strategy_name == "baseline_pid":
            return {
                "estimator_mode": "simple",
                "target_selector": "nearest",
                "guidance_mode": "fixed",
                "controller_mode": "pursuit",
            }
        if strategy_name == "greedy_intercept":
            return {
                "estimator_mode": "simple",
                "target_selector": "score_based",
                "guidance_mode": "fixed",
                "controller_mode": "pursuit",
            }
        if strategy_name == "predictive_intercept":
            return {
                "estimator_mode": "simple",
                "target_selector": "score_based",
                "guidance_mode": "predictive",
                "controller_mode": "pursuit",
            }
        if strategy_name == "score_based":
            return {
                "estimator_mode": "alpha_beta",
                "target_selector": "score_based",
                "guidance_mode": "fixed",
                "controller_mode": "adaptive",
            }
        if strategy_name == "mpc_light":
            return {
                "estimator_mode": "wind_aware",
                "target_selector": "reachable",
                "guidance_mode": "short_horizon",
                "controller_mode": "adaptive",
            }
        if strategy_name == "cem_planner":
            return {
                "estimator_mode": "wind_aware",
                "target_selector": "reachable",
                "guidance_mode": "cem",
                "controller_mode": "adaptive",
            }
        if strategy_name == "rl_policy_wrapper":
            return {
                "estimator_mode": "alpha_beta",
                "target_selector": "score_based",
                "guidance_mode": "rl",
                "controller_mode": "adaptive",
            }
        return {}

    @staticmethod
    def _choose_strategy_name(
        params: dict[str, Any],
        left_strategy: str,
        right_strategy: str,
    ) -> str:
        guidance_mode = str(params.get("guidance_mode", "fixed"))
        target_selector = str(params.get("target_selector", "score_based"))
        if guidance_mode == "cem":
            return "cem_planner"
        if guidance_mode == "short_horizon":
            return "mpc_light"
        if guidance_mode == "rl":
            return "rl_policy_wrapper"
        if guidance_mode == "predictive":
            return "predictive_intercept"
        if target_selector in {"score_based", "reachable"}:
            return "score_based"
        if left_strategy == right_strategy:
            return left_strategy
        return "baseline_pid"
