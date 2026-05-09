from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from rocket_auto_research.auto_research.failure_analyzer import FailureReport


@dataclass(slots=True)
class ResearchHypothesis:
    hypothesis_id: str
    rationale: str
    adjustments: dict[str, Any]
    preferred_strategies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def as_text(self) -> str:
        strategy_hint = ""
        if self.preferred_strategies:
            strategy_hint = f" Prefer {', '.join(self.preferred_strategies)}."
        return f"{self.rationale}{strategy_hint}"


def generate_hypotheses(failure_report: FailureReport) -> list[ResearchHypothesis]:
    hypotheses: list[ResearchHypothesis] = []
    rates = failure_report.rates

    if rates.get("tvc_saturation", 0.0) > 0.15:
        hypotheses.append(
            ResearchHypothesis(
                hypothesis_id="smooth_turn_demand",
                rationale="Guidance is too aggressive and is repeatedly saturating TVC.",
                adjustments={
                    "kp": -0.14,
                    "lookahead_time": 0.22,
                    "switching_penalty": 0.12,
                    "target_lock_duration": 0.35,
                },
                preferred_strategies=["score_based", "predictive_intercept"],
            )
        )

    if rates.get("near_miss", 0.0) > 0.15 or rates.get("wrong_target_or_timing", 0.0) > 0.1:
        hypotheses.append(
            ResearchHypothesis(
                hypothesis_id="improve_intercept_timing",
                rationale="The rocket is getting close but is still biasing intercept timing or choosing the wrong reachable point.",
                adjustments={
                    "guidance_mode": "predictive",
                    "lookahead_time": 0.12,
                    "target_selector": "reachable",
                    "wind_comp_gain": 0.08,
                    "ascent_targeting_turn_scale": 0.14,
                    "ascent_targeting_altitude_m": -120.0,
                },
                preferred_strategies=["predictive_intercept", "mpc_light", "energy_aware"],
            )
        )

    if rates.get("late_launch", 0.0) > 0.1:
        hypotheses.append(
            ResearchHypothesis(
                hypothesis_id="launch_earlier",
                rationale="Launch timing is too conservative under current release patterns.",
                adjustments={
                    "launch_wait_time": -0.08,
                    "stabilize_duration": -0.2,
                    "throttle": 0.03,
                },
                preferred_strategies=["score_based", "predictive_intercept"],
            )
        )

    if rates.get("target_chattering", 0.0) > 0.15:
        hypotheses.append(
            ResearchHypothesis(
                hypothesis_id="stabilize_target_lock",
                rationale="Frequent target switches are wasting control authority.",
                adjustments={
                    "switching_penalty": 0.2,
                    "target_lock_duration": 0.5,
                    "target_selector": "reachable",
                },
                preferred_strategies=["score_based", "mpc_light"],
            )
        )

    if rates.get("wind_drift", 0.0) > 0.15:
        hypotheses.append(
            ResearchHypothesis(
                hypothesis_id="compensate_wind",
                rationale="Crosswind drift is large enough to dominate mission loss.",
                adjustments={
                    "estimator_mode": "wind_aware",
                    "guidance_mode": "short_horizon",
                    "wind_comp_gain": 0.18,
                },
                preferred_strategies=["mpc_light", "cem_planner"],
            )
        )

    if rates.get("sensor_jitter", 0.0) > 0.15 or rates.get("nan_or_invalid_action", 0.0) > 0.02:
        hypotheses.append(
            ResearchHypothesis(
                hypothesis_id="filter_noisy_observations",
                rationale="Noise is leaking into control outputs and destabilizing action quality.",
                adjustments={
                    "estimator_mode": "alpha_beta",
                    "alpha": -0.08,
                    "beta": -0.05,
                    "policy_temperature": -0.04,
                },
                preferred_strategies=["score_based", "rl_policy_wrapper"],
            )
        )

    if rates.get("altitude_shortfall", 0.0) > 0.15 or rates.get("velocity_shortfall", 0.0) > 0.15:
        hypotheses.append(
            ResearchHypothesis(
                hypothesis_id="recover_energy_margin",
                rationale="The rocket is not carrying enough energy into the engagement window.",
                adjustments={
                    "throttle": 0.08,
                    "desired_speed": 0.14,
                    "launch_wait_time": -0.03,
                    "climb_bias_altitude_m": 80.0,
                    "target_energy_weight": 0.1,
                    "ascent_targeting_turn_scale": -0.1,
                    "ascent_targeting_altitude_m": 120.0,
                },
                preferred_strategies=["predictive_intercept", "energy_aware", "score_based"],
            )
        )

    if rates.get("crash", 0.0) > 0.1 or rates.get("recovery_fail", 0.0) > 0.05:
        hypotheses.append(
            ResearchHypothesis(
                hypothesis_id="reduce_instability",
                rationale="The controller is entering unrecoverable states too often.",
                adjustments={
                    "kp": -0.18,
                    "kd": 0.03,
                    "throttle": -0.07,
                    "failsafe_tilt_rate": -0.5,
                    "controller_mode": "adaptive",
                },
                preferred_strategies=["score_based", "mpc_light"],
            )
        )

    if not hypotheses:
        hypotheses.append(
            ResearchHypothesis(
                hypothesis_id="local_refinement",
                rationale="No single failure mode dominates; continue local exploration around the current elite.",
                adjustments={"lookahead_time": 0.05, "switching_penalty": 0.05},
                preferred_strategies=[],
            )
        )
    return hypotheses
