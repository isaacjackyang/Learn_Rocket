from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from rocket_auto_research.auto_research.experiment_spec import ExperimentSpec
from rocket_auto_research.auto_research.simulation import EpisodeResult


@dataclass(slots=True)
class StageHypothesisSeed:
    hypothesis_id: str
    rationale: str
    adjustments: dict[str, Any]
    preferred_strategies: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ResearchStage:
    stage_id: str
    title: str
    description: str
    success_threshold: float
    preferred_strategies: list[str] = field(default_factory=list)
    planner_hypotheses: list[StageHypothesisSeed] = field(default_factory=list)


@dataclass(slots=True)
class StageEvaluation:
    stage_id: str
    fitness: float
    success_rate: float
    success_count: int
    notes: list[str]


def default_problem_stages() -> list[ResearchStage]:
    return [
        ResearchStage(
            stage_id="launch_valid",
            title="Launch Validity",
            description="First objective: issue a legal launch promptly and avoid invalid or immediately broken episodes.",
            success_threshold=0.8,
            preferred_strategies=["baseline_pid", "greedy_intercept", "energy_aware"],
            planner_hypotheses=[
                StageHypothesisSeed(
                    hypothesis_id="stage_launch_validity",
                    rationale="Prioritize fast valid launch, near-vertical rail exit, and conservative early control.",
                    adjustments={
                        "launch_wait_time": -0.2,
                        "launch_inclination_deg": 89.0,
                        "max_tvc": -1.0,
                        "kp": -0.1,
                        "throttle": 0.08,
                    },
                    preferred_strategies=["baseline_pid", "greedy_intercept", "energy_aware"],
                )
            ],
        ),
        ResearchStage(
            stage_id="ascent_stable",
            title="Stable Ascent",
            description="Second objective: survive ascent without crashing and keep the stack stable long enough to build altitude.",
            success_threshold=0.7,
            preferred_strategies=["energy_aware", "score_based", "baseline_pid"],
            planner_hypotheses=[
                StageHypothesisSeed(
                    hypothesis_id="stage_ascent_stable",
                    rationale="Bias toward stable climb with limited gimbal demand and early energy preservation.",
                    adjustments={
                        "kp": -0.14,
                        "kd": 0.05,
                        "max_tvc": -1.5,
                        "throttle": 0.05,
                        "target_lock_duration": 0.4,
                    },
                    preferred_strategies=["energy_aware", "score_based"],
                )
            ],
        ),
        ResearchStage(
            stage_id="energy_margin",
            title="Energy Margin",
            description="Third objective: carry enough altitude and speed into the engagement window to make interception possible.",
            success_threshold=0.65,
            preferred_strategies=["energy_aware", "predictive_intercept", "score_based"],
            planner_hypotheses=[
                StageHypothesisSeed(
                    hypothesis_id="stage_energy_margin",
                    rationale="Increase usable ascent energy before spending control authority on turning toward targets.",
                    adjustments={
                        "throttle": 0.08,
                        "launch_wait_time": -0.05,
                        "lookahead_time": 0.12,
                        "target_angle_weight": -0.12,
                    },
                    preferred_strategies=["energy_aware", "predictive_intercept"],
                )
            ],
        ),
        ResearchStage(
            stage_id="approach_window",
            title="Approach Window",
            description="Fourth objective: reach near-intercept geometry consistently, even before the first pop appears.",
            success_threshold=0.55,
            preferred_strategies=["predictive_intercept", "score_based", "mpc_light", "energy_aware"],
            planner_hypotheses=[
                StageHypothesisSeed(
                    hypothesis_id="stage_approach_window",
                    rationale="Shift focus from pure ascent to reachable-target selection and predictive approach shaping.",
                    adjustments={
                        "lookahead_time": 0.15,
                        "target_distance_weight": 0.1,
                        "target_angle_weight": 0.08,
                        "switching_penalty": 0.2,
                    },
                    preferred_strategies=["predictive_intercept", "score_based", "mpc_light"],
                )
            ],
        ),
        ResearchStage(
            stage_id="balloon_pop",
            title="Balloon Pop",
            description="Final objective: maximize popped balloons under the official reward contract.",
            success_threshold=0.3,
            preferred_strategies=["energy_aware", "mpc_light", "cem_planner", "predictive_intercept"],
            planner_hypotheses=[
                StageHypothesisSeed(
                    hypothesis_id="stage_balloon_pop",
                    rationale="Use the full mission objective once the system consistently survives and reaches targets.",
                    adjustments={
                        "lookahead_time": 0.08,
                        "target_distance_weight": 0.12,
                        "target_angle_weight": 0.12,
                        "switching_penalty": 0.1,
                    },
                    preferred_strategies=["energy_aware", "mpc_light", "cem_planner"],
                )
            ],
        ),
    ]


def evaluate_stage(stage: ResearchStage, episodes: list[EpisodeResult]) -> StageEvaluation:
    total = max(1, len(episodes))
    successes = 0
    episode_scores: list[float] = []

    for episode in episodes:
        apogee = float(episode.metadata.get("apogee_agl_m", 0.0))
        relative_speed = float(episode.metadata.get("mean_relative_speed_mps", 0.0))
        launched = not episode.late_launch and episode.time_to_launch < min(max(episode.duration, 0.5), 2.0)
        if stage.stage_id == "launch_valid":
            success = launched and not episode.nan_detected and episode.duration >= 0.5
            score = 100.0 * float(success) - 80.0 * float(episode.nan_detected) - 55.0 * float(episode.crashed)
            score += max(0.0, 2.0 - episode.time_to_launch) * 12.0 + min(episode.duration, 3.0) * 4.0
        elif stage.stage_id == "ascent_stable":
            success = not episode.crashed and apogee >= 120.0 and episode.duration >= 2.0
            score = apogee * 0.08 + episode.duration * 12.0 - 110.0 * float(episode.crashed)
            score -= 25.0 * episode.tvc_saturation_ratio - 10.0 * float(episode.nan_detected)
        elif stage.stage_id == "energy_margin":
            success = not episode.crashed and apogee >= 250.0 and relative_speed >= 20.0
            score = apogee * 0.06 + relative_speed * 3.5 + episode.duration * 2.0
            score -= 110.0 * float(episode.crashed) + 20.0 * float(episode.late_launch)
        elif stage.stage_id == "approach_window":
            success = not episode.crashed and (episode.popped > 0 or episode.min_distance_to_any_balloon <= 20.0)
            proximity_bonus = 180.0 / max(1.0, episode.min_distance_to_any_balloon)
            score = proximity_bonus + apogee * 0.03 + relative_speed * 1.5
            score -= 90.0 * float(episode.crashed) + 18.0 * episode.target_switch_count
        else:
            success = episode.popped > 0
            proximity_bonus = 120.0 / max(1.0, episode.min_distance_to_any_balloon)
            score = episode.score + episode.popped * 180.0 + proximity_bonus
            score -= 100.0 * float(episode.crashed) + 30.0 * float(episode.nan_detected)
        successes += int(success)
        episode_scores.append(score)

    success_rate = successes / total
    notes: list[str] = []
    if success_rate < stage.success_threshold:
        notes.append(
            f"Stage '{stage.stage_id}' is below promotion threshold {stage.success_threshold:.2f} with success rate {success_rate:.2f}."
        )
    return StageEvaluation(
        stage_id=stage.stage_id,
        fitness=round(mean(episode_scores), 4),
        success_rate=round(success_rate, 4),
        success_count=successes,
        notes=notes,
    )


def build_stage_bootstrap_specs(
    stage: ResearchStage,
    generation: int,
    seeds: list[int],
    fixed_params: dict[str, Any],
    available_strategies: list[str],
) -> list[ExperimentSpec]:
    templates: list[tuple[list[str], dict[str, Any]]] = []
    if stage.stage_id == "launch_valid":
        templates = [
            (
                ["baseline_pid", "greedy_intercept", "energy_aware"],
                {
                    "throttle": 0.92,
                    "kp": 0.55,
                    "kd": 0.28,
                    "max_tvc": 5.5,
                    "launch_wait_time": 0.0,
                    "launch_inclination_deg": 89.5,
                    "launch_heading_deg": 0.0,
                    "target_lock_duration": 1.2,
                    "switching_penalty": 1.4,
                    "lookahead_time": 1.1,
                },
            ),
            (
                ["energy_aware", "score_based"],
                {
                    "throttle": 0.88,
                    "kp": 0.6,
                    "kd": 0.32,
                    "max_tvc": 6.0,
                    "launch_wait_time": 0.0,
                    "launch_inclination_deg": 88.5,
                    "launch_heading_deg": 0.0,
                    "target_lock_duration": 1.0,
                    "switching_penalty": 1.2,
                    "lookahead_time": 1.25,
                },
            ),
        ]
    elif stage.stage_id == "ascent_stable":
        templates = [
            (
                ["energy_aware", "score_based", "baseline_pid"],
                {
                    "throttle": 0.86,
                    "kp": 0.65,
                    "kd": 0.35,
                    "max_tvc": 6.2,
                    "launch_wait_time": 0.0,
                    "launch_inclination_deg": 88.8,
                    "switching_penalty": 1.5,
                    "target_lock_duration": 1.3,
                    "lookahead_time": 1.35,
                },
            )
        ]
    elif stage.stage_id == "energy_margin":
        templates = [
            (
                ["energy_aware", "predictive_intercept", "score_based"],
                {
                    "throttle": 0.94,
                    "kp": 0.72,
                    "kd": 0.32,
                    "max_tvc": 7.0,
                    "launch_wait_time": 0.0,
                    "launch_inclination_deg": 88.0,
                    "lookahead_time": 1.4,
                    "target_distance_weight": 1.1,
                    "target_angle_weight": 0.5,
                },
            )
        ]
    elif stage.stage_id == "approach_window":
        templates = [
            (
                ["predictive_intercept", "score_based", "mpc_light", "energy_aware"],
                {
                    "throttle": 0.9,
                    "kp": 0.8,
                    "kd": 0.3,
                    "max_tvc": 8.5,
                    "lookahead_time": 1.55,
                    "target_distance_weight": 1.3,
                    "target_angle_weight": 0.8,
                    "switching_penalty": 1.6,
                },
            )
        ]
    else:
        templates = [
            (
                ["energy_aware", "mpc_light", "cem_planner", "predictive_intercept"],
                {
                    "throttle": 0.88,
                    "kp": 0.85,
                    "kd": 0.28,
                    "max_tvc": 9.0,
                    "lookahead_time": 1.7,
                    "target_distance_weight": 1.4,
                    "target_angle_weight": 0.9,
                    "switching_penalty": 1.5,
                },
            )
        ]

    bootstrap_specs: list[ExperimentSpec] = []
    for index, (strategy_candidates, params) in enumerate(templates):
        strategy_name = next((name for name in strategy_candidates if name in available_strategies), None)
        if strategy_name is None:
            continue
        merged_params = dict(params)
        merged_params.update(fixed_params)
        bootstrap_specs.append(
            ExperimentSpec(
                strategy_name=strategy_name,
                params=merged_params,
                seeds=seeds,
                note=f"stage_bootstrap:{stage.stage_id}:{index}",
                generation=generation,
            )
        )
    return bootstrap_specs
