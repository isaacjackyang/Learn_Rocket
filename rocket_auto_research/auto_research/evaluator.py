from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean, median, pstdev
from typing import Any

from rocket_auto_research.auto_research.problem_definition import ResearchStage, evaluate_stage
from rocket_auto_research.auto_research.simulation import EpisodeResult


@dataclass(slots=True)
class EvaluationSummary:
    stage_id: str
    mean_score: float
    median_score: float
    min_score: float
    max_score: float
    mean_popped: float
    median_popped: float
    crash_rate: float
    nan_rate: float
    score_std: float
    robustness_score: float
    mission_fitness: float
    stage_fitness: float
    stage_success_rate: float
    stage_success_count: int
    final_fitness: float
    mean_time_to_first_pop: float | None
    pop_per_second: float
    late_launch_rate: float
    near_miss_rate: float
    mean_duration: float
    mean_min_distance: float
    mean_target_switches: float
    mean_tvc_saturation: float
    wind_sensitivity: float
    seed_sensitivity: float
    overall_progress: float
    score_is_proxy: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_episode_set(
    episodes: list[EpisodeResult],
    stage: ResearchStage | None = None,
) -> EvaluationSummary:
    if not episodes:
        raise ValueError("Cannot evaluate an empty episode set.")

    scores = [episode.score for episode in episodes]
    popped = [episode.popped for episode in episodes]
    crash_rate = mean(1.0 if episode.crashed else 0.0 for episode in episodes)
    nan_rate = mean(1.0 if episode.nan_detected else 0.0 for episode in episodes)
    late_launch_rate = mean(1.0 if episode.late_launch else 0.0 for episode in episodes)
    near_miss_rate = mean(1.0 if episode.near_miss else 0.0 for episode in episodes)
    score_std = pstdev(scores) if len(scores) > 1 else 0.0
    time_to_first_pop_values = [episode.time_to_first_pop for episode in episodes if episode.time_to_first_pop is not None]
    mean_time_to_first_pop = mean(time_to_first_pop_values) if time_to_first_pop_values else None
    pop_per_second = mean(
        episode.popped / episode.duration if episode.duration > 1e-9 else 0.0 for episode in episodes
    )
    mean_duration = mean(episode.duration for episode in episodes)
    mean_min_distance = mean(episode.min_distance_to_any_balloon for episode in episodes)
    mean_target_switches = mean(episode.target_switch_count for episode in episodes)
    mean_tvc_saturation = mean(episode.tvc_saturation_ratio for episode in episodes)
    wind_sensitivity = mean(float(episode.metadata.get("wind_drift_m", 0.0)) for episode in episodes)
    seed_sensitivity = score_std / max(abs(mean(scores)), 1.0)
    score_is_proxy = any(bool(episode.metadata.get("score_is_proxy", False)) for episode in episodes)

    robustness_score = (
        mean(scores)
        - 0.6 * score_std
        - 140.0 * crash_rate
        - 180.0 * nan_rate
        - 45.0 * mean_tvc_saturation
        - 8.0 * mean_target_switches
    )
    mission_score = (
        150.0 * mean(popped)
        + 30.0 * pop_per_second
        - 0.12 * mean_min_distance
        - 20.0 * late_launch_rate
        - 18.0 * near_miss_rate
        - 0.03 * wind_sensitivity
    )
    mission_fitness = mission_score + robustness_score
    if stage is None:
        stage_fitness = mission_fitness
        stage_success_rate = 1.0 if mean(popped) > 0 or crash_rate < 1.0 else 0.0
        stage_success_count = len(episodes)
        stage_id = "mission"
    else:
        stage_evaluation = evaluate_stage(stage, episodes)
        stage_fitness = stage_evaluation.fitness
        stage_success_rate = stage_evaluation.success_rate
        stage_success_count = stage_evaluation.success_count
        stage_id = stage.stage_id
    final_fitness = stage_fitness if stage is not None else mission_fitness
    overall_progress = min(
        1.0,
        max(
            0.0,
            0.65 * stage_success_rate
            + 0.35 * max(0.0, min(1.0, (mission_fitness + 250.0) / 500.0)),
        ),
    )

    return EvaluationSummary(
        stage_id=stage_id,
        mean_score=round(mean(scores), 4),
        median_score=round(median(scores), 4),
        min_score=round(min(scores), 4),
        max_score=round(max(scores), 4),
        mean_popped=round(mean(popped), 4),
        median_popped=round(median(popped), 4),
        crash_rate=round(crash_rate, 4),
        nan_rate=round(nan_rate, 4),
        score_std=round(score_std, 4),
        robustness_score=round(robustness_score, 4),
        mission_fitness=round(mission_fitness, 4),
        stage_fitness=round(stage_fitness, 4),
        stage_success_rate=round(stage_success_rate, 4),
        stage_success_count=stage_success_count,
        final_fitness=round(final_fitness, 4),
        mean_time_to_first_pop=round(mean_time_to_first_pop, 4) if mean_time_to_first_pop is not None else None,
        pop_per_second=round(pop_per_second, 4),
        late_launch_rate=round(late_launch_rate, 4),
        near_miss_rate=round(near_miss_rate, 4),
        mean_duration=round(mean_duration, 4),
        mean_min_distance=round(mean_min_distance, 4),
        mean_target_switches=round(mean_target_switches, 4),
        mean_tvc_saturation=round(mean_tvc_saturation, 4),
        wind_sensitivity=round(wind_sensitivity, 4),
        seed_sensitivity=round(seed_sensitivity, 4),
        overall_progress=round(overall_progress, 4),
        score_is_proxy=score_is_proxy,
    )
