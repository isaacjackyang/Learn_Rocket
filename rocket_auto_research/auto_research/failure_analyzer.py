from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any

from rocket_auto_research.auto_research.evaluator import EvaluationSummary
from rocket_auto_research.auto_research.simulation import EpisodeResult


@dataclass(slots=True)
class FailureReport:
    counts: dict[str, int]
    rates: dict[str, float]
    dominant_failure: str | None
    notes: list[str]
    examples: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_failures(episodes: list[EpisodeResult], summary: EvaluationSummary) -> FailureReport:
    counts: Counter[str] = Counter()
    examples: dict[str, dict[str, Any]] = {}
    total = max(1, len(episodes))

    for episode in episodes:
        metadata = episode.metadata
        _flag(counts, examples, "nan_or_invalid_action", episode.nan_detected, episode)
        _flag(counts, examples, "crash", episode.crashed, episode)
        _flag(counts, examples, "late_launch", episode.late_launch, episode)
        _flag(
            counts,
            examples,
            "early_launch",
            episode.time_to_launch < float(metadata.get("recommended_launch_floor_s", 0.05)),
            episode,
        )
        _flag(counts, examples, "near_miss", episode.near_miss, episode)
        _flag(
            counts,
            examples,
            "tvc_saturation",
            episode.tvc_saturation_ratio >= float(metadata.get("tvc_saturation_threshold", 0.72)),
            episode,
        )
        _flag(
            counts,
            examples,
            "target_chattering",
            episode.target_switch_count >= int(metadata.get("target_chatter_threshold", 8)),
            episode,
        )
        _flag(
            counts,
            examples,
            "wrong_target_or_timing",
            episode.popped == 0
            and episode.min_distance_to_any_balloon
            <= float(metadata.get("near_intercept_threshold_m", 75.0)),
            episode,
        )
        _flag(
            counts,
            examples,
            "altitude_shortfall",
            float(metadata.get("apogee_agl_m", 0.0)) < float(metadata.get("target_altitude_floor_m", 900.0)),
            episode,
        )
        _flag(
            counts,
            examples,
            "velocity_shortfall",
            float(metadata.get("mean_relative_speed_mps", 0.0)) < float(metadata.get("velocity_floor_mps", 35.0)),
            episode,
        )
        _flag(
            counts,
            examples,
            "wind_drift",
            float(metadata.get("wind_drift_m", 0.0)) > float(metadata.get("wind_drift_threshold_m", 300.0)),
            episode,
        )
        _flag(
            counts,
            examples,
            "sensor_jitter",
            float(metadata.get("sensor_jitter_index", 0.0)) > float(metadata.get("sensor_jitter_threshold", 1.25)),
            episode,
        )
        _flag(
            counts,
            examples,
            "recovery_fail",
            bool(metadata.get("recovery_triggered", False)) and episode.crashed,
            episode,
        )
        _flag(
            counts,
            examples,
            "no_intercept",
            episode.popped == 0 and not episode.crashed and int(metadata.get("balloons_released", 0)) > 0,
            episode,
        )

    rates = {name: round(value / total, 4) for name, value in counts.items()}
    dominant_failure = counts.most_common(1)[0][0] if counts else None
    notes = _build_notes(episodes, summary, rates)
    return FailureReport(
        counts=dict(counts),
        rates=rates,
        dominant_failure=dominant_failure,
        notes=notes,
        examples=examples,
    )


def _flag(
    counts: Counter[str],
    examples: dict[str, dict[str, Any]],
    key: str,
    triggered: bool,
    episode: EpisodeResult,
) -> None:
    if not triggered:
        return
    counts[key] += 1
    examples.setdefault(
        key,
        {
            "seed": episode.seed,
            "score": episode.score,
            "popped": episode.popped,
            "time_to_launch": episode.time_to_launch,
            "target_switch_count": episode.target_switch_count,
            "tvc_saturation_ratio": episode.tvc_saturation_ratio,
        },
    )


def _build_notes(
    episodes: list[EpisodeResult],
    summary: EvaluationSummary,
    rates: dict[str, float],
) -> list[str]:
    notes: list[str] = []
    if summary.crash_rate > 0.18:
        notes.append("Crash rate is materially high; reduce aggressiveness or tighten failsafe behavior.")
    if summary.nan_rate > 0.02:
        notes.append("Invalid action rate is non-trivial; sanitation and controller clamps still need work.")
    if summary.near_miss_rate > 0.2:
        notes.append("Near misses dominate; intercept timing is close, but target prediction is still biased.")
    if rates.get("target_chattering", 0.0) > 0.2:
        notes.append("Target lock is unstable; add or increase switching penalty and target lock duration.")
    if rates.get("wind_drift", 0.0) > 0.2:
        notes.append("Wind drift is causing mission loss; promote wind-aware estimation and forward-looking guidance.")
    if rates.get("sensor_jitter", 0.0) > 0.2:
        notes.append("Sensor jitter is propagating into control; stronger filtering is warranted.")
    if rates.get("late_launch", 0.0) > 0.15:
        notes.append("Launch policy remains conservative under this seed set.")
    if summary.mean_popped < 1.0:
        notes.append("The strategy is still struggling to convert approaches into balloon pops.")
    active_wind = [
        float(episode.metadata.get("wind_drift_m", 0.0))
        for episode in episodes
        if "wind_drift_m" in episode.metadata
    ]
    if active_wind and mean(active_wind) > 180.0:
        notes.append("Average wind drift is elevated; benchmark seeds are stressing crosswind robustness.")
    return notes
