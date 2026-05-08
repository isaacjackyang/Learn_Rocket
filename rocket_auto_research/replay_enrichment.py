from __future__ import annotations

import math
from typing import Any

from rocket_auto_research.auto_research.activerocketpy_adapter import ActiveRocketPySimulationAdapter
from rocket_auto_research.auto_research.competition_platform_api import SimulatedCompetitionEnv
from rocket_auto_research.auto_research.experiment_spec import ExperimentSpec
from rocket_auto_research.gnc.state import Vector3


def _build_spec(spec_payload: dict[str, Any], seed: int) -> ExperimentSpec:
    return ExperimentSpec(
        strategy_name=spec_payload["strategy_name"],
        params=dict(spec_payload.get("params", {})),
        seeds=[seed],
        note=str(spec_payload.get("note", "")),
        generation=int(spec_payload.get("generation", 0) or 0),
        parent_ids=list(spec_payload.get("parent_ids", [])),
        experiment_id=str(spec_payload.get("experiment_id", "")),
    )


def _interpolate_profile(profile: list[tuple[float, float]], altitude_agl_m: float) -> float:
    if not profile:
        return 0.0
    if altitude_agl_m <= profile[0][0]:
        return profile[0][1]
    for left, right in zip(profile, profile[1:]):
        if left[0] <= altitude_agl_m <= right[0]:
            span = max(right[0] - left[0], 1e-6)
            ratio = (altitude_agl_m - left[0]) / span
            return left[1] + ratio * (right[1] - left[1])
    return profile[-1][1]


def _competition_snapshot_builder(spec: ExperimentSpec, seed: int):
    env = SimulatedCompetitionEnv(spec, seed)

    def wind_at(z_asl_m: float) -> Vector3:
        return env._wind_at_altitude(z_asl_m)

    def balloon_position(balloon, time_s: float, ambient_wind: Vector3) -> Vector3:
        return balloon.position_at(time_s, ambient_wind)

    return env.balloons, wind_at, balloon_position


def _activerocketpy_snapshot_builder(spec: ExperimentSpec, seed: int):
    balloons = ActiveRocketPySimulationAdapter._seeded_balloon_field(spec, seed)
    profile = ActiveRocketPySimulationAdapter._seeded_wind_profile(spec, seed)
    elevation = float(spec.params.get("elevation_m", 1400.0))

    def wind_at(z_asl_m: float) -> Vector3:
        altitude_agl = max(0.0, z_asl_m - elevation)
        return Vector3(
            x=_interpolate_profile(profile["u"], altitude_agl),
            y=_interpolate_profile(profile["v"], altitude_agl),
            z=0.0,
        )

    def balloon_position(balloon, time_s: float, ambient_wind: Vector3) -> Vector3:
        position = balloon.position_at(time_s, ambient_wind)
        return Vector3(x=position.x, y=position.y, z=position.z + elevation)

    return balloons, wind_at, balloon_position


def augment_trajectory_with_balloon_snapshots(
    spec_payload: dict[str, Any],
    adapter_name: str,
    seed: int,
    trajectory: list[dict[str, Any]],
    top_k: int = 20,
) -> list[dict[str, Any]]:
    if not trajectory:
        return trajectory
    if adapter_name == "competition_platform":
        balloons, wind_at, balloon_position = _competition_snapshot_builder(_build_spec(spec_payload, seed), seed)
    elif adapter_name == "activerocketpy":
        balloons, wind_at, balloon_position = _activerocketpy_snapshot_builder(_build_spec(spec_payload, seed), seed)
    else:
        return trajectory

    enriched: list[dict[str, Any]] = []
    for point in trajectory:
        time_s = float(point.get("time_s", 0.0))
        rocket_position = Vector3.from_mapping(point.get("rocket_position", {}))
        ambient_wind = wind_at(rocket_position.z)
        released_candidates: list[tuple[float, dict[str, Any]]] = []
        target_snapshot: dict[str, Any] | None = None
        target_id = point.get("target_id")
        for balloon in balloons:
            if not balloon.is_released(time_s):
                continue
            position = balloon_position(balloon, time_s, ambient_wind)
            distance = math.sqrt(
                (position.x - rocket_position.x) ** 2
                + (position.y - rocket_position.y) ** 2
                + (position.z - rocket_position.z) ** 2
            )
            snapshot = {
                "balloon_id": balloon.balloon_id,
                "position": position.to_dict(),
                "distance_to_rocket_m": round(distance, 3),
                "is_target": balloon.balloon_id == target_id,
            }
            if snapshot["is_target"]:
                target_snapshot = snapshot
            released_candidates.append((distance, snapshot))
        released_candidates.sort(key=lambda item: item[0])
        selected = [snapshot for _, snapshot in released_candidates[: max(1, top_k)]]
        if target_snapshot is not None and all(snapshot["balloon_id"] != target_snapshot["balloon_id"] for snapshot in selected):
            if len(selected) >= max(1, top_k):
                selected[-1] = target_snapshot
            else:
                selected.append(target_snapshot)
        enriched.append(
            {
                **point,
                "balloons": selected,
                "visible_balloon_count": len(released_candidates),
            }
        )
    return enriched
