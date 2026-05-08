from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rocket_auto_research.gnc.state import BalloonState, GuidanceCommand, Vector3, WorldState


@dataclass(slots=True)
class FixedLookaheadGuidance:
    lookahead_time: float = 1.0
    desired_speed: float = 1.0

    def command(self, world_state: WorldState, target: BalloonState | None) -> GuidanceCommand:
        if target is None:
            return GuidanceCommand()
        intercept_position = target.position + target.velocity.scale(self.lookahead_time)
        direction = intercept_position - world_state.rocket.position
        return GuidanceCommand(
            desired_direction=direction.normalized(),
            desired_speed=self.desired_speed,
            target_id=target.balloon_id,
            target_position=intercept_position,
        )


@dataclass(slots=True)
class PredictiveInterceptGuidance:
    min_intercept_time: float = 0.25
    max_intercept_time: float = 4.0
    speed_scale: float = 1.0

    def command(self, world_state: WorldState, target: BalloonState | None) -> GuidanceCommand:
        if target is None:
            return GuidanceCommand()
        relative = target.position - world_state.rocket.position
        rocket_speed = max(world_state.rocket.velocity.norm(), 1.0)
        intercept_time = max(self.min_intercept_time, min(self.max_intercept_time, relative.norm() / rocket_speed))
        intercept_position = target.position + target.velocity.scale(intercept_time)
        desired_direction = (intercept_position - world_state.rocket.position).normalized()
        return GuidanceCommand(
            desired_direction=desired_direction,
            desired_speed=rocket_speed * self.speed_scale,
            target_id=target.balloon_id,
            target_position=intercept_position,
        )


@dataclass(slots=True)
class ShortHorizonGuidance:
    horizon_s: float = 1.2
    branch_count: int = 5
    desired_speed: float = 1.0

    def command(self, world_state: WorldState, target: BalloonState | None) -> GuidanceCommand:
        if target is None:
            return GuidanceCommand()
        best_direction = None
        best_score = float("-inf")
        best_target_position = target.position
        rocket_position = world_state.rocket.position
        rocket_speed = max(1.0, world_state.rocket.velocity.norm())
        for branch in range(self.branch_count):
            lookahead = self.horizon_s * (0.5 + branch / max(1, self.branch_count - 1))
            predicted_target = target.position + target.velocity.scale(lookahead)
            desired_direction = (predicted_target - rocket_position).normalized()
            velocity_alignment = (
                world_state.rocket.velocity.x * desired_direction.x
                + world_state.rocket.velocity.y * desired_direction.y
                + world_state.rocket.velocity.z * desired_direction.z
            )
            score = velocity_alignment - (predicted_target - rocket_position).norm() * 0.01
            if score > best_score:
                best_score = score
                best_direction = desired_direction
                best_target_position = predicted_target
        return GuidanceCommand(
            desired_direction=best_direction or (target.position - rocket_position).normalized(),
            desired_speed=max(self.desired_speed, rocket_speed),
            target_id=target.balloon_id,
            target_position=best_target_position,
        )


@dataclass(slots=True)
class EnergyAwareGuidance:
    min_intercept_time: float = 0.3
    max_intercept_time: float = 4.0
    intercept_lead_scale: float = 1.0
    wind_comp_gain: float = 0.15
    climb_bias_altitude_m: float = 1100.0
    climb_vertical_gain: float = 0.65
    climb_velocity_floor_mps: float = 50.0
    desired_speed_floor: float = 1.2

    def command(self, world_state: WorldState, target: BalloonState | None) -> GuidanceCommand:
        if target is None:
            return GuidanceCommand()
        rocket = world_state.rocket
        rocket_speed = max(rocket.velocity.norm(), 1.0)
        relative = target.position - rocket.position
        distance = max(1.0, relative.norm())
        closure_basis = max(rocket_speed + max(target.velocity.norm(), 0.5) * 0.35, 8.0)
        intercept_time = max(
            self.min_intercept_time,
            min(self.max_intercept_time, distance / closure_basis),
        )
        intercept_position = (
            target.position
            + target.velocity.scale(intercept_time * self.intercept_lead_scale)
            + world_state.wind.scale(intercept_time * self.wind_comp_gain)
        )
        target_vector = intercept_position - rocket.position
        direction = target_vector.normalized()
        altitude_agl = float(world_state.metadata.get("altitude_agl_m", rocket.position.z))
        vertical_gap = max(0.0, intercept_position.z - rocket.position.z)
        if altitude_agl < self.climb_bias_altitude_m and (
            vertical_gap > 60.0 or rocket.velocity.z < self.climb_velocity_floor_mps
        ):
            climb_bias = min(
                0.85,
                max(
                    0.15,
                    ((self.climb_bias_altitude_m - altitude_agl) / max(self.climb_bias_altitude_m, 1.0))
                    * self.climb_vertical_gain,
                ),
            )
            direction = _blend_direction(direction, Vector3(z=1.0), climb_bias)
        desired_speed = max(
            self.desired_speed_floor,
            rocket_speed * 0.85 + min(0.5, vertical_gap / 600.0),
        )
        return GuidanceCommand(
            desired_direction=direction,
            desired_speed=desired_speed,
            target_id=target.balloon_id,
            target_position=intercept_position,
        )


def _blend_direction(primary: Vector3, secondary: Vector3, secondary_weight: float) -> Vector3:
    weight = max(0.0, min(0.95, secondary_weight))
    return Vector3(
        x=primary.x * (1.0 - weight) + secondary.x * weight,
        y=primary.y * (1.0 - weight) + secondary.y * weight,
        z=primary.z * (1.0 - weight) + secondary.z * weight,
    ).normalized()


def build_guidance(params: dict[str, Any]):
    mode = str(params.get("guidance_mode", "fixed")).lower()
    if mode in {"fixed", "lookahead"}:
        return FixedLookaheadGuidance(
            lookahead_time=float(params.get("lookahead_time", 1.0)),
            desired_speed=float(params.get("desired_speed", 1.0)),
        )
    if mode in {"predictive", "predictive_intercept"}:
        return PredictiveInterceptGuidance(
            min_intercept_time=float(params.get("min_intercept_time", 0.25)),
            max_intercept_time=float(params.get("max_intercept_time", 4.0)),
            speed_scale=float(params.get("speed_scale", 1.0)),
        )
    if mode in {"short_horizon", "shorthorizon", "mpc_light", "cem", "rl"}:
        return ShortHorizonGuidance(
            horizon_s=float(params.get("horizon_s", params.get("lookahead_time", 1.2))),
            branch_count=int(params.get("branch_count", 5)),
            desired_speed=float(params.get("desired_speed", 1.0)),
        )
    if mode in {"energy_aware", "energy", "ascent_intercept"}:
        return EnergyAwareGuidance(
            min_intercept_time=float(params.get("min_intercept_time", 0.3)),
            max_intercept_time=float(params.get("max_intercept_time", 4.0)),
            intercept_lead_scale=float(params.get("intercept_lead_scale", 1.0)),
            wind_comp_gain=float(params.get("wind_comp_gain", 0.15)),
            climb_bias_altitude_m=float(params.get("climb_bias_altitude_m", 1100.0)),
            climb_vertical_gain=float(params.get("climb_vertical_gain", 0.65)),
            climb_velocity_floor_mps=float(params.get("climb_velocity_floor_mps", 50.0)),
            desired_speed_floor=float(params.get("desired_speed_floor", 1.2)),
        )
    raise ValueError(f"Unknown guidance_mode '{mode}'.")
