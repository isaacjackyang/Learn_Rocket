from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rocket_auto_research.gnc.state import ControlAction, GuidanceCommand, WorldState


@dataclass(slots=True)
class PursuitController:
    kp: float = 1.4
    kd: float = 0.15
    throttle: float = 0.9
    max_tvc: float = 1.0

    def command(self, world_state: WorldState, guidance: GuidanceCommand, launch: bool) -> ControlAction:
        velocity = world_state.rocket.velocity
        error = guidance.desired_direction - velocity.normalized()
        damping_x = world_state.rocket.angular_rate.y * self.kd
        damping_y = world_state.rocket.angular_rate.x * self.kd
        return ControlAction(
            launch=launch,
            throttle=self.throttle if launch else 0.0,
            tvc_x=max(-self.max_tvc, min(self.max_tvc, self.kp * error.x - damping_x)),
            tvc_y=max(-self.max_tvc, min(self.max_tvc, self.kp * error.y - damping_y)),
            roll=0.0,
        )


@dataclass(slots=True)
class AdaptivePursuitController:
    kp: float = 1.1
    kd: float = 0.15
    throttle: float = 0.9
    max_tvc: float = 1.0
    throttle_floor: float = 0.35

    def command(self, world_state: WorldState, guidance: GuidanceCommand, launch: bool) -> ControlAction:
        velocity = world_state.rocket.velocity
        error = guidance.desired_direction - velocity.normalized()
        altitude_agl = float(world_state.metadata.get("altitude_agl_m", world_state.rocket.position.z))
        agility_scale = 0.7 if altitude_agl < 150 else 1.0
        throttle = self.throttle_floor + (self.throttle - self.throttle_floor) * min(1.0, guidance.desired_speed / 50.0)
        return ControlAction(
            launch=launch,
            throttle=throttle if launch else 0.0,
            tvc_x=max(-self.max_tvc, min(self.max_tvc, agility_scale * self.kp * error.x - self.kd * world_state.rocket.angular_rate.y)),
            tvc_y=max(-self.max_tvc, min(self.max_tvc, agility_scale * self.kp * error.y - self.kd * world_state.rocket.angular_rate.x)),
            roll=max(-1.0, min(1.0, -0.2 * world_state.rocket.angular_rate.z)),
        )


def build_controller(params: dict[str, Any]):
    mode = str(params.get("controller_mode", "pursuit")).lower()
    common = dict(
        kp=float(params.get("kp", 1.2)),
        kd=float(params.get("kd", 0.15)),
        throttle=float(params.get("throttle", 0.9)),
        max_tvc=float(params.get("max_tvc", 1.0)),
    )
    if mode == "pursuit":
        return PursuitController(**common)
    if mode in {"adaptive", "adaptive_pursuit"}:
        return AdaptivePursuitController(
            **common,
            throttle_floor=float(params.get("throttle_floor", 0.35)),
        )
    raise ValueError(f"Unknown controller_mode '{mode}'.")
