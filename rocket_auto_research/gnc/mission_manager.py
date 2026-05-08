from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from rocket_auto_research.gnc.state import WorldState


class MissionPhase(StrEnum):
    WAIT_FOR_LAUNCH = "WAIT_FOR_LAUNCH"
    ASCENT_STABILIZE = "ASCENT_STABILIZE"
    TARGET_INTERCEPT = "TARGET_INTERCEPT"
    DESCENT_RECENTER = "DESCENT_RECENTER"
    RECOVERY_OR_FAILSAFE = "RECOVERY_OR_FAILSAFE"


@dataclass(slots=True)
class MissionManager:
    launch_wait_time: float = 0.0
    stabilize_duration: float = 1.5
    failsafe_tilt_rate: float = 4.0
    minimum_intercept_altitude_m: float = 180.0

    def phase_for(self, world_state: WorldState) -> MissionPhase:
        altitude_agl = float(world_state.metadata.get("altitude_agl_m", world_state.rocket.position.z))
        balloons_popped = int(
            world_state.metadata.get(
                "balloons_popped",
                world_state.metadata.get("virtual_waypoints_popped", 0),
            )
        )
        if not world_state.rocket.valid:
            return MissionPhase.RECOVERY_OR_FAILSAFE
        if abs(world_state.rocket.angular_rate.x) > self.failsafe_tilt_rate or abs(world_state.rocket.angular_rate.y) > self.failsafe_tilt_rate:
            return MissionPhase.RECOVERY_OR_FAILSAFE
        if not world_state.rocket.launched and world_state.time_s < self.launch_wait_time:
            return MissionPhase.WAIT_FOR_LAUNCH
        if not world_state.rocket.launched:
            return MissionPhase.TARGET_INTERCEPT
        if world_state.time_s < self.launch_wait_time + self.stabilize_duration:
            return MissionPhase.ASCENT_STABILIZE
        if altitude_agl < self.minimum_intercept_altitude_m and world_state.rocket.velocity.z > 0.0:
            return MissionPhase.ASCENT_STABILIZE
        if world_state.rocket.velocity.z < -5.0 and balloons_popped == 0:
            return MissionPhase.DESCENT_RECENTER
        return MissionPhase.TARGET_INTERCEPT
