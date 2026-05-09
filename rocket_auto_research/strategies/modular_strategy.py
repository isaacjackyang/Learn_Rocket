from __future__ import annotations

from math import sqrt
from typing import Any

from rocket_auto_research.gnc.controller import build_controller
from rocket_auto_research.gnc.estimator import build_estimator
from rocket_auto_research.gnc.guidance import build_guidance
from rocket_auto_research.gnc.mission_manager import MissionManager, MissionPhase
from rocket_auto_research.gnc.safety_guard import SafetyGuard
from rocket_auto_research.gnc.state import ControlAction, Vector3
from rocket_auto_research.gnc.target_selector import build_target_selector
from rocket_auto_research.strategies.base import Strategy


class ModularStrategy(Strategy):
    def __init__(self, name: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(name, params)
        params = params or {}
        self.estimator = build_estimator(params)
        self.selector = build_target_selector(params)
        self.guidance = build_guidance(params)
        self.controller = build_controller(params)
        self.mission = MissionManager(
            launch_wait_time=float(params.get("launch_wait_time", 0.0)),
            stabilize_duration=float(params.get("stabilize_duration", 1.0)),
            failsafe_tilt_rate=float(params.get("failsafe_tilt_rate", 4.0)),
            minimum_intercept_altitude_m=float(params.get("minimum_intercept_altitude_m", 180.0)),
        )
        self.guard = SafetyGuard(max_tvc=float(params.get("max_tvc", 1.0)))
        self.ascent_turn_scale = float(params.get("ascent_turn_scale", 0.35))
        self.ascent_targeting_turn_scale = max(
            self.ascent_turn_scale,
            float(params.get("ascent_targeting_turn_scale", 0.72)),
        )
        self.ascent_throttle_floor = float(params.get("ascent_throttle_floor", 0.82))
        self.ascent_targeting_altitude_m = max(
            1.0,
            float(
                params.get(
                    "ascent_targeting_altitude_m",
                    max(120.0, float(params.get("minimum_intercept_altitude_m", 180.0)) * 1.5),
                )
            ),
        )
        self.recenter_turn_scale = float(params.get("recenter_turn_scale", 0.2))

    def act(self, world_state):
        estimated = self.estimator.update(world_state)
        phase = self.mission.phase_for(estimated)
        target = self.selector.select(estimated, self.context.current_target_id)
        guidance = self.guidance.command(estimated, target)
        self.context.current_target_id = guidance.target_id
        if guidance.target_id is not None:
            self.context.metadata["last_target_id"] = guidance.target_id
        action = self.controller.command(
            estimated,
            guidance,
            launch=phase != MissionPhase.WAIT_FOR_LAUNCH,
        )
        action = self._apply_phase_adjustments(estimated, action, phase, guidance.target_position)
        self.context.step_count += 1
        return self.guard.sanitize(action)

    def _apply_phase_adjustments(
        self,
        world_state,
        action: ControlAction,
        phase: MissionPhase,
        target_position: Vector3 | None = None,
    ) -> ControlAction:
        if phase == MissionPhase.ASCENT_STABILIZE:
            turn_scale = self._ascent_turn_scale_for(world_state, target_position)
            action.tvc_x *= turn_scale
            action.tvc_y *= turn_scale
            action.throttle = max(action.throttle, self.ascent_throttle_floor)
        if phase == MissionPhase.DESCENT_RECENTER:
            action.tvc_x *= self.recenter_turn_scale
            action.tvc_y *= self.recenter_turn_scale
        if phase == MissionPhase.RECOVERY_OR_FAILSAFE:
            action.tvc_x = 0.0
            action.tvc_y = 0.0
            action.roll = 0.0
            action.throttle *= 0.5
        return action

    def _ascent_turn_scale_for(self, world_state, target_position: Vector3 | None) -> float:
        base_scale = self.ascent_turn_scale
        max_scale = self.ascent_targeting_turn_scale
        if target_position is None or max_scale <= base_scale:
            return base_scale
        target_vector = target_position - world_state.rocket.position
        target_distance = target_vector.norm()
        if target_distance <= 1e-6:
            return max_scale
        altitude_agl = float(world_state.metadata.get("altitude_agl_m", world_state.rocket.position.z))
        altitude_progress = _clamp01(altitude_agl / self.ascent_targeting_altitude_m)
        horizontal_distance = sqrt(target_vector.x**2 + target_vector.y**2)
        lateral_ratio = horizontal_distance / target_distance
        planning_progress = _clamp01(max(altitude_progress, lateral_ratio * 0.35 + altitude_progress * 0.65))
        return base_scale + (max_scale - base_scale) * planning_progress


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
