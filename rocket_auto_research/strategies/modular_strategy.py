from __future__ import annotations

from typing import Any

from rocket_auto_research.gnc.controller import build_controller
from rocket_auto_research.gnc.estimator import build_estimator
from rocket_auto_research.gnc.guidance import build_guidance
from rocket_auto_research.gnc.mission_manager import MissionManager, MissionPhase
from rocket_auto_research.gnc.safety_guard import SafetyGuard
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
        self.ascent_throttle_floor = float(params.get("ascent_throttle_floor", 0.82))
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
        if phase == MissionPhase.ASCENT_STABILIZE:
            action.tvc_x *= self.ascent_turn_scale
            action.tvc_y *= self.ascent_turn_scale
            action.throttle = max(action.throttle, self.ascent_throttle_floor)
        if phase == MissionPhase.DESCENT_RECENTER:
            action.tvc_x *= self.recenter_turn_scale
            action.tvc_y *= self.recenter_turn_scale
        if phase == MissionPhase.RECOVERY_OR_FAILSAFE:
            action.tvc_x = 0.0
            action.tvc_y = 0.0
            action.roll = 0.0
            action.throttle *= 0.5
        self.context.step_count += 1
        return self.guard.sanitize(action)
