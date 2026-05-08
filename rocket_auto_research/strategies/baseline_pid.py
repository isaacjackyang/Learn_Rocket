from __future__ import annotations

from rocket_auto_research.gnc.controller import PursuitController
from rocket_auto_research.gnc.estimator import SimpleEstimator
from rocket_auto_research.gnc.guidance import FixedLookaheadGuidance
from rocket_auto_research.gnc.mission_manager import MissionManager, MissionPhase
from rocket_auto_research.gnc.safety_guard import SafetyGuard
from rocket_auto_research.gnc.target_selector import NearestTargetSelector
from rocket_auto_research.strategies.base import Strategy


class BaselinePIDStrategy(Strategy):
    def __init__(self, params: dict[str, float] | None = None) -> None:
        super().__init__("baseline_pid", params)
        params = params or {}
        self.estimator = SimpleEstimator(alpha=float(params.get("alpha", 0.45)))
        self.selector = NearestTargetSelector()
        self.guidance = FixedLookaheadGuidance(
            lookahead_time=float(params.get("lookahead_time", 0.8)),
            desired_speed=float(params.get("desired_speed", 1.0)),
        )
        self.controller = PursuitController(
            kp=float(params.get("kp", 1.2)),
            kd=float(params.get("kd", 0.15)),
            throttle=float(params.get("throttle", 0.85)),
            max_tvc=float(params.get("max_tvc", 0.6)),
        )
        self.mission = MissionManager(
            launch_wait_time=float(params.get("launch_wait_time", 0.0)),
            stabilize_duration=float(params.get("stabilize_duration", 1.5)),
        )
        self.guard = SafetyGuard(max_tvc=float(params.get("max_tvc", 1.0)))

    def act(self, world_state):
        estimated = self.estimator.update(world_state)
        phase = self.mission.phase_for(estimated)
        target = self.selector.select(estimated, self.context.current_target_id)
        guidance = self.guidance.command(estimated, target)
        self.context.current_target_id = guidance.target_id
        launch = phase != MissionPhase.WAIT_FOR_LAUNCH
        action = self.controller.command(estimated, guidance, launch=launch)
        self.context.step_count += 1
        return self.guard.sanitize(action)
