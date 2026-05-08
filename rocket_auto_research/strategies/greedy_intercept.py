from __future__ import annotations

from rocket_auto_research.gnc.controller import PursuitController
from rocket_auto_research.gnc.estimator import SimpleEstimator
from rocket_auto_research.gnc.guidance import FixedLookaheadGuidance
from rocket_auto_research.gnc.mission_manager import MissionManager, MissionPhase
from rocket_auto_research.gnc.safety_guard import SafetyGuard
from rocket_auto_research.gnc.target_selector import ScoreBasedTargetSelector
from rocket_auto_research.strategies.base import Strategy


class GreedyInterceptStrategy(Strategy):
    def __init__(self, params: dict[str, float] | None = None) -> None:
        super().__init__("greedy_intercept", params)
        params = params or {}
        self.estimator = SimpleEstimator(alpha=float(params.get("alpha", 0.55)))
        self.selector = ScoreBasedTargetSelector(
            distance_weight=float(params.get("target_distance_weight", 1.0)),
            angle_weight=float(params.get("target_angle_weight", 0.7)),
            height_weight=float(params.get("target_height_weight", 0.25)),
            switching_penalty=float(params.get("switching_penalty", 0.6)),
        )
        self.guidance = FixedLookaheadGuidance(
            lookahead_time=float(params.get("lookahead_time", 1.0)),
            desired_speed=float(params.get("desired_speed", 1.1)),
        )
        self.controller = PursuitController(
            kp=float(params.get("kp", 1.45)),
            kd=float(params.get("kd", 0.18)),
            throttle=float(params.get("throttle", 0.9)),
            max_tvc=float(params.get("max_tvc", 0.8)),
        )
        self.mission = MissionManager(
            launch_wait_time=float(params.get("launch_wait_time", 0.1)),
            stabilize_duration=float(params.get("stabilize_duration", 1.0)),
        )
        self.guard = SafetyGuard(max_tvc=float(params.get("max_tvc", 1.0)))

    def act(self, world_state):
        estimated = self.estimator.update(world_state)
        phase = self.mission.phase_for(estimated)
        target = self.selector.select(estimated, self.context.current_target_id)
        guidance = self.guidance.command(estimated, target)
        self.context.current_target_id = guidance.target_id
        action = self.controller.command(estimated, guidance, launch=phase != MissionPhase.WAIT_FOR_LAUNCH)
        self.context.step_count += 1
        return self.guard.sanitize(action)
