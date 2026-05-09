from __future__ import annotations

from pathlib import Path

from rocket_auto_research.gnc.mission_manager import MissionManager, MissionPhase
from rocket_auto_research.gnc.safety_guard import SafetyGuard
from rocket_auto_research.gnc.target_selector import build_target_selector
from rocket_auto_research.learning.behavioral_cloning import load_model, predict_action
from rocket_auto_research.strategies.modular_strategy import ModularStrategy


class RLPolicyWrapperStrategy(ModularStrategy):
    def __init__(self, params=None) -> None:
        merged = {
            "target_selector": "score_based",
            "guidance_mode": "rl",
            "controller_mode": "adaptive",
            "estimator_mode": "alpha_beta",
            "policy_temperature": 0.25,
        }
        if params:
            merged.update(params)
        super().__init__("rl_policy_wrapper", merged)
        self.policy_path = merged.get("policy_path")
        self.learned_policy = None
        if self.policy_path:
            policy_file = Path(self.policy_path)
            if policy_file.exists():
                self.learned_policy = load_model(policy_file)
        self.selector = build_target_selector(merged)
        self.mission = MissionManager(
            launch_wait_time=float(merged.get("launch_wait_time", 0.0)),
            stabilize_duration=float(merged.get("stabilize_duration", 1.0)),
            failsafe_tilt_rate=float(merged.get("failsafe_tilt_rate", 4.0)),
            minimum_intercept_altitude_m=float(merged.get("minimum_intercept_altitude_m", 180.0)),
        )
        self.guard = SafetyGuard(max_tvc=float(merged.get("max_tvc", 1.0)))

    def act(self, world_state):
        if self.learned_policy is None:
            return super().act(world_state)
        estimated = self.estimator.update(world_state)
        phase = self.mission.phase_for(estimated)
        target = self.selector.select(estimated, self.context.current_target_id)
        self.context.current_target_id = target.balloon_id if target is not None else None
        if self.context.current_target_id is not None:
            self.context.metadata["last_target_id"] = self.context.current_target_id
        action = predict_action(self.learned_policy, estimated, target)
        action.launch = phase != MissionPhase.WAIT_FOR_LAUNCH
        target_position = target.position if target is not None else None
        action = self._apply_phase_adjustments(estimated, action, phase, target_position)
        self.context.step_count += 1
        return self.guard.sanitize(action)
