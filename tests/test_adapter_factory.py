import unittest

from rocket_auto_research.auto_research.adapter_factory import build_simulation_adapter
from rocket_auto_research.auto_research.experiment_spec import ExperimentSpec


class AdapterFactoryTests(unittest.TestCase):
    def test_stage_router_uses_profile_from_spec_params(self) -> None:
        adapter = build_simulation_adapter(
            {
                "adapter": "stage_router",
                "default_adapter_profile": "mock_a",
                "adapter_profiles": {
                    "mock_a": {"adapter": "mock"},
                    "mock_b": {"adapter": "mock"},
                },
            }
        )
        spec = ExperimentSpec(
            strategy_name="greedy_intercept",
            params={
                "simulation_adapter": "mock_b",
                "kp": 0.8,
                "kd": 0.2,
                "lookahead_time": 1.2,
                "throttle": 0.8,
                "target_angle_weight": 0.6,
                "switching_penalty": 0.3,
                "launch_wait_time": 0.0,
            },
            seeds=[0],
            note="router_test",
        )
        result = adapter.run_episode(spec, 0)
        self.assertIn("mock_adapter", result.metadata)
        self.assertTrue(result.metadata["mock_adapter"])

