import unittest

from rocket_auto_research.auto_research.competition_platform_api import SimulatedCompetitionEnv
from rocket_auto_research.auto_research.experiment_spec import ExperimentSpec


class CompetitionPlatformApiTests(unittest.TestCase):
    def test_reset_and_step_contract(self) -> None:
        spec = ExperimentSpec(
            strategy_name="energy_aware",
            params={"balloon_count": 12, "api_max_time_s": 1.0},
            seeds=[0],
            note="competition_api_test",
        )
        env = SimulatedCompetitionEnv(spec, seed=0)
        observation, info = env.reset()
        self.assertIn("simulation_time", observation)
        self.assertIn("balloon_status", observation)
        self.assertIn("balloon_states", observation)
        self.assertIn("rocket_sensors", observation)
        self.assertEqual(info["adapter"], "competition_platform")
        next_observation, reward, terminated, truncated, step_info = env.step(
            {
                "launch": True,
                "launch_inclination_heading": [90.0, 0.0],
                "tvc": [0.0, 0.0],
                "throttle": 0.8,
                "roll": 0.0,
            }
        )
        self.assertIn("balloon_status", next_observation)
        self.assertIn("rocket_states", step_info)
        self.assertIsInstance(reward, float)
        self.assertIn("tvc_saturation_ratio", step_info)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)

    def test_challenge_scenario_defaults_apply(self) -> None:
        spec = ExperimentSpec(
            strategy_name="energy_aware",
            params={"challenge_repo_root": ".external/BalloonPoppingChallenge", "challenge_scenario_number": 1},
            seeds=[0],
            note="challenge_defaults_test",
        )
        env = SimulatedCompetitionEnv(spec, seed=0)
        observation, _ = env.reset()
        self.assertEqual(len(observation["balloon_status"]), 100)
        self.assertAlmostEqual(env.balloon_release_interval_s, 0.5)
        self.assertEqual(observation["metadata"]["challenge_scenario_number"], 1)
