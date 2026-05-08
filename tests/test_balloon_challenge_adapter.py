import unittest
from pathlib import Path

import numpy as np

from rocket_auto_research.auto_research.balloon_challenge_adapter import (
    BalloonChallengeSimulationAdapter,
)
from rocket_auto_research.auto_research.experiment_spec import ExperimentSpec


class BalloonChallengeAdapterTests(unittest.TestCase):
    def test_popped_count_handles_numpy_status_rows(self) -> None:
        observation = {
            "balloon_status": np.array([[0], [1], [2], [2]], dtype=int),
        }
        self.assertEqual(BalloonChallengeSimulationAdapter._popped_count(observation), 2)

    def test_run_episode_smoke_on_real_env(self) -> None:
        if not Path(".external/BalloonPoppingChallenge").exists():
            raise unittest.SkipTest("External BalloonPoppingChallenge dependency is not installed.")
        spec = ExperimentSpec(
            strategy_name="greedy_intercept",
            params={
                "challenge_repo_root": ".external/BalloonPoppingChallenge",
                "challenge_scenario_number": 0,
                "lookahead_time": 1.5,
                "kp": 1.2,
                "kd": 0.3,
                "throttle": 0.8,
                "max_tvc": 10.0,
                "target_distance_weight": 1.0,
                "target_angle_weight": 0.6,
                "launch_wait_time": 0.0,
                "launch_inclination_deg": 88.0,
                "launch_heading_deg": 0.0,
            },
            seeds=[7],
            note="balloon_challenge_smoke",
        )
        adapter = BalloonChallengeSimulationAdapter(repo_root=".external/BalloonPoppingChallenge", scenario_number=0)
        result = adapter.run_episode(spec=spec, seed=7)
        self.assertEqual(result.metadata["adapter"], "balloon_challenge")
        self.assertEqual(result.metadata["score_is_proxy"], False)
        self.assertGreaterEqual(result.metadata["balloons_total"], 1)
        self.assertGreater(len(result.metadata["trajectory"]), 0)
