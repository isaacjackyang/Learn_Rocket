import unittest

from rocket_auto_research.auto_research.balloon_challenge_loader import load_balloon_challenge_scenario


class BalloonChallengeLoaderTests(unittest.TestCase):
    def test_load_scenario_one_defaults(self) -> None:
        scenario = load_balloon_challenge_scenario(
            repo_root=".external/BalloonPoppingChallenge",
            scenario_number=1,
        )
        self.assertIsNotNone(scenario)
        assert scenario is not None
        self.assertEqual(scenario.scenario_number, 1)
        self.assertEqual(int(scenario.balloon["num"]), 100)
        self.assertAlmostEqual(float(scenario.balloon["radius"]), 1.5)
        self.assertAlmostEqual(float(scenario.balloon["release_interval"]), 0.5)


if __name__ == "__main__":
    unittest.main()
