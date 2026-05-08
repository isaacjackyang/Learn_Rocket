import unittest

from rocket_auto_research.auto_research.evaluator import evaluate_episode_set
from rocket_auto_research.auto_research.simulation import EpisodeResult


class EvaluatorTests(unittest.TestCase):
    def test_evaluator_penalizes_crashes_and_nan(self) -> None:
        episodes = [
            EpisodeResult(
                seed=0,
                score=40.0,
                popped=20,
                crashed=False,
                nan_detected=False,
                duration=40.0,
                time_to_launch=0.0,
                time_to_first_pop=5.0,
                min_distance_to_any_balloon=1.0,
                target_switch_count=3,
                tvc_saturation_ratio=0.3,
                metadata={"score_is_proxy": True, "wind_drift_m": 20.0},
            ),
            EpisodeResult(
                seed=1,
                score=40.0,
                popped=20,
                crashed=True,
                nan_detected=True,
                duration=40.0,
                time_to_launch=0.0,
                time_to_first_pop=5.0,
                min_distance_to_any_balloon=1.0,
                target_switch_count=3,
                tvc_saturation_ratio=0.3,
                metadata={"score_is_proxy": True, "wind_drift_m": 20.0},
            ),
        ]
        summary = evaluate_episode_set(episodes)
        self.assertEqual(summary.mean_score, 40.0)
        self.assertEqual(summary.crash_rate, 0.5)
        self.assertEqual(summary.nan_rate, 0.5)
        self.assertTrue(summary.score_is_proxy)
        self.assertLess(summary.robustness_score, summary.mean_score)
