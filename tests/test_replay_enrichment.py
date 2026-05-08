import unittest

from rocket_auto_research.replay_enrichment import augment_trajectory_with_balloon_snapshots


class ReplayEnrichmentTests(unittest.TestCase):
    def test_competition_platform_enrichment_adds_balloon_snapshots(self) -> None:
        spec_payload = {
            "experiment_id": "exp123",
            "strategy_name": "energy_aware",
            "params": {
                "balloon_count": 10,
                "release_window_s": 5.0,
                "elevation_m": 1400.0,
            },
            "seeds": [7],
        }
        trajectory = [
            {
                "time_s": 4.0,
                "rocket_position": {"x": 0.0, "y": 0.0, "z": 1500.0},
                "rocket_velocity": {"x": 0.0, "y": 0.0, "z": 10.0},
                "target_id": "balloon_000",
                "popped": 0,
                "released": 4,
            }
        ]
        enriched = augment_trajectory_with_balloon_snapshots(spec_payload, "competition_platform", 7, trajectory, top_k=5)
        self.assertEqual(len(enriched), 1)
        self.assertIn("balloons", enriched[0])
        self.assertLessEqual(len(enriched[0]["balloons"]), 5)


if __name__ == "__main__":
    unittest.main()
