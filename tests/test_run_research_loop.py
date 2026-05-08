import json
import tempfile
import unittest
from pathlib import Path

from experiments.run_research_loop import _load_bootstrap_specs


class RunResearchLoopTests(unittest.TestCase):
    def test_load_bootstrap_specs_auto_includes_best_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            best_dir = root / "results" / "best_agents"
            best_dir.mkdir(parents=True, exist_ok=True)
            (best_dir / "best_config.json").write_text(
                json.dumps(
                    {
                        "experiment_id": "best-1",
                        "strategy_name": "energy_aware",
                        "params": {"throttle": 0.9},
                    }
                ),
                encoding="utf-8",
            )

            specs = _load_bootstrap_specs([], repo_root=root, auto_best=True)

            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0].strategy_name, "energy_aware")
            self.assertEqual(specs[0].params["throttle"], 0.9)
            self.assertEqual(specs[0].experiment_id, "best-1")

    def test_load_bootstrap_specs_deduplicates_explicit_best_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            best_dir = root / "results" / "best_agents"
            best_dir.mkdir(parents=True, exist_ok=True)
            best_path = best_dir / "best_config.json"
            best_path.write_text(
                json.dumps(
                    {
                        "experiment_id": "best-2",
                        "strategy_name": "score_based",
                        "params": {"kp": 1.2},
                    }
                ),
                encoding="utf-8",
            )

            specs = _load_bootstrap_specs(
                ["results/best_agents/best_config.json"],
                repo_root=root,
                auto_best=True,
            )

            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0].experiment_id, "best-2")


if __name__ == "__main__":
    unittest.main()
