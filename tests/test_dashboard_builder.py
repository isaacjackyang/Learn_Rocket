import json
import tempfile
import unittest
from pathlib import Path

from rocket_auto_research.dashboard_builder import build_dashboard


class DashboardBuilderTests(unittest.TestCase):
    def test_build_dashboard_generates_index_and_trajectory_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            results_dir = root / "results"
            (results_dir / "best_agents").mkdir(parents=True)
            (results_dir / "leaderboards").mkdir(parents=True)
            (results_dir / "reports").mkdir(parents=True)
            run_dir = results_dir / "runs" / "exp123"
            run_dir.mkdir(parents=True)

            (results_dir / "best_agents" / "best_config.json").write_text(
                json.dumps({"experiment_id": "exp123", "strategy_name": "energy_aware", "final_fitness": 123.4}),
                encoding="utf-8",
            )
            (results_dir / "leaderboards" / "generation_0000.json").write_text(
                json.dumps(
                    [
                        {
                            "rank": 1,
                            "experiment_id": "exp123",
                            "strategy_name": "energy_aware",
                            "final_fitness": 123.4,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (results_dir / "reports" / "generation_0000.md").write_text("# Report\n\nPreview text", encoding="utf-8")
            (run_dir / "spec.json").write_text(
                json.dumps({"experiment_id": "exp123", "strategy_name": "energy_aware", "params": {}, "seeds": [7]}),
                encoding="utf-8",
            )
            (run_dir / "summary.json").write_text(
                json.dumps({"final_fitness": 123.4, "mean_popped": 4.0, "mean_score": 456.7, "mean_target_switches": 2.0}),
                encoding="utf-8",
            )
            (run_dir / "failure_report.json").write_text(
                json.dumps({"dominant_failure": "near_miss", "notes": ["Timing drift observed."]}),
                encoding="utf-8",
            )
            (run_dir / "trajectory_seed_007.jsonl").write_text(
                json.dumps(
                    {
                        "summary": {"seed": 7, "score": 456.7, "popped": 4, "duration": 10.0, "target_switch_count": 2},
                        "metadata": {"adapter": "competition_platform"},
                        "trajectory": [
                            {
                                "time_s": 0.0,
                                "rocket_position": {"x": 0.0, "y": 0.0, "z": 100.0},
                                "rocket_velocity": {"x": 0.0, "y": 0.0, "z": 1.0},
                                "target_id": "balloon_0",
                                "popped": 0,
                                "released": 1,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            index_path = build_dashboard(results_dir=results_dir, output_dir=results_dir / "dashboard")

            self.assertTrue(index_path.exists())
            self.assertTrue((results_dir / "dashboard" / "app.js").exists())
            self.assertTrue((results_dir / "dashboard" / "styles.css").exists())
            self.assertTrue((results_dir / "dashboard" / "data" / "index.js").exists())
            self.assertTrue((results_dir / "dashboard" / "data" / "trajectories" / "exp123_7.js").exists())
            index_html = index_path.read_text(encoding="utf-8")
            data_js = (results_dir / "dashboard" / "data" / "index.js").read_text(encoding="utf-8")
            app_js = (results_dir / "dashboard" / "app.js").read_text(encoding="utf-8")
            self.assertIn("自動研究控制台", index_html)
            self.assertIn("theme-select", index_html)
            self.assertIn("worker-count", index_html)
            self.assertIn("progress-chart", index_html)
            self.assertIn("progress-window", index_html)
            self.assertIn("progress-metric", index_html)
            self.assertIn("/api/start", app_js)
            self.assertIn("parallel_workers", app_js)
            self.assertIn("research_progress", data_js)
            self.assertIn("running_best_popped", data_js)
            self.assertIn("annotation_fitness", data_js)
            self.assertIn("research-status-pill", index_html)


if __name__ == "__main__":
    unittest.main()
