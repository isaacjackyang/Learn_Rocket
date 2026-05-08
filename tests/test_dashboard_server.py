import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rocket_auto_research.dashboard_server import ResearchDashboardManager, describe_config_entry


class DashboardServerTests(unittest.TestCase):
    def test_describe_config_entry_groups_search_space_configs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "configs" / "auto_research.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text("", encoding="utf-8")
            entry = describe_config_entry(config_path, root)
            self.assertEqual(entry["mode"], "auto_research_loop")
            self.assertEqual(entry["group"], "Auto Research Loops")

    def test_describe_config_entry_groups_single_strategy_configs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "configs" / "energy_aware.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text("", encoding="utf-8")
            entry = describe_config_entry(config_path, root)
            self.assertEqual(entry["mode"], "single_strategy_run")
            self.assertEqual(entry["group"], "Single Strategy Runs")

    def test_start_returns_status_payload_without_deadlock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "configs" / "auto_research.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text("strategies: []\n", encoding="utf-8")
            fake_dashboard = root / "results" / "dashboard" / "index.html"
            fake_dashboard.parent.mkdir(parents=True, exist_ok=True)
            fake_dashboard.write_text("<html></html>", encoding="utf-8")

            with patch.object(ResearchDashboardManager, "build_dashboard", return_value=fake_dashboard):
                with patch("rocket_auto_research.dashboard_server.subprocess.Popen") as popen_mock:
                    process = popen_mock.return_value
                    process.pid = 12345
                    process.poll.return_value = None

                    manager = ResearchDashboardManager(repo_root=root)
                    status = manager.start("configs/auto_research.yaml", parallel_workers=8)
                    if manager._log_handle is not None:
                        manager._log_handle.close()
                        manager._log_handle = None

            self.assertEqual(status["status"], "starting")
            self.assertTrue(status["running"])
            self.assertEqual(status["config_mode"], "auto_research_loop")
            self.assertEqual(status["pid"], 12345)
            launch_cmd = popen_mock.call_args.args[0]
            self.assertIn("--parallel-workers", launch_cmd)
            self.assertIn("8", launch_cmd)
            self.assertEqual(status["configured_workers"], 8)

    def test_status_uses_pid_fallback_when_process_handle_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_dashboard = root / "results" / "dashboard" / "index.html"
            fake_dashboard.parent.mkdir(parents=True, exist_ok=True)
            fake_dashboard.write_text("<html></html>", encoding="utf-8")

            with patch.object(ResearchDashboardManager, "build_dashboard", return_value=fake_dashboard):
                with patch("rocket_auto_research.dashboard_server.subprocess.run") as run_mock:
                    run_mock.return_value.stdout = '"python.exe","4321","Console","1","12,000 K"\n'
                    manager = ResearchDashboardManager(repo_root=root)
                    manager.control.update_status(status="running", message="running", pid=4321)
                    status = manager.status()

            self.assertTrue(status["running"])
            self.assertEqual(status["status"], "running")

    def test_list_configs_only_exposes_root_configs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "configs").mkdir(parents=True, exist_ok=True)
            (root / "configs" / "legacy").mkdir(parents=True, exist_ok=True)
            (root / "configs" / "auto_research.yaml").write_text("", encoding="utf-8")
            (root / "configs" / "energy_aware.yaml").write_text("", encoding="utf-8")
            (root / "configs" / "legacy" / "search_space.yaml").write_text("", encoding="utf-8")
            fake_dashboard = root / "results" / "dashboard" / "index.html"
            fake_dashboard.parent.mkdir(parents=True, exist_ok=True)
            fake_dashboard.write_text("<html></html>", encoding="utf-8")

            with patch.object(ResearchDashboardManager, "build_dashboard", return_value=fake_dashboard):
                manager = ResearchDashboardManager(repo_root=root)
                values = [entry["value"] for entry in manager.list_configs()]

            self.assertIn("configs/auto_research.yaml", values)
            self.assertIn("configs/energy_aware.yaml", values)
            self.assertNotIn("configs/legacy/search_space.yaml", values)

    def test_status_includes_stage_context_from_latest_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_dashboard = root / "results" / "dashboard" / "index.html"
            fake_dashboard.parent.mkdir(parents=True, exist_ok=True)
            fake_dashboard.write_text("<html></html>", encoding="utf-8")
            latest_plan = root / "results" / "research_memory" / "latest_plan.json"
            latest_plan.parent.mkdir(parents=True, exist_ok=True)
            latest_plan.write_text(
                '{"stage_id":"energy_margin","recent_stage_success_rate":0.5,"rationale":"test rationale","planner_notes":["n1"]}',
                encoding="utf-8",
            )
            run_dir = root / "results" / "runs" / "exp-1"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "failure_report.json").write_text(
                '{"counts":{"crash":2,"altitude_shortfall":1},"rates":{"crash":1.0,"altitude_shortfall":0.5}}',
                encoding="utf-8",
            )

            with patch.object(ResearchDashboardManager, "build_dashboard", return_value=fake_dashboard):
                manager = ResearchDashboardManager(repo_root=root)
                manager.control.update_status(
                    status="running",
                    current_stage="energy_margin",
                    current_experiment_id="exp-1",
                    message="running",
                )
                status = manager.status()

            stage_context = status["stage_context"]
            self.assertEqual(stage_context["stage_id"], "energy_margin")
            self.assertEqual(stage_context["stage_title"], "Energy Margin")
            self.assertEqual(stage_context["recent_stage_success_rate"], 0.5)
            self.assertEqual(stage_context["next_stage_id"], "approach_window")
            self.assertEqual(stage_context["promotion_decision"], "stay")
            self.assertTrue(stage_context["pipeline"])
            self.assertEqual(stage_context["failure_breakdowns"]["current_experiment"][0]["name"], "crash")

    def test_status_exposes_worker_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fake_dashboard = root / "results" / "dashboard" / "index.html"
            fake_dashboard.parent.mkdir(parents=True, exist_ok=True)
            fake_dashboard.write_text("<html></html>", encoding="utf-8")

            with patch.object(ResearchDashboardManager, "build_dashboard", return_value=fake_dashboard):
                manager = ResearchDashboardManager(repo_root=root)
                status = manager.status()

            self.assertEqual(status["worker_limits"]["min"], 1)
            self.assertEqual(status["worker_limits"]["max"], 32)


if __name__ == "__main__":
    unittest.main()
