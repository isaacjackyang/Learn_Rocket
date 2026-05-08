import tempfile
import threading
import unittest
from pathlib import Path

from rocket_auto_research.auto_research.runtime_control import ResearchRuntimeControl, ResearchStopRequested


class RuntimeControlTests(unittest.TestCase):
    def test_begin_run_marks_continuous_when_total_generations_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            control = ResearchRuntimeControl(Path(temp_dir), poll_interval_s=0.01)
            control.begin_run(config_path="configs/auto_research.yaml", total_generations=None, population_size=6)
            status = control.read_status()
            self.assertTrue(status.get("continuous"))
            self.assertIsNone(status.get("total_generations"))

    def test_pause_and_resume_cycle_updates_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            control = ResearchRuntimeControl(Path(temp_dir), poll_interval_s=0.01)
            control.begin_run(config_path="configs/search_space.yaml", total_generations=3, population_size=8)
            control.request_pause()
            threading.Timer(0.05, control.clear_command).start()
            control.wait_if_paused(allow_stop=False)
            self.assertEqual(control.read_status().get("status"), "running")

    def test_stop_request_raises(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            control = ResearchRuntimeControl(Path(temp_dir), poll_interval_s=0.01)
            control.begin_run(config_path="configs/search_space.yaml", total_generations=3, population_size=8)
            control.request_stop()
            with self.assertRaises(ResearchStopRequested):
                control.check_stop_requested()

    def test_reader_sees_status_updates_from_separate_instance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            control_dir = Path(temp_dir)
            writer = ResearchRuntimeControl(control_dir, poll_interval_s=0.01, min_status_write_interval_s=0.0)
            reader = ResearchRuntimeControl(control_dir, poll_interval_s=0.01)
            writer.begin_run(config_path="configs/auto_research.yaml", total_generations=None, population_size=6)
            writer.update_status(message="phase-1", force=True)
            self.assertEqual(reader.read_status().get("message"), "phase-1")
            writer.update_status(message="phase-2", force=True)
            self.assertEqual(reader.read_status().get("message"), "phase-2")


if __name__ == "__main__":
    unittest.main()
