import unittest
import tempfile
from pathlib import Path

from experiments.run_research_loop import _parse_parallel_workers
from rocket_auto_research.auto_research.experiment_runner import ExperimentRunner, _is_parallel_safe_for_spec
from rocket_auto_research.auto_research.experiment_spec import ExperimentSpec
from rocket_auto_research.auto_research.simulation import MockRocketSimAdapter


class ParallelExecutionTests(unittest.TestCase):
    def test_parse_parallel_workers_auto_returns_positive_int(self) -> None:
        value = _parse_parallel_workers("auto")
        self.assertGreaterEqual(value, 1)
        self.assertLessEqual(value, 32)

    def test_parse_parallel_workers_caps_large_values(self) -> None:
        self.assertEqual(_parse_parallel_workers(128), 32)

    def test_competition_platform_is_parallel_safe(self) -> None:
        spec = ExperimentSpec(strategy_name="energy_aware", params={}, seeds=[1, 2])
        self.assertTrue(_is_parallel_safe_for_spec({"adapter": "competition_platform"}, spec))

    def test_balloon_challenge_is_parallel_safe(self) -> None:
        spec = ExperimentSpec(strategy_name="energy_aware", params={}, seeds=[1, 2])
        self.assertTrue(_is_parallel_safe_for_spec({"adapter": "balloon_challenge"}, spec))

    def test_stage_router_uses_profile_safety(self) -> None:
        spec_safe = ExperimentSpec(
            strategy_name="energy_aware",
            params={"simulation_adapter": "competition_platform_stage"},
            seeds=[1, 2],
        )
        spec_unsafe = ExperimentSpec(
            strategy_name="energy_aware",
            params={"simulation_adapter": "balloon_challenge_s0"},
            seeds=[1, 2],
        )
        router_config = {
            "adapter": "stage_router",
            "default_adapter_profile": "competition_platform_stage",
            "adapter_profiles": {
                "competition_platform_stage": {"adapter": "competition_platform"},
                "balloon_challenge_s0": {"adapter": "balloon_challenge"},
            },
        }
        self.assertTrue(_is_parallel_safe_for_spec(router_config, spec_safe))
        self.assertTrue(_is_parallel_safe_for_spec(router_config, spec_unsafe))

    def test_population_parallel_worker_count_uses_safe_specs(self) -> None:
        runner = ExperimentRunner(
            MockRocketSimAdapter(),
            adapter_config={"adapter": "competition_platform"},
            max_workers=4,
        )
        specs = [
            ExperimentSpec(strategy_name="energy_aware", params={}, seeds=[1, 2]),
            ExperimentSpec(strategy_name="score_based", params={}, seeds=[3, 4]),
        ]
        self.assertEqual(runner._parallel_population_worker_count(specs), 2)

    def test_runner_buffers_persistence_until_flush(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir) / "runs"
            runner = ExperimentRunner(
                MockRocketSimAdapter(),
                results_dir=results_dir,
                adapter_config={"adapter": "competition_platform"},
                max_workers=1,
                persist_flush_interval_s=9999,
            )
            spec = ExperimentSpec(strategy_name="energy_aware", params={}, seeds=[1])
            result = runner.run(spec)
            self.assertEqual(result.spec.experiment_id, spec.experiment_id)
            self.assertFalse((results_dir / spec.experiment_id).exists())
            runner.flush_pending(force=True)
            self.assertTrue((results_dir / spec.experiment_id / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
