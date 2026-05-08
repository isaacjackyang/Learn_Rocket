import unittest

from rocket_auto_research.auto_research.experiment_spec import ExperimentSpec


class ExperimentSpecTests(unittest.TestCase):
    def test_experiment_spec_is_stable(self) -> None:
        left = ExperimentSpec(strategy_name="baseline_pid", params={"kp": 1.0}, seeds=[0, 1], note="x")
        right = ExperimentSpec(strategy_name="baseline_pid", params={"kp": 1.0}, seeds=[0, 1], note="x")
        self.assertEqual(left.experiment_id, right.experiment_id)
