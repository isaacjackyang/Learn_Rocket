import unittest

from rocket_auto_research.auto_research.experiment_spec import ExperimentSpec
from rocket_auto_research.auto_research.failure_analyzer import FailureReport
from rocket_auto_research.auto_research.hypothesis_generator import generate_hypotheses
from rocket_auto_research.auto_research.mutation_engine import MutationEngine
from rocket_auto_research.auto_research.strategy_crossover import StrategyCrossover


class ResearchLoopComponentTests(unittest.TestCase):
    def test_hypotheses_include_target_lock_for_chattering(self) -> None:
        report = FailureReport(
            counts={"target_chattering": 4},
            rates={"target_chattering": 0.4},
            dominant_failure="target_chattering",
            notes=[],
        )
        hypotheses = generate_hypotheses(report)
        self.assertTrue(any(h.hypothesis_id == "stabilize_target_lock" for h in hypotheses))

    def test_mutation_engine_supports_choice_parameters(self) -> None:
        engine = MutationEngine(
            parameter_space={
                "guidance_mode": {"choices": ["fixed", "predictive", "short_horizon"]},
                "kp": {"min": 0.1, "max": 1.0, "sigma": 0.1},
            },
            mutation_rate=1.0,
            strategy_choices=["score_based"],
            fixed_params={"challenge_scenario_number": 1},
        )
        parent = ExperimentSpec(
            strategy_name="score_based",
            params={"guidance_mode": "fixed", "kp": 0.5},
            seeds=[0, 1],
            note="parent",
        )
        child = engine.mutate(parent, generation=1, seeds=[2, 3], hypotheses=None)
        self.assertIn(child.params["guidance_mode"], ["fixed", "predictive", "short_horizon"])
        self.assertEqual(child.params["challenge_scenario_number"], 1)
        self.assertNotEqual(child.experiment_id, parent.experiment_id)

    def test_crossover_mixes_module_level_settings(self) -> None:
        crossover = StrategyCrossover()
        left = ExperimentSpec(
            strategy_name="score_based",
            params={"target_selector": "score_based", "guidance_mode": "fixed", "kp": 0.4},
            seeds=[0],
            note="left",
        )
        right = ExperimentSpec(
            strategy_name="mpc_light",
            params={"target_selector": "reachable", "guidance_mode": "short_horizon", "branch_count": 7},
            seeds=[0],
            note="right",
        )
        child = crossover.crossover(left, right, generation=1, seeds=[1, 2])
        self.assertIn(child.params["target_selector"], {"score_based", "reachable"})
        self.assertIn(child.params["guidance_mode"], {"fixed", "short_horizon"})
