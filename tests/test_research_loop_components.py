import unittest

from rocket_auto_research.auto_research.experiment_spec import ExperimentSpec
from rocket_auto_research.auto_research.failure_analyzer import FailureReport
from rocket_auto_research.auto_research.hypothesis_generator import ResearchHypothesis, generate_hypotheses
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

    def test_mutation_engine_can_add_ascent_targeting_params_from_hypothesis(self) -> None:
        engine = MutationEngine(
            parameter_space={
                "ascent_targeting_turn_scale": {"min": 0.35, "max": 1.0, "default": 0.72, "sigma": 0.08},
                "ascent_targeting_altitude_m": {"min": 120.0, "max": 900.0, "default": 270.0, "sigma": 45.0},
            },
            mutation_rate=0.0,
            strategy_choices=["energy_aware"],
        )
        parent = ExperimentSpec(
            strategy_name="energy_aware",
            params={},
            seeds=[0, 1],
            note="parent",
        )
        child = engine.mutate(
            parent,
            generation=1,
            seeds=[2, 3],
            hypotheses=[
                ResearchHypothesis(
                    hypothesis_id="early_route_planning",
                    rationale="Start route planning during ascent.",
                    adjustments={
                        "ascent_targeting_turn_scale": 0.1,
                        "ascent_targeting_altitude_m": -80.0,
                    },
                )
            ],
        )
        self.assertIn("ascent_targeting_turn_scale", child.params)
        self.assertIn("ascent_targeting_altitude_m", child.params)
        self.assertGreater(child.params["ascent_targeting_turn_scale"], 0.72)
        self.assertLess(child.params["ascent_targeting_altitude_m"], 270.0)

    def test_mutation_engine_avoids_blocked_region(self) -> None:
        engine = MutationEngine(
            parameter_space={
                "kp": {"min": 0.1, "max": 1.0, "sigma": 0.05},
                "lookahead_time": {"min": 0.5, "max": 2.5, "sigma": 0.1},
            },
            mutation_rate=1.0,
            strategy_choices=["score_based"],
        )
        blocked_region = {
            "strategy_name": "score_based",
            "numeric": {
                "kp": {"min": 0.35, "max": 0.65},
                "lookahead_time": {"min": 1.0, "max": 1.4},
            },
            "categorical": {},
        }
        parent = ExperimentSpec(
            strategy_name="score_based",
            params={"kp": 0.5, "lookahead_time": 1.2},
            seeds=[0, 1],
            note="parent",
        )
        child = engine.mutate(
            parent,
            generation=1,
            seeds=[2, 3],
            hypotheses=None,
            blocked_regions=[blocked_region],
        )
        self.assertFalse(engine.is_blocked(child.strategy_name, child.params, [blocked_region]))

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

    def test_hypotheses_bias_intercept_timing_toward_earlier_ascent_targeting(self) -> None:
        report = FailureReport(
            counts={"near_miss": 3},
            rates={"near_miss": 0.3},
            dominant_failure="near_miss",
            notes=[],
        )
        hypotheses = generate_hypotheses(report)
        target_hypothesis = next(h for h in hypotheses if h.hypothesis_id == "improve_intercept_timing")
        self.assertIn("ascent_targeting_turn_scale", target_hypothesis.adjustments)
        self.assertIn("ascent_targeting_altitude_m", target_hypothesis.adjustments)
