import unittest

from rocket_auto_research.auto_research.evaluator import evaluate_episode_set
from rocket_auto_research.auto_research.next_step_planner import NextStepPlanner
from rocket_auto_research.auto_research.problem_definition import default_problem_stages
from rocket_auto_research.auto_research.research_memory import MemoryRecord, ResearchMemory
from rocket_auto_research.auto_research.simulation import EpisodeResult


class StagedAutoResearchTests(unittest.TestCase):
    def test_stage_evaluation_uses_stage_specific_fitness(self) -> None:
        stage = default_problem_stages()[0]
        episodes = [
            EpisodeResult(
                seed=0,
                score=0.0,
                popped=0,
                crashed=False,
                nan_detected=False,
                duration=1.2,
                time_to_launch=0.2,
                time_to_first_pop=None,
                min_distance_to_any_balloon=999.0,
                target_switch_count=0,
                tvc_saturation_ratio=0.0,
                metadata={"score_is_proxy": False},
            )
        ]
        summary = evaluate_episode_set(episodes, stage=stage)
        self.assertEqual(summary.stage_id, "launch_valid")
        self.assertGreater(summary.stage_fitness, 0.0)
        self.assertEqual(summary.stage_success_rate, 1.0)
        self.assertEqual(summary.final_fitness, summary.stage_fitness)

    def test_memory_stage_stats_aggregate_recent_best_records(self) -> None:
        records = [
            MemoryRecord(
                generation=0,
                rank=1,
                experiment_id="a",
                strategy_name="energy_aware",
                stage_id="launch_valid",
                stage_title="Launch Validity",
                final_fitness=10.0,
                stage_fitness=10.0,
                stage_success_rate=0.3,
                mean_score=0.0,
                mean_popped=0.0,
                crash_rate=1.0,
                dominant_failure="crash",
                note="x",
                params={},
                hypothesis_ids=[],
            ),
            MemoryRecord(
                generation=1,
                rank=1,
                experiment_id="b",
                strategy_name="energy_aware",
                stage_id="launch_valid",
                stage_title="Launch Validity",
                final_fitness=20.0,
                stage_fitness=20.0,
                stage_success_rate=0.9,
                mean_score=0.0,
                mean_popped=0.0,
                crash_rate=0.0,
                dominant_failure=None,
                note="y",
                params={},
                hypothesis_ids=[],
            ),
        ]
        stats = ResearchMemory.stage_stats(records, "launch_valid")
        self.assertEqual(stats["best_stage_success_rate"], 0.9)
        self.assertEqual(stats["mean_stage_success_rate"], 0.6)

    def test_planner_stays_on_unpassed_stage(self) -> None:
        planner = NextStepPlanner(default_problem_stages())
        records = [
            MemoryRecord(
                generation=0,
                rank=1,
                experiment_id="a",
                strategy_name="energy_aware",
                stage_id="launch_valid",
                stage_title="Launch Validity",
                final_fitness=10.0,
                stage_fitness=10.0,
                stage_success_rate=0.2,
                mean_score=0.0,
                mean_popped=0.0,
                crash_rate=1.0,
                dominant_failure="crash",
                note="x",
                params={},
                hypothesis_ids=[],
            )
        ]
        plan = planner.plan_generation(
            generation=1,
            records=records,
            fixed_params={"challenge_scenario_number": 0},
            seeds=[1],
            available_strategies=["greedy_intercept", "score_based", "energy_aware"],
            stage_stats_provider=ResearchMemory.stage_stats,
            plateau_detector=ResearchMemory.detect_stage_plateau,
        )
        self.assertEqual(plan.stage_id, "launch_valid")
        self.assertGreater(len(plan.bootstrap_specs), 0)

    def test_plateau_detector_flags_flat_stage(self) -> None:
        records = [
            MemoryRecord(
                generation=index,
                rank=1,
                experiment_id=f"e{index}",
                strategy_name="energy_aware",
                stage_id="approach_window",
                stage_title="Approach Window",
                final_fitness=-70.0,
                stage_fitness=-67.8,
                stage_success_rate=0.0,
                mean_score=0.0,
                mean_popped=0.0,
                crash_rate=1.0,
                dominant_failure="crash",
                note="x",
                params={},
                hypothesis_ids=[],
            )
            for index in range(4)
        ]
        plateau = ResearchMemory.detect_stage_plateau(records, "approach_window")
        self.assertTrue(plateau["is_plateau"])
        self.assertEqual(plateau["mean_crash_rate"], 1.0)

    def test_planner_falls_back_when_stage_plateaus_with_crashes(self) -> None:
        planner = NextStepPlanner(default_problem_stages())
        records = []
        for generation in range(4):
            records.extend(
                [
                    MemoryRecord(
                        generation=0,
                        rank=1,
                        experiment_id="lv",
                        strategy_name="baseline_pid",
                        stage_id="launch_valid",
                        stage_title="Launch Validity",
                        final_fitness=10.0,
                        stage_fitness=10.0,
                        stage_success_rate=1.0,
                        mean_score=0.0,
                        mean_popped=0.0,
                        crash_rate=0.0,
                        dominant_failure=None,
                        note="ok",
                        params={},
                        hypothesis_ids=[],
                    ),
                    MemoryRecord(
                        generation=0,
                        rank=1,
                        experiment_id="as",
                        strategy_name="energy_aware",
                        stage_id="ascent_stable",
                        stage_title="Stable Ascent",
                        final_fitness=20.0,
                        stage_fitness=20.0,
                        stage_success_rate=1.0,
                        mean_score=0.0,
                        mean_popped=0.0,
                        crash_rate=0.0,
                        dominant_failure=None,
                        note="ok",
                        params={},
                        hypothesis_ids=[],
                    ),
                    MemoryRecord(
                        generation=0,
                        rank=1,
                        experiment_id="em",
                        strategy_name="energy_aware",
                        stage_id="energy_margin",
                        stage_title="Energy Margin",
                        final_fitness=30.0,
                        stage_fitness=30.0,
                        stage_success_rate=1.0,
                        mean_score=0.0,
                        mean_popped=0.0,
                        crash_rate=0.0,
                        dominant_failure=None,
                        note="ok",
                        params={},
                        hypothesis_ids=[],
                    ),
                    MemoryRecord(
                        generation=generation,
                        rank=1,
                        experiment_id=f"aw{generation}",
                        strategy_name="energy_aware",
                        stage_id="approach_window",
                        stage_title="Approach Window",
                        final_fitness=-70.0,
                        stage_fitness=-67.8,
                        stage_success_rate=0.0,
                        mean_score=0.0,
                        mean_popped=0.0,
                        crash_rate=1.0,
                        dominant_failure="crash",
                        note="stuck",
                        params={},
                        hypothesis_ids=[],
                    ),
                ]
            )
        plan = planner.plan_generation(
            generation=5,
            records=records,
            fixed_params={"challenge_scenario_number": 0},
            seeds=[1, 2],
            available_strategies=["greedy_intercept", "score_based", "energy_aware", "predictive_intercept"],
            stage_stats_provider=ResearchMemory.stage_stats,
            plateau_detector=ResearchMemory.detect_stage_plateau,
        )
        self.assertEqual(plan.stage_id, "energy_margin")
        self.assertTrue(plan.stuck_mode)
        self.assertGreater(plan.mutation_scale, 1.0)
