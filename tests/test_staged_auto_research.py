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
            parameter_space={},
            base_population_size=6,
            base_seeds_per_experiment=2,
            stage_stats_provider=ResearchMemory.stage_stats,
            plateau_detector=ResearchMemory.detect_stage_plateau,
            plateau_streak_provider=ResearchMemory.plateau_streak,
            blocked_region_provider=ResearchMemory.build_blocked_regions,
            reentry_boost_provider=ResearchMemory.reentry_boost_state,
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

    def test_planner_falls_back_with_hard_reset_when_approach_window_repeatedly_plateaus(self) -> None:
        planner = NextStepPlanner(default_problem_stages())
        records = []
        for generation in range(5):
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
                        strategy_name="predictive_intercept",
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
                        params={
                            "kp": 0.8,
                            "kd": 0.3,
                            "lookahead_time": 1.55,
                            "target_distance_weight": 1.3,
                            "target_angle_weight": 0.8,
                            "switching_penalty": 1.6,
                        },
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
            parameter_space={
                "kp": {"min": 0.1, "max": 2.0, "sigma": 0.1},
                "kd": {"min": 0.0, "max": 1.0, "sigma": 0.05},
                "lookahead_time": {"min": 0.5, "max": 2.5, "sigma": 0.1},
                "target_distance_weight": {"min": 0.0, "max": 2.0, "sigma": 0.1},
                "target_angle_weight": {"min": 0.0, "max": 2.0, "sigma": 0.1},
                "switching_penalty": {"min": 0.0, "max": 3.0, "sigma": 0.1},
            },
            base_population_size=6,
            base_seeds_per_experiment=2,
            stage_stats_provider=ResearchMemory.stage_stats,
            plateau_detector=ResearchMemory.detect_stage_plateau,
            plateau_streak_provider=ResearchMemory.plateau_streak,
            blocked_region_provider=ResearchMemory.build_blocked_regions,
            reentry_boost_provider=ResearchMemory.reentry_boost_state,
        )
        self.assertEqual(plan.stage_id, "energy_margin")
        self.assertTrue(plan.stuck_mode)
        self.assertTrue(plan.hard_reset_mode)
        self.assertGreater(plan.mutation_scale, 1.0)
        self.assertGreaterEqual(len(plan.blocked_regions), 1)
        self.assertEqual(plan.blocked_region_stage_id, "approach_window")

    def test_reentry_boost_increases_population_and_seeds_for_approach_window(self) -> None:
        planner = NextStepPlanner(default_problem_stages())
        records = [
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
                generation=1,
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
                generation=2,
                rank=1,
                experiment_id="em1",
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
        ]
        for generation in range(3, 7):
            records.append(
                MemoryRecord(
                    generation=generation,
                    rank=1,
                    experiment_id=f"aw-old-{generation}",
                    strategy_name="predictive_intercept",
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
                    params={"lookahead_time": 1.55, "target_distance_weight": 1.3},
                    hypothesis_ids=[],
                )
            )
        records.extend(
            [
            MemoryRecord(
                generation=7,
                rank=1,
                experiment_id="em2",
                strategy_name="energy_aware",
                stage_id="energy_margin",
                stage_title="Energy Margin",
                final_fitness=31.0,
                stage_fitness=31.0,
                stage_success_rate=1.0,
                mean_score=0.0,
                mean_popped=0.0,
                crash_rate=0.0,
                dominant_failure=None,
                note="fallback",
                params={},
                hypothesis_ids=[],
            ),
            MemoryRecord(
                generation=8,
                rank=1,
                experiment_id="em3",
                strategy_name="energy_aware",
                stage_id="energy_margin",
                stage_title="Energy Margin",
                final_fitness=32.0,
                stage_fitness=32.0,
                stage_success_rate=1.0,
                mean_score=0.0,
                mean_popped=0.0,
                crash_rate=0.0,
                dominant_failure=None,
                note="fallback",
                params={},
                hypothesis_ids=[],
            ),
        ]
        )
        plan = planner.plan_generation(
            generation=9,
            records=records,
            fixed_params={"challenge_scenario_number": 0},
            seeds=[1, 2],
            available_strategies=["greedy_intercept", "score_based", "energy_aware", "predictive_intercept"],
            parameter_space={
                "lookahead_time": {"min": 0.5, "max": 2.5, "sigma": 0.1},
                "target_distance_weight": {"min": 0.0, "max": 2.0, "sigma": 0.1},
            },
            base_population_size=6,
            base_seeds_per_experiment=2,
            stage_stats_provider=ResearchMemory.stage_stats,
            plateau_detector=ResearchMemory.detect_stage_plateau,
            plateau_streak_provider=ResearchMemory.plateau_streak,
            blocked_region_provider=ResearchMemory.build_blocked_regions,
            reentry_boost_provider=ResearchMemory.reentry_boost_state,
        )
        self.assertEqual(plan.stage_id, "approach_window")
        self.assertGreater(plan.effective_population_size or 0, 6)
        self.assertGreater(plan.effective_seeds_per_experiment or 0, 2)
        self.assertGreaterEqual(len(plan.blocked_regions), 1)


if __name__ == "__main__":
    unittest.main()
