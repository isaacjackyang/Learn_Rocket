from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rocket_auto_research.auto_research.experiment_runner import ExperimentResult, ExperimentRunner
from rocket_auto_research.auto_research.experiment_spec import ExperimentSpec
from rocket_auto_research.auto_research.hypothesis_generator import ResearchHypothesis, generate_hypotheses
from rocket_auto_research.auto_research.leaderboard import LeaderboardEntry, write_leaderboard
from rocket_auto_research.auto_research.mutation_engine import MutationEngine
from rocket_auto_research.auto_research.next_step_planner import NextStepPlan, NextStepPlanner
from rocket_auto_research.auto_research.problem_definition import default_problem_stages
from rocket_auto_research.auto_research.research_memory import ResearchMemory
from rocket_auto_research.auto_research.report_writer import write_generation_report
from rocket_auto_research.auto_research.runtime_control import ResearchRuntimeControl, ResearchStopRequested
from rocket_auto_research.auto_research.strategy_crossover import StrategyCrossover


@dataclass(slots=True)
class ResearchConfig:
    strategies: list[str]
    parameter_space: dict[str, dict[str, Any]]
    fixed_params: dict[str, Any]
    stage_policy: dict[str, dict[str, Any]]
    population_size: int
    elite_count: int
    mutation_rate: float
    crossover_rate: float
    seeds_per_experiment: int
    generations: int
    continuous: bool = False
    base_seed: int = 0
    bootstrap_specs: list[ExperimentSpec] | None = None


class ResearcherLoop:
    def __init__(
        self,
        runner: ExperimentRunner,
        config: ResearchConfig,
        results_root: str | Path = "results",
        runtime_control: ResearchRuntimeControl | None = None,
    ) -> None:
        self.runner = runner
        self.config = config
        self.results_root = Path(results_root)
        self.runtime_control = runtime_control
        self.mutator = MutationEngine(
            parameter_space=config.parameter_space,
            mutation_rate=config.mutation_rate,
            strategy_choices=config.strategies,
            fixed_params=config.fixed_params,
        )
        self.crossover = StrategyCrossover()
        self.problem_stages = default_problem_stages()
        self.planner = NextStepPlanner(self.problem_stages)
        self.memory = ResearchMemory(self.results_root / "research_memory")

    def run(self) -> ExperimentResult:
        last_best: ExperimentResult | None = None
        previous_best_entry: LeaderboardEntry | None = None
        completed_experiments = 0

        try:
            generation = 0
            population: list[ExperimentSpec] | None = None
            while True:
                records = self.memory.load_records()
                current_plan = self._plan_generation(generation, records)
                if population is None:
                    population = self._initial_population(current_plan)
                if self.runtime_control is not None:
                    self.runtime_control.wait_if_paused()
                    self.runtime_control.check_stop_requested()
                    population_workers = self.runner._parallel_population_worker_count(population)
                    seed_workers = self.runner._parallel_worker_count(population[0]) if population else 1
                    self.runtime_control.update_status(
                        status="running",
                        current_generation=generation,
                        generation_label=self._generation_label(generation),
                        current_stage=current_plan.stage_id,
                        message=(
                            f"Running generation {generation + 1} "
                            f"for stage {current_plan.stage_id}."
                        ),
                        completed_experiments=completed_experiments,
                        active_workers=population_workers if population_workers > 1 else seed_workers,
                        worker_mode="generation" if population_workers > 1 else "seed" if seed_workers > 1 else "serial",
                    )
                experiment_index_map = {spec.experiment_id: index for index, spec in enumerate(population, start=1)}
                if self.runtime_control is not None and population:
                    first_spec = population[0]
                    self.runtime_control.update_status(
                        status="running",
                        current_generation=generation,
                        generation_label=self._generation_label(generation),
                        current_experiment_id=first_spec.experiment_id,
                        current_strategy_name=first_spec.strategy_name,
                        current_experiment_index=1,
                        current_stage=current_plan.stage_id,
                        message=(
                            f"Evaluating {len(population)} experiments under stage {current_plan.stage_id} "
                            f"using {self.runtime_control.read_status().get('active_workers', 1)} workers."
                        ),
                        completed_experiments=completed_experiments,
                    )

                evaluated: list[ExperimentResult] = []

                def _on_completed(spec: ExperimentSpec, result: ExperimentResult) -> None:
                    nonlocal completed_experiments
                    completed_experiments += 1
                    evaluated.append(result)
                    if self.runtime_control is not None:
                        self.runtime_control.update_status(
                            completed_experiments=completed_experiments,
                            last_finished_experiment_id=spec.experiment_id,
                            current_experiment_id=spec.experiment_id,
                            current_strategy_name=spec.strategy_name,
                            current_experiment_index=experiment_index_map.get(spec.experiment_id, 0),
                            message=(
                                f"Completed {completed_experiments} experiments in generation {generation + 1}; "
                                f"latest finished {spec.strategy_name}."
                            ),
                        )

                self.runner.run_population(
                    population,
                    stage=self._stage_by_id(current_plan.stage_id),
                    on_completed=_on_completed,
                )
                ranked = sorted(evaluated, key=lambda result: result.summary.final_fitness, reverse=True)
                entries = self._entries_for_generation(ranked)
                hypotheses = list(current_plan.planner_hypotheses) + generate_hypotheses(ranked[0].failure_report)
                write_leaderboard(entries, self.results_root / "leaderboards", generation)
                write_generation_report(
                    self.results_root / "reports",
                    generation,
                    current_plan,
                    ranked[0],
                    entries[0],
                    ranked[0].failure_report,
                    hypotheses,
                    previous_best=previous_best_entry,
                )
                self.memory.append_generation(generation, current_plan, ranked, hypotheses)
                self._write_best_snapshot(ranked[0])
                self._refresh_dashboard_assets()
                last_best = ranked[0]
                previous_best_entry = entries[0]
                if not self.config.continuous and generation >= self.config.generations - 1:
                    break
                next_plan = self._plan_generation(generation + 1, self.memory.load_records())
                population = self._next_population(ranked, generation + 1, hypotheses, next_plan)
                generation += 1
        except ResearchStopRequested:
            self._refresh_dashboard_assets()
            if self.runtime_control is not None:
                self.runtime_control.mark_stopped()
            raise
        except Exception as exc:
            self._refresh_dashboard_assets()
            if self.runtime_control is not None:
                self.runtime_control.mark_error(f"Auto research failed: {exc}")
            raise

        if last_best is None:
            raise RuntimeError("Research loop did not produce any result.")
        self._refresh_dashboard_assets()
        if self.runtime_control is not None:
            self.runtime_control.mark_completed(
                best_experiment_id=last_best.spec.experiment_id,
                strategy_name=last_best.spec.strategy_name,
            )
        return last_best

    def _generation_label(self, generation: int) -> str:
        if self.config.continuous:
            return f"{generation + 1}/ongoing"
        return f"{generation + 1}/{self.config.generations}"

    def _initial_population(self, plan: NextStepPlan) -> list[ExperimentSpec]:
        seeds = self._generation_seeds(0)
        population: list[ExperimentSpec] = []
        for bootstrap in plan.bootstrap_specs:
            if len(population) >= self.config.population_size:
                break
            population.append(self._apply_stage_context(bootstrap, plan, seeds))
        for index, bootstrap in enumerate(self.config.bootstrap_specs or []):
            if len(population) >= self.config.population_size:
                break
            population.append(
                self._apply_stage_context(
                    ExperimentSpec(
                        strategy_name=bootstrap.strategy_name,
                        params=dict(bootstrap.params),
                        seeds=seeds,
                        note=f"bootstrap:{bootstrap.experiment_id}:{index}",
                        generation=0,
                        parent_ids=[bootstrap.experiment_id],
                    ),
                    plan,
                    seeds,
                )
            )
        while len(population) < self.config.population_size:
            population.append(
                self._apply_stage_context(
                    self.mutator.random_spec(
                        generation=0,
                        seeds=seeds,
                        preferred_strategies=plan.preferred_strategies,
                        salt=f"init:{len(population)}:{plan.stage_id}",
                    ),
                    plan,
                    seeds,
                )
            )
        return population

    def _next_population(
        self,
        ranked: list[ExperimentResult],
        generation: int,
        hypotheses: list[ResearchHypothesis],
        plan: NextStepPlan,
    ) -> list[ExperimentSpec]:
        effective_elite_count = self.config.elite_count if not plan.stuck_mode else max(1, self.config.elite_count - 1)
        elites = ranked[:effective_elite_count]
        seeds = self._generation_seeds(generation)
        next_population: list[ExperimentSpec] = list(plan.bootstrap_specs)
        next_population = [self._apply_stage_context(spec, plan, seeds) for spec in next_population]
        next_population.extend(
            [
            self._apply_stage_context(
                ExperimentSpec(
                    strategy_name=elite.spec.strategy_name,
                    params=dict(elite.spec.params),
                    seeds=seeds,
                    note=f"elite_from:{elite.spec.experiment_id}",
                    generation=generation,
                    parent_ids=[elite.spec.experiment_id],
                ),
                plan,
                seeds,
            )
            for elite in elites
            ]
        )

        for index, hypothesis in enumerate(hypotheses):
            if len(next_population) >= self.config.population_size:
                break
            elite = elites[index % len(elites)]
            next_population.append(
                self._apply_stage_context(
                    self.mutator.mutate(
                        elite.spec,
                        generation=generation,
                        seeds=seeds,
                        hypotheses=[hypothesis],
                        mutation_scale=plan.mutation_scale,
                    ),
                    plan,
                    seeds,
                )
            )

        random_injection_target = max(1, int(round(self.config.population_size * plan.random_injection_ratio))) if plan.stuck_mode else 0
        random_injected = 0
        while len(next_population) < self.config.population_size:
            if plan.stuck_mode and random_injected < random_injection_target:
                candidate = self.mutator.random_spec(
                    generation=generation,
                    seeds=seeds,
                    preferred_strategies=plan.preferred_strategies or None,
                    salt=f"stuck-fill:{generation}:{len(next_population)}:{plan.stage_id}",
                )
                next_population.append(self._apply_stage_context(candidate, plan, seeds))
                random_injected += 1
                continue
            index = len(next_population) - effective_elite_count
            left = elites[index % len(elites)].spec
            right = elites[(index + 1) % len(elites)].spec
            rng = random.Random(f"next-pop:{generation}:{index}:{left.experiment_id}:{right.experiment_id}")
            if rng.random() < self.config.crossover_rate:
                candidate = self.crossover.crossover(left, right, generation=generation, seeds=seeds)
            elif rng.random() < 0.8:
                candidate = self.mutator.mutate(
                    left,
                    generation=generation,
                    seeds=seeds,
                    hypotheses=hypotheses,
                    mutation_scale=plan.mutation_scale,
                )
            else:
                preferred = [strategy for hypothesis in hypotheses for strategy in hypothesis.preferred_strategies]
                candidate = self.mutator.random_spec(
                    generation=generation,
                    seeds=seeds,
                    preferred_strategies=preferred or plan.preferred_strategies or None,
                    salt=f"fill:{generation}:{len(next_population)}:{plan.stage_id}",
                )
            next_population.append(self._apply_stage_context(candidate, plan, seeds))
        return next_population[: self.config.population_size]

    def _generation_seeds(self, generation: int) -> list[int]:
        start = self.config.base_seed + generation * self.config.seeds_per_experiment
        return list(range(start, start + self.config.seeds_per_experiment))

    def _entries_for_generation(self, ranked: list[ExperimentResult]) -> list[LeaderboardEntry]:
        entries: list[LeaderboardEntry] = []
        for rank, result in enumerate(ranked, start=1):
            entries.append(
                LeaderboardEntry(
                    rank=rank,
                    experiment_id=result.spec.experiment_id,
                    strategy_name=result.spec.strategy_name,
                    final_fitness=result.summary.final_fitness,
                    mean_score=result.summary.mean_score,
                    median_score=result.summary.median_score,
                    mean_popped=result.summary.mean_popped,
                    crash_rate=result.summary.crash_rate,
                    nan_rate=result.summary.nan_rate,
                    score_std=result.summary.score_std,
                    params=result.spec.params,
                    note=result.spec.note,
                )
            )
        return entries

    def _plan_generation(self, generation: int, records) -> NextStepPlan:
        seeds = self._generation_seeds(generation)
        plan = self.planner.plan_generation(
            generation=generation,
            records=records,
            fixed_params=self.config.fixed_params,
            seeds=seeds,
            available_strategies=self.config.strategies,
            stage_stats_provider=self.memory.stage_stats,
            plateau_detector=self.memory.detect_stage_plateau,
        )
        override = dict(self.config.stage_policy.get(plan.stage_id, {}))
        if "preferred_strategies" in override:
            preferred = [name for name in list(override.get("preferred_strategies", [])) if name in self.config.strategies]
            if preferred:
                plan.preferred_strategies = preferred
        plan.adapter_profile = override.get("adapter_profile", plan.adapter_profile)
        plan.fixed_params = dict(self.config.fixed_params)
        plan.fixed_params.update(dict(override.get("fixed_params", {})))
        if "notes" in override:
            plan.notes.extend(list(override.get("notes", [])))
        return plan

    def _stage_by_id(self, stage_id: str):
        for stage in self.problem_stages:
            if stage.stage_id == stage_id:
                return stage
        raise KeyError(f"Unknown stage id '{stage_id}'.")

    def _apply_stage_context(self, spec: ExperimentSpec, plan: NextStepPlan, seeds: list[int]) -> ExperimentSpec:
        params = dict(spec.params)
        params.update(plan.fixed_params)
        if plan.adapter_profile:
            params["simulation_adapter"] = plan.adapter_profile
        note = spec.note
        if f"stage:{plan.stage_id}" not in note:
            note = f"{note};stage:{plan.stage_id}" if note else f"stage:{plan.stage_id}"
        return ExperimentSpec(
            strategy_name=spec.strategy_name,
            params=params,
            seeds=seeds,
            note=note,
            generation=spec.generation,
            parent_ids=list(spec.parent_ids),
        )

    def _write_best_snapshot(self, result: ExperimentResult) -> None:
        best_dir = self.results_root / "best_agents"
        best_dir.mkdir(parents=True, exist_ok=True)
        adapter_name = str(result.episodes[0].metadata.get("adapter", "")) if result.episodes else ""
        if adapter_name in {"competition_platform", "balloon_challenge"}:
            observation_schema = "balloon_challenge"
            action_schema = "balloon_challenge"
        else:
            observation_schema = "flat_competition"
            action_schema = "flat_competition"
        snapshot = {
            "strategy_name": result.spec.strategy_name,
            "params": result.spec.params,
            "experiment_id": result.spec.experiment_id,
            "final_fitness": result.summary.final_fitness,
            "observation_schema": observation_schema,
            "action_schema": action_schema,
        }
        (best_dir / "best_config.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        snapshot_module = "\n".join(
            [
                "from __future__ import annotations",
                "",
                "BEST_AGENT_SNAPSHOT = " + json.dumps(snapshot, indent=2, ensure_ascii=False),
                "",
                "def load_best_agent_snapshot() -> dict[str, object]:",
                "    return dict(BEST_AGENT_SNAPSHOT)",
                "",
            ]
        )
        (best_dir / "best_agent_snapshot.py").write_text(snapshot_module, encoding="utf-8")

    def _refresh_dashboard_assets(self) -> None:
        try:
            from rocket_auto_research.dashboard_builder import build_dashboard

            build_dashboard(results_dir=self.results_root, output_dir=self.results_root / "dashboard")
        except Exception:
            return
