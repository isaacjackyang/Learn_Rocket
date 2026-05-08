from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from rocket_auto_research.auto_research.experiment_spec import ExperimentSpec
from rocket_auto_research.auto_research.hypothesis_generator import ResearchHypothesis
from rocket_auto_research.auto_research.problem_definition import (
    ResearchStage,
    build_stage_bootstrap_specs,
)

if TYPE_CHECKING:
    from rocket_auto_research.auto_research.research_memory import MemoryRecord


@dataclass(slots=True)
class NextStepPlan:
    generation: int
    stage_id: str
    stage_title: str
    rationale: str
    preferred_strategies: list[str]
    adapter_profile: str | None = None
    fixed_params: dict[str, Any] = field(default_factory=dict)
    bootstrap_specs: list[ExperimentSpec] = field(default_factory=list)
    planner_hypotheses: list[ResearchHypothesis] = field(default_factory=list)
    recent_stage_success_rate: float = 0.0
    stuck_mode: bool = False
    mutation_scale: float = 1.0
    random_injection_ratio: float = 0.25
    plateau_reason: str | None = None
    notes: list[str] = field(default_factory=list)


class NextStepPlanner:
    def __init__(self, stages: list[ResearchStage]) -> None:
        self.stages = stages

    def plan_generation(
        self,
        generation: int,
        records: list["MemoryRecord"],
        fixed_params: dict[str, object],
        seeds: list[int],
        available_strategies: list[str],
        stage_stats_provider,
        plateau_detector,
    ) -> NextStepPlan:
        selected_stage = self.stages[-1]
        rationale = "All prerequisite stages already passed; continue optimizing final mission performance."
        recent_success_rate = 0.0
        stuck_mode = False
        mutation_scale = 1.0
        random_injection_ratio = 0.25
        plateau_reason = None
        notes: list[str] = []

        for stage in self.stages:
            stats = stage_stats_provider(records, stage.stage_id)
            recent_success_rate = float(stats["mean_stage_success_rate"])
            if stats["record_count"] == 0.0:
                selected_stage = stage
                rationale = f"No prior evidence for stage '{stage.stage_id}', so start there."
                break
            if stats["best_stage_success_rate"] < stage.success_threshold:
                selected_stage = stage
                rationale = (
                    f"Stage '{stage.stage_id}' has not yet cleared its promotion threshold "
                    f"({stats['best_stage_success_rate']:.2f} < {stage.success_threshold:.2f})."
                )
                notes.append(
                    f"Recent stage fitness mean={stats['mean_stage_fitness']:.2f}, "
                    f"success mean={stats['mean_stage_success_rate']:.2f}."
                )
                break
            notes.append(
                f"Stage '{stage.stage_id}' already met promotion criteria with best success "
                f"{stats['best_stage_success_rate']:.2f}."
            )

        selected_stage_index = next(
            (index for index, stage in enumerate(self.stages) if stage.stage_id == selected_stage.stage_id),
            len(self.stages) - 1,
        )
        plateau = plateau_detector(records, selected_stage.stage_id)
        if bool(plateau.get("is_plateau")):
            stuck_mode = True
            mutation_scale = 2.5
            random_injection_ratio = 0.5
            plateau_reason = (
                f"Stage '{selected_stage.stage_id}' plateaued over {int(plateau['record_count'])} recent generations: "
                f"fitness span {plateau['fitness_span']:.2f}, fitness delta {plateau['fitness_delta']:.2f}, "
                f"mean success {plateau['mean_stage_success_rate']:.2f}."
            )
            notes.append(plateau_reason)
            if selected_stage_index > 0 and float(plateau.get("mean_crash_rate", 0.0)) >= 0.8:
                fallback_stage = self.stages[selected_stage_index - 1]
                notes.append(
                    f"Fallback activated: crash-heavy plateau in '{selected_stage.stage_id}', "
                    f"returning to '{fallback_stage.stage_id}' to rebuild survivability."
                )
                selected_stage = fallback_stage
                selected_stage_index -= 1
                rationale = (
                    f"Stage '{fallback_stage.stage_id}' was reselected because "
                    f"'{self.stages[selected_stage_index + 1].stage_id}' plateaued with high crash rate."
                )

        preferred = [name for name in selected_stage.preferred_strategies if name in available_strategies]
        bootstrap_specs = build_stage_bootstrap_specs(
            selected_stage,
            generation=generation,
            seeds=seeds,
            fixed_params=fixed_params,
            available_strategies=available_strategies,
        )
        return NextStepPlan(
            generation=generation,
            stage_id=selected_stage.stage_id,
            stage_title=selected_stage.title,
            rationale=rationale,
            preferred_strategies=preferred or list(available_strategies),
            bootstrap_specs=bootstrap_specs,
            planner_hypotheses=[
                ResearchHypothesis(
                    hypothesis_id=seed.hypothesis_id,
                    rationale=seed.rationale,
                    adjustments=dict(seed.adjustments),
                    preferred_strategies=list(seed.preferred_strategies),
                )
                for seed in selected_stage.planner_hypotheses
            ],
            recent_stage_success_rate=recent_success_rate,
            stuck_mode=stuck_mode,
            mutation_scale=mutation_scale,
            random_injection_ratio=random_injection_ratio,
            plateau_reason=plateau_reason,
            notes=notes,
        )
