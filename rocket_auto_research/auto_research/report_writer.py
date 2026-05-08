from __future__ import annotations

from pathlib import Path

from rocket_auto_research.auto_research.experiment_runner import ExperimentResult
from rocket_auto_research.auto_research.failure_analyzer import FailureReport
from rocket_auto_research.auto_research.hypothesis_generator import ResearchHypothesis
from rocket_auto_research.auto_research.leaderboard import LeaderboardEntry
from rocket_auto_research.auto_research.next_step_planner import NextStepPlan


def write_generation_report(
    output_dir: Path,
    generation: int,
    plan: NextStepPlan,
    best_result: ExperimentResult,
    best_entry: LeaderboardEntry,
    failure_report: FailureReport,
    hypotheses: list[ResearchHypothesis],
    previous_best: LeaderboardEntry | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    delta_fitness = None if previous_best is None else round(best_entry.final_fitness - previous_best.final_fitness, 4)
    lines = [
        f"# Generation {generation:04d}",
        "",
        "## Stage Plan",
        f"- Stage: `{plan.stage_id}` ({plan.stage_title})",
        f"- Rationale: {plan.rationale}",
        f"- Preferred strategies: `{plan.preferred_strategies}`",
        f"- Recent stage success rate: `{plan.recent_stage_success_rate}`",
        f"- Stuck mode: `{plan.stuck_mode}`",
        f"- Mutation scale: `{plan.mutation_scale}`",
        f"- Random injection ratio: `{plan.random_injection_ratio}`",
        *( [f"- Plateau reason: {plan.plateau_reason}"] if plan.plateau_reason else [] ),
        *(f"- Planner note: {note}" for note in plan.notes),
        "",
        "## Best Strategy",
        f"- Experiment: `{best_entry.experiment_id}`",
        f"- Strategy: `{best_entry.strategy_name}`",
        f"- Fitness: `{best_entry.final_fitness}`",
        f"- Stage fitness: `{best_result.summary.stage_fitness}`",
        f"- Mission fitness: `{best_result.summary.mission_fitness}`",
        f"- Stage success rate: `{best_result.summary.stage_success_rate}`",
        f"- Mean score: `{best_entry.mean_score}`",
        f"- Mean popped: `{best_result.summary.mean_popped}`",
        f"- Median popped: `{best_result.summary.median_popped}`",
        f"- Crash rate: `{best_entry.crash_rate}`",
        f"- Near miss rate: `{best_result.summary.near_miss_rate}`",
        f"- Mean TVC saturation: `{best_result.summary.mean_tvc_saturation}`",
        f"- Wind sensitivity: `{best_result.summary.wind_sensitivity}`",
        f"- Overall progress: `{best_result.summary.overall_progress}`",
        f"- Score is proxy: `{best_result.summary.score_is_proxy}`",
    ]
    if delta_fitness is not None:
        lines.append(f"- Fitness delta vs previous best: `{delta_fitness}`")
    lines.extend(
        [
            "",
            "## Dominant Failures",
            *(f"- {name}: count={count}, rate={failure_report.rates.get(name, 0.0)}" for name, count in sorted(failure_report.counts.items())),
            "",
            "## Analysis Notes",
            *(f"- {note}" for note in failure_report.notes),
            "",
            "## Next Hypotheses",
            *(f"- `{hypothesis.hypothesis_id}`: {hypothesis.as_text()}" for hypothesis in hypotheses),
            "",
            "## Recommended Review",
            f"- Best params: `{best_result.spec.params}`",
            f"- Failure examples: `{failure_report.examples}`",
            "",
        ]
    )
    (output_dir / f"generation_{generation:04d}.md").write_text("\n".join(lines), encoding="utf-8")
