from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from rocket_auto_research.auto_research.experiment_runner import ExperimentResult
from rocket_auto_research.auto_research.hypothesis_generator import ResearchHypothesis
from rocket_auto_research.auto_research.next_step_planner import NextStepPlan


@dataclass(slots=True)
class MemoryRecord:
    generation: int
    rank: int
    experiment_id: str
    strategy_name: str
    stage_id: str
    stage_title: str
    final_fitness: float
    stage_fitness: float
    stage_success_rate: float
    mean_score: float
    mean_popped: float
    crash_rate: float
    dominant_failure: str | None
    note: str
    params: dict[str, Any]
    hypothesis_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResearchMemory:
    def __init__(self, root: str | Path = "results/research_memory") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.records_path = self.root / "experiments.jsonl"

    def load_records(self) -> list[MemoryRecord]:
        if not self.records_path.exists():
            return []
        records: list[MemoryRecord] = []
        for line in self.records_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(MemoryRecord(**payload))
        return records

    def append_generation(
        self,
        generation: int,
        plan: NextStepPlan,
        ranked: list[ExperimentResult],
        hypotheses: list[ResearchHypothesis],
    ) -> None:
        records = [
            MemoryRecord(
                generation=generation,
                rank=rank,
                experiment_id=result.spec.experiment_id,
                strategy_name=result.spec.strategy_name,
                stage_id=plan.stage_id,
                stage_title=plan.stage_title,
                final_fitness=result.summary.final_fitness,
                stage_fitness=result.summary.stage_fitness,
                stage_success_rate=result.summary.stage_success_rate,
                mean_score=result.summary.mean_score,
                mean_popped=result.summary.mean_popped,
                crash_rate=result.summary.crash_rate,
                dominant_failure=result.failure_report.dominant_failure,
                note=result.spec.note,
                params=dict(result.spec.params),
                hypothesis_ids=[hypothesis.hypothesis_id for hypothesis in hypotheses],
            )
            for rank, result in enumerate(ranked, start=1)
        ]
        with self.records_path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        generation_payload = {
            "generation": generation,
            "stage_id": plan.stage_id,
            "stage_title": plan.stage_title,
            "rationale": plan.rationale,
            "records": [record.to_dict() for record in records],
        }
        (self.root / f"generation_{generation:04d}.json").write_text(
            json.dumps(generation_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        latest_payload = {
            "generation": generation,
            "stage_id": plan.stage_id,
            "stage_title": plan.stage_title,
            "rationale": plan.rationale,
            "recent_stage_success_rate": plan.recent_stage_success_rate,
            "stuck_mode": plan.stuck_mode,
            "mutation_scale": plan.mutation_scale,
            "random_injection_ratio": plan.random_injection_ratio,
            "plateau_reason": plan.plateau_reason,
            "planner_notes": plan.notes,
        }
        (self.root / "latest_plan.json").write_text(
            json.dumps(latest_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def stage_stats(records: list[MemoryRecord], stage_id: str, window: int = 2) -> dict[str, float]:
        stage_records = [record for record in records if record.stage_id == stage_id]
        if not stage_records:
            return {
                "record_count": 0.0,
                "best_stage_success_rate": 0.0,
                "mean_stage_success_rate": 0.0,
                "best_stage_fitness": 0.0,
                "mean_stage_fitness": 0.0,
            }
        generations = sorted({record.generation for record in stage_records})
        recent_generations = set(generations[-window:])
        recent = [record for record in stage_records if record.generation in recent_generations and record.rank == 1]
        if not recent:
            recent = stage_records[-window:]
        return {
            "record_count": float(len(recent)),
            "best_stage_success_rate": round(max(record.stage_success_rate for record in recent), 4),
            "mean_stage_success_rate": round(mean(record.stage_success_rate for record in recent), 4),
            "best_stage_fitness": round(max(record.stage_fitness for record in recent), 4),
            "mean_stage_fitness": round(mean(record.stage_fitness for record in recent), 4),
        }

    @staticmethod
    def detect_stage_plateau(
        records: list[MemoryRecord],
        stage_id: str,
        window: int = 4,
        fitness_epsilon: float = 1.0,
    ) -> dict[str, float | bool]:
        stage_records = [record for record in records if record.stage_id == stage_id and record.rank == 1]
        if len(stage_records) < window:
            return {
                "is_plateau": False,
                "record_count": float(len(stage_records)),
                "fitness_span": 0.0,
                "fitness_delta": 0.0,
                "mean_stage_success_rate": 0.0,
                "mean_crash_rate": 0.0,
            }
        recent = stage_records[-window:]
        fitness_values = [record.stage_fitness for record in recent]
        success_values = [record.stage_success_rate for record in recent]
        crash_values = [record.crash_rate for record in recent]
        fitness_span = max(fitness_values) - min(fitness_values)
        fitness_delta = recent[-1].stage_fitness - recent[0].stage_fitness
        mean_success = mean(success_values)
        mean_crash = mean(crash_values)
        is_plateau = fitness_span <= fitness_epsilon and abs(fitness_delta) <= fitness_epsilon and mean_success <= 0.1
        return {
            "is_plateau": is_plateau,
            "record_count": float(len(recent)),
            "fitness_span": round(fitness_span, 4),
            "fitness_delta": round(fitness_delta, 4),
            "mean_stage_success_rate": round(mean_success, 4),
            "mean_crash_rate": round(mean_crash, 4),
        }
