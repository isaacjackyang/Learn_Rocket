from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from rocket_auto_research.auto_research.experiment_runner import ExperimentResult
from rocket_auto_research.auto_research.hypothesis_generator import ResearchHypothesis
from rocket_auto_research.auto_research.next_step_planner import NextStepPlan

BLOCK_REGION_KEYS = {
    "target_selector",
    "guidance_mode",
    "controller_mode",
    "lookahead_time",
    "target_distance_weight",
    "target_angle_weight",
    "target_height_weight",
    "target_lock_duration",
    "switching_penalty",
    "kp",
    "kd",
    "throttle",
    "max_tvc",
    "desired_speed",
    "branch_count",
    "horizon_s",
    "launch_wait_time",
    "launch_inclination_deg",
    "launch_heading_deg",
    "stabilize_duration",
}


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
            "hard_reset_mode": plan.hard_reset_mode,
            "mutation_scale": plan.mutation_scale,
            "random_injection_ratio": plan.random_injection_ratio,
            "plateau_reason": plan.plateau_reason,
            "blocked_region_stage_id": plan.blocked_region_stage_id,
            "blocked_region_count": len(plan.blocked_regions),
            "effective_population_size": plan.effective_population_size,
            "effective_seeds_per_experiment": plan.effective_seeds_per_experiment,
            "reentry_boost_remaining": plan.reentry_boost_remaining,
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

    @staticmethod
    def plateau_streak(
        records: list[MemoryRecord],
        stage_id: str,
        window: int = 4,
        max_windows: int = 3,
        fitness_epsilon: float = 1.0,
    ) -> int:
        stage_records = [record for record in records if record.stage_id == stage_id and record.rank == 1]
        if len(stage_records) < window:
            return 0
        streak = 0
        for offset in range(max_windows):
            end = len(stage_records) - offset
            start = end - window
            if start < 0:
                break
            metrics = ResearchMemory._plateau_metrics(stage_records[start:end], fitness_epsilon=fitness_epsilon)
            if not metrics["is_plateau"]:
                break
            streak += 1
        return streak

    @staticmethod
    def build_blocked_regions(
        records: list[MemoryRecord],
        stage_id: str,
        parameter_space: dict[str, dict[str, Any]],
        top_k: int = 3,
        recent_generations: int = 8,
    ) -> list[dict[str, Any]]:
        stage_records = [record for record in records if record.stage_id == stage_id and record.rank == 1]
        if not stage_records:
            return []
        latest_generation = max(record.generation for record in stage_records)
        recent = [
            record
            for record in stage_records
            if record.generation >= latest_generation - recent_generations + 1
        ]
        selected = sorted(recent, key=lambda record: record.generation, reverse=True)[:top_k]
        regions: list[dict[str, Any]] = []
        for record in selected:
            numeric: dict[str, dict[str, float]] = {}
            categorical: dict[str, Any] = {}
            for key, value in record.params.items():
                if key not in BLOCK_REGION_KEYS:
                    continue
                spec = parameter_space.get(key)
                if spec is None:
                    continue
                if "choices" in spec:
                    categorical[key] = value
                    continue
                if not isinstance(value, (int, float)):
                    continue
                min_value = float(spec["min"])
                max_value = float(spec["max"])
                span = max_value - min_value
                default_sigma = span * 0.05 if span > 0.0 else 0.01
                radius = max(float(spec.get("sigma", default_sigma)) * 1.5, span * 0.08, 0.01)
                numeric[key] = {
                    "min": round(max(min_value, float(value) - radius), 4),
                    "max": round(min(max_value, float(value) + radius), 4),
                    "center": round(float(value), 4),
                }
            if not numeric and not categorical:
                continue
            regions.append(
                {
                    "stage_id": stage_id,
                    "source_experiment_id": record.experiment_id,
                    "strategy_name": record.strategy_name,
                    "numeric": numeric,
                    "categorical": categorical,
                }
            )
        return regions

    @staticmethod
    def reentry_boost_state(
        records: list[MemoryRecord],
        stage_id: str,
        boost_generations: int = 3,
        lookback_generations: int = 8,
    ) -> dict[str, Any]:
        stage_sequence = ResearchMemory._best_records_by_generation(records)
        if not stage_sequence:
            return {"active": False, "remaining_generations": 0, "from_stage_id": None}

        generations = [record.generation for record in stage_sequence]
        stage_ids = [record.stage_id for record in stage_sequence]
        if stage_id not in stage_ids:
            return {"active": False, "remaining_generations": 0, "from_stage_id": None}

        if stage_ids[-1] == stage_id:
            tail_length = 0
            index = len(stage_ids) - 1
            while index >= 0 and stage_ids[index] == stage_id:
                tail_length += 1
                index -= 1
            if index >= 0 and stage_id in stage_ids[:index]:
                remaining = max(0, boost_generations - tail_length)
                return {
                    "active": remaining > 0,
                    "remaining_generations": remaining,
                    "from_stage_id": stage_ids[index],
                }
            return {"active": False, "remaining_generations": 0, "from_stage_id": None}

        last_target_index = max(index for index, value in enumerate(stage_ids) if value == stage_id)
        generations_since_target = generations[-1] - generations[last_target_index]
        if 1 <= generations_since_target <= lookback_generations:
            return {
                "active": True,
                "remaining_generations": boost_generations,
                "from_stage_id": stage_ids[-1],
            }
        return {"active": False, "remaining_generations": 0, "from_stage_id": None}

    @staticmethod
    def _best_records_by_generation(records: list[MemoryRecord]) -> list[MemoryRecord]:
        best_by_generation: dict[int, MemoryRecord] = {}
        for record in sorted(records, key=lambda item: (item.generation, item.rank)):
            if record.rank != 1:
                continue
            best_by_generation.setdefault(record.generation, record)
        return [best_by_generation[generation] for generation in sorted(best_by_generation)]

    @staticmethod
    def _plateau_metrics(records: list[MemoryRecord], fitness_epsilon: float) -> dict[str, float | bool]:
        if not records:
            return {
                "is_plateau": False,
                "fitness_span": 0.0,
                "fitness_delta": 0.0,
                "mean_stage_success_rate": 0.0,
                "mean_crash_rate": 0.0,
            }
        fitness_values = [record.stage_fitness for record in records]
        success_values = [record.stage_success_rate for record in records]
        crash_values = [record.crash_rate for record in records]
        fitness_span = max(fitness_values) - min(fitness_values)
        fitness_delta = records[-1].stage_fitness - records[0].stage_fitness
        mean_success = mean(success_values)
        mean_crash = mean(crash_values)
        return {
            "is_plateau": fitness_span <= fitness_epsilon
            and abs(fitness_delta) <= fitness_epsilon
            and mean_success <= 0.1,
            "fitness_span": round(fitness_span, 4),
            "fitness_delta": round(fitness_delta, 4),
            "mean_stage_success_rate": round(mean_success, 4),
            "mean_crash_rate": round(mean_crash, 4),
        }
