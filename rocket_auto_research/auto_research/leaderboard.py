from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class LeaderboardEntry:
    rank: int
    experiment_id: str
    strategy_name: str
    final_fitness: float
    mean_score: float
    median_score: float
    mean_popped: float
    crash_rate: float
    nan_rate: float
    score_std: float
    params: dict[str, Any]
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "experiment_id": self.experiment_id,
            "strategy_name": self.strategy_name,
            "final_fitness": self.final_fitness,
            "mean_score": self.mean_score,
            "median_score": self.median_score,
            "mean_popped": self.mean_popped,
            "crash_rate": self.crash_rate,
            "nan_rate": self.nan_rate,
            "score_std": self.score_std,
            "params": self.params,
            "note": self.note,
        }


def write_leaderboard(entries: list[LeaderboardEntry], output_dir: Path, generation: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"generation_{generation:04d}.json"
    csv_path = output_dir / f"generation_{generation:04d}.csv"

    json_path.write_text(
        json.dumps([entry.to_dict() for entry in entries], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "experiment_id",
                "strategy_name",
                "final_fitness",
                "mean_score",
                "median_score",
                "mean_popped",
                "crash_rate",
                "nan_rate",
                "score_std",
                "params",
                "note",
            ],
        )
        writer.writeheader()
        for entry in entries:
            row = entry.to_dict()
            row["params"] = json.dumps(row["params"], ensure_ascii=False, sort_keys=True)
            writer.writerow(row)
