from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ExperimentSpec:
    strategy_name: str
    params: dict[str, Any]
    seeds: list[int]
    note: str = ""
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    experiment_id: str = ""

    def __post_init__(self) -> None:
        if not self.experiment_id:
            digest_source = json.dumps(
                {
                    "strategy_name": self.strategy_name,
                    "params": self.params,
                    "seeds": self.seeds,
                    "note": self.note,
                    "generation": self.generation,
                    "parent_ids": self.parent_ids,
                },
                sort_keys=True,
            ).encode("utf-8")
            self.experiment_id = hashlib.sha1(digest_source).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
