from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any

from rocket_auto_research.auto_research.experiment_spec import ExperimentSpec
from rocket_auto_research.auto_research.hypothesis_generator import ResearchHypothesis


@dataclass(slots=True)
class MutationEngine:
    parameter_space: dict[str, dict[str, Any]]
    mutation_rate: float
    strategy_choices: list[str]
    fixed_params: dict[str, Any] | None = None

    def mutate(
        self,
        parent: ExperimentSpec,
        generation: int,
        seeds: list[int],
        hypotheses: list[ResearchHypothesis] | None = None,
        mutation_scale: float = 1.0,
    ) -> ExperimentSpec:
        rng = random.Random(f"mutate:{parent.experiment_id}:{generation}:{','.join(str(seed) for seed in seeds)}")
        params = copy.deepcopy(parent.params)
        for key, value in (self.fixed_params or {}).items():
            params[key] = copy.deepcopy(value)
        chosen_hypothesis = rng.choice(hypotheses) if hypotheses else None
        if chosen_hypothesis is not None:
            params = self._apply_adjustments(params, chosen_hypothesis.adjustments, rng, mutation_scale=mutation_scale)

        for key, spec in self.parameter_space.items():
            if rng.random() > self.mutation_rate:
                continue
            params[key] = self._mutate_value(key, params.get(key), spec, rng, mutation_scale=mutation_scale)

        strategy_name = parent.strategy_name
        if chosen_hypothesis and chosen_hypothesis.preferred_strategies and rng.random() < 0.6:
            strategy_name = rng.choice(chosen_hypothesis.preferred_strategies)

        note = f"mutation_from:{parent.experiment_id}"
        if chosen_hypothesis is not None:
            note = f"{note};hypothesis:{chosen_hypothesis.hypothesis_id}"

        return ExperimentSpec(
            strategy_name=strategy_name,
            params=params,
            seeds=seeds,
            note=note,
            generation=generation,
            parent_ids=[parent.experiment_id],
        )

    def random_spec(
        self,
        generation: int,
        seeds: list[int],
        preferred_strategies: list[str] | None = None,
        salt: str = "",
    ) -> ExperimentSpec:
        rng = random.Random(f"random:{generation}:{','.join(str(seed) for seed in seeds)}:{salt}")
        params = {key: self._sample_value(spec, rng) for key, spec in self.parameter_space.items()}
        for key, value in (self.fixed_params or {}).items():
            params[key] = copy.deepcopy(value)
        strategies = preferred_strategies or self.strategy_choices
        strategy_name = rng.choice(strategies)
        return ExperimentSpec(
            strategy_name=strategy_name,
            params=params,
            seeds=seeds,
            note="random_initialization",
            generation=generation,
        )

    def _apply_adjustments(
        self,
        params: dict[str, Any],
        adjustments: dict[str, Any],
        rng: random.Random,
        mutation_scale: float = 1.0,
    ) -> dict[str, Any]:
        updated = copy.deepcopy(params)
        for key, delta in adjustments.items():
            spec = self.parameter_space.get(key)
            current = updated.get(key)
            if spec is None:
                updated[key] = delta if current is None else delta
                continue
            if "choices" in spec:
                if isinstance(delta, str):
                    updated[key] = delta
                elif current in spec["choices"] and rng.random() < 0.5:
                    updated[key] = current
                else:
                    updated[key] = rng.choice(list(spec["choices"]))
                continue
            base = float(current if current is not None else spec.get("default", spec.get("min", 0.0)))
            proposal = base + float(delta) * mutation_scale
            updated[key] = self._clamp_numeric(proposal, spec)
        return updated

    def _mutate_value(
        self,
        key: str,
        current: Any,
        spec: dict[str, Any],
        rng: random.Random,
        mutation_scale: float = 1.0,
    ) -> Any:
        if "choices" in spec:
            choices = list(spec["choices"])
            if not choices:
                return current
            alternatives = [choice for choice in choices if choice != current]
            if alternatives:
                return rng.choice(alternatives)
            return current if current is not None else choices[0]

        value = float(current if current is not None else spec.get("default", spec["min"]))
        spread = float(spec.get("sigma", max((float(spec["max"]) - float(spec["min"])) * 0.1, 0.01))) * mutation_scale
        proposal = value + rng.gauss(0.0, spread)
        clamped = self._clamp_numeric(proposal, spec)
        if spec.get("type") == "int":
            return int(round(clamped))
        return round(float(clamped), 4)

    def _sample_value(self, spec: dict[str, Any], rng: random.Random) -> Any:
        if "choices" in spec:
            return rng.choice(list(spec["choices"]))
        if spec.get("type") == "int":
            return int(rng.randint(int(spec["min"]), int(spec["max"])))
        value = rng.uniform(float(spec["min"]), float(spec["max"]))
        return round(value, 4)

    @staticmethod
    def _clamp_numeric(value: float, spec: dict[str, Any]) -> float:
        return max(float(spec["min"]), min(float(spec["max"]), value))
