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
        blocked_regions: list[dict[str, Any]] | None = None,
        max_attempts: int = 12,
    ) -> ExperimentSpec:
        rng = random.Random(f"mutate:{parent.experiment_id}:{generation}:{','.join(str(seed) for seed in seeds)}")
        chosen_hypothesis = rng.choice(hypotheses) if hypotheses else None
        fallback_candidate: ExperimentSpec | None = None
        for attempt in range(max_attempts):
            params = copy.deepcopy(parent.params)
            for key, value in (self.fixed_params or {}).items():
                params[key] = copy.deepcopy(value)
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

            candidate = ExperimentSpec(
                strategy_name=strategy_name,
                params=params,
                seeds=seeds,
                note=note,
                generation=generation,
                parent_ids=[parent.experiment_id],
            )
            if fallback_candidate is None:
                fallback_candidate = candidate
            if not self.is_blocked(candidate.strategy_name, candidate.params, blocked_regions):
                return candidate

        return self.random_spec(
            generation=generation,
            seeds=seeds,
            preferred_strategies=[fallback_candidate.strategy_name] if fallback_candidate is not None else None,
            salt=f"blocked-fallback:{parent.experiment_id}",
            blocked_regions=blocked_regions,
        )

    def random_spec(
        self,
        generation: int,
        seeds: list[int],
        preferred_strategies: list[str] | None = None,
        salt: str = "",
        blocked_regions: list[dict[str, Any]] | None = None,
        max_attempts: int = 24,
    ) -> ExperimentSpec:
        rng = random.Random(f"random:{generation}:{','.join(str(seed) for seed in seeds)}:{salt}")
        strategies = preferred_strategies or self.strategy_choices
        fallback_candidate: ExperimentSpec | None = None
        for attempt in range(max_attempts):
            params = {key: self._sample_value(spec, rng) for key, spec in self.parameter_space.items()}
            for key, value in (self.fixed_params or {}).items():
                params[key] = copy.deepcopy(value)
            strategy_name = rng.choice(strategies)
            candidate = ExperimentSpec(
                strategy_name=strategy_name,
                params=params,
                seeds=seeds,
                note="random_initialization",
                generation=generation,
            )
            if fallback_candidate is None:
                fallback_candidate = candidate
            if not self.is_blocked(candidate.strategy_name, candidate.params, blocked_regions):
                return candidate
        if fallback_candidate is not None:
            return fallback_candidate
        raise RuntimeError("Unable to generate random experiment candidate.")

    @staticmethod
    def is_blocked(
        strategy_name: str,
        params: dict[str, Any],
        blocked_regions: list[dict[str, Any]] | None = None,
    ) -> bool:
        if not blocked_regions:
            return False
        for region in blocked_regions:
            region_strategy = region.get("strategy_name")
            if region_strategy and str(region_strategy) != strategy_name:
                continue
            categorical = dict(region.get("categorical", {}))
            if any(params.get(key) != value for key, value in categorical.items()):
                continue
            numeric = dict(region.get("numeric", {}))
            matched = True
            for key, bounds in numeric.items():
                value = params.get(key)
                if not isinstance(value, (int, float)):
                    matched = False
                    break
                if float(value) < float(bounds["min"]) or float(value) > float(bounds["max"]):
                    matched = False
                    break
            if matched:
                return True
        return False

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
