from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rocket_auto_research.strategies.base import Strategy
from rocket_auto_research.strategies.baseline_pid import BaselinePIDStrategy
from rocket_auto_research.strategies.cem_planner import CEMPlannerStrategy
from rocket_auto_research.strategies.energy_aware import EnergyAwareStrategy
from rocket_auto_research.strategies.greedy_intercept import GreedyInterceptStrategy
from rocket_auto_research.strategies.mpc_light import MPCLightStrategy
from rocket_auto_research.strategies.predictive_intercept import PredictiveInterceptStrategy
from rocket_auto_research.strategies.rl_policy_wrapper import RLPolicyWrapperStrategy
from rocket_auto_research.strategies.score_based import ScoreBasedStrategy

StrategyFactory = Callable[[dict[str, Any] | None], Strategy]

REGISTRY: dict[str, StrategyFactory] = {
    "baseline_pid": BaselinePIDStrategy,
    "greedy_intercept": GreedyInterceptStrategy,
    "predictive_intercept": PredictiveInterceptStrategy,
    "score_based": ScoreBasedStrategy,
    "mpc_light": MPCLightStrategy,
    "cem_planner": CEMPlannerStrategy,
    "energy_aware": EnergyAwareStrategy,
    "rl_policy_wrapper": RLPolicyWrapperStrategy,
}


def build_strategy(strategy_name: str, params: dict[str, Any] | None = None) -> Strategy:
    if strategy_name not in REGISTRY:
        available = ", ".join(sorted(REGISTRY))
        raise KeyError(f"Unknown strategy '{strategy_name}'. Available: {available}")
    return REGISTRY[strategy_name](params)
