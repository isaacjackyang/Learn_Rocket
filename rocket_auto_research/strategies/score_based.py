from __future__ import annotations

from rocket_auto_research.strategies.modular_strategy import ModularStrategy


class ScoreBasedStrategy(ModularStrategy):
    def __init__(self, params=None) -> None:
        merged = {
            "target_selector": "score_based",
            "guidance_mode": "fixed",
            "controller_mode": "adaptive",
            "estimator_mode": "alpha_beta",
        }
        if params:
            merged.update(params)
        super().__init__("score_based", merged)
