from __future__ import annotations

from rocket_auto_research.strategies.modular_strategy import ModularStrategy


class CEMPlannerStrategy(ModularStrategy):
    def __init__(self, params=None) -> None:
        merged = {
            "target_selector": "reachable",
            "guidance_mode": "cem",
            "controller_mode": "adaptive",
            "estimator_mode": "wind_aware",
            "branch_count": 9,
            "horizon_s": 1.6,
        }
        if params:
            merged.update(params)
        super().__init__("cem_planner", merged)
