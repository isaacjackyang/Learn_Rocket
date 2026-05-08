from __future__ import annotations

from rocket_auto_research.strategies.modular_strategy import ModularStrategy


class MPCLightStrategy(ModularStrategy):
    def __init__(self, params=None) -> None:
        merged = {
            "target_selector": "reachable",
            "guidance_mode": "short_horizon",
            "controller_mode": "adaptive",
            "estimator_mode": "wind_aware",
            "branch_count": 7,
            "horizon_s": 1.4,
        }
        if params:
            merged.update(params)
        super().__init__("mpc_light", merged)
