from __future__ import annotations

from rocket_auto_research.strategies.modular_strategy import ModularStrategy


class EnergyAwareStrategy(ModularStrategy):
    def __init__(self, params=None) -> None:
        merged = {
            "target_selector": "energy_aware",
            "guidance_mode": "energy_aware",
            "controller_mode": "adaptive",
            "estimator_mode": "wind_aware",
            "target_lock_duration": 6,
            "switching_penalty": 1.1,
            "wind_comp_gain": 0.2,
            "climb_bias_altitude_m": 1100.0,
            "climb_vertical_gain": 0.7,
            "climb_velocity_floor_mps": 48.0,
            "desired_speed_floor": 1.25,
            "ascent_turn_scale": 0.28,
            "ascent_throttle_floor": 0.88,
            "minimum_intercept_altitude_m": 220.0,
        }
        if params:
            merged.update(params)
        super().__init__("energy_aware", merged)
