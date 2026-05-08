from __future__ import annotations

from math import isfinite

from rocket_auto_research.gnc.state import ControlAction


def _safe_float(value: float, lower: float, upper: float) -> float:
    if not isfinite(value):
        return 0.0
    return max(lower, min(upper, value))


class SafetyGuard:
    def __init__(self, max_tvc: float = 1.0, max_roll: float = 1.0, throttle_range: tuple[float, float] = (0.0, 1.0)) -> None:
        self.max_tvc = max_tvc
        self.max_roll = max_roll
        self.throttle_range = throttle_range

    def sanitize(self, action: ControlAction) -> ControlAction:
        return ControlAction(
            launch=bool(action.launch),
            launch_inclination_deg=_safe_float(action.launch_inclination_deg, 0.0, 90.0),
            launch_heading_deg=_safe_float(action.launch_heading_deg, 0.0, 360.0),
            throttle=_safe_float(action.throttle, self.throttle_range[0], self.throttle_range[1]),
            tvc_x=_safe_float(action.tvc_x, -self.max_tvc, self.max_tvc),
            tvc_y=_safe_float(action.tvc_y, -self.max_tvc, self.max_tvc),
            roll=_safe_float(action.roll, -self.max_roll, self.max_roll),
        )
