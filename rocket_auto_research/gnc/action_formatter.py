from __future__ import annotations

from typing import Any

from rocket_auto_research.gnc.state import ControlAction


def format_action(action: ControlAction, schema: str = "internal") -> dict[str, Any]:
    if schema == "internal":
        return action.to_dict()
    if schema == "flat_competition":
        return {
            "launch": bool(action.launch),
            "throttle": float(action.throttle),
            "tvc_x": float(action.tvc_x),
            "tvc_y": float(action.tvc_y),
            "roll": float(action.roll),
        }
    if schema == "balloon_challenge":
        return {
            "launch": bool(action.launch),
            "launch_inclination_heading": [
                float(action.launch_inclination_deg),
                float(action.launch_heading_deg),
            ],
            "tvc": [float(action.tvc_x), float(action.tvc_y)],
            "throttle": float(action.throttle),
            "roll": float(action.roll),
        }
    raise ValueError(f"Unsupported action schema '{schema}'.")
