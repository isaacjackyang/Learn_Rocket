from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class CompetitionContract:
    observation_schema: str = "flat_competition"
    action_schema: str = "flat_competition"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REQUIRED_FLAT_OBSERVATION_KEYS = {"time_s", "balloons"}
REQUIRED_FLAT_ACTION_KEYS = {"launch", "throttle", "tvc_x", "tvc_y", "roll"}
REQUIRED_BALLOON_CHALLENGE_OBSERVATION_KEYS = {
    "simulation_time",
    "balloon_status",
    "balloon_states",
    "rocket_sensors",
}
REQUIRED_BALLOON_CHALLENGE_ACTION_KEYS = {
    "launch",
    "launch_inclination_heading",
    "tvc",
    "throttle",
    "roll",
}


def validate_observation_payload(observation: dict[str, Any], schema: str) -> None:
    if schema == "flat_competition":
        missing = [key for key in REQUIRED_FLAT_OBSERVATION_KEYS if key not in observation]
        if missing:
            raise ValueError(f"Observation payload is missing required keys for '{schema}': {missing}")
        if not any(key in observation for key in ("rocket_position", "x", "pos_x")):
            raise ValueError("Observation payload must include rocket position data.")
        if not any(key in observation for key in ("rocket_velocity", "vx")):
            raise ValueError("Observation payload must include rocket velocity data.")
        return
    if schema == "balloon_challenge":
        missing = [key for key in REQUIRED_BALLOON_CHALLENGE_OBSERVATION_KEYS if key not in observation]
        if missing:
            raise ValueError(f"Observation payload is missing required keys for '{schema}': {missing}")
        return


def validate_action_payload(action: dict[str, Any], schema: str) -> None:
    if schema == "flat_competition":
        missing = [key for key in REQUIRED_FLAT_ACTION_KEYS if key not in action]
        if missing:
            raise ValueError(f"Action payload is missing required keys for '{schema}': {missing}")
        throttle = float(action["throttle"])
        if not 0.0 <= throttle <= 1.0:
            raise ValueError(f"Throttle is out of range: {throttle}")
        return
    if schema == "balloon_challenge":
        missing = [key for key in REQUIRED_BALLOON_CHALLENGE_ACTION_KEYS if key not in action]
        if missing:
            raise ValueError(f"Action payload is missing required keys for '{schema}': {missing}")
        throttle = float(action["throttle"])
        if not 0.0 <= throttle <= 1.0:
            raise ValueError(f"Throttle is out of range: {throttle}")
        launch_vector = action["launch_inclination_heading"]
        if len(launch_vector) != 2:
            raise ValueError("launch_inclination_heading must contain exactly 2 elements.")
        tvc = action["tvc"]
        if len(tvc) != 2:
            raise ValueError("tvc must contain exactly 2 elements.")
