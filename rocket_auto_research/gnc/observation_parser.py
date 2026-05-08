from __future__ import annotations

from math import isfinite
from typing import Any

from rocket_auto_research.gnc.state import BalloonState, RocketState, Vector3, WorldState


def _sanitize_vector(payload: Any) -> Vector3:
    vector = Vector3.from_mapping(payload)
    for axis in ("x", "y", "z"):
        value = getattr(vector, axis)
        if not isfinite(value):
            setattr(vector, axis, 0.0)
    return vector


def _parse_internal_schema(observation: dict[str, Any]) -> WorldState:
    rocket_raw = observation.get("rocket", {})
    balloons_raw = observation.get("balloons", [])

    rocket = RocketState(
        position=_sanitize_vector(rocket_raw.get("position", {})),
        velocity=_sanitize_vector(rocket_raw.get("velocity", {})),
        attitude=_sanitize_vector(rocket_raw.get("attitude", {})),
        angular_rate=_sanitize_vector(rocket_raw.get("angular_rate", {})),
        launched=bool(rocket_raw.get("launched", False)),
        valid=bool(rocket_raw.get("valid", True)),
    )

    balloons: list[BalloonState] = []
    for index, balloon_raw in enumerate(balloons_raw):
        balloons.append(
            BalloonState(
                balloon_id=str(balloon_raw.get("id", index)),
                position=_sanitize_vector(balloon_raw.get("position", {})),
                velocity=_sanitize_vector(balloon_raw.get("velocity", {})),
                released=bool(balloon_raw.get("released", True)),
                popped=bool(balloon_raw.get("popped", False)),
            )
        )

    return WorldState(
        time_s=float(observation.get("time_s", 0.0) or 0.0),
        rocket=rocket,
        balloons=balloons,
        wind=_sanitize_vector(observation.get("wind", {})),
        metadata=dict(observation.get("metadata", {})),
    )


def _parse_flat_competition_schema(observation: dict[str, Any]) -> WorldState:
    balloons_raw = observation.get("balloons", observation.get("targets", []))
    rocket_position = observation.get("rocket_position", {})
    rocket_velocity = observation.get("rocket_velocity", {})
    rocket_attitude = observation.get("rocket_attitude", {})
    rocket_angular_rate = observation.get("rocket_angular_rate", {})
    wind_raw = observation.get("wind", {})
    rocket = RocketState(
        position=_sanitize_vector(
            {
                "x": rocket_position.get("x", observation.get("x", observation.get("pos_x", 0.0))),
                "y": rocket_position.get("y", observation.get("y", observation.get("pos_y", 0.0))),
                "z": rocket_position.get("z", observation.get("z", observation.get("pos_z", 0.0))),
            }
        ),
        velocity=_sanitize_vector(
            {
                "x": rocket_velocity.get("x", observation.get("vx", 0.0)),
                "y": rocket_velocity.get("y", observation.get("vy", 0.0)),
                "z": rocket_velocity.get("z", observation.get("vz", 0.0)),
            }
        ),
        attitude=_sanitize_vector(
            {
                "x": rocket_attitude.get("x", observation.get("pitch", 0.0)),
                "y": rocket_attitude.get("y", observation.get("yaw", 0.0)),
                "z": rocket_attitude.get("z", observation.get("roll", 0.0)),
            }
        ),
        angular_rate=_sanitize_vector(
            {
                "x": rocket_angular_rate.get("x", observation.get("wx", 0.0)),
                "y": rocket_angular_rate.get("y", observation.get("wy", 0.0)),
                "z": rocket_angular_rate.get("z", observation.get("wz", 0.0)),
            }
        ),
        launched=bool(observation.get("rocket_launched", observation.get("launched", True))),
        valid=not bool(observation.get("invalid", False)),
    )
    balloons = []
    for index, balloon_raw in enumerate(balloons_raw):
        balloons.append(
            BalloonState(
                balloon_id=str(balloon_raw.get("balloon_id", balloon_raw.get("id", index))),
                position=_sanitize_vector(balloon_raw.get("position", balloon_raw)),
                velocity=_sanitize_vector(balloon_raw.get("velocity", {})),
                released=bool(balloon_raw.get("released", True)),
                popped=bool(balloon_raw.get("popped", False)),
            )
        )
    metadata = dict(observation.get("metadata", {}))
    metadata["schema_detected"] = "flat_competition"
    return WorldState(
        time_s=float(observation.get("time_s", observation.get("t", 0.0)) or 0.0),
        rocket=rocket,
        balloons=balloons,
        wind=_sanitize_vector(
            {
                "x": wind_raw.get("x", observation.get("wind_x", 0.0)),
                "y": wind_raw.get("y", observation.get("wind_y", 0.0)),
                "z": wind_raw.get("z", observation.get("wind_z", 0.0)),
            }
        ),
        metadata=metadata,
    )


def _row_values(matrix: Any, index: int, count: int) -> list[float]:
    if isinstance(matrix, list) and index < len(matrix):
        row = matrix[index]
    else:
        try:
            row = matrix[index]
        except Exception:
            row = []
    values = list(row) if isinstance(row, (list, tuple)) else list(getattr(row, "tolist", lambda: [])())
    if len(values) < count:
        values.extend([0.0] * (count - len(values)))
    return [float(values[i] or 0.0) for i in range(count)]


def _scalar_status(raw_value: Any) -> int:
    if isinstance(raw_value, (list, tuple)) and raw_value:
        raw_value = raw_value[0]
    elif hasattr(raw_value, "shape"):
        values = getattr(raw_value, "tolist", lambda: raw_value)()
        if isinstance(values, list) and values:
            raw_value = values[0]
    try:
        return int(raw_value)
    except Exception:
        return 0


def _parse_balloon_challenge_schema(observation: dict[str, Any]) -> WorldState:
    balloon_status = observation.get("balloon_status", [])
    balloon_states = observation.get("balloon_states", [])
    rocket_sensors = list(observation.get("rocket_sensors", []))
    if len(rocket_sensors) < 12:
        rocket_sensors.extend([0.0] * (12 - len(rocket_sensors)))

    rocket = RocketState(
        position=Vector3(
            x=float(rocket_sensors[6] or 0.0),
            y=float(rocket_sensors[7] or 0.0),
            z=float(rocket_sensors[8] or 0.0),
        ),
        velocity=Vector3(
            x=float(rocket_sensors[9] or 0.0),
            y=float(rocket_sensors[10] or 0.0),
            z=float(rocket_sensors[11] or 0.0),
        ),
        attitude=Vector3(),
        angular_rate=Vector3(
            x=float(rocket_sensors[0] or 0.0),
            y=float(rocket_sensors[1] or 0.0),
            z=float(rocket_sensors[2] or 0.0),
        ),
        launched=any(isfinite(float(value or 0.0)) and abs(float(value or 0.0)) > 1e-9 for value in rocket_sensors),
        valid=True,
    )

    balloon_count = max(len(balloon_status), len(balloon_states))
    balloons: list[BalloonState] = []
    for index in range(balloon_count):
        status_value = _scalar_status(balloon_status[index] if index < len(balloon_status) else 0)
        row = _row_values(balloon_states, index, 6)
        balloons.append(
            BalloonState(
                balloon_id=f"balloon_{index:03d}",
                position=Vector3(x=row[0], y=row[1], z=row[2]),
                velocity=Vector3(x=row[3], y=row[4], z=row[5]),
                released=status_value >= 1,
                popped=status_value >= 2,
            )
        )

    metadata = dict(observation.get("metadata", {}))
    metadata["schema_detected"] = "balloon_challenge"
    metadata["rocket_sensors"] = rocket_sensors
    return WorldState(
        time_s=float(observation.get("simulation_time", 0.0) or 0.0),
        rocket=rocket,
        balloons=balloons,
        wind=Vector3(),
        metadata=metadata,
    )


def detect_observation_schema(observation: dict[str, Any]) -> str:
    if "rocket" in observation:
        return "internal"
    if all(key in observation for key in ("simulation_time", "balloon_status", "balloon_states", "rocket_sensors")):
        return "balloon_challenge"
    if any(
        key in observation
        for key in ("vx", "vy", "vz", "x", "y", "z", "targets", "rocket_position", "rocket_velocity")
    ):
        return "flat_competition"
    return "internal"


def parse_observation(observation: dict[str, Any], schema: str | None = None) -> WorldState:
    selected_schema = schema or detect_observation_schema(observation)
    if selected_schema == "internal":
        return _parse_internal_schema(observation)
    if selected_schema == "flat_competition":
        return _parse_flat_competition_schema(observation)
    if selected_schema == "balloon_challenge":
        return _parse_balloon_challenge_schema(observation)
    raise ValueError(f"Unsupported observation schema '{selected_schema}'.")
