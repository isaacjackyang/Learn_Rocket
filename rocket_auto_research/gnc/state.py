from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any


@dataclass(slots=True)
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def scale(self, factor: float) -> "Vector3":
        return Vector3(self.x * factor, self.y * factor, self.z * factor)

    def norm(self) -> float:
        return sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> "Vector3":
        magnitude = self.norm()
        if magnitude <= 1e-9:
            return Vector3()
        return self.scale(1.0 / magnitude)

    def to_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "z": self.z}

    @classmethod
    def from_mapping(cls, payload: Any) -> "Vector3":
        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, dict):
            return cls()
        return cls(
            x=float(payload.get("x", 0.0) or 0.0),
            y=float(payload.get("y", 0.0) or 0.0),
            z=float(payload.get("z", 0.0) or 0.0),
        )


@dataclass(slots=True)
class BalloonState:
    balloon_id: str
    position: Vector3
    velocity: Vector3 = field(default_factory=Vector3)
    released: bool = True
    popped: bool = False


@dataclass(slots=True)
class RocketState:
    position: Vector3 = field(default_factory=Vector3)
    velocity: Vector3 = field(default_factory=Vector3)
    attitude: Vector3 = field(default_factory=Vector3)
    angular_rate: Vector3 = field(default_factory=Vector3)
    launched: bool = False
    valid: bool = True


@dataclass(slots=True)
class WorldState:
    time_s: float
    rocket: RocketState
    balloons: list[BalloonState]
    wind: Vector3 = field(default_factory=Vector3)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GuidanceCommand:
    desired_direction: Vector3 = field(default_factory=Vector3)
    desired_speed: float = 0.0
    target_id: str | None = None
    target_position: Vector3 | None = None


@dataclass(slots=True)
class ControlAction:
    launch: bool = False
    launch_inclination_deg: float = 90.0
    launch_heading_deg: float = 0.0
    throttle: float = 0.0
    tvc_x: float = 0.0
    tvc_y: float = 0.0
    roll: float = 0.0

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "launch": self.launch,
            "launch_inclination_deg": self.launch_inclination_deg,
            "launch_heading_deg": self.launch_heading_deg,
            "throttle": self.throttle,
            "tvc_x": self.tvc_x,
            "tvc_y": self.tvc_y,
            "roll": self.roll,
        }
