from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rocket_auto_research.gnc.state import BalloonState, RocketState, Vector3, WorldState


def _blend(previous: Vector3, current: Vector3, alpha: float) -> Vector3:
    return Vector3(
        x=(1.0 - alpha) * previous.x + alpha * current.x,
        y=(1.0 - alpha) * previous.y + alpha * current.y,
        z=(1.0 - alpha) * previous.z + alpha * current.z,
    )


@dataclass(slots=True)
class SimpleEstimator:
    alpha: float = 0.45
    _last_velocity: Vector3 = field(default_factory=Vector3)
    _last_wind: Vector3 = field(default_factory=Vector3)

    def update(self, world_state: WorldState) -> WorldState:
        rocket = RocketState(
            position=world_state.rocket.position,
            velocity=_blend(self._last_velocity, world_state.rocket.velocity, self.alpha),
            attitude=world_state.rocket.attitude,
            angular_rate=world_state.rocket.angular_rate,
            launched=world_state.rocket.launched,
            valid=world_state.rocket.valid,
        )
        wind = _blend(self._last_wind, world_state.wind, self.alpha)
        self._last_velocity = rocket.velocity
        self._last_wind = wind
        return WorldState(
            time_s=world_state.time_s,
            rocket=rocket,
            balloons=world_state.balloons,
            wind=wind,
            metadata=world_state.metadata,
        )


@dataclass(slots=True)
class AlphaBetaEstimator:
    alpha: float = 0.55
    beta: float = 0.18
    _last_time_s: float | None = None
    _position_estimate: Vector3 = field(default_factory=Vector3)
    _velocity_estimate: Vector3 = field(default_factory=Vector3)

    def update(self, world_state: WorldState) -> WorldState:
        if self._last_time_s is None:
            self._last_time_s = world_state.time_s
            self._position_estimate = world_state.rocket.position
            self._velocity_estimate = world_state.rocket.velocity
            return world_state

        dt = max(1e-3, world_state.time_s - self._last_time_s)
        predicted_position = self._position_estimate + self._velocity_estimate.scale(dt)
        residual = world_state.rocket.position - predicted_position
        position_estimate = predicted_position + residual.scale(self.alpha)
        velocity_estimate = self._velocity_estimate + residual.scale(self.beta / dt)
        self._last_time_s = world_state.time_s
        self._position_estimate = position_estimate
        self._velocity_estimate = velocity_estimate
        return WorldState(
            time_s=world_state.time_s,
            rocket=RocketState(
                position=position_estimate,
                velocity=velocity_estimate,
                attitude=world_state.rocket.attitude,
                angular_rate=world_state.rocket.angular_rate,
                launched=world_state.rocket.launched,
                valid=world_state.rocket.valid,
            ),
            balloons=world_state.balloons,
            wind=world_state.wind,
            metadata=world_state.metadata,
        )


@dataclass(slots=True)
class WindAwareEstimator:
    alpha: float = 0.45
    wind_bias_gain: float = 0.15
    _simple: SimpleEstimator = field(default_factory=SimpleEstimator)
    _wind_bias: Vector3 = field(default_factory=Vector3)

    def update(self, world_state: WorldState) -> WorldState:
        filtered = self._simple.update(world_state)
        apparent_wind = filtered.wind - filtered.rocket.velocity.scale(0.05)
        self._wind_bias = _blend(self._wind_bias, apparent_wind, self.wind_bias_gain)
        metadata = dict(filtered.metadata)
        metadata["estimated_wind_bias"] = self._wind_bias.to_dict()
        return WorldState(
            time_s=filtered.time_s,
            rocket=filtered.rocket,
            balloons=filtered.balloons,
            wind=filtered.wind + self._wind_bias.scale(0.25),
            metadata=metadata,
        )


@dataclass(slots=True)
class JitterAwareEstimator:
    alpha: float = 0.35
    balloon_alpha: float = 0.28
    jump_threshold_m: float = 45.0
    _rocket_position: Vector3 = field(default_factory=Vector3)
    _rocket_velocity: Vector3 = field(default_factory=Vector3)
    _last_balloons: dict[str, BalloonState] = field(default_factory=dict)
    _initialized: bool = False

    def update(self, world_state: WorldState) -> WorldState:
        if not self._initialized:
            self._rocket_position = world_state.rocket.position
            self._rocket_velocity = world_state.rocket.velocity
            self._initialized = True
        self._rocket_position = _blend(self._rocket_position, world_state.rocket.position, self.alpha)
        self._rocket_velocity = _blend(self._rocket_velocity, world_state.rocket.velocity, self.alpha)
        filtered_balloons: list[BalloonState] = []
        new_cache: dict[str, BalloonState] = {}
        for balloon in world_state.balloons:
            previous = self._last_balloons.get(balloon.balloon_id)
            if previous is None:
                filtered = balloon
            else:
                jump = (balloon.position - previous.position).norm()
                blend_alpha = self.balloon_alpha * 0.5 if jump > self.jump_threshold_m else self.balloon_alpha
                filtered = BalloonState(
                    balloon_id=balloon.balloon_id,
                    position=_blend(previous.position, balloon.position, blend_alpha),
                    velocity=_blend(previous.velocity, balloon.velocity, blend_alpha),
                    released=balloon.released,
                    popped=balloon.popped,
                )
            filtered_balloons.append(filtered)
            new_cache[balloon.balloon_id] = filtered
        self._last_balloons = new_cache
        metadata = dict(world_state.metadata)
        metadata["jitter_filtered"] = True
        return WorldState(
            time_s=world_state.time_s,
            rocket=RocketState(
                position=self._rocket_position,
                velocity=self._rocket_velocity,
                attitude=world_state.rocket.attitude,
                angular_rate=world_state.rocket.angular_rate,
                launched=world_state.rocket.launched,
                valid=world_state.rocket.valid,
            ),
            balloons=filtered_balloons,
            wind=world_state.wind,
            metadata=metadata,
        )


def build_estimator(params: dict[str, Any]):
    mode = str(params.get("estimator_mode", "simple")).lower()
    if mode == "simple":
        return SimpleEstimator(alpha=float(params.get("alpha", 0.45)))
    if mode in {"alpha_beta", "alphabeta"}:
        return AlphaBetaEstimator(
            alpha=float(params.get("alpha", 0.55)),
            beta=float(params.get("beta", 0.18)),
        )
    if mode in {"wind_aware", "windaware"}:
        return WindAwareEstimator(
            alpha=float(params.get("alpha", 0.45)),
            wind_bias_gain=float(params.get("wind_bias_gain", 0.15)),
            _simple=SimpleEstimator(alpha=float(params.get("alpha", 0.45))),
        )
    if mode in {"jitter_aware", "jitteraware"}:
        return JitterAwareEstimator(
            alpha=float(params.get("alpha", 0.35)),
            balloon_alpha=float(params.get("balloon_alpha", 0.28)),
            jump_threshold_m=float(params.get("jump_threshold_m", 45.0)),
        )
    raise ValueError(f"Unknown estimator_mode '{mode}'.")
