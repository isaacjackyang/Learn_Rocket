from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from rocket_auto_research.auto_research.activerocketpy_adapter import (
    BalloonFieldBalloon,
    BalloonFieldCourse,
    _noise_vector,
)
from rocket_auto_research.auto_research.balloon_challenge_loader import load_balloon_challenge_scenario
from rocket_auto_research.auto_research.experiment_spec import ExperimentSpec
from rocket_auto_research.gnc.state import ControlAction, Vector3


@dataclass(slots=True)
class CompetitionStepState:
    time_s: float = 0.0
    step_count: int = 0
    position: Vector3 = field(default_factory=Vector3)
    velocity: Vector3 = field(default_factory=Vector3)
    attitude: Vector3 = field(default_factory=Vector3)
    angular_rate: Vector3 = field(default_factory=Vector3)
    launched: bool = False
    valid: bool = True
    burn_time_s: float = 0.0
    popped: int = 0
    launch_time: float | None = None
    first_pop_time: float | None = None
    target_switch_count: int = 0
    target_history: list[str] = field(default_factory=list)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    min_distance_to_any_balloon: float = float("inf")
    tvc_saturation_samples: list[float] = field(default_factory=list)
    last_target_id: str | None = None


class SimulatedCompetitionEnv:
    def __init__(self, spec: ExperimentSpec, seed: int) -> None:
        self.spec = spec
        self.seed = seed
        self.rng = random.Random(f"{spec.experiment_id}:{seed}:competition-platform")
        self.challenge_scenario = load_balloon_challenge_scenario(
            repo_root=spec.params.get("challenge_repo_root", ".external/BalloonPoppingChallenge"),
            scenario_number=int(spec.params.get("challenge_scenario_number", 1)),
            scenario_path=spec.params.get("challenge_scenario_path"),
        )
        scenario_sim = self.challenge_scenario.simulation if self.challenge_scenario is not None else {}
        scenario_env = self.challenge_scenario.environment if self.challenge_scenario is not None else {}
        scenario_balloon = self.challenge_scenario.balloon if self.challenge_scenario is not None else {}
        scenario_control = (
            self.challenge_scenario.rocket.get("control", {}) if self.challenge_scenario is not None else {}
        )
        self.dt = float(spec.params.get("api_dt_s", scenario_sim.get("time_step", 0.1)))
        self.max_time_s = float(spec.params.get("api_max_time_s", scenario_sim.get("max_time", 35.0)))
        self.max_tvc = float(spec.params.get("max_tvc", scenario_control.get("gimbal_range", 1.0)))
        self.max_roll = float(spec.params.get("max_roll", scenario_control.get("max_roll_torque", 1.0)))
        self.max_accel_mps2 = float(spec.params.get("api_max_accel_mps2", 55.0))
        self.drag_coeff = float(spec.params.get("api_drag_coeff", 0.02))
        self.wind_coupling = float(spec.params.get("api_wind_coupling", 0.04))
        self.angular_gain = float(spec.params.get("api_angular_gain", 1.15))
        self.angular_damping = float(spec.params.get("api_angular_damping", 0.22))
        self.burn_duration_s = float(spec.params.get("api_burn_duration_s", 16.0))
        self.elevation_m = float(spec.params.get("elevation_m", scenario_env.get("elevation", 1400.0)))
        sensor_cfg = self.challenge_scenario.rocket.get("sensors", {}) if self.challenge_scenario is not None else {}
        self.position_noise_std_m = float(spec.params.get("rocket_position_noise_std_m", sensor_cfg.get("gnss_position_accuracy", 0.7)))
        self.velocity_noise_std_mps = float(spec.params.get("rocket_velocity_noise_std_mps", sensor_cfg.get("gnss_velocity_accuracy", 0.25)))
        self.wind_noise_std_mps = float(spec.params.get("wind_sensor_noise_std_mps", 0.3))
        self.balloon_position_noise_std_m = float(spec.params.get("position_noise_std_m", 2.2))
        self.balloon_velocity_noise_std_mps = float(spec.params.get("velocity_noise_std_mps", 0.45))
        self.balloon_release_interval_s = float(spec.params.get("release_interval_s", scenario_balloon.get("release_interval", 0.5)))
        self.challenge_reward_mode = str(spec.params.get("challenge_reward_mode", "popped_count"))
        self.balloons = self._seeded_balloon_field()
        self.course = BalloonFieldCourse(self.balloons)
        self.wind_profile = self._seeded_wind_profile()
        self.state = CompetitionStepState(position=Vector3(z=self.elevation_m))

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        self.course = BalloonFieldCourse(self.balloons)
        self.state = CompetitionStepState(position=Vector3(z=self.elevation_m))
        observation = self._build_observation()
        info = {"seed": self.seed, "adapter": "competition_platform"}
        return observation, info

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        launch_vector = action.get("launch_inclination_heading", [90.0, 0.0])
        if not isinstance(launch_vector, (list, tuple)) or len(launch_vector) != 2:
            launch_vector = [90.0, 0.0]
        tvc_vector = action.get("tvc", None)
        if isinstance(tvc_vector, (list, tuple)) and len(tvc_vector) == 2:
            tvc_x = float(tvc_vector[0])
            tvc_y = float(tvc_vector[1])
        else:
            tvc_x = float(action.get("tvc_x", 0.0))
            tvc_y = float(action.get("tvc_y", 0.0))
        control = ControlAction(
            launch=bool(action.get("launch", False)),
            launch_inclination_deg=float(launch_vector[0]),
            launch_heading_deg=float(launch_vector[1]),
            throttle=max(0.0, min(1.0, float(action.get("throttle", 0.0)))),
            tvc_x=max(-self.max_tvc, min(self.max_tvc, tvc_x)),
            tvc_y=max(-self.max_tvc, min(self.max_tvc, tvc_y)),
            roll=max(-self.max_roll, min(self.max_roll, float(action.get("roll", 0.0)))),
        )
        reward = self._integrate(control)
        terminated = self._is_terminated()
        truncated = self.state.time_s >= self.max_time_s
        observation = self._build_observation()
        info = self._build_info()
        return observation, reward, terminated, truncated, info

    def record_target(self, target_id: str | None) -> None:
        if target_id != self.state.last_target_id:
            if self.state.last_target_id is not None:
                self.state.target_switch_count += 1
            self.state.last_target_id = target_id
            if target_id is not None:
                self.state.target_history.append(target_id)

    def _integrate(self, action: ControlAction) -> float:
        previous_popped = self.state.popped
        previous_position = Vector3(self.state.position.x, self.state.position.y, self.state.position.z)
        previous_time_s = self.state.time_s
        if action.launch and not self.state.launched:
            self.state.launched = True
            self.state.launch_time = self.state.time_s
            inclination_rad = math.radians(max(0.0, min(90.0, action.launch_inclination_deg)))
            heading_rad = math.radians(action.launch_heading_deg)
            horizontal = math.cos(inclination_rad)
            self.state.attitude = Vector3(
                x=horizontal * math.cos(heading_rad),
                y=horizontal * math.sin(heading_rad),
                z=math.sin(inclination_rad),
            )
        wind = self._wind_at_altitude(self.state.position.z)
        if self.state.launched and self.state.burn_time_s < self.burn_duration_s:
            self.state.burn_time_s += self.dt * max(0.2, action.throttle)
            self.state.angular_rate = Vector3(
                x=self.state.angular_rate.x * (1.0 - self.angular_damping * self.dt)
                + self.angular_gain * action.tvc_y * self.dt,
                y=self.state.angular_rate.y * (1.0 - self.angular_damping * self.dt)
                + self.angular_gain * action.tvc_x * self.dt,
                z=self.state.angular_rate.z * (1.0 - self.angular_damping * self.dt)
                + 0.6 * action.roll * self.dt,
            )
            self.state.attitude = Vector3(
                x=self.state.attitude.x + self.state.angular_rate.x * self.dt,
                y=self.state.attitude.y + self.state.angular_rate.y * self.dt,
                z=self.state.attitude.z + self.state.angular_rate.z * self.dt,
            )
            thrust_axis = Vector3(
                x=0.28 * self.state.attitude.y + 0.85 * action.tvc_x,
                y=0.28 * self.state.attitude.x + 0.85 * action.tvc_y,
                z=1.0,
            ).normalized()
            thrust_accel = thrust_axis.scale(self.max_accel_mps2 * action.throttle)
        else:
            thrust_accel = Vector3()
        drag = self.state.velocity.scale(-self.drag_coeff * self.state.velocity.norm())
        wind_push = (wind - self.state.velocity).scale(self.wind_coupling)
        gravity = Vector3(z=-9.81)
        acceleration = thrust_accel + drag + wind_push + gravity
        self.state.velocity = self.state.velocity + acceleration.scale(self.dt)
        self.state.position = self.state.position + self.state.velocity.scale(self.dt)
        if self.state.position.z < self.elevation_m:
            self.state.position.z = self.elevation_m
        self.state.time_s += self.dt
        self.state.step_count += 1
        self.course.update(
            self.state.position,
            self.state.time_s,
            wind,
            previous_rocket_position=previous_position,
            previous_time_s=previous_time_s,
        )
        self.state.popped = self.course.popped_count
        self.state.first_pop_time = self.course.first_pop_time
        self.state.min_distance_to_any_balloon = self.course.min_distance_to_any_balloon
        self.state.tvc_saturation_samples.append(max(abs(action.tvc_x), abs(action.tvc_y)) / max(self.max_tvc, 1e-6))
        if self.state.step_count % 3 == 0:
            self.state.trajectory.append(
                {
                    "time_s": round(self.state.time_s, 4),
                    "rocket_position": self.state.position.to_dict(),
                    "rocket_velocity": self.state.velocity.to_dict(),
                    "popped": self.state.popped,
                    "released": self.course.release_count,
                    "target_id": self.state.last_target_id,
                }
            )
        pop_reward = 950.0 * max(0, self.state.popped - previous_popped)
        shaping = max(0.0, 18.0 - self.course.min_distance_to_any_balloon * 0.08)
        return pop_reward + shaping

    def _build_observation(self) -> dict[str, Any]:
        wind = self._wind_at_altitude(self.state.position.z)
        balloon_status: list[list[int]] = []
        balloon_states: list[list[float]] = []
        balloons_payload: list[dict[str, Any]] = []
        for balloon in self.balloons:
            released = balloon.is_released(self.state.time_s)
            popped = balloon.balloon_id in self.course.popped_ids
            position = balloon.position_at(self.state.time_s, wind) if released else balloon.spawn_position
            velocity = balloon.velocity_at(wind) if released else balloon.drift_velocity
            noisy_position = _noise_vector(position, self.balloon_position_noise_std_m, self.rng)
            noisy_velocity = _noise_vector(velocity, self.balloon_velocity_noise_std_mps, self.rng)
            status = 2 if popped else 1 if released else 0
            balloon_status.append([status])
            balloon_states.append(
                [
                    float(noisy_position.x),
                    float(noisy_position.y),
                    float(noisy_position.z),
                    float(noisy_velocity.x),
                    float(noisy_velocity.y),
                    float(noisy_velocity.z),
                ]
            )
            balloons_payload.append(
                {
                    "balloon_id": balloon.balloon_id,
                    "position": noisy_position.to_dict(),
                    "velocity": noisy_velocity.to_dict(),
                    "released": released,
                    "popped": popped,
                }
            )
        rocket_position = _noise_vector(self.state.position, self.position_noise_std_m, self.rng)
        rocket_velocity = _noise_vector(self.state.velocity, self.velocity_noise_std_mps, self.rng)
        wind_noisy = _noise_vector(wind, self.wind_noise_std_mps, self.rng)
        rocket_sensors = [
            float(self.state.angular_rate.x),
            float(self.state.angular_rate.y),
            float(self.state.angular_rate.z),
            0.0,
            0.0,
            -9.81,
            float(rocket_position.x),
            float(rocket_position.y),
            float(rocket_position.z),
            float(rocket_velocity.x),
            float(rocket_velocity.y),
            float(rocket_velocity.z),
        ]
        return {
            "time_s": round(self.state.time_s, 4),
            "simulation_time": round(self.state.time_s, 4),
            "rocket_position": rocket_position.to_dict(),
            "rocket_velocity": rocket_velocity.to_dict(),
            "rocket_attitude": self.state.attitude.to_dict(),
            "rocket_angular_rate": self.state.angular_rate.to_dict(),
            "rocket_launched": self.state.launched,
            "wind": wind_noisy.to_dict(),
            "balloon_status": balloon_status,
            "balloon_states": balloon_states,
            "rocket_sensors": rocket_sensors,
            "balloons": balloons_payload,
            "metadata": {
                "altitude_agl_m": self.state.position.z - self.elevation_m,
                "balloons_total": len(self.balloons),
                "balloons_released": self.course.release_count,
                "balloons_popped": self.state.popped,
                "score_is_proxy": True,
                "challenge_scenario_number": self.challenge_scenario.scenario_number if self.challenge_scenario is not None else None,
                "challenge_scenario_path": str(self.challenge_scenario.source_path) if self.challenge_scenario is not None else None,
            },
        }

    def _build_info(self) -> dict[str, Any]:
        wind = self._wind_at_altitude(self.state.position.z)
        return {
            "time_s": round(self.state.time_s, 4),
            "popped": self.state.popped,
            "released": self.course.release_count,
            "remaining": self.course.remaining_count,
            "first_pop_time": self.state.first_pop_time,
            "launch_time": self.state.launch_time,
            "min_distance_to_any_balloon": self.course.min_distance_to_any_balloon,
            "wind_drift_m": math.hypot(self.state.position.x, self.state.position.y),
            "mean_relative_speed_mps": self.state.velocity.norm(),
            "sensor_jitter_index": self.position_noise_std_m + self.balloon_position_noise_std_m * 0.08,
            "trajectory": self.state.trajectory,
            "target_history": self.state.target_history,
            "target_switch_count": self.state.target_switch_count,
            "tvc_saturation_ratio": sum(self.state.tvc_saturation_samples) / max(len(self.state.tvc_saturation_samples), 1),
            "wind_vector": wind.to_dict(),
            "score_is_proxy": True,
            "rocket_states": {
                "position": self.state.position.to_dict(),
                "velocity": self.state.velocity.to_dict(),
                "attitude": self.state.attitude.to_dict(),
                "angular_rate": self.state.angular_rate.to_dict(),
            },
            "challenge_scenario_number": self.challenge_scenario.scenario_number if self.challenge_scenario is not None else None,
            "challenge_scenario_path": str(self.challenge_scenario.source_path) if self.challenge_scenario is not None else None,
            "balloons_total": len(self.balloons),
            "balloons_released": self.course.release_count,
            "balloons_popped": self.state.popped,
            "target_chatter_threshold": 10,
            "tvc_saturation_threshold": 0.72,
            "recommended_launch_floor_s": 0.05,
            "recommended_launch_ceiling_s": 0.75,
            "target_altitude_floor_m": 850.0,
            "velocity_floor_mps": 28.0,
            "near_intercept_threshold_m": 60.0,
        }

    def final_score(self) -> float:
        if self.challenge_reward_mode == "popped_count":
            return float(self.state.popped)
        info = self._build_info()
        return max(
            0.0,
            self.state.popped * 1000.0
            + max(0.0, 180.0 - (self.state.first_pop_time or 30.0) * 7.0)
            - 0.07 * float(info["wind_drift_m"])
            - 55.0 * float(info["tvc_saturation_ratio"])
            - 0.08 * self.course.min_distance_to_any_balloon,
        )

    def _is_terminated(self) -> bool:
        if self.state.launched and self.state.position.z <= self.elevation_m and self.state.time_s > 1.5:
            return True
        if self.state.popped >= len(self.balloons):
            return True
        return False

    def _wind_at_altitude(self, altitude_asl_m: float) -> Vector3:
        return Vector3(
            x=self._interpolate_profile(self.wind_profile["u"], altitude_asl_m - self.elevation_m),
            y=self._interpolate_profile(self.wind_profile["v"], altitude_asl_m - self.elevation_m),
            z=0.0,
        )

    @staticmethod
    def _interpolate_profile(profile: list[tuple[float, float]], altitude_agl_m: float) -> float:
        if not profile:
            return 0.0
        if altitude_agl_m <= profile[0][0]:
            return profile[0][1]
        for left, right in zip(profile, profile[1:]):
            if left[0] <= altitude_agl_m <= right[0]:
                span = max(right[0] - left[0], 1e-6)
                ratio = (altitude_agl_m - left[0]) / span
                return left[1] + ratio * (right[1] - left[1])
        return profile[-1][1]

    def _seeded_balloon_field(self) -> list[BalloonFieldBalloon]:
        scenario_balloon = self.challenge_scenario.balloon if self.challenge_scenario is not None else {}
        scenario_number = self.challenge_scenario.scenario_number if self.challenge_scenario is not None else -1
        balloon_count = int(self.spec.params.get("balloon_count", scenario_balloon.get("num", 100)))
        release_step_spacing = max(1, int(round(self.balloon_release_interval_s / max(self.dt, 1e-6))))
        release_steps = [index * release_step_spacing for index in range(balloon_count)]
        self.rng.shuffle(release_steps)
        balloons: list[BalloonFieldBalloon] = []
        for index in range(balloon_count):
            if scenario_number == 0:
                altitude = self.elevation_m + 10.0 + index * 40.0
                radial_distance = 0.0
                azimuth = 0.0
                release_time = 0.0
                drift_speed = 0.0
                drift_heading = 0.0
                vertical_rate = 0.0
            else:
                altitude = self.elevation_m + 120.0 + (index / max(balloon_count - 1, 1)) * 2200.0 + self.rng.uniform(-35.0, 35.0)
                radial_distance = max(0.0, (self.rng.random() ** 2.0) * min(220.0, 25.0 + (altitude - self.elevation_m) * 0.055))
                azimuth = self.rng.uniform(0.0, 2.0 * math.pi)
                release_time = release_steps[index] * self.dt
                drift_speed = self.rng.uniform(0.1, 2.4)
                drift_heading = self.rng.uniform(0.0, 2.0 * math.pi)
                vertical_rate = self.rng.uniform(-0.15, 0.7)
            balloons.append(
                BalloonFieldBalloon(
                    balloon_id=f"balloon_{index:03d}",
                    spawn_position=Vector3(
                        x=radial_distance * math.cos(azimuth),
                        y=radial_distance * math.sin(azimuth),
                        z=altitude,
                    ),
                    drift_velocity=Vector3(
                        x=drift_speed * math.cos(drift_heading),
                        y=drift_speed * math.sin(drift_heading),
                        z=vertical_rate,
                    ),
                    release_time_s=release_time,
                    radius_m=float(self.spec.params.get("balloon_radius_m", scenario_balloon.get("radius", 1.5))),
                )
            )
        return balloons

    def _seeded_wind_profile(self) -> dict[str, list[tuple[float, float]]]:
        altitudes = [0.0, 200.0, 700.0, 1400.0, 2200.0, 3200.0]
        surface_wind = float(self.spec.params.get("wind_surface_mps", 2.5))
        shear = float(self.spec.params.get("wind_shear_mps", 4.5))
        gust = float(self.spec.params.get("wind_gust_mps", 0.8))
        heading = self.rng.uniform(0.0, 2.0 * math.pi)
        veer = self.rng.uniform(-0.28, 0.28)
        wind_u: list[tuple[float, float]] = []
        wind_v: list[tuple[float, float]] = []
        for altitude in altitudes:
            ratio = altitude / max(altitudes[-1], 1.0)
            speed = surface_wind + shear * ratio + self.rng.uniform(-gust, gust)
            local_heading = heading + veer * ratio
            wind_u.append((altitude, round(speed * math.cos(local_heading), 4)))
            wind_v.append((altitude, round(speed * math.sin(local_heading), 4)))
        return {"u": wind_u, "v": wind_v}
