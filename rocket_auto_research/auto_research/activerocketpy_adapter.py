from __future__ import annotations

import math
import random
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rocket_auto_research.auto_research.external_paths import (
    DEFAULT_CHALLENGE_ACTIVEROCKETPY,
    DEFAULT_VENDOR_ACTIVEROCKETPY,
    activerocketpy_setup_hint,
    resolve_first_existing_path,
)
from rocket_auto_research.auto_research.experiment_spec import ExperimentSpec
from rocket_auto_research.auto_research.simulation import EpisodeResult, SimulationAdapter
from rocket_auto_research.gnc.state import BalloonState, ControlAction, RocketState, Vector3, WorldState
from rocket_auto_research.strategies.registry import build_strategy


def _import_rocketpy(vendor_repo: str | Path | None = None):
    repo_path = resolve_first_existing_path(
        vendor_repo,
        DEFAULT_VENDOR_ACTIVEROCKETPY,
        DEFAULT_CHALLENGE_ACTIVEROCKETPY,
    )
    if repo_path is None:
        raise FileNotFoundError(activerocketpy_setup_hint())
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    import rocketpy  # type: ignore

    return rocketpy


@dataclass(slots=True)
class BalloonFieldBalloon:
    balloon_id: str
    spawn_position: Vector3
    drift_velocity: Vector3
    release_time_s: float
    radius_m: float

    def is_released(self, time_s: float) -> bool:
        return time_s >= self.release_time_s

    def position_at(self, time_s: float, ambient_wind: Vector3) -> Vector3:
        dt = max(0.0, time_s - self.release_time_s)
        wind_drift = ambient_wind.scale(0.55 * dt)
        return self.spawn_position + self.drift_velocity.scale(dt) + wind_drift

    def velocity_at(self, ambient_wind: Vector3) -> Vector3:
        return self.drift_velocity + ambient_wind.scale(0.55)


@dataclass(slots=True)
class BalloonFieldCourse:
    balloons: list[BalloonFieldBalloon]
    popped_ids: set[str] = field(default_factory=set)
    near_miss_ids: set[str] = field(default_factory=set)
    first_pop_time: float | None = None
    min_distance_to_any_balloon: float = float("inf")
    near_miss_count: int = 0
    miss_distances: list[float] = field(default_factory=list)
    release_count: int = 0

    @staticmethod
    def _closest_distance_between_segments(
        start_a: Vector3,
        end_a: Vector3,
        start_b: Vector3,
        end_b: Vector3,
    ) -> float:
        epsilon = 1e-9
        u = end_a - start_a
        v = end_b - start_b
        w = start_a - start_b
        a = u.x * u.x + u.y * u.y + u.z * u.z
        b = u.x * v.x + u.y * v.y + u.z * v.z
        c = v.x * v.x + v.y * v.y + v.z * v.z
        d = u.x * w.x + u.y * w.y + u.z * w.z
        e = v.x * w.x + v.y * w.y + v.z * w.z
        denom = a * c - b * b

        if a <= epsilon and c <= epsilon:
            return (start_a - start_b).norm()
        if a <= epsilon:
            s = 0.0
            t = max(0.0, min(1.0, e / max(c, epsilon)))
        elif c <= epsilon:
            t = 0.0
            s = max(0.0, min(1.0, -d / max(a, epsilon)))
        else:
            if abs(denom) <= epsilon:
                s = 0.0
            else:
                s = max(0.0, min(1.0, (b * e - c * d) / denom))
            t = (b * s + e) / c
            if t < 0.0:
                t = 0.0
                s = max(0.0, min(1.0, -d / a))
            elif t > 1.0:
                t = 1.0
                s = max(0.0, min(1.0, (b - d) / a))

        closest_a = start_a + u.scale(s)
        closest_b = start_b + v.scale(t)
        return (closest_a - closest_b).norm()

    def update(
        self,
        rocket_position: Vector3,
        time_s: float,
        ambient_wind: Vector3,
        previous_rocket_position: Vector3 | None = None,
        previous_time_s: float | None = None,
    ) -> None:
        self.release_count = sum(1 for balloon in self.balloons if balloon.is_released(time_s))
        for balloon in self.balloons:
            if not balloon.is_released(time_s) or balloon.balloon_id in self.popped_ids:
                continue
            position = balloon.position_at(time_s, ambient_wind)
            if previous_rocket_position is not None and previous_time_s is not None:
                previous_balloon_position = balloon.position_at(previous_time_s, ambient_wind)
                distance = self._closest_distance_between_segments(
                    previous_rocket_position,
                    rocket_position,
                    previous_balloon_position,
                    position,
                )
            else:
                distance = (position - rocket_position).norm()
            self.min_distance_to_any_balloon = min(self.min_distance_to_any_balloon, distance)
            if distance <= balloon.radius_m:
                self.popped_ids.add(balloon.balloon_id)
                if self.first_pop_time is None:
                    self.first_pop_time = time_s
                continue
            if distance <= balloon.radius_m * 2.25 and balloon.balloon_id not in self.near_miss_ids:
                self.near_miss_ids.add(balloon.balloon_id)
                self.near_miss_count += 1
                self.miss_distances.append(distance)

    def balloons_for_observation(
        self,
        time_s: float,
        ambient_wind: Vector3,
        position_noise_std_m: float,
        velocity_noise_std_mps: float,
        rng: random.Random,
    ) -> list[BalloonState]:
        observed: list[BalloonState] = []
        for balloon in self.balloons:
            released = balloon.is_released(time_s)
            position = balloon.position_at(time_s, ambient_wind) if released else balloon.spawn_position
            velocity = balloon.velocity_at(ambient_wind) if released else balloon.drift_velocity
            observed.append(
                BalloonState(
                    balloon_id=balloon.balloon_id,
                    position=_noise_vector(position, position_noise_std_m, rng) if released else position,
                    velocity=_noise_vector(velocity, velocity_noise_std_mps, rng) if released else velocity,
                    released=released,
                    popped=balloon.balloon_id in self.popped_ids,
                )
            )
        return observed

    @property
    def popped_count(self) -> int:
        return len(self.popped_ids)

    @property
    def remaining_count(self) -> int:
        return len(self.balloons) - len(self.popped_ids)

    @property
    def miss_distance_p50(self) -> float:
        if not self.miss_distances:
            return float("inf")
        return float(statistics.median(self.miss_distances))


@dataclass(slots=True)
class StrategyLoopState:
    last_action: ControlAction = field(default_factory=ControlAction)
    last_target_id: str | None = None
    target_switch_count: int = 0
    target_history: list[str] = field(default_factory=list)
    controller_log: list[dict[str, float | str | None]] = field(default_factory=list)
    launch_time: float | None = None
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    sensor_jitter_samples: list[float] = field(default_factory=list)
    relative_speed_samples: list[float] = field(default_factory=list)
    last_observed_world: WorldState | None = None


@dataclass(slots=True)
class ActiveRocketPySimulationAdapter(SimulationAdapter):
    vendor_repo: str | Path = str(DEFAULT_VENDOR_ACTIVEROCKETPY)
    terminate_on_apogee: bool = True

    def run_episode(self, spec: ExperimentSpec, seed: int) -> EpisodeResult:
        rocketpy = _import_rocketpy(self.vendor_repo)
        strategy_params = dict(spec.params)
        strategy = build_strategy(spec.strategy_name, strategy_params)
        rng = random.Random(f"{spec.experiment_id}:{seed}:balloon-field")
        course = BalloonFieldCourse(self._seeded_balloon_field(spec, seed))
        loop_state = StrategyLoopState()
        noise_config = self._noise_config(spec)
        wind_profile = self._seeded_wind_profile(spec, seed)

        env, rocket = self._build_calisto_scenario(
            rocketpy=rocketpy,
            spec=spec,
            seed=seed,
            strategy=strategy,
            course=course,
            loop_state=loop_state,
            noise_config=noise_config,
            wind_profile=wind_profile,
            rng=rng,
        )

        try:
            flight = rocketpy.Flight(
                rocket=rocket,
                environment=env,
                rail_length=float(spec.params.get("rail_length_m", 5.2)),
                inclination=float(spec.params.get("launch_inclination_deg", 85.0)),
                heading=float(spec.params.get("launch_heading_deg", 0.0)),
                terminate_on_apogee=self.terminate_on_apogee,
                run_simulation=False,
            )
            safety_counter = 0
            while not flight._step_state["finished"]:
                flight.step_simulation()
                safety_counter += 1
                if safety_counter > 25000:  # pragma: no cover
                    raise RuntimeError("ActiveRocketPy step simulation exceeded safety iteration limit.")
        except Exception as exc:  # pragma: no cover
            return EpisodeResult(
                seed=seed,
                score=0.0,
                popped=course.popped_count,
                crashed=True,
                nan_detected=True,
                duration=0.0,
                time_to_launch=round(float(spec.params.get("launch_wait_time", 0.0)), 4),
                time_to_first_pop=course.first_pop_time,
                min_distance_to_any_balloon=999999.0,
                target_switch_count=loop_state.target_switch_count,
                tvc_saturation_ratio=self._tvc_saturation_ratio(loop_state.controller_log, float(spec.params.get("max_tvc", 3.0))),
                late_launch=True,
                near_miss=False,
                metadata={
                    "adapter": "activerocketpy",
                    "scenario": "balloon_field_benchmark",
                    "score_is_proxy": True,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )

        apogee_agl = float(flight.apogee - env.elevation)
        apogee_time = float(flight.apogee_time)
        apogee_x = float(flight.x(apogee_time))
        apogee_y = float(flight.y(apogee_time))
        lateral_drift = math.hypot(apogee_x, apogee_y)
        launch_time = loop_state.launch_time if loop_state.launch_time is not None else float("inf")
        saturation_ratio = self._tvc_saturation_ratio(loop_state.controller_log, float(spec.params.get("max_tvc", 3.0)))
        wind_drift = self._mean_wind_drift(loop_state.trajectory)
        score = max(
            0.0,
            course.popped_count * 1000.0
            + max(0.0, 220.0 - (course.first_pop_time or 35.0) * 8.0)
            + max(0.0, 140.0 - launch_time * 45.0)
            - 90.0 * saturation_ratio
            - 0.05 * lateral_drift
            - 0.08 * wind_drift
            - 12.0 * course.near_miss_count,
        )
        near_miss = course.near_miss_count > max(2, course.popped_count)
        crashed = not math.isfinite(apogee_agl) or apogee_agl <= 0.0
        min_distance_value = course.min_distance_to_any_balloon if math.isfinite(course.min_distance_to_any_balloon) else 999999.0

        metadata = {
            "adapter": "activerocketpy",
            "scenario": "balloon_field_benchmark",
            "score_mode": "competition_shaped_balloon_field",
            "score_is_proxy": True,
            "apogee_agl_m": round(apogee_agl, 4),
            "apogee_time_s": round(apogee_time, 4),
            "lateral_drift_m": round(lateral_drift, 4),
            "wind_drift_m": round(wind_drift, 4),
            "balloons_total": len(course.balloons),
            "balloons_released": course.release_count,
            "balloons_remaining": course.remaining_count,
            "miss_distance_p50_m": round(course.miss_distance_p50, 4) if math.isfinite(course.miss_distance_p50) else None,
            "near_intercept_threshold_m": 75.0,
            "target_history": loop_state.target_history,
            "target_chatter_threshold": 10,
            "tvc_saturation_threshold": 0.72,
            "recommended_launch_floor_s": 0.05,
            "recommended_launch_ceiling_s": 0.75,
            "target_altitude_floor_m": 900.0,
            "velocity_floor_mps": 35.0,
            "mean_relative_speed_mps": round(mean_or_zero(loop_state.relative_speed_samples), 4),
            "sensor_jitter_index": round(mean_or_zero(loop_state.sensor_jitter_samples), 4),
            "sensor_jitter_threshold": 1.25,
            "trajectory": loop_state.trajectory,
            "objective_note": (
                "Real 6-DoF flight loop with a 100-balloon stochastic benchmark. "
                "This now reflects multi-target balloon popping, but it is still not the official competition platform API."
            ),
        }

        return EpisodeResult(
            seed=seed,
            score=round(score, 4),
            popped=course.popped_count,
            crashed=crashed,
            nan_detected=not math.isfinite(score),
            duration=round(float(flight.t_final), 4),
            time_to_launch=round(float(launch_time), 4) if math.isfinite(launch_time) else 999.0,
            time_to_first_pop=round(course.first_pop_time, 4) if course.first_pop_time is not None else None,
            min_distance_to_any_balloon=round(min_distance_value, 4),
            target_switch_count=loop_state.target_switch_count,
            tvc_saturation_ratio=round(saturation_ratio, 4),
            late_launch=launch_time > 0.75,
            near_miss=near_miss,
            metadata=metadata,
        )

    def _build_calisto_scenario(
        self,
        rocketpy,
        spec: ExperimentSpec,
        seed: int,
        strategy,
        course: BalloonFieldCourse,
        loop_state: StrategyLoopState,
        noise_config: dict[str, float],
        wind_profile: dict[str, list[tuple[float, float]]],
        rng: random.Random,
    ):
        root = Path(self.vendor_repo).resolve()
        env = rocketpy.Environment(
            latitude=float(spec.params.get("latitude_deg", 32.990254)),
            longitude=float(spec.params.get("longitude_deg", -106.974998)),
            elevation=float(spec.params.get("elevation_m", 1400.0)),
        )
        env.set_atmospheric_model(
            type="custom_atmosphere",
            wind_u=wind_profile["u"],
            wind_v=wind_profile["v"],
        )

        motor = rocketpy.SolidMotor(
            thrust_source=str(root / "data" / "motors" / "cesaroni" / "Cesaroni_M1670.eng"),
            burn_time=3.9,
            dry_mass=1.815,
            dry_inertia=(0.125, 0.125, 0.002),
            center_of_dry_mass_position=0.317,
            nozzle_position=0,
            grain_number=5,
            grain_density=1815,
            nozzle_radius=33 / 1000,
            throat_radius=11 / 1000,
            grain_separation=5 / 1000,
            grain_outer_radius=33 / 1000,
            grain_initial_height=120 / 1000,
            grains_center_of_mass_position=0.397,
            grain_initial_inner_radius=15 / 1000,
            interpolation_method="linear",
            coordinate_system_orientation="nozzle_to_combustion_chamber",
        )
        rocket = rocketpy.Rocket(
            radius=0.0635,
            mass=14.426,
            inertia=(6.321, 6.321, 0.034),
            power_off_drag=str(root / "data" / "rockets" / "calisto" / "powerOffDragCurve.csv"),
            power_on_drag=str(root / "data" / "rockets" / "calisto" / "powerOnDragCurve.csv"),
            center_of_mass_without_motor=0,
            coordinate_system_orientation="tail_to_nose",
        )
        rocket.add_motor(motor, position=-1.373)
        rocket.add_surfaces(
            rocketpy.NoseCone(
                length=0.55829,
                kind="vonkarman",
                base_radius=0.0635,
                rocket_radius=0.0635,
                name="calisto_nose_cone",
            ),
            1.160,
        )
        rocket.add_surfaces(
            rocketpy.Tail(
                top_radius=0.0635,
                bottom_radius=0.0435,
                length=0.060,
                rocket_radius=0.0635,
                name="calisto_tail",
            ),
            -1.313,
        )
        rocket.add_surfaces(
            rocketpy.TrapezoidalFins(
                n=4,
                span=0.100,
                root_chord=0.120,
                tip_chord=0.040,
                rocket_radius=0.0635,
                name="calisto_trapezoidal_fins",
                cant_angle=0,
            ),
            -1.168,
        )
        rocket.set_rail_buttons(
            upper_button_position=0.082,
            lower_button_position=-0.618,
            angular_position=0,
        )

        sampling_rate = float(spec.params.get("controller_rate_hz", 20.0))
        max_tvc = float(spec.params.get("max_tvc", 3.0))

        def tvc_controller(
            time: float,
            sampling_rate_hz: float,
            state: list[float],
            state_history: list[list[float]],
            observed_variables: list[Any],
            tvc,
            sensors,
            environment,
        ):
            del observed_variables, sensors, sampling_rate_hz, state_history
            altitude_asl = float(state[2])
            ambient_wind = Vector3(
                x=float(environment.wind_velocity_x(altitude_asl)),
                y=float(environment.wind_velocity_y(altitude_asl)),
                z=0.0,
            )
            rocket_position = Vector3(x=float(state[0]), y=float(state[1]), z=altitude_asl)
            course.update(rocket_position, time, ambient_wind)
            observed_world = self._world_state_from_flight(
                time=time,
                state=state,
                environment=environment,
                course=course,
                ambient_wind=ambient_wind,
                noise_config=noise_config,
                rng=rng,
                launch_active=loop_state.launch_time is not None,
            )
            loop_state.last_observed_world = observed_world
            action = strategy.act(observed_world)
            launch_active = loop_state.launch_time is not None or action.launch
            if launch_active and loop_state.launch_time is None:
                loop_state.launch_time = time
            if not launch_active:
                action.throttle = 0.0
                action.tvc_x = 0.0
                action.tvc_y = 0.0
                action.roll = 0.0
            loop_state.last_action = action

            if strategy.context.current_target_id != loop_state.last_target_id:
                if loop_state.last_target_id is not None:
                    loop_state.target_switch_count += 1
                loop_state.last_target_id = strategy.context.current_target_id
                if strategy.context.current_target_id is not None:
                    loop_state.target_history.append(strategy.context.current_target_id)

            loop_state.relative_speed_samples.append(math.sqrt(float(state[3]) ** 2 + float(state[4]) ** 2 + float(state[5]) ** 2))
            observed_rocket = observed_world.rocket
            true_rocket_position = Vector3(x=float(state[0]), y=float(state[1]), z=float(state[2]))
            loop_state.sensor_jitter_samples.append((observed_rocket.position - true_rocket_position).norm())

            tvc.gimbal_angle_x = float(action.tvc_x)
            tvc.gimbal_angle_y = float(action.tvc_y)
            loop_state.controller_log.append(
                {
                    "time": float(time),
                    "tvc_x": float(tvc.gimbal_angle_x),
                    "tvc_y": float(tvc.gimbal_angle_y),
                    "throttle": float(action.throttle),
                    "target_id": strategy.context.current_target_id,
                }
            )
            if len(loop_state.controller_log) % 5 == 0:
                loop_state.trajectory.append(
                    {
                        "time_s": round(float(time), 4),
                        "rocket_position": true_rocket_position.to_dict(),
                        "rocket_velocity": {"x": float(state[3]), "y": float(state[4]), "z": float(state[5])},
                        "target_id": strategy.context.current_target_id,
                        "popped": course.popped_count,
                        "released": course.release_count,
                    }
                )
            return (
                time,
                tvc.gimbal_angle_x,
                tvc.gimbal_angle_y,
                action.throttle,
                observed_world.rocket.position.z - environment.elevation,
                math.hypot(observed_world.rocket.position.x, observed_world.rocket.position.y),
                state[5],
            )

        def throttle_controller(
            time: float,
            sampling_rate_hz: float,
            state: list[float],
            state_history: list[list[float]],
            observed_variables: list[Any],
            throttle_control,
            sensors,
            environment,
        ):
            del time, sampling_rate_hz, state, state_history, observed_variables, sensors, environment
            throttle_control.throttle = float(loop_state.last_action.throttle)
            return throttle_control.throttle

        def roll_controller(
            time: float,
            sampling_rate_hz: float,
            state: list[float],
            state_history: list[list[float]],
            observed_variables: list[Any],
            roll_control,
            sensors,
            environment,
        ):
            del time, sampling_rate_hz, state, state_history, observed_variables, sensors, environment
            roll_control.roll_torque = float(loop_state.last_action.roll)
            return roll_control.roll_torque

        rocket.add_tvc(
            gimbal_range=max_tvc,
            gimbal_rate_limit=float(spec.params.get("tvc_rate_limit", 40.0)),
            controller_function=tvc_controller,
            sampling_rate=sampling_rate,
            clamp=True,
        )
        rocket.add_throttle_control(
            controller_function=throttle_controller,
            sampling_rate=sampling_rate,
            throttle_range=(0.0, 1.0),
            throttle=0.0,
            clamp=True,
            throttle_rate_limit=float(spec.params.get("throttle_rate_limit", 5.0)),
        )
        rocket.add_roll_control(
            max_roll_torque=float(spec.params.get("max_roll_torque", 0.0)),
            torque_rate_limit=float(spec.params.get("roll_torque_rate_limit", 10.0)),
            controller_function=roll_controller,
            sampling_rate=sampling_rate,
            clamp=True,
        )
        return env, rocket

    def _world_state_from_flight(
        self,
        time: float,
        state: list[float],
        environment,
        course: BalloonFieldCourse,
        ambient_wind: Vector3,
        noise_config: dict[str, float],
        rng: random.Random,
        launch_active: bool,
    ) -> WorldState:
        altitude_asl = float(state[2])
        rocket_position = _noise_vector(
            Vector3(x=float(state[0]), y=float(state[1]), z=altitude_asl),
            noise_config["rocket_position_noise_std_m"],
            rng,
        )
        rocket_velocity = _noise_vector(
            Vector3(x=float(state[3]), y=float(state[4]), z=float(state[5])),
            noise_config["rocket_velocity_noise_std_mps"],
            rng,
        )
        return WorldState(
            time_s=float(time),
            rocket=RocketState(
                position=rocket_position,
                velocity=rocket_velocity,
                attitude=Vector3(x=float(state[6]), y=float(state[7]), z=float(state[8])),
                angular_rate=Vector3(x=float(state[10]), y=float(state[11]), z=float(state[12])),
                launched=launch_active,
                valid=all(math.isfinite(float(value)) for value in state),
            ),
            balloons=course.balloons_for_observation(
                time_s=float(time),
                ambient_wind=ambient_wind,
                position_noise_std_m=noise_config["balloon_position_noise_std_m"],
                velocity_noise_std_mps=noise_config["balloon_velocity_noise_std_mps"],
                rng=rng,
            ),
            wind=_noise_vector(ambient_wind, noise_config["wind_sensor_noise_std_mps"], rng),
            metadata={
                "altitude_agl_m": altitude_asl - environment.elevation,
                "balloons_total": len(course.balloons),
                "balloons_released": course.release_count,
                "balloons_popped": course.popped_count,
                "sensor_noise_level": noise_config["balloon_position_noise_std_m"],
            },
        )

    @staticmethod
    def _seeded_balloon_field(spec: ExperimentSpec, seed: int) -> list[BalloonFieldBalloon]:
        rng = random.Random(f"{spec.experiment_id}:{seed}:balloon-field-layout")
        balloon_count = int(spec.params.get("balloon_count", 100))
        balloons: list[BalloonFieldBalloon] = []
        for index in range(balloon_count):
            altitude = 120.0 + (index / max(balloon_count - 1, 1)) * 2400.0 + rng.uniform(-40.0, 40.0)
            max_radius = min(260.0, 35.0 + altitude * 0.06)
            radial_distance = max(0.0, (rng.random() ** 2.2) * max_radius)
            azimuth = rng.uniform(0.0, 2.0 * math.pi)
            release_time = rng.uniform(0.0, float(spec.params.get("release_window_s", 35.0)))
            drift_speed = rng.uniform(0.1, 3.2)
            drift_heading = rng.uniform(0.0, 2.0 * math.pi)
            vertical_rate = rng.uniform(-0.2, 0.8)
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
                    radius_m=float(spec.params.get("balloon_radius_m", 42.0)),
                )
            )
        return balloons

    @staticmethod
    def _seeded_wind_profile(spec: ExperimentSpec, seed: int) -> dict[str, list[tuple[float, float]]]:
        rng = random.Random(f"{spec.experiment_id}:{seed}:wind-profile")
        altitudes = [0.0, 200.0, 800.0, 1600.0, 2600.0, 3600.0]
        surface_wind = float(spec.params.get("wind_surface_mps", 3.0))
        shear = float(spec.params.get("wind_shear_mps", 5.5))
        gust = float(spec.params.get("wind_gust_mps", 1.2))
        heading = rng.uniform(0.0, 2.0 * math.pi)
        veer = rng.uniform(-0.35, 0.35)
        wind_u: list[tuple[float, float]] = []
        wind_v: list[tuple[float, float]] = []
        for altitude in altitudes:
            altitude_ratio = altitude / max(altitudes[-1], 1.0)
            speed = surface_wind + shear * altitude_ratio + rng.uniform(-gust, gust)
            local_heading = heading + veer * altitude_ratio
            wind_u.append((altitude, round(speed * math.cos(local_heading), 4)))
            wind_v.append((altitude, round(speed * math.sin(local_heading), 4)))
        return {"u": wind_u, "v": wind_v}

    @staticmethod
    def _noise_config(spec: ExperimentSpec) -> dict[str, float]:
        return {
            "balloon_position_noise_std_m": float(spec.params.get("position_noise_std_m", 3.5)),
            "balloon_velocity_noise_std_mps": float(spec.params.get("velocity_noise_std_mps", 0.65)),
            "rocket_position_noise_std_m": float(spec.params.get("rocket_position_noise_std_m", 0.8)),
            "rocket_velocity_noise_std_mps": float(spec.params.get("rocket_velocity_noise_std_mps", 0.3)),
            "wind_sensor_noise_std_mps": float(spec.params.get("wind_sensor_noise_std_mps", 0.45)),
        }

    @staticmethod
    def _tvc_saturation_ratio(history: list[dict[str, float | str | None]], max_tvc: float) -> float:
        if not history or max_tvc <= 1e-9:
            return 0.0
        saturated = 0
        for entry in history:
            gimbal_x = abs(float(entry["tvc_x"]))
            gimbal_y = abs(float(entry["tvc_y"]))
            if gimbal_x >= 0.98 * max_tvc or gimbal_y >= 0.98 * max_tvc:
                saturated += 1
        return saturated / len(history)

    @staticmethod
    def _mean_wind_drift(trajectory: list[dict[str, Any]]) -> float:
        if not trajectory:
            return 0.0
        distances = [
            math.hypot(
                float(snapshot["rocket_position"]["x"]),
                float(snapshot["rocket_position"]["y"]),
            )
            for snapshot in trajectory
        ]
        return mean_or_zero(distances)


def _noise_vector(base: Vector3, stddev: float, rng: random.Random) -> Vector3:
    if stddev <= 1e-9:
        return base
    return Vector3(
        x=base.x + rng.gauss(0.0, stddev),
        y=base.y + rng.gauss(0.0, stddev),
        z=base.z + rng.gauss(0.0, stddev),
    )


def mean_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))
