from __future__ import annotations

from dataclasses import dataclass
from math import acos
from typing import Any

from rocket_auto_research.gnc.state import BalloonState, RocketState, Vector3, WorldState


def _angle_between(a: Vector3, b: Vector3) -> float:
    if a.norm() <= 1e-9 or b.norm() <= 1e-9:
        return 0.0
    dot = a.x * b.x + a.y * b.y + a.z * b.z
    cosine = max(-1.0, min(1.0, dot / (a.norm() * b.norm())))
    return acos(cosine)


def _available_balloons(world_state: WorldState) -> list[BalloonState]:
    return [balloon for balloon in world_state.balloons if balloon.released and not balloon.popped]


def _dot(a: Vector3, b: Vector3) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z


@dataclass(slots=True)
class NearestTargetSelector:
    def select(self, world_state: WorldState, current_target_id: str | None = None) -> BalloonState | None:
        candidates = _available_balloons(world_state)
        if not candidates:
            return None
        rocket_position = world_state.rocket.position
        return min(candidates, key=lambda balloon: (balloon.position - rocket_position).norm())


@dataclass(slots=True)
class ScoreBasedTargetSelector:
    distance_weight: float = 1.0
    angle_weight: float = 0.75
    height_weight: float = 0.25
    switching_penalty: float = 0.5
    switch_margin: float = 0.0
    switch_cooldown_steps: int = 0
    _cooldown_remaining: int = 0

    def _score(self, rocket: RocketState, balloon: BalloonState, current_target_id: str | None) -> float:
        relative_position = balloon.position - rocket.position
        distance = relative_position.norm()
        angle = _angle_between(rocket.velocity, relative_position)
        height_bonus = max(0.0, balloon.position.z - rocket.position.z)
        switch_cost = self.switching_penalty if current_target_id and balloon.balloon_id != current_target_id else 0.0
        return (
            (-self.distance_weight * distance)
            + (-self.angle_weight * angle)
            + (self.height_weight * height_bonus)
            - switch_cost
        )

    def select(self, world_state: WorldState, current_target_id: str | None = None) -> BalloonState | None:
        candidates = _available_balloons(world_state)
        if not candidates:
            return None
        scored = [
            (balloon, self._score(world_state.rocket, balloon, current_target_id))
            for balloon in candidates
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        chosen_balloon, chosen_score = scored[0]
        if current_target_id is not None:
            current_candidate = next((balloon for balloon, _ in scored if balloon.balloon_id == current_target_id), None)
            if current_candidate is not None:
                current_score = self._score(world_state.rocket, current_candidate, current_target_id)
                if self._cooldown_remaining > 0 or chosen_score - current_score < self.switch_margin:
                    self._cooldown_remaining = max(0, self._cooldown_remaining - 1)
                    return current_candidate
        self._cooldown_remaining = self.switch_cooldown_steps
        return chosen_balloon


@dataclass(slots=True)
class ReachableTargetSelector:
    distance_weight: float = 0.8
    angle_weight: float = 0.7
    closure_weight: float = 0.4
    switching_penalty: float = 0.7
    target_lock_duration: int = 3
    switch_margin: float = 0.0
    switch_cooldown_steps: int = 0
    _locked_target_id: str | None = None
    _lock_remaining: int = 0
    _cooldown_remaining: int = 0

    def select(self, world_state: WorldState, current_target_id: str | None = None) -> BalloonState | None:
        candidates = _available_balloons(world_state)
        if not candidates:
            self._locked_target_id = None
            self._lock_remaining = 0
            return None
        if self._locked_target_id is not None and self._lock_remaining > 0:
            for candidate in candidates:
                if candidate.balloon_id == self._locked_target_id:
                    self._lock_remaining -= 1
                    return candidate
        rocket = world_state.rocket
        scored = [
            (balloon, self._score(rocket, balloon, current_target_id))
            for balloon in candidates
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        selected, selected_score = scored[0]
        current_candidate = next((balloon for balloon, _ in scored if balloon.balloon_id == current_target_id), None)
        if current_candidate is not None:
            current_score = self._score(rocket, current_candidate, current_target_id)
            if self._cooldown_remaining > 0 or selected_score - current_score < self.switch_margin:
                self._cooldown_remaining = max(0, self._cooldown_remaining - 1)
                selected = current_candidate
        self._locked_target_id = selected.balloon_id
        self._lock_remaining = self.target_lock_duration
        self._cooldown_remaining = self.switch_cooldown_steps
        return selected

    def _score(self, rocket: RocketState, balloon: BalloonState, current_target_id: str | None) -> float:
        relative = balloon.position - rocket.position
        distance = max(1e-6, relative.norm())
        direction = relative.normalized()
        angle = _angle_between(rocket.velocity, relative)
        closure = rocket.velocity.x * direction.x + rocket.velocity.y * direction.y + rocket.velocity.z * direction.z
        switch_cost = self.switching_penalty if current_target_id and balloon.balloon_id != current_target_id else 0.0
        return (
            -self.distance_weight * distance
            -self.angle_weight * angle
            +self.closure_weight * closure
            -switch_cost
        )


@dataclass(slots=True)
class EnergyAwareTargetSelector:
    distance_weight: float = 0.45
    angle_weight: float = 0.55
    closure_weight: float = 0.7
    energy_weight: float = 0.9
    time_weight: float = 0.45
    crosswind_weight: float = 0.18
    switching_penalty: float = 0.9
    target_lock_duration: int = 5
    switch_margin: float = 0.0
    switch_cooldown_steps: int = 0
    _locked_target_id: str | None = None
    _lock_remaining: int = 0
    _cooldown_remaining: int = 0

    def select(self, world_state: WorldState, current_target_id: str | None = None) -> BalloonState | None:
        candidates = _available_balloons(world_state)
        if not candidates:
            self._locked_target_id = None
            self._lock_remaining = 0
            return None
        if self._locked_target_id is not None and self._lock_remaining > 0:
            for candidate in candidates:
                if candidate.balloon_id == self._locked_target_id:
                    self._lock_remaining -= 1
                    return candidate
        rocket = world_state.rocket
        scored = [
            (balloon, self._score(world_state, rocket, balloon, current_target_id))
            for balloon in candidates
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        selected, selected_score = scored[0]
        current_candidate = next((balloon for balloon, _ in scored if balloon.balloon_id == current_target_id), None)
        if current_candidate is not None:
            current_score = self._score(world_state, rocket, current_candidate, current_target_id)
            if self._cooldown_remaining > 0 or selected_score - current_score < self.switch_margin:
                self._cooldown_remaining = max(0, self._cooldown_remaining - 1)
                selected = current_candidate
        self._locked_target_id = selected.balloon_id
        self._lock_remaining = self.target_lock_duration
        self._cooldown_remaining = self.switch_cooldown_steps
        return selected

    def _score(
        self,
        world_state: WorldState,
        rocket: RocketState,
        balloon: BalloonState,
        current_target_id: str | None,
    ) -> float:
        relative = balloon.position - rocket.position
        distance = max(1e-6, relative.norm())
        rocket_speed = max(rocket.velocity.norm(), 1.0)
        direction = relative.normalized()
        angle = _angle_between(rocket.velocity, relative)
        closure = _dot(rocket.velocity, direction)
        estimated_time = distance / max(closure + rocket_speed * 0.55, 6.0)
        vertical_gap = max(0.0, balloon.position.z - rocket.position.z)
        required_vertical_rate = vertical_gap / max(estimated_time, 0.4)
        energy_margin = rocket.velocity.z - required_vertical_rate
        crosswind = abs(world_state.wind.x * direction.y - world_state.wind.y * direction.x)
        switch_cost = self.switching_penalty if current_target_id and balloon.balloon_id != current_target_id else 0.0
        return (
            -self.distance_weight * distance
            -self.angle_weight * angle
            +self.closure_weight * closure
            +self.energy_weight * energy_margin
            -self.time_weight * estimated_time
            -self.crosswind_weight * crosswind
            -switch_cost
        )


def build_target_selector(params: dict[str, Any]):
    mode = str(params.get("target_selector", "nearest")).lower()
    if mode == "nearest":
        return NearestTargetSelector()
    if mode in {"score", "score_based"}:
        return ScoreBasedTargetSelector(
            distance_weight=float(params.get("target_distance_weight", 1.0)),
            angle_weight=float(params.get("target_angle_weight", 0.75)),
            height_weight=float(params.get("target_height_weight", 0.25)),
            switching_penalty=float(params.get("switching_penalty", 0.5)),
            switch_margin=float(params.get("target_switch_margin", 0.0)),
            switch_cooldown_steps=int(params.get("target_switch_cooldown_steps", 0)),
        )
    if mode in {"reachable", "predictive_reachable"}:
        return ReachableTargetSelector(
            distance_weight=float(params.get("target_distance_weight", 0.8)),
            angle_weight=float(params.get("target_angle_weight", 0.7)),
            closure_weight=float(params.get("target_closure_weight", 0.4)),
            switching_penalty=float(params.get("switching_penalty", 0.7)),
            target_lock_duration=int(params.get("target_lock_duration", 3)),
            switch_margin=float(params.get("target_switch_margin", 0.0)),
            switch_cooldown_steps=int(params.get("target_switch_cooldown_steps", 0)),
        )
    if mode in {"energy_aware", "energy", "reachable_energy"}:
        return EnergyAwareTargetSelector(
            distance_weight=float(params.get("target_distance_weight", 0.45)),
            angle_weight=float(params.get("target_angle_weight", 0.55)),
            closure_weight=float(params.get("target_closure_weight", 0.7)),
            energy_weight=float(params.get("target_energy_weight", 0.9)),
            time_weight=float(params.get("target_time_weight", 0.45)),
            crosswind_weight=float(params.get("target_crosswind_weight", 0.18)),
            switching_penalty=float(params.get("switching_penalty", 0.9)),
            target_lock_duration=int(params.get("target_lock_duration", 5)),
            switch_margin=float(params.get("target_switch_margin", 0.0)),
            switch_cooldown_steps=int(params.get("target_switch_cooldown_steps", 0)),
        )
    raise ValueError(f"Unknown target_selector '{mode}'.")
