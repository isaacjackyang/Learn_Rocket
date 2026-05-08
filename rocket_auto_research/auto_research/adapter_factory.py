from __future__ import annotations

from pathlib import Path
from typing import Any

from rocket_auto_research.auto_research.activerocketpy_adapter import ActiveRocketPySimulationAdapter
from rocket_auto_research.auto_research.balloon_challenge_adapter import BalloonChallengeSimulationAdapter
from rocket_auto_research.auto_research.competition_platform_adapter import CompetitionPlatformSimulationAdapter
from rocket_auto_research.auto_research.simulation import MockRocketSimAdapter, SimulationAdapter


class StageRouterSimulationAdapter(SimulationAdapter):
    def __init__(self, adapter_profiles: dict[str, dict[str, Any]], default_profile: str | None = None) -> None:
        self.adapter_profiles = dict(adapter_profiles)
        self.default_profile = default_profile
        self._cache: dict[str, SimulationAdapter] = {}

    def run_episode(self, spec, seed):
        profile_name = str(spec.params.get("simulation_adapter", self.default_profile or "")).strip()
        if not profile_name:
            raise ValueError("StageRouterSimulationAdapter requires 'simulation_adapter' in spec params or a default profile.")
        adapter = self._adapter_for_profile(profile_name)
        return adapter.run_episode(spec, seed)

    def _adapter_for_profile(self, profile_name: str) -> SimulationAdapter:
        if profile_name in self._cache:
            return self._cache[profile_name]
        if profile_name not in self.adapter_profiles:
            raise KeyError(f"Unknown adapter profile '{profile_name}'. Available: {sorted(self.adapter_profiles)}")
        adapter = _build_named_adapter(self.adapter_profiles[profile_name])
        self._cache[profile_name] = adapter
        return adapter


def _build_named_adapter(config: dict[str, Any]) -> SimulationAdapter:
    adapter_name = str(config.get("adapter", "mock")).lower()
    if adapter_name == "mock":
        return MockRocketSimAdapter()
    if adapter_name == "activerocketpy":
        return ActiveRocketPySimulationAdapter(
            vendor_repo=config.get("vendor_repo", Path("vendor/ActiveRocketPy")),
            terminate_on_apogee=bool(config.get("terminate_on_apogee", True)),
        )
    if adapter_name == "competition_platform":
        return CompetitionPlatformSimulationAdapter()
    if adapter_name == "balloon_challenge":
        return BalloonChallengeSimulationAdapter(
            repo_root=config.get("challenge_repo_root", Path(".external/BalloonPoppingChallenge")),
            scenario_number=int(config.get("challenge_scenario_number", 1)),
        )
    raise ValueError(f"Unsupported adapter '{adapter_name}'.")


def build_simulation_adapter(config: dict[str, Any]) -> SimulationAdapter:
    adapter_name = str(config.get("adapter", "mock")).lower()
    if adapter_name in {"stage_router", "auto_research_router"}:
        profiles = dict(config.get("adapter_profiles", {}))
        if not profiles:
            raise ValueError("Router adapter requires non-empty 'adapter_profiles'.")
        default_profile = config.get("default_adapter_profile")
        return StageRouterSimulationAdapter(adapter_profiles=profiles, default_profile=default_profile)
    return _build_named_adapter(config)
