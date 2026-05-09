from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any

from rocket_auto_research.auto_research.adapter_factory import build_simulation_adapter
from rocket_auto_research.auto_research.evaluator import EvaluationSummary, evaluate_episode_set
from rocket_auto_research.auto_research.experiment_spec import ExperimentSpec
from rocket_auto_research.auto_research.failure_analyzer import FailureReport, analyze_failures
from rocket_auto_research.auto_research.problem_definition import ResearchStage
from rocket_auto_research.auto_research.runtime_control import ResearchRuntimeControl
from rocket_auto_research.auto_research.simulation import EpisodeResult, SimulationAdapter


@dataclass(slots=True)
class ExperimentResult:
    spec: ExperimentSpec
    episodes: list[EpisodeResult]
    summary: EvaluationSummary
    failure_report: FailureReport


class ExperimentRunner:
    def __init__(
        self,
        adapter: SimulationAdapter,
        results_dir: str | Path = "results/runs",
        runtime_control: ResearchRuntimeControl | None = None,
        adapter_config: dict[str, Any] | None = None,
        max_workers: int = 1,
        persist_flush_interval_s: float = 180.0,
    ) -> None:
        self.adapter = adapter
        self.results_dir = Path(results_dir)
        self.runtime_control = runtime_control
        self.adapter_config = dict(adapter_config or {})
        self.max_workers = max(1, int(max_workers))
        self.persist_flush_interval_s = max(1.0, float(persist_flush_interval_s))
        self._pending_persist: list[ExperimentResult] = []
        self._last_persist_flush_monotonic = time.monotonic()
        self._persisted_results_count = 0

    def run(self, spec: ExperimentSpec, stage: ResearchStage | None = None, *, persist: bool = True) -> ExperimentResult:
        episodes = self._run_episodes(spec)
        summary = evaluate_episode_set(episodes, stage=stage)
        failure_report = analyze_failures(episodes, summary)
        result = ExperimentResult(spec=spec, episodes=episodes, summary=summary, failure_report=failure_report)
        if persist:
            self._queue_persist(result)
        return result

    def run_population(
        self,
        specs: list[ExperimentSpec],
        stage: ResearchStage | None = None,
        on_completed=None,
    ) -> list[ExperimentResult]:
        worker_count = self._parallel_population_worker_count(specs)
        if worker_count <= 1:
            results: list[ExperimentResult] = []
            for spec in specs:
                result = self.run(spec, stage=stage, persist=False)
                self._queue_persist(result)
                results.append(result)
                if on_completed is not None:
                    on_completed(spec, result)
            return results

        futures = {}
        stage_id = stage.stage_id if stage is not None else None
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for spec in specs:
                future = executor.submit(
                    _run_spec_worker,
                    self.adapter_config,
                    spec.to_dict(),
                    stage_id,
                    str(self.results_dir),
                )
                futures[future] = spec

            ordered: dict[str, ExperimentResult] = {}
            for future in as_completed(futures):
                spec = futures[future]
                payload = future.result()
                result = _result_from_payload(payload)
                self._queue_persist(result)
                ordered[spec.experiment_id] = result
                if on_completed is not None:
                    on_completed(spec, result)
        return [ordered[spec.experiment_id] for spec in specs if spec.experiment_id in ordered]

    def flush_pending(self, *, force: bool = False) -> None:
        if not self._pending_persist:
            return
        now = time.monotonic()
        if not force and (now - self._last_persist_flush_monotonic) < self.persist_flush_interval_s:
            return
        pending = list(self._pending_persist)
        self._pending_persist.clear()
        if self.runtime_control is not None:
            self.runtime_control.update_status(
                buffered_results_pending=len(pending),
                buffered_results_flushing=True,
                buffered_results_flush_total=len(pending),
                buffered_results_flush_completed=0,
                persisted_results_count=self._persisted_results_count,
                message=f"Flushing {len(pending)} buffered experiment results to disk...",
                force=True,
            )
        last_progress_update = time.monotonic()
        for index, result in enumerate(pending, start=1):
            self._persist(result.spec, result.episodes, result.summary, result.failure_report)
            if self.runtime_control is not None:
                current_time = time.monotonic()
                if index == len(pending) or (current_time - last_progress_update) >= 1.0:
                    self.runtime_control.update_status(
                        buffered_results_pending=max(0, len(pending) - index),
                        buffered_results_flushing=index < len(pending),
                        buffered_results_flush_total=len(pending),
                        buffered_results_flush_completed=index,
                        persisted_results_count=self._persisted_results_count + index,
                        message=(
                            f"Flushing buffered experiment results to disk: {index}/{len(pending)} completed."
                            if index < len(pending)
                            else "Buffered experiment results saved to disk."
                        ),
                        force=index == len(pending),
                    )
                    last_progress_update = current_time
        self._persisted_results_count += len(pending)
        self._last_persist_flush_monotonic = now

    def _queue_persist(self, result: ExperimentResult) -> None:
        self._pending_persist.append(result)
        if self.runtime_control is not None:
            self.runtime_control.update_status(
                buffered_results_pending=len(self._pending_persist),
                buffered_results_flushing=False,
                buffered_results_flush_total=0,
                buffered_results_flush_completed=0,
                persisted_results_count=self._persisted_results_count,
            )
        self.flush_pending(force=False)

    def _run_episodes(self, spec: ExperimentSpec) -> list[EpisodeResult]:
        if self._parallel_worker_count(spec) <= 1:
            episodes: list[EpisodeResult] = []
            for seed in spec.seeds:
                if self.runtime_control is not None:
                    self.runtime_control.wait_if_paused(allow_stop=False)
                    self.runtime_control.update_status(current_seed=seed)
                episodes.append(self.adapter.run_episode(spec, seed))
            return episodes

        if self.runtime_control is not None:
            self.runtime_control.update_status(current_seed="parallel")
        spec_payload = spec.to_dict()
        worker_count = self._parallel_worker_count(spec)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            episode_payloads = list(
                executor.map(
                    _run_episode_worker,
                    [self.adapter_config] * len(spec.seeds),
                    [spec_payload] * len(spec.seeds),
                    spec.seeds,
                )
            )
        episodes = [EpisodeResult(**payload) for payload in episode_payloads]
        episodes.sort(key=lambda episode: episode.seed)
        return episodes

    def _parallel_worker_count(self, spec: ExperimentSpec) -> int:
        if self.max_workers <= 1 or len(spec.seeds) <= 1 or not self.adapter_config:
            return 1
        if not _is_parallel_safe_for_spec(self.adapter_config, spec):
            return 1
        return min(self.max_workers, len(spec.seeds))

    def _parallel_population_worker_count(self, specs: list[ExperimentSpec]) -> int:
        if self.max_workers <= 1 or len(specs) <= 1 or not self.adapter_config:
            return 1
        if not all(_is_parallel_safe_for_spec(self.adapter_config, spec) for spec in specs):
            return 1
        return min(self.max_workers, len(specs))

    def _persist(
        self,
        spec: ExperimentSpec,
        episodes: list[EpisodeResult],
        summary: EvaluationSummary,
        failure_report: FailureReport,
    ) -> None:
        run_dir = self.results_dir / spec.experiment_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "spec.json").write_text(json.dumps(spec.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (run_dir / "summary.json").write_text(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (run_dir / "failure_report.json").write_text(
            json.dumps(failure_report.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        for episode in episodes:
            path = run_dir / f"trajectory_seed_{episode.seed:03d}.jsonl"
            metadata = dict(episode.metadata)
            trajectory = metadata.pop("trajectory", [])
            payload = {
                "summary": asdict(episode),
                "metadata": metadata,
                "trajectory": trajectory,
            }
            path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _run_episode_worker(adapter_config: dict[str, Any], spec_payload: dict[str, Any], seed: int) -> dict[str, Any]:
    adapter = build_simulation_adapter(adapter_config)
    spec = ExperimentSpec(**spec_payload)
    episode = adapter.run_episode(spec, seed)
    return episode.to_dict()


def _run_spec_worker(
    adapter_config: dict[str, Any],
    spec_payload: dict[str, Any],
    stage_id: str | None,
    results_dir: str,
) -> dict[str, Any]:
    spec = ExperimentSpec(**spec_payload)
    adapter = build_simulation_adapter(adapter_config)
    runner = ExperimentRunner(adapter, results_dir=results_dir, adapter_config=adapter_config, max_workers=1)
    stage = _stage_by_id(stage_id) if stage_id else None
    result = runner.run(spec, stage=stage, persist=False)
    return {
        "spec": result.spec.to_dict(),
        "episodes": [episode.to_dict() for episode in result.episodes],
        "summary": result.summary.to_dict(),
        "failure_report": result.failure_report.to_dict(),
    }


def _result_from_payload(payload: dict[str, Any]) -> ExperimentResult:
    spec = ExperimentSpec(**payload["spec"])
    episodes = [EpisodeResult(**episode_payload) for episode_payload in payload["episodes"]]
    summary = EvaluationSummary(**payload["summary"])
    failure_report = FailureReport(**payload["failure_report"])
    return ExperimentResult(spec=spec, episodes=episodes, summary=summary, failure_report=failure_report)


def _stage_by_id(stage_id: str | None) -> ResearchStage | None:
    if not stage_id:
        return None
    from rocket_auto_research.auto_research.problem_definition import default_problem_stages

    for stage in default_problem_stages():
        if stage.stage_id == stage_id:
            return stage
    raise KeyError(f"Unknown stage id '{stage_id}'.")


def _is_parallel_safe_for_spec(adapter_config: dict[str, Any], spec: ExperimentSpec) -> bool:
    adapter_name = str(adapter_config.get("adapter", "mock")).lower()
    if adapter_name in {"mock", "competition_platform", "balloon_challenge"}:
        return True
    if adapter_name not in {"stage_router", "auto_research_router"}:
        return False
    profile_name = str(spec.params.get("simulation_adapter", adapter_config.get("default_adapter_profile", ""))).strip()
    if not profile_name:
        return False
    profile = dict(adapter_config.get("adapter_profiles", {}).get(profile_name, {}))
    return str(profile.get("adapter", "")).lower() in {"mock", "competition_platform", "balloon_challenge"}
