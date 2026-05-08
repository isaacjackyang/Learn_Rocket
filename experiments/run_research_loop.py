from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rocket_auto_research.auto_research.adapter_factory import build_simulation_adapter
from rocket_auto_research.auto_research.experiment_runner import ExperimentRunner
from rocket_auto_research.auto_research.experiment_spec import ExperimentSpec
from rocket_auto_research.auto_research.researcher_loop import ResearchConfig, ResearcherLoop
from rocket_auto_research.auto_research.runtime_control import ResearchRuntimeControl, ResearchStopRequested
from rocket_auto_research.config import load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the auto research loop.")
    parser.add_argument("--config", required=True, help="Path to search-space YAML config.")
    parser.add_argument("--parallel-workers", type=int, default=None, help="Override parallel worker count.")
    parser.add_argument("--population-size", type=int, default=None, help="Override population size for each generation.")
    parser.add_argument("--control-dir", default=None, help="Optional runtime control directory for dashboard orchestration.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_yaml_config(config_path)
    bootstrap_specs = _load_bootstrap_specs(
        config.get("bootstrap_configs", []),
        repo_root=ROOT,
        auto_best=bool(config.get("auto_bootstrap_best", True)),
    )
    runtime_control = ResearchRuntimeControl(args.control_dir) if args.control_dir else None
    parameter_space = dict(config.get("global_parameter_space", config.get("parameter_space", {})))
    research_config = ResearchConfig(
        strategies=list(config["strategies"]),
        parameter_space=parameter_space,
        fixed_params=dict(config.get("fixed_params", {})),
        stage_policy=dict(config.get("stage_policy", {})),
        population_size=_parse_population_size(
            args.population_size if args.population_size is not None else config["population_size"]
        ),
        elite_count=int(config["elite_count"]),
        mutation_rate=float(config["mutation_rate"]),
        crossover_rate=float(config["crossover_rate"]),
        seeds_per_experiment=int(config["seeds_per_experiment"]),
        generations=int(config["generations"]),
        continuous=bool(config.get("continuous", False)),
        base_seed=int(config.get("base_seed", 0)),
        bootstrap_specs=bootstrap_specs,
    )
    parallel_workers = _parse_parallel_workers(
        args.parallel_workers if args.parallel_workers is not None else config.get("parallel_workers", 1)
    )
    if runtime_control is not None:
        runtime_control.begin_run(
            config_path=str(Path(args.config).resolve()),
            total_generations=None if research_config.continuous else research_config.generations,
            population_size=research_config.population_size,
            configured_workers=parallel_workers,
            log_path=str((Path(args.control_dir) / "research.log").resolve()),
        )
    try:
        best = ResearcherLoop(
            ExperimentRunner(
                build_simulation_adapter(config),
                runtime_control=runtime_control,
                adapter_config=config,
                max_workers=parallel_workers,
            ),
            research_config,
            runtime_control=runtime_control,
        ).run()
    except ResearchStopRequested:
        print(json.dumps({"status": "stopped"}, ensure_ascii=False))
        raise SystemExit(130)
    payload = {
        "best_experiment_id": best.spec.experiment_id,
        "strategy_name": best.spec.strategy_name,
        "summary": best.summary.to_dict(),
        "params": best.spec.params,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _load_bootstrap_specs(
    paths: list[str],
    *,
    repo_root: Path,
    auto_best: bool = True,
) -> list[ExperimentSpec]:
    specs: list[ExperimentSpec] = []
    seen_paths: set[Path] = set()
    resolved_paths: list[Path] = []
    for raw_path in paths:
        path = _resolve_bootstrap_path(raw_path, repo_root)
        if path in seen_paths:
            continue
        seen_paths.add(path)
        resolved_paths.append(path)
    if auto_best:
        best_path = (repo_root / "results" / "best_agents" / "best_config.json").resolve()
        if best_path.exists() and best_path not in seen_paths:
            resolved_paths.append(best_path)
            seen_paths.add(best_path)
    for path in resolved_paths:
        if not path.exists():
            continue
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            specs.append(
                ExperimentSpec(
                    strategy_name=payload["strategy_name"],
                    params=dict(payload.get("params", {})),
                    seeds=[],
                    note=f"bootstrap_source:{path}",
                    experiment_id=str(payload.get("experiment_id", "")),
                )
            )
            continue
        payload = load_yaml_config(path)
        specs.append(
            ExperimentSpec(
                strategy_name=payload["strategy_name"],
                params=dict(payload.get("params", {})),
                seeds=[],
                note=f"bootstrap_source:{path}",
            )
        )
    return specs


def _resolve_bootstrap_path(raw_path: str, repo_root: Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = (repo_root / path).resolve()
    else:
        path = path.resolve()
    return path


def _parse_parallel_workers(raw_value: object) -> int:
    if raw_value in {None, "", False}:
        return 1
    if isinstance(raw_value, str) and raw_value.strip().lower() == "auto":
        import os

        return min(32, max(1, (os.cpu_count() or 1) - 1))
    try:
        return min(32, max(1, int(raw_value)))
    except (TypeError, ValueError):
        return 1


def _parse_population_size(raw_value: object) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return 6
    return min(64, max(4, value))


if __name__ == "__main__":
    main()
