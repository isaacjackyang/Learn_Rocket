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
from rocket_auto_research.auto_research.runtime_control import ResearchRuntimeControl, ResearchStopRequested
from rocket_auto_research.config import load_yaml_config
from experiments.run_research_loop import _parse_parallel_workers


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one strategy on multiple seeds.")
    parser.add_argument("--config", required=True, help="Path to strategy YAML config.")
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="List of seeds.")
    parser.add_argument("--parallel-workers", type=int, default=None, help="Override parallel worker count.")
    parser.add_argument("--control-dir", default=None, help="Optional runtime control directory for dashboard orchestration.")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    runtime_control = ResearchRuntimeControl(args.control_dir) if args.control_dir else None
    parallel_workers = _parse_parallel_workers(
        args.parallel_workers if args.parallel_workers is not None else config.get("parallel_workers", 1)
    )
    spec = ExperimentSpec(
        strategy_name=config["strategy_name"],
        params=config.get("params", {}),
        seeds=args.seeds,
        note="run_batch",
    )
    if runtime_control is not None:
        runtime_control.begin_run(
            config_path=str(Path(args.config).resolve()),
            total_generations=1,
            population_size=1,
            configured_workers=parallel_workers,
            log_path=str((Path(args.control_dir) / "research.log").resolve()),
        )
        runtime_control.update_status(
            status="running",
            generation_label="1/1",
            current_generation=0,
            current_experiment_id=spec.experiment_id,
            current_strategy_name=spec.strategy_name,
            current_experiment_index=1,
            message=f"Running single strategy batch for {spec.strategy_name}.",
        )
    try:
        runner = ExperimentRunner(
            build_simulation_adapter(config),
            runtime_control=runtime_control,
            adapter_config=config,
            max_workers=parallel_workers,
        )
        result = runner.run(spec)
        runner.flush_pending(force=True)
    except ResearchStopRequested:
        if runtime_control is not None:
            runtime_control.flush_status()
            runtime_control.mark_stopped()
        print(json.dumps({"status": "stopped"}, ensure_ascii=False))
        raise SystemExit(130)
    except Exception as exc:
        if runtime_control is not None:
            runtime_control.flush_status()
            runtime_control.mark_error(f"Single strategy batch failed: {exc}")
        raise
    if runtime_control is not None:
        runtime_control.flush_status()
        runtime_control.mark_completed(
            best_experiment_id=result.spec.experiment_id,
            strategy_name=result.spec.strategy_name,
        )
    run_dir = Path("results/runs") / result.spec.experiment_id
    payload = {
        "experiment_id": result.spec.experiment_id,
        "strategy_name": result.spec.strategy_name,
        "summary": result.summary.to_dict(),
        "failures": result.failure_report.to_dict(),
        "seed_count": len(args.seeds),
        "run_dir": str(run_dir),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
