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
from rocket_auto_research.config import load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one strategy on one seed.")
    parser.add_argument("--config", required=True, help="Path to strategy YAML config.")
    parser.add_argument("--seed", type=int, default=0, help="Simulation seed.")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    spec = ExperimentSpec(
        strategy_name=config["strategy_name"],
        params=config.get("params", {}),
        seeds=[args.seed],
        note="run_single",
    )
    result = ExperimentRunner(build_simulation_adapter(config)).run(spec)
    run_dir = Path("results/runs") / result.spec.experiment_id
    payload = {
        "experiment_id": result.spec.experiment_id,
        "strategy_name": result.spec.strategy_name,
        "summary": result.summary.to_dict(),
        "failures": result.failure_report.to_dict(),
        "run_dir": str(run_dir),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
