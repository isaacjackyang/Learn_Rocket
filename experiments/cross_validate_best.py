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
    parser = argparse.ArgumentParser(description="Cross-validate the current best config across adapters.")
    parser.add_argument("--best-config", default="results/best_agents/best_config.json")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/auto_research.yaml",
            "configs/legacy/search_space_activerocketpy.yaml",
        ],
        help="Adapter/search-space configs to validate against.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="Fresh validation seeds.")
    args = parser.parse_args()

    best_payload = json.loads(Path(args.best_config).read_text(encoding="utf-8"))
    results: list[dict[str, object]] = []

    for config_path in args.configs:
        config = load_yaml_config(config_path)
        spec = ExperimentSpec(
            strategy_name=best_payload["strategy_name"],
            params=best_payload["params"],
            seeds=args.seeds,
            note=f"cross_validate:{best_payload['experiment_id']}:{Path(config_path).stem}",
        )
        result = ExperimentRunner(build_simulation_adapter(config)).run(spec)
        results.append(
            {
                "config": config_path,
                "adapter": config.get("adapter", "mock"),
                "experiment_id": result.spec.experiment_id,
                "summary": result.summary.to_dict(),
                "failure_report": result.failure_report.to_dict(),
            }
        )

    output_dir = Path("results/cross_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{best_payload['experiment_id']}_cross_validation.json"
    payload = {
        "best_experiment_id": best_payload["experiment_id"],
        "strategy_name": best_payload["strategy_name"],
        "seeds": args.seeds,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_path": str(output_path), **payload}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
