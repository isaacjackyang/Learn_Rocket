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
    parser = argparse.ArgumentParser(description="Replay the current best config with fresh seeds.")
    parser.add_argument("--best-config", default="results/best_agents/best_config.json")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--config", default="configs/auto_research.yaml")
    args = parser.parse_args()

    best_payload = json.loads(Path(args.best_config).read_text(encoding="utf-8"))
    spec = ExperimentSpec(
        strategy_name=best_payload["strategy_name"],
        params=best_payload["params"],
        seeds=args.seeds,
        note=f"replay_of:{best_payload['experiment_id']}",
    )
    config = load_yaml_config(args.config)
    result = ExperimentRunner(build_simulation_adapter(config)).run(spec)
    trajectory_path = None
    if result.episodes:
        trajectory_path = f"results/runs/{result.spec.experiment_id}/trajectory_seed_{result.episodes[0].seed:03d}.jsonl"
    print(
        json.dumps(
            {
                "replay_of": best_payload["experiment_id"],
                "new_experiment_id": result.spec.experiment_id,
                "strategy_name": result.spec.strategy_name,
                "summary": result.summary.to_dict(),
                "failure_report": result.failure_report.to_dict(),
                "first_episode_metadata": result.episodes[0].metadata if result.episodes else {},
                "trajectory_path": trajectory_path,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
