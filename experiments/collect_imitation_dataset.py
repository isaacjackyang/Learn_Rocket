from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rocket_auto_research.auto_research.competition_platform_api import SimulatedCompetitionEnv
from rocket_auto_research.auto_research.experiment_spec import ExperimentSpec
from rocket_auto_research.gnc.action_formatter import format_action
from rocket_auto_research.gnc.observation_parser import parse_observation
from rocket_auto_research.strategies.registry import build_strategy


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect imitation-learning data from the current best strategy.")
    parser.add_argument("--best-config", default="results/best_agents/best_config.json")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", default="results/datasets")
    args = parser.parse_args()

    best_payload = json.loads(Path(args.best_config).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / f"imitation_{best_payload['experiment_id']}_{len(args.seeds)}seeds.jsonl"
    manifest_path = output_dir / f"imitation_{best_payload['experiment_id']}_{len(args.seeds)}seeds_manifest.json"

    transition_count = 0
    episode_summaries: list[dict[str, object]] = []
    with dataset_path.open("w", encoding="utf-8") as handle:
        for seed in args.seeds:
            spec = ExperimentSpec(
                strategy_name=best_payload["strategy_name"],
                params=best_payload["params"],
                seeds=[seed],
                note=f"imitation_collect:{best_payload['experiment_id']}:{seed}",
            )
            env = SimulatedCompetitionEnv(spec, seed)
            strategy = build_strategy(spec.strategy_name, dict(spec.params))
            observation, reset_info = env.reset()
            terminated = False
            truncated = False
            episode_reward = 0.0
            steps = 0

            while not terminated and not truncated:
                world_state = parse_observation(observation, schema="flat_competition")
                action_obj = strategy.act(world_state)
                env.record_target(strategy.context.current_target_id)
                action = format_action(action_obj, schema="flat_competition")
                next_observation, reward, terminated, truncated, info = env.step(action)
                transition = {
                    "seed": seed,
                    "step_index": steps,
                    "strategy_name": spec.strategy_name,
                    "observation": observation,
                    "action": action,
                    "selected_target_id": strategy.context.current_target_id,
                    "reward": reward,
                    "next_observation": next_observation,
                    "terminated": terminated,
                    "truncated": truncated,
                    "info": info,
                }
                handle.write(json.dumps(transition, ensure_ascii=False) + "\n")
                observation = next_observation
                episode_reward += reward
                steps += 1
                transition_count += 1

            episode_summaries.append(
                {
                    "seed": seed,
                    "steps": steps,
                    "episode_reward": round(episode_reward, 4),
                    "final_score": round(env.final_score(), 4),
                    "popped": env.state.popped,
                    "reset_info": reset_info,
                }
            )

    manifest = {
        "source_best_experiment_id": best_payload["experiment_id"],
        "strategy_name": best_payload["strategy_name"],
        "observation_schema": "flat_competition",
        "action_schema": "flat_competition",
        "seeds": args.seeds,
        "transition_count": transition_count,
        "dataset_path": str(dataset_path),
        "episodes": episode_summaries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"dataset_path": str(dataset_path), "manifest_path": str(manifest_path), **manifest}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
