from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rocket_auto_research.auto_research.competition_platform_api import SimulatedCompetitionEnv
from rocket_auto_research.auto_research.experiment_spec import ExperimentSpec
from rocket_auto_research.config import load_yaml_config
from rocket_auto_research.gnc.action_formatter import format_action
from rocket_auto_research.gnc.observation_parser import parse_observation
from rocket_auto_research.learning.behavioral_cloning import extract_features, fit_mlp_policy, save_model
from rocket_auto_research.strategies.registry import build_strategy


def _find_latest_imitation_dataset(dataset_dir: Path) -> Path | None:
    candidates = sorted(dataset_dir.glob("imitation_*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _load_training_samples(dataset_path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            world_state = parse_observation(payload["observation"], schema="flat_competition")
            selected_target_id = payload.get("selected_target_id")
            target = None
            if selected_target_id is not None:
                target = next((balloon for balloon in world_state.balloons if balloon.balloon_id == selected_target_id), None)
            if target is None:
                released = [balloon for balloon in world_state.balloons if balloon.released and not balloon.popped]
                target = released[0] if released else None
            samples.append(
                {
                    "features": extract_features(world_state, target).tolist(),
                    "action": payload["action"],
                }
            )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one DAgger-style imitation iteration using learner rollouts and expert labels.")
    parser.add_argument("--expert-best-config", default="results/best_agents/best_config.json")
    parser.add_argument("--learner-config", default="configs/rl_policy_wrapper_mlp.yaml")
    parser.add_argument("--base-dataset", help="Optional base imitation dataset JSONL. Defaults to the newest results/datasets/imitation_*.jsonl.")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", default="results/datasets")
    parser.add_argument("--model-output", default="results/models/mlp_policy_dagger.json")
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[64, 64])
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--l2", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    expert_payload = json.loads(Path(args.expert_best_config).read_text(encoding="utf-8"))
    learner_config = load_yaml_config(args.learner_config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_dataset_path = Path(args.base_dataset) if args.base_dataset else _find_latest_imitation_dataset(output_dir)
    dagger_dataset_path = output_dir / f"dagger_{expert_payload['experiment_id']}_{len(args.seeds)}seeds.jsonl"
    manifest_path = output_dir / f"dagger_{expert_payload['experiment_id']}_{len(args.seeds)}seeds_manifest.json"

    transition_count = 0
    episode_summaries: list[dict[str, Any]] = []
    with dagger_dataset_path.open("w", encoding="utf-8") as handle:
        for rollout_seed in args.seeds:
            spec = ExperimentSpec(
                strategy_name=expert_payload["strategy_name"],
                params=expert_payload["params"],
                seeds=[rollout_seed],
                note=f"dagger_rollout:{expert_payload['experiment_id']}:{rollout_seed}",
            )
            env = SimulatedCompetitionEnv(spec, rollout_seed)
            expert_strategy = build_strategy(expert_payload["strategy_name"], dict(expert_payload["params"]))
            learner_strategy = build_strategy(learner_config["strategy_name"], dict(learner_config.get("params", {})))
            observation, reset_info = env.reset()
            terminated = False
            truncated = False
            rollout_reward = 0.0
            steps = 0

            while not terminated and not truncated:
                world_state = parse_observation(observation, schema="flat_competition")
                learner_action_obj = learner_strategy.act(world_state)
                expert_action_obj = expert_strategy.act(world_state)
                env.record_target(learner_strategy.context.current_target_id)
                learner_action = format_action(learner_action_obj, schema="flat_competition")
                expert_action = format_action(expert_action_obj, schema="flat_competition")
                next_observation, reward, terminated, truncated, info = env.step(learner_action)
                handle.write(
                    json.dumps(
                        {
                            "seed": rollout_seed,
                            "step_index": steps,
                            "strategy_name": learner_config["strategy_name"],
                            "observation": observation,
                            "action": expert_action,
                            "behavior_action": learner_action,
                            "selected_target_id": expert_strategy.context.current_target_id,
                            "behavior_target_id": learner_strategy.context.current_target_id,
                            "reward": reward,
                            "next_observation": next_observation,
                            "terminated": terminated,
                            "truncated": truncated,
                            "info": info,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                observation = next_observation
                rollout_reward += reward
                steps += 1
                transition_count += 1

            episode_summaries.append(
                {
                    "seed": rollout_seed,
                    "steps": steps,
                    "episode_reward": round(rollout_reward, 4),
                    "final_score": round(env.final_score(), 4),
                    "popped": env.state.popped,
                    "reset_info": reset_info,
                }
            )

    training_samples: list[dict[str, Any]] = []
    if base_dataset_path is not None and base_dataset_path.exists():
        training_samples.extend(_load_training_samples(base_dataset_path))
    training_samples.extend(_load_training_samples(dagger_dataset_path))
    model = fit_mlp_policy(
        training_samples,
        hidden_sizes=tuple(args.hidden_sizes),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        l2_lambda=args.l2,
        seed=args.seed,
    )
    save_model(model, args.model_output)

    manifest = {
        "expert_best_experiment_id": expert_payload["experiment_id"],
        "learner_strategy_name": learner_config["strategy_name"],
        "learner_config": args.learner_config,
        "base_dataset_path": str(base_dataset_path) if base_dataset_path is not None else None,
        "dagger_dataset_path": str(dagger_dataset_path),
        "model_output": args.model_output,
        "transition_count": transition_count,
        "training_sample_count": len(training_samples),
        "seeds": args.seeds,
        "episodes": episode_summaries,
        "model_metadata": model.metadata,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"manifest_path": str(manifest_path), **manifest}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
