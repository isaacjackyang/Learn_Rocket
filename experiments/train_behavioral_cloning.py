from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rocket_auto_research.gnc.observation_parser import parse_observation
from rocket_auto_research.learning.behavioral_cloning import (
    extract_features,
    fit_linear_policy,
    fit_mlp_policy,
    save_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight behavioral cloning policy from collected transitions.")
    parser.add_argument("--dataset", required=True, help="Path to imitation dataset JSONL.")
    parser.add_argument("--output", required=True, help="Path to save the trained policy JSON.")
    parser.add_argument("--model-type", choices=("linear", "mlp"), default="mlp", help="Model family to train.")
    parser.add_argument("--ridge", type=float, default=1e-3, help="Ridge regularization strength.")
    parser.add_argument("--hidden-sizes", nargs="+", type=int, default=[64, 64], help="Hidden layer sizes for MLP training.")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs for MLP training.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate for MLP training.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for MLP training.")
    parser.add_argument("--l2", type=float, default=1e-5, help="L2 regularization strength for MLP training.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for MLP initialization.")
    args = parser.parse_args()

    samples: list[dict[str, object]] = []
    dataset_path = Path(args.dataset)
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            world_state = parse_observation(payload["observation"], schema="flat_competition")
            selected_target_id = payload.get("selected_target_id")
            target = None
            if selected_target_id is not None:
                target = next(
                    (balloon for balloon in world_state.balloons if balloon.balloon_id == selected_target_id),
                    None,
                )
            if target is None:
                released = [balloon for balloon in world_state.balloons if balloon.released and not balloon.popped]
                target = released[0] if released else None
            features = extract_features(world_state, target).tolist()
            samples.append(
                {
                    "features": features,
                    "action": payload["action"],
                }
            )

    if args.model_type == "linear":
        model = fit_linear_policy(samples, ridge_lambda=args.ridge)
    else:
        model = fit_mlp_policy(
            samples,
            hidden_sizes=tuple(args.hidden_sizes),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            l2_lambda=args.l2,
            seed=args.seed,
        )
    save_model(model, args.output)
    print(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "output": args.output,
                "model_type": args.model_type,
                "sample_count": len(samples),
                "ridge_lambda": args.ridge if args.model_type == "linear" else None,
                "hidden_sizes": args.hidden_sizes if args.model_type == "mlp" else None,
                "epochs": args.epochs if args.model_type == "mlp" else None,
                "feature_count": len(model.feature_names),
                "metadata": model.metadata,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
