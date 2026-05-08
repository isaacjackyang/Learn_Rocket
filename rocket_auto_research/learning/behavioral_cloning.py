from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

from rocket_auto_research.gnc.state import BalloonState, ControlAction, WorldState


FEATURE_NAMES = [
    "altitude_agl_m",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "angular_rate_x",
    "angular_rate_y",
    "angular_rate_z",
    "wind_x",
    "wind_y",
    "rel_target_x",
    "rel_target_y",
    "rel_target_z",
    "rel_target_vx",
    "rel_target_vy",
    "rel_target_vz",
    "target_distance",
    "target_closure",
    "launched_flag",
]

ACTION_NAMES = ["launch", "throttle", "tvc_x", "tvc_y", "roll"]


@dataclass(slots=True)
class LinearPolicyModel:
    feature_names: list[str]
    action_names: list[str]
    weights: dict[str, list[float]]
    bias: dict[str, float]
    metadata: dict[str, Any]
    model_type: str = "linear"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class NeuralPolicyModel:
    feature_names: list[str]
    action_names: list[str]
    hidden_sizes: list[int]
    weights: list[list[list[float]]]
    biases: list[list[float]]
    feature_mean: list[float]
    feature_std: list[float]
    metadata: dict[str, Any]
    model_type: str = "mlp"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PolicyModel: TypeAlias = LinearPolicyModel | NeuralPolicyModel


def extract_features(world_state: WorldState, target: BalloonState | None) -> np.ndarray:
    target = target or BalloonState(balloon_id="none", position=world_state.rocket.position)
    relative_position = target.position - world_state.rocket.position
    relative_velocity = target.velocity - world_state.rocket.velocity
    direction = relative_position.normalized()
    target_closure = (
        world_state.rocket.velocity.x * direction.x
        + world_state.rocket.velocity.y * direction.y
        + world_state.rocket.velocity.z * direction.z
    )
    features = np.array(
        [
            float(world_state.metadata.get("altitude_agl_m", world_state.rocket.position.z)),
            world_state.rocket.velocity.x,
            world_state.rocket.velocity.y,
            world_state.rocket.velocity.z,
            world_state.rocket.angular_rate.x,
            world_state.rocket.angular_rate.y,
            world_state.rocket.angular_rate.z,
            world_state.wind.x,
            world_state.wind.y,
            relative_position.x,
            relative_position.y,
            relative_position.z,
            relative_velocity.x,
            relative_velocity.y,
            relative_velocity.z,
            relative_position.norm(),
            target_closure,
            1.0 if world_state.rocket.launched else 0.0,
        ],
        dtype=float,
    )
    return features


def fit_linear_policy(samples: list[dict[str, Any]], ridge_lambda: float = 1e-3) -> LinearPolicyModel:
    if not samples:
        raise ValueError("Cannot fit behavioral cloning model on an empty sample set.")
    feature_matrix = np.array([sample["features"] for sample in samples], dtype=float)
    outputs = {name: np.array([sample["action"][name] for sample in samples], dtype=float) for name in ACTION_NAMES}
    augmented = np.column_stack([np.ones(len(feature_matrix)), feature_matrix])
    regularizer = ridge_lambda * np.eye(augmented.shape[1])
    regularizer[0, 0] = 0.0
    gram = augmented.T @ augmented + regularizer
    gram_inv = np.linalg.pinv(gram)
    weights: dict[str, list[float]] = {}
    bias: dict[str, float] = {}
    for action_name, target in outputs.items():
        coeffs = gram_inv @ augmented.T @ target
        bias[action_name] = float(coeffs[0])
        weights[action_name] = [float(value) for value in coeffs[1:]]
    metadata = {
        "sample_count": len(samples),
        "ridge_lambda": ridge_lambda,
    }
    return LinearPolicyModel(
        feature_names=list(FEATURE_NAMES),
        action_names=list(ACTION_NAMES),
        weights=weights,
        bias=bias,
        metadata=metadata,
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _prepare_dataset(samples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    if not samples:
        raise ValueError("Cannot fit behavioral cloning model on an empty sample set.")
    feature_matrix = np.array([sample["features"] for sample in samples], dtype=float)
    targets = np.array(
        [
            [
                1.0 if sample["action"]["launch"] else 0.0,
                float(sample["action"]["throttle"]),
                float(sample["action"]["tvc_x"]),
                float(sample["action"]["tvc_y"]),
                float(sample["action"]["roll"]),
            ]
            for sample in samples
        ],
        dtype=float,
    )
    return feature_matrix, targets


def _predict_mlp_outputs(model: NeuralPolicyModel, features: np.ndarray) -> np.ndarray:
    standardized = (features - np.array(model.feature_mean, dtype=float)) / np.array(model.feature_std, dtype=float)
    activations = standardized
    weight_matrices = [np.array(layer, dtype=float) for layer in model.weights]
    bias_vectors = [np.array(layer, dtype=float) for layer in model.biases]
    for weight_matrix, bias_vector in zip(weight_matrices[:-1], bias_vectors[:-1], strict=True):
        activations = np.tanh(activations @ weight_matrix.T + bias_vector)
    raw_outputs = activations @ weight_matrices[-1].T + bias_vectors[-1]
    launch = _sigmoid(raw_outputs[:, [0]])
    throttle = _sigmoid(raw_outputs[:, [1]])
    tvc = raw_outputs[:, 2:4]
    roll = np.tanh(raw_outputs[:, [4]])
    return np.hstack([launch, throttle, tvc, roll])


def fit_mlp_policy(
    samples: list[dict[str, Any]],
    hidden_sizes: tuple[int, ...] = (64, 64),
    epochs: int = 250,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    l2_lambda: float = 1e-5,
    seed: int = 0,
) -> NeuralPolicyModel:
    feature_matrix, targets = _prepare_dataset(samples)
    feature_mean = feature_matrix.mean(axis=0)
    feature_std = feature_matrix.std(axis=0)
    feature_std = np.where(feature_std < 1e-6, 1.0, feature_std)
    normalized = (feature_matrix - feature_mean) / feature_std

    layer_sizes = [normalized.shape[1], *hidden_sizes, len(ACTION_NAMES)]
    rng = np.random.default_rng(seed)
    weights = [
        rng.uniform(
            low=-np.sqrt(6.0 / (layer_sizes[index] + layer_sizes[index + 1])),
            high=np.sqrt(6.0 / (layer_sizes[index] + layer_sizes[index + 1])),
            size=(layer_sizes[index + 1], layer_sizes[index]),
        )
        for index in range(len(layer_sizes) - 1)
    ]
    biases = [np.zeros(size, dtype=float) for size in layer_sizes[1:]]
    adam_m_w = [np.zeros_like(weight_matrix) for weight_matrix in weights]
    adam_v_w = [np.zeros_like(weight_matrix) for weight_matrix in weights]
    adam_m_b = [np.zeros_like(bias_vector) for bias_vector in biases]
    adam_v_b = [np.zeros_like(bias_vector) for bias_vector in biases]
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    step = 0
    loss_history: list[float] = []

    for epoch in range(max(1, epochs)):
        permutation = rng.permutation(len(normalized))
        epoch_losses: list[float] = []
        for start in range(0, len(normalized), max(1, batch_size)):
            batch_indices = permutation[start : start + max(1, batch_size)]
            inputs = normalized[batch_indices]
            batch_targets = targets[batch_indices]
            activations = [inputs]
            pre_activations: list[np.ndarray] = []
            current = inputs
            for weight_matrix, bias_vector in zip(weights[:-1], biases[:-1], strict=True):
                pre_activation = current @ weight_matrix.T + bias_vector
                current = np.tanh(pre_activation)
                pre_activations.append(pre_activation)
                activations.append(current)
            raw_outputs = current @ weights[-1].T + biases[-1]
            launch = _sigmoid(raw_outputs[:, [0]])
            throttle = _sigmoid(raw_outputs[:, [1]])
            tvc = raw_outputs[:, 2:4]
            roll = np.tanh(raw_outputs[:, [4]])
            predictions = np.hstack([launch, throttle, tvc, roll])

            launch_targets = batch_targets[:, [0]]
            aux_targets = batch_targets[:, 1:]
            launch_loss = -np.mean(
                launch_targets * np.log(np.clip(launch, 1e-6, 1.0 - 1e-6))
                + (1.0 - launch_targets) * np.log(np.clip(1.0 - launch, 1e-6, 1.0))
            )
            aux_loss = 0.5 * np.mean((predictions[:, 1:] - aux_targets) ** 2)
            l2_penalty = 0.5 * l2_lambda * sum(float(np.sum(weight_matrix**2)) for weight_matrix in weights)
            batch_loss = float(launch_loss + aux_loss + l2_penalty)
            epoch_losses.append(batch_loss)

            batch_scale = float(len(inputs))
            output_gradient = np.zeros_like(predictions)
            output_gradient[:, [0]] = (launch - launch_targets) / batch_scale
            output_gradient[:, [1]] = ((throttle - batch_targets[:, [1]]) * throttle * (1.0 - throttle)) / batch_scale
            output_gradient[:, 2:4] = (tvc - batch_targets[:, 2:4]) / (batch_scale * 4.0)
            output_gradient[:, [4]] = ((roll - batch_targets[:, [4]]) * (1.0 - roll**2)) / batch_scale

            gradients_w = [np.zeros_like(weight_matrix) for weight_matrix in weights]
            gradients_b = [np.zeros_like(bias_vector) for bias_vector in biases]
            gradients_w[-1] = output_gradient.T @ activations[-1] + l2_lambda * weights[-1]
            gradients_b[-1] = output_gradient.sum(axis=0)
            hidden_gradient = output_gradient @ weights[-1]
            for layer_index in range(len(hidden_sizes) - 1, -1, -1):
                hidden_activation = activations[layer_index + 1]
                hidden_delta = hidden_gradient * (1.0 - hidden_activation**2)
                gradients_w[layer_index] = hidden_delta.T @ activations[layer_index] + l2_lambda * weights[layer_index]
                gradients_b[layer_index] = hidden_delta.sum(axis=0)
                hidden_gradient = hidden_delta @ weights[layer_index]

            step += 1
            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            for layer_index in range(len(weights)):
                adam_m_w[layer_index] = beta1 * adam_m_w[layer_index] + (1.0 - beta1) * gradients_w[layer_index]
                adam_v_w[layer_index] = beta2 * adam_v_w[layer_index] + (1.0 - beta2) * (gradients_w[layer_index] ** 2)
                adam_m_b[layer_index] = beta1 * adam_m_b[layer_index] + (1.0 - beta1) * gradients_b[layer_index]
                adam_v_b[layer_index] = beta2 * adam_v_b[layer_index] + (1.0 - beta2) * (gradients_b[layer_index] ** 2)
                corrected_m_w = adam_m_w[layer_index] / bias_correction1
                corrected_v_w = adam_v_w[layer_index] / bias_correction2
                corrected_m_b = adam_m_b[layer_index] / bias_correction1
                corrected_v_b = adam_v_b[layer_index] / bias_correction2
                weights[layer_index] -= learning_rate * corrected_m_w / (np.sqrt(corrected_v_w) + epsilon)
                biases[layer_index] -= learning_rate * corrected_m_b / (np.sqrt(corrected_v_b) + epsilon)
        loss_history.append(float(np.mean(epoch_losses) if epoch_losses else 0.0))

    metadata = {
        "sample_count": len(samples),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "l2_lambda": l2_lambda,
        "seed": seed,
        "final_loss": loss_history[-1] if loss_history else None,
        "loss_tail": [round(value, 6) for value in loss_history[-5:]],
    }
    return NeuralPolicyModel(
        feature_names=list(FEATURE_NAMES),
        action_names=list(ACTION_NAMES),
        hidden_sizes=list(hidden_sizes),
        weights=[weight_matrix.tolist() for weight_matrix in weights],
        biases=[bias_vector.tolist() for bias_vector in biases],
        feature_mean=feature_mean.tolist(),
        feature_std=feature_std.tolist(),
        metadata=metadata,
    )


def predict_action(model: PolicyModel, world_state: WorldState, target: BalloonState | None) -> ControlAction:
    features = extract_features(world_state, target)
    action_values: dict[str, float] = {}
    if isinstance(model, LinearPolicyModel):
        for action_name in model.action_names:
            weight_vector = np.array(model.weights[action_name], dtype=float)
            action_values[action_name] = float(model.bias[action_name] + np.dot(weight_vector, features))
    else:
        outputs = _predict_mlp_outputs(model, features.reshape(1, -1))[0]
        action_values = {
            "launch": float(outputs[0]),
            "throttle": float(outputs[1]),
            "tvc_x": float(outputs[2]),
            "tvc_y": float(outputs[3]),
            "roll": float(outputs[4]),
        }
    return ControlAction(
        launch=action_values["launch"] >= 0.5,
        throttle=max(0.0, min(1.0, action_values["throttle"])),
        tvc_x=action_values["tvc_x"],
        tvc_y=action_values["tvc_y"],
        roll=max(-1.0, min(1.0, action_values["roll"])),
    )


def save_model(model: PolicyModel, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(model.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_model(path: str | Path) -> PolicyModel:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    model_type = payload.get("model_type", "linear")
    if model_type == "mlp":
        return NeuralPolicyModel(
            feature_names=list(payload["feature_names"]),
            action_names=list(payload["action_names"]),
            hidden_sizes=[int(size) for size in payload["hidden_sizes"]],
            weights=[[list(row) for row in layer] for layer in payload["weights"]],
            biases=[list(layer) for layer in payload["biases"]],
            feature_mean=[float(value) for value in payload["feature_mean"]],
            feature_std=[float(value) for value in payload["feature_std"]],
            metadata=dict(payload.get("metadata", {})),
            model_type="mlp",
        )
    return LinearPolicyModel(
        feature_names=list(payload["feature_names"]),
        action_names=list(payload["action_names"]),
        weights={key: list(value) for key, value in payload["weights"].items()},
        bias={key: float(value) for key, value in payload["bias"].items()},
        metadata=dict(payload.get("metadata", {})),
        model_type="linear",
    )
