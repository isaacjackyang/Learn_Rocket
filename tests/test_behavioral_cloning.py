import tempfile
import unittest
from pathlib import Path

from rocket_auto_research.gnc.state import BalloonState, RocketState, Vector3, WorldState
from rocket_auto_research.learning.behavioral_cloning import (
    LinearPolicyModel,
    NeuralPolicyModel,
    extract_features,
    fit_mlp_policy,
    load_model,
    predict_action,
    save_model,
)


class BehavioralCloningTests(unittest.TestCase):
    def test_model_round_trip_and_prediction(self) -> None:
        world_state = WorldState(
            time_s=0.0,
            rocket=RocketState(
                position=Vector3(z=100.0),
                velocity=Vector3(x=2.0, y=0.0, z=10.0),
                angular_rate=Vector3(x=0.1, y=0.0, z=0.0),
                launched=True,
            ),
            balloons=[
                BalloonState(
                    balloon_id="b0",
                    position=Vector3(x=10.0, y=0.0, z=120.0),
                    velocity=Vector3(x=1.0, y=0.0, z=0.0),
                )
            ],
            wind=Vector3(x=1.0, y=0.5, z=0.0),
            metadata={"altitude_agl_m": 100.0},
        )
        model = LinearPolicyModel(
            feature_names=[],
            action_names=["launch", "throttle", "tvc_x", "tvc_y", "roll"],
            weights={name: [0.0] * len(extract_features(world_state, world_state.balloons[0])) for name in ["launch", "throttle", "tvc_x", "tvc_y", "roll"]},
            bias={"launch": 1.0, "throttle": 0.7, "tvc_x": 0.1, "tvc_y": -0.1, "roll": 0.0},
            metadata={},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "policy.json"
            save_model(model, path)
            loaded = load_model(path)
        action = predict_action(loaded, world_state, world_state.balloons[0])
        self.assertTrue(action.launch)
        self.assertAlmostEqual(action.throttle, 0.7)

    def test_neural_model_round_trip_and_prediction(self) -> None:
        world_state = WorldState(
            time_s=0.0,
            rocket=RocketState(
                position=Vector3(z=100.0),
                velocity=Vector3(x=2.0, y=1.0, z=10.0),
                angular_rate=Vector3(x=0.1, y=0.0, z=0.0),
                launched=True,
            ),
            balloons=[
                BalloonState(
                    balloon_id="b0",
                    position=Vector3(x=20.0, y=10.0, z=130.0),
                    velocity=Vector3(x=1.0, y=0.5, z=0.0),
                )
            ],
            wind=Vector3(x=1.0, y=0.5, z=0.0),
            metadata={"altitude_agl_m": 100.0},
        )
        model = NeuralPolicyModel(
            feature_names=[],
            action_names=["launch", "throttle", "tvc_x", "tvc_y", "roll"],
            hidden_sizes=[4],
            weights=[
                [[0.0] * len(extract_features(world_state, world_state.balloons[0])) for _ in range(4)],
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
            ],
            biases=[
                [0.0, 0.0, 0.0, 0.0],
                [3.0, 0.7, 0.2, -0.2, 0.1],
            ],
            feature_mean=[0.0] * len(extract_features(world_state, world_state.balloons[0])),
            feature_std=[1.0] * len(extract_features(world_state, world_state.balloons[0])),
            metadata={},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mlp_policy.json"
            save_model(model, path)
            loaded = load_model(path)
        action = predict_action(loaded, world_state, world_state.balloons[0])
        self.assertTrue(action.launch)
        self.assertGreater(action.throttle, 0.65)
        self.assertAlmostEqual(action.tvc_x, 0.2)
        self.assertAlmostEqual(action.tvc_y, -0.2)

    def test_fit_mlp_policy_returns_neural_model(self) -> None:
        samples = [
            {
                "features": [0.1] * 18,
                "action": {"launch": True, "throttle": 0.9, "tvc_x": 0.3, "tvc_y": -0.1, "roll": 0.0},
            },
            {
                "features": [0.2] * 18,
                "action": {"launch": True, "throttle": 0.8, "tvc_x": 0.2, "tvc_y": -0.05, "roll": 0.1},
            },
            {
                "features": [-0.1] * 18,
                "action": {"launch": False, "throttle": 0.2, "tvc_x": -0.2, "tvc_y": 0.1, "roll": -0.1},
            },
        ]
        model = fit_mlp_policy(
            samples,
            hidden_sizes=(8,),
            epochs=5,
            learning_rate=1e-2,
            batch_size=2,
            l2_lambda=1e-5,
            seed=7,
        )
        self.assertEqual(model.model_type, "mlp")
        self.assertEqual(model.hidden_sizes, [8])
        self.assertEqual(len(model.weights), 2)
