import unittest

from rocket_auto_research.auto_research.competition_contract import (
    validate_action_payload,
    validate_observation_payload,
)


class CompetitionContractTests(unittest.TestCase):
    def test_flat_competition_observation_validation_passes(self) -> None:
        observation = {
            "time_s": 0.0,
            "rocket_position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rocket_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
            "balloons": [],
        }
        validate_observation_payload(observation, schema="flat_competition")

    def test_flat_competition_action_validation_rejects_bad_throttle(self) -> None:
        with self.assertRaises(ValueError):
            validate_action_payload(
                {
                    "launch": True,
                    "throttle": 1.5,
                    "tvc_x": 0.0,
                    "tvc_y": 0.0,
                    "roll": 0.0,
                },
                schema="flat_competition",
            )

    def test_balloon_challenge_observation_validation_passes(self) -> None:
        observation = {
            "simulation_time": 0.0,
            "balloon_status": [[0], [1]],
            "balloon_states": [
                [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
                [1.0, 2.0, 20.0, 0.1, 0.2, 0.3],
            ],
            "rocket_sensors": [0.0] * 12,
        }
        validate_observation_payload(observation, schema="balloon_challenge")

    def test_balloon_challenge_action_validation_rejects_bad_vectors(self) -> None:
        with self.assertRaises(ValueError):
            validate_action_payload(
                {
                    "launch": True,
                    "launch_inclination_heading": [90.0],
                    "tvc": [0.0, 0.0],
                    "throttle": 0.5,
                    "roll": 0.0,
                },
                schema="balloon_challenge",
            )
