import json
import tempfile
import unittest
from pathlib import Path

from rocket_auto_research.agents.competition_agent import CompetitionAgent


class CompetitionAgentTests(unittest.TestCase):
    def test_agent_parses_and_formats_flat_competition_schema(self) -> None:
        payload = {
            "strategy_name": "score_based",
            "params": {"kp": 0.4, "lookahead_time": 0.8},
            "observation_schema": "flat_competition",
            "action_schema": "flat_competition",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "best_config.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            agent = CompetitionAgent(config_path)
            observation = {
                "time_s": 0.2,
                "rocket_position": {"x": 0.0, "y": 0.0, "z": 10.0},
                "rocket_velocity": {"x": 0.0, "y": 0.0, "z": 20.0},
                "rocket_attitude": {"x": 0.0, "y": 0.0, "z": 0.0},
                "rocket_angular_rate": {"x": 0.0, "y": 0.0, "z": 0.0},
                "rocket_launched": True,
                "wind": {"x": 1.0, "y": 0.0, "z": 0.0},
                "balloons": [
                    {
                        "balloon_id": "b0",
                        "position": {"x": 50.0, "y": 0.0, "z": 100.0},
                        "velocity": {"x": 0.5, "y": 0.0, "z": 0.0},
                        "released": True,
                        "popped": False,
                    }
                ],
            }
            action = agent.act(observation)
        self.assertIn("tvc_x", action)
        self.assertIn("tvc_y", action)
        self.assertIn("throttle", action)

    def test_agent_parses_and_formats_balloon_challenge_schema(self) -> None:
        payload = {
            "strategy_name": "score_based",
            "params": {"kp": 0.4, "lookahead_time": 0.8},
            "observation_schema": "balloon_challenge",
            "action_schema": "balloon_challenge",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "best_config.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            agent = CompetitionAgent(config_path)
            observation = {
                "simulation_time": 0.2,
                "balloon_status": [[1]],
                "balloon_states": [[50.0, 0.0, 100.0, 0.5, 0.0, 0.0]],
                "rocket_sensors": [0.0, 0.0, 0.0, 0.0, 0.0, -9.81, 0.0, 0.0, 10.0, 0.0, 0.0, 20.0],
            }
            action = agent.act(observation)
        self.assertIn("launch_inclination_heading", action)
        self.assertIn("tvc", action)
        self.assertEqual(len(action["launch_inclination_heading"]), 2)
        self.assertEqual(len(action["tvc"]), 2)
