import unittest

from rocket_auto_research.gnc.state import RocketState, Vector3, WorldState
from rocket_auto_research.strategies.energy_aware import EnergyAwareStrategy


class ModularStrategyTests(unittest.TestCase):
    def test_ascent_turn_scale_grows_with_altitude_progress(self) -> None:
        strategy = EnergyAwareStrategy(
            {
                "ascent_turn_scale": 0.25,
                "ascent_targeting_turn_scale": 0.85,
                "ascent_targeting_altitude_m": 300.0,
            }
        )
        target_position = Vector3(x=220.0, y=0.0, z=260.0)
        low_altitude_state = WorldState(
            time_s=1.0,
            rocket=RocketState(
                position=Vector3(x=0.0, y=0.0, z=40.0),
                velocity=Vector3(x=0.0, y=0.0, z=30.0),
                launched=True,
            ),
            balloons=[],
            metadata={"altitude_agl_m": 40.0},
        )
        high_altitude_state = WorldState(
            time_s=1.0,
            rocket=RocketState(
                position=Vector3(x=0.0, y=0.0, z=220.0),
                velocity=Vector3(x=0.0, y=0.0, z=30.0),
                launched=True,
            ),
            balloons=[],
            metadata={"altitude_agl_m": 220.0},
        )

        low_scale = strategy._ascent_turn_scale_for(low_altitude_state, target_position)
        high_scale = strategy._ascent_turn_scale_for(high_altitude_state, target_position)

        self.assertGreater(low_scale, 0.25)
        self.assertGreater(high_scale, low_scale)
        self.assertLessEqual(high_scale, 0.85)


if __name__ == "__main__":
    unittest.main()
