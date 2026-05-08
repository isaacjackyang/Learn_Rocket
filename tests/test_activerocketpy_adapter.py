import unittest

from rocket_auto_research.auto_research.activerocketpy_adapter import (
    ActiveRocketPySimulationAdapter,
    BalloonFieldCourse,
)
from rocket_auto_research.auto_research.experiment_spec import ExperimentSpec
from rocket_auto_research.gnc.state import Vector3


class ActiveRocketPyAdapterTests(unittest.TestCase):
    def test_seeded_balloon_field_is_stable(self) -> None:
        spec = ExperimentSpec(
            strategy_name="score_based",
            params={"kp": 0.4},
            seeds=[0],
            note="adapter_test",
        )
        left = ActiveRocketPySimulationAdapter._seeded_balloon_field(spec, 3)
        right = ActiveRocketPySimulationAdapter._seeded_balloon_field(spec, 3)
        self.assertEqual(
            [
                (
                    balloon.balloon_id,
                    balloon.spawn_position.to_dict(),
                    balloon.drift_velocity.to_dict(),
                    balloon.release_time_s,
                    balloon.radius_m,
                )
                for balloon in left[:5]
            ],
            [
                (
                    balloon.balloon_id,
                    balloon.spawn_position.to_dict(),
                    balloon.drift_velocity.to_dict(),
                    balloon.release_time_s,
                    balloon.radius_m,
                )
                for balloon in right[:5]
            ],
        )

    def test_balloon_field_marks_hits_after_release(self) -> None:
        spec = ExperimentSpec(
            strategy_name="score_based",
            params={},
            seeds=[0],
            note="adapter_test",
        )
        balloons = ActiveRocketPySimulationAdapter._seeded_balloon_field(spec, 0)
        first = balloons[0]
        first.release_time_s = 0.0
        course = BalloonFieldCourse(balloons)
        course.update(first.spawn_position, time_s=0.25, ambient_wind=Vector3())
        self.assertEqual(course.popped_count, 1)
        self.assertEqual(course.first_pop_time, 0.25)
