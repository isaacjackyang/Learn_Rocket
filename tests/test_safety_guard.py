import unittest

from rocket_auto_research.gnc.safety_guard import SafetyGuard
from rocket_auto_research.gnc.state import ControlAction


class SafetyGuardTests(unittest.TestCase):
    def test_safety_guard_clamps_ranges(self) -> None:
        sanitized = SafetyGuard().sanitize(
            ControlAction(
                launch=True,
                throttle=2.0,
                tvc_x=3.0,
                tvc_y=-3.0,
                roll=4.0,
            )
        )
        self.assertEqual(sanitized.throttle, 1.0)
        self.assertEqual(sanitized.tvc_x, 1.0)
        self.assertEqual(sanitized.tvc_y, -1.0)
        self.assertEqual(sanitized.roll, 1.0)
