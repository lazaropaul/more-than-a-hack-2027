import unittest

import numpy as np
import pandas as pd

from src.possession import (
    AWAY,
    DEAD,
    HOME,
    UNKNOWN,
    _mark_boundary_gaps,
    _confirm_switches,
    _fill_play_unknowns,
    _tagged_team,
)


class PossessionSmoothingTests(unittest.TestCase):
    def test_short_opposing_blip_does_not_change_possession(self):
        labels = np.array([HOME] * 60 + [AWAY] * 20 + [HOME] * 60)

        result = _confirm_switches(labels, min_switch_frames=50)

        self.assertEqual(result.tolist(), [HOME] * 140)

    def test_sustained_switch_is_backdated_to_first_evidence(self):
        labels = np.array([HOME] * 60 + [AWAY] * 50)

        result = _confirm_switches(labels, min_switch_frames=50)

        self.assertEqual(result[60:].tolist(), [AWAY] * 50)

    def test_dead_ball_resets_switch_state(self):
        labels = np.array([HOME] * 10 + [DEAD] * 5 + [AWAY] * 3)

        result = _confirm_switches(labels, min_switch_frames=50)

        self.assertEqual(result[10:15].tolist(), [DEAD] * 5)
        self.assertEqual(result[15:].tolist(), [AWAY] * 3)

    def test_unknown_gaps_carry_state_and_leading_gap_uses_first_state(self):
        labels = np.array([UNKNOWN, HOME, UNKNOWN, DEAD, UNKNOWN, AWAY, UNKNOWN])

        result = _fill_play_unknowns(labels)

        self.assertEqual(result.tolist(), [HOME, HOME, HOME, DEAD, DEAD, AWAY, AWAY])

    def test_entirely_unknown_half_stays_unknown(self):
        labels = np.array([UNKNOWN, UNKNOWN, UNKNOWN])

        result = _fill_play_unknowns(labels)

        self.assertEqual(result.tolist(), [UNKNOWN, UNKNOWN, UNKNOWN])


class SmartTagTeamTests(unittest.TestCase):
    def test_defensive_transition_means_opponent_possession(self):
        tag = {"code": "DEFENSIVE TRANSITION", "tags": {"Team": "Barcelona"}}

        self.assertEqual(_tagged_team(tag), AWAY)

    def test_unassigned_team_is_not_used_as_evidence(self):
        tag = {"code": "BUILD UP", "tags": {"Team": "N/A"}}

        self.assertIsNone(_tagged_team(tag))


class DeadBallTests(unittest.TestCase):
    def test_long_gap_after_ball_reaches_boundary_is_dead(self):
        features = {
            "ball_visible": np.array([True] * 40 + [False] * 80),
            "ball_x_m": np.array([1.0] * 40 + [np.nan] * 80),
            "ball_y_m": np.array([34.0] * 40 + [np.nan] * 80),
        }
        dead = np.zeros(120, dtype=bool)

        _mark_boundary_gaps(pd.DataFrame(features), dead, (105.3, 68.0))

        self.assertFalse(dead[:40].any())
        self.assertTrue(dead[40:].all())

    def test_short_gap_in_play_is_not_dead(self):
        features = {
            "ball_visible": np.array([True] * 40 + [False] * 10),
            "ball_x_m": np.array([50.0] * 40 + [np.nan] * 10),
            "ball_y_m": np.array([34.0] * 40 + [np.nan] * 10),
        }
        dead = np.zeros(50, dtype=bool)

        _mark_boundary_gaps(pd.DataFrame(features), dead, (105.3, 68.0))

        self.assertFalse(dead.any())

    def test_boundary_gap_shorter_than_three_seconds_is_not_dead(self):
        features = {
            "ball_visible": np.array([True] * 40 + [False] * 74),
            "ball_x_m": np.array([1.0] * 40 + [np.nan] * 74),
            "ball_y_m": np.array([34.0] * 40 + [np.nan] * 74),
        }
        dead = np.zeros(114, dtype=bool)

        _mark_boundary_gaps(pd.DataFrame(features), dead, (105.3, 68.0))

        self.assertFalse(dead.any())


if __name__ == "__main__":
    unittest.main()
