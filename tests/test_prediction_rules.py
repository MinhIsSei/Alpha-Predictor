import unittest

import pandas as pd

from src.prediction_rules import target_fits_session


class TestTargetFitsSession(unittest.TestCase):
    def test_target_before_close_is_allowed(self):
        prediction_end = pd.Timestamp("2026-09-11 15:55", tz="UTC")
        session_close = pd.Timestamp("2026-09-11 16:00", tz="UTC")

        self.assertTrue(target_fits_session(prediction_end, session_close))

    def test_target_at_close_is_allowed(self):
        session_close = pd.Timestamp("2026-09-11 16:00", tz="UTC")

        self.assertTrue(target_fits_session(session_close, session_close))

    def test_target_after_close_is_rejected(self):
        prediction_end = pd.Timestamp("2026-09-11 16:15", tz="UTC")
        session_close = pd.Timestamp("2026-09-11 16:00", tz="UTC")

        self.assertFalse(target_fits_session(prediction_end, session_close))

    def test_early_close_uses_supplied_session_end(self):
        prediction_end = pd.Timestamp("2026-09-11 13:15", tz="UTC")
        session_close = pd.Timestamp("2026-09-11 13:00", tz="UTC")

        self.assertFalse(target_fits_session(prediction_end, session_close))
