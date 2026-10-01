import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


class TestDashboardApp(unittest.TestCase):
    """Smoke tests: the app renders end to end without raising.

    Runs against whichever databases exist — in CI (where data/predictions/
    is gitignored) that's only the checked-in demo database.
    """

    def _run(self):
        # The model-comparison section re-fits models when the processed
        # table exists locally, which takes a few seconds.
        return AppTest.from_file(APP_PATH, default_timeout=60).run()

    def test_renders_without_exceptions(self):
        app = self._run()
        self.assertEqual(len(app.exception), 0, [e.value for e in app.exception])
        self.assertEqual(app.title[0].value, "Alpha-Predictor")

    def test_shows_headline_tiles(self):
        app = self._run()
        labels = [metric.label for metric in app.metric]
        self.assertIn("Predictions", labels)
        self.assertIn("Accuracy (evaluated only)", labels)

    def test_prediction_history_and_walk_forward_are_separate_labelled_sections(self):
        app = self._run()
        headers = [h.value for h in app.header]
        self.assertTrue(any("prediction history" in h and "records" in h for h in headers), headers)
        self.assertIn("Historical walk-forward evaluation", headers)

    def test_demo_snapshot_states_its_source_and_dates(self):
        app = self._run()
        database_control = app.get("button_group")[0]
        if "Demo database" not in database_control.options:
            self.skipTest("Demo database not present.")

        app = database_control.set_value("Demo database").run()
        captions = " ".join(c.value for c in app.caption)
        self.assertIn("Source: demo database", captions)
        self.assertIn("2026", captions)
        self.assertIn("not a live feed", captions)
        self.assertIn("Latest prediction in this selection", captions)

    def test_switching_to_demo_database_renders(self):
        app = self._run()
        database_control = app.get("button_group")[0]
        if "Demo database" not in database_control.options:
            self.skipTest("Demo database not present.")

        app = database_control.set_value("Demo database").run()
        self.assertEqual(len(app.exception), 0, [e.value for e in app.exception])

    def test_fresh_clone_renders_from_checked_in_demo_data_only(self):
        """What a Streamlit Community Cloud deploy sees: no live database and
        no processed training table, since both are gitignored."""
        import src.dashboard as dashboard

        missing = Path(__file__).resolve().parent / "does-not-exist"
        demo_only = {"Demo database": dashboard.DATABASES["Demo database"]}

        with (
            patch.object(dashboard, "DATABASES", demo_only),
            patch.object(dashboard, "PROCESSED_PATH", missing),
        ):
            app = self._run()

        self.assertEqual(len(app.exception), 0, [e.value for e in app.exception])
        labels = [metric.label for metric in app.metric]
        self.assertIn("Predictions", labels)
        # The model comparison still renders, from the precomputed snapshot.
        self.assertIn("LR beats gradient boosting", labels)
        self.assertTrue(any("precomputed historical snapshot" in info.value for info in app.info))


if __name__ == "__main__":
    unittest.main()
