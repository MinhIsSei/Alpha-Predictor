import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "src" / "dashboard.py")


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

    def test_switching_to_demo_database_renders(self):
        app = self._run()
        database_control = app.get("button_group")[0]
        if "Demo database" not in database_control.options:
            self.skipTest("Demo database not present.")

        app = database_control.set_value("Demo database").run()
        self.assertEqual(len(app.exception), 0, [e.value for e in app.exception])


if __name__ == "__main__":
    unittest.main()
