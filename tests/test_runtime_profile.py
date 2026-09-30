import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
from runtime_profile import RuntimeProfile


class RuntimeProfileTest(unittest.TestCase):
    def test_stage_is_persisted_before_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "timings.json"
            profile = RuntimeProfile(path)
            with profile.stage("small", rows=3):
                sum(range(100))

            partial = json.loads(path.read_text())
            self.assertFalse(partial["finished"])
            self.assertEqual(partial["stages"][0]["name"], "small")
            self.assertEqual(partial["stages"][0]["rows"], 3)

            profile.finish()
            final = json.loads(path.read_text())
            self.assertTrue(final["finished"])
            self.assertGreaterEqual(final["elapsed_wall_seconds"], 0)

    def test_disabled_profile_is_noop(self):
        profile = RuntimeProfile()
        with profile.stage("ignored"):
            pass
        self.assertEqual(profile.stages, [])


if __name__ == "__main__":
    unittest.main()
