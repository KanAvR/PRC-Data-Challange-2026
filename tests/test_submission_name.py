import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import submission


class RemoteNameTest(unittest.TestCase):
    def test_numbered_versions(self):
        self.assertEqual(submission.remote_name("48"), "humble-giraffe_v48.parquet")
        self.assertEqual(submission.remote_name("v48"), "humble-giraffe_v48.parquet")

    def test_final_is_not_numbered(self):
        # ranking.html: the one final submission is <team-name>_final.parquet
        self.assertEqual(submission.remote_name("final"), "humble-giraffe_final.parquet")


if __name__ == "__main__":
    unittest.main()
