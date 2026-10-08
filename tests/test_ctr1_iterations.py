import argparse
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import model  # noqa: E402


class Ctr1IterationsTest(unittest.TestCase):
    def tearDown(self):
        model.CAT_CTR1_PARAMS["iterations"] = 3500

    def parse(self, argv):
        ap = argparse.ArgumentParser()
        model.add_model_args(ap)
        args = ap.parse_args(argv)
        model.apply_model_args(args)
        return args

    def test_default_reproduces_3500(self):
        self.parse([])
        self.assertEqual(model.CAT_CTR1_PARAMS["iterations"], 3500)

    def test_flag_sets_ctr1_only(self):
        self.parse(["--ctr1-iterations", "8000", "--cat-iterations", "10000"])
        self.assertEqual(model.CAT_CTR1_PARAMS["iterations"], 8000)
        self.assertEqual(model.CAT_MU_PARAMS["iterations"], 10000)


if __name__ == "__main__":
    unittest.main()
