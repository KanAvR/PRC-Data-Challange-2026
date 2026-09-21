import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, 'output/experiments/v9_src')
import numpy as np
from normal_correction import apply_normal_correction
from model import combine

def test_exact_mixture_component_change():
    mu = np.array([1000., 1000., 1000., 1000.])
    new = mu + 400
    p = np.array([.5, 0., .5, 1.])
    sched = np.array([3000., 12000., 12000., 12000.])
    old = combine(mu, p, sched)
    actual = apply_normal_correction(old, np.full(4, np.nan), sched, p, mu, new)
    np.testing.assert_allclose(actual, combine(mu + .25 * (new - mu), p, sched))
    assert actual[-1] == old[-1]

def test_matched_support_and_specialist_preserved():
    old = np.array([1000., 3660., 8000.])
    gap = np.array([1000., 0., 8000.])
    result = apply_normal_correction(old, gap, gap, np.zeros(3), old, old + 1000)
    np.testing.assert_allclose(result, [1250., 3660., 8000.])

def test_identity():
    old = np.array([1000., 4000.])
    result = apply_normal_correction(old, np.array([2000., np.nan]), old, np.array([0., .5]), old, old)
    np.testing.assert_array_equal(result, old)
