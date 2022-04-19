import numpy as np

from cmip6_downscaling.analysis.metrics import spell_length_stat


def test_spell_length_stat():
    series = np.array([True, True, True, True, False, False, True, False, True])
    assert spell_length_stat(series) == 2.0
    series = np.array([False, True, True, True, True, False, False, True, False, True])
    assert spell_length_stat(series) == 2.0
    series = np.array([True, True, True, True, False, False, True, False, True, False])
    assert spell_length_stat(series) == 2.0
    series = np.array([True, True, True, True, False, False, True, False, True])
    assert spell_length_stat(series) == 2.0
    series = np.array([False, True, True, True, True, False, False, True, False, True, False])
    assert spell_length_stat(series) == 2.0
