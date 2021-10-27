# import pytest

from cmip6_downscaling.workflows.ERA5 import ERA5_resample


def test_get_ERA5_zstore_list():
    ERA5_zstore_list = ERA5_resample.get_ERA5_zstore_list()
    assert (
        len(ERA5_zstore_list) == 6552
    ), "The length of the zstore list for ERA5 does not seem to match the expected 72324 entries. Perhaps some were added or removed? #13 (vars) * 12 (months) * 42 (years 1979-2020) = 6552 stores"
