import numpy as np
import pytest
import xarray as xr

from cmip6_downscaling.tasks.common_tasks import to_standard_calendar


@pytest.fixture
def da_noleap(val=1.0):
    time = xr.cftime_range(start='2020-01-01', end='2020-12-31', freq='1D', calendar='noleap')
    return xr.DataArray(
        val,
        dims=['lat', 'lon', 'time'],
        coords={'lat': np.arange(19, 56, 1), 'lon': np.arange(-133, -61, 2), 'time': time},
    )


# def test_to_standard_calendar(da_noleap):

#     da_std = to_standard_calendar.run(da_noleap)
#     assert da_noleap.sizes['time'] == 365
#     assert da_std.sizes['time'] == 366
#     assert not da_std.isnull().any().item()
