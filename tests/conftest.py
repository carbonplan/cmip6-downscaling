import numpy as np
import pytest
import xarray as xr

from cmip6_downscaling import config


@pytest.fixture(scope="session", autouse=True)
def set_test_config():
    config.set(
        {
            'storage.intermediate.uri': '/tmp/intermediate',
            'storage.results.uri': '/tmp/results',
            'storage.temporary.uri': '/tmp/temporary',
        }
    )


def example_3d_dataarray_us_domain(val=0):
    return xr.DataArray(
        val,
        dims=['lat', 'lon', 'time'],
        coords={'lat': np.arange(19, 56, 1), 'lon': np.arange(-133, -61, 2), 'time': np.arange(5)},
    )


@pytest.fixture(scope="session", autouse=True)
def example_3d_data_us_domain():
    ds = xr.Dataset()
    for v in ['var1', 'var2']:
        ds[v] = example_3d_dataarray_us_domain()
    return ds


@pytest.fixture
def da_noleap(val=1.0):
    time = xr.cftime_range(start='2020-01-01', end='2020-12-31', freq='1D', calendar='noleap')
    return xr.DataArray(
        val,
        dims=['lat', 'lon', 'time'],
        coords={'lat': np.arange(19, 56, 1), 'lon': np.arange(-133, -61, 2), 'time': time},
    )
