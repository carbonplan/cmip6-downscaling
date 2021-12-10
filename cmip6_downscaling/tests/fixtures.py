import numpy as np
import xarray as xr


def example_3d_dataarray_us_domain(val=0):
    return xr.DataArray(
        val,
        dims=['lat', 'lon', 'time'],
        coords={'lat': np.arange(19, 56, 1), 'lon': np.arange(-133, -61, 2), 'time': np.arange(5)},
    )


def example_3d_data_us_domain():
    ds = xr.Dataset()
    for v in ['var1', 'var2']:
        ds[v] = example_3d_dataarray_us_domain()
    return ds
