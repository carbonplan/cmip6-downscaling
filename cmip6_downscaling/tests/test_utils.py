import numpy as np
import pytest
import xarray as xr

from cmip6_downscaling.methods.common.utils import lon_to_180


def create_test_ds(xname, yname, zname, xlen, ylen, zlen):
    x = np.linspace(0, 359, xlen)
    y = np.linspace(-90, 89, ylen)
    z = np.linspace(0, 5000, zlen)

    data = np.random.rand(len(x), len(y), len(z))
    ds = xr.DataArray(data, coords=[(xname, x), (yname, y), (zname, z)]).to_dataset(name="test")
    ds.attrs["source_id"] = "test_id"
    # if x and y are not lon and lat, add lon and lat to make sure there are no conflicts
    lon = ds[xname] * xr.ones_like(ds[yname])
    lat = xr.ones_like(ds[xname]) * ds[yname]
    if xname != "lon" and yname != "lat":
        ds = ds.assign_coords(lon=lon, lat=lat)
    else:
        ds = ds.assign_coords(longitude=lon, latitude=lat)
    return ds


@pytest.mark.parametrize(
    "shift",
    [
        0,
        -180,
    ],
)  # cant handle positive shifts yet
def test_lon_to_180(shift):
    xlen, ylen, zlen = (40, 20, 6)
    ds = create_test_ds("lon", "lat", "time", xlen, ylen, zlen)

    ds = ds.assign_coords(lon=ds["lon"].data + shift)
    lon = ds["lon"].reset_coords(drop=True)
    ds = ds.assign_coords(lon=lon + shift)

    ds_lon_corrected = lon_to_180(ds)
    assert ds_lon_corrected.lon.min() < -1
    assert ds_lon_corrected.lon.max() <= 180
    assert (ds_lon_corrected.lon.diff(dim='lon') > 0).all()
