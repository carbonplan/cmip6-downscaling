import numpy as np
import pytest
import xarray as xr

from cmip6_downscaling.data.utils import lon_to_180, to_standard_calendar
from cmip6_downscaling.utils import str_to_hash, write


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


def test_to_standard_calendar(da_noleap):

    da_std = to_standard_calendar(da_noleap)
    assert da_noleap.sizes['time'] == 365
    assert da_std.sizes['time'] == 366
    assert not da_std.isnull().any().compute().item()


def test_str_to_hash():
    s = "test"
    h = str_to_hash(s)
    assert h == "96ad3bb4a2d666d3"


def test_write(tmp_path):
    ds = create_test_ds("lon", "lat", "time", 40, 20, 6)
    target = tmp_path / "test.zarr"
    write(ds, target)
    assert target.is_dir()
    assert target.exists()

    xr.testing.assert_identical(ds, xr.open_zarr(target))
