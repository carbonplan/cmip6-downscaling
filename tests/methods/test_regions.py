import itertools

import numpy as np
import xarray as xr

from cmip6_downscaling.methods.regions import combine_outputs, generate_subdomains


def test_generate_subdomains(example_3d_data_us_domain):
    ds = example_3d_data_us_domain

    region_def = 'ar6'
    buffer_size = 4
    subdomains, mask = generate_subdomains(
        ex_output_grid=ds.isel(time=0), buffer_size=buffer_size, region_def=region_def
    )

    # make sure the correct regions are there
    region_codes = list(subdomains.keys())
    assert sorted(region_codes) == [1, 2, 3, 4, 5, 6, 7, 8]

    # make sure the subdomain bounds are as expected
    assert subdomains[1] == (-180, 46.0, -101.0, 85.0)
    assert subdomains[2] == (-109.0, 46.0, -46.0, 89.0)

    # make sure the regions are consistent between both outputs
    mask_regions = np.unique(mask.values)
    mask_regions = mask_regions[np.isfinite(mask_regions)]
    for r in mask_regions:
        assert r in region_codes

    # make sure the output domain is as expected
    assert mask.lat.min().values == ds.lat.min().values
    assert mask.lat.max().values == ds.lat.max().values
    assert mask.lon.min().values == ds.lon.min().values
    assert mask.lon.max().values == ds.lon.max().values


def test_combine_outputs(example_3d_data_us_domain):
    ds = example_3d_data_us_domain

    # get mask using test data
    region_def = 'ar6'
    buffer_size = 4
    subdomains, mask = generate_subdomains(
        ex_output_grid=ds.isel(time=0), buffer_size=buffer_size, region_def=region_def
    )

    # make mock output data
    ds_dict = {}
    for k, bounds in subdomains.items():
        # for region k, the values in the mock dataset is k
        min_lon, min_lat, max_lon, max_lat = bounds
        temp = ds.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
        for v in ds.data_vars:
            temp[v] = xr.DataArray(k, dims=temp[v].dims, coords=temp[v].coords)
        ds_dict[k] = temp

    # test that the correct combined output is present
    out = combine_outputs(ds_dict, mask)
    for k, v in itertools.product(subdomains.keys(), ds.data_vars):
        unique_val = np.unique(out[v].where(mask == k).values)
        unique_val = unique_val[np.isfinite(unique_val)]
        # ensure that there is only one value for the region
        assert len(unique_val) == 1
        # ensure that for region k, the value we get is k
        assert unique_val[0] == k


def test_region_28():
    # region 28 of ar6 spans across the -180 longitude line, thus making the min/max longitude hard to determine
    # we hard coded the correct boundary in the code and testing it here
    ds = xr.DataArray(
        1,
        dims=['lat', 'lon', 'time'],
        coords={'lat': [66, 67], 'lon': [50, 51], 'time': np.arange(5)},
    )
    buffer_size = 4
    subdomains, mask = generate_subdomains(
        ex_output_grid=ds.isel(time=0), buffer_size=buffer_size, region_def='ar6'
    )

    assert subdomains[28] == (36.0, 61.0, 180, 86.0)
