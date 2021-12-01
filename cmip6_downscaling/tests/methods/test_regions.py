import numpy as np
import xarray as xr

from cmip6_downscaling.methods.regions import combine_outputs, generate_subdomains


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


def test_generate_subdomains():
    ds = example_3d_data_us_domain()

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
    mask_regions = mask_regions[~np.isnan(mask_regions)]
    for r in mask_regions:
        assert r in region_codes

    # make sure the output domain is as expected
    assert mask.lat.min().values == ds.lat.min().values
    assert mask.lat.max().values == ds.lat.max().values
    assert mask.lon.min().values == ds.lon.min().values
    assert mask.lon.max().values == ds.lon.max().values


def test_combine_outputs():
    ds = example_3d_data_us_domain()

    # get mask
    region_def = 'ar6'
    buffer_size = 4
    subdomains, mask = generate_subdomains(
        ex_output_grid=ds.isel(time=0), buffer_size=buffer_size, region_def=region_def
    )

    # make mock output data
    ds_dict = {}
    for k, bounds in subdomains.items():
        min_lon, min_lat, max_lon, max_lat = bounds
        temp = ds.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
        for v in ds.data_vars:
            temp[v] = xr.DataArray(k, dims=temp[v].dims, coords=temp[v].coords)
        ds_dict[k] = temp

    out = combine_outputs(ds_dict, mask)
    for k in subdomains.keys():
        for v in ds.data_vars:
            unique_val = np.unique(out[v].where(mask == k).values)
            unique_val = unique_val[~np.isnan(unique_val)]
            assert len(unique_val) == 1
            assert unique_val[0] == k
