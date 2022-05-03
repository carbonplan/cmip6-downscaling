from __future__ import annotations
from this import d

from typing import Any

import numpy as np
from cmip6_downscaling.methods.common.containers import RunParameters
import regionmask
import pandas as pd
import xarray as xr


def generate_subdomains(
    ex_output_grid: xr.Dataset | xr.DataArray,
    buffer_size: float | int,
    region_def: str = 'ar6',
) -> tuple[dict[int | float, Any], xr.DataArray]:
    """
    Given an example output grid, determine all subdomains that need to be process in order to generate the final output.
    Outputs the list of bounding boxes for each subdomain considering the buffer size, as well as a mask in the resolution of the example output grid specifying
    which subdomain's value to use for each grid cell. Longitudes are in -180 to 180

    Parameters
    ----------
    ex_output_grid : xarray.DataArray or xarray.Dataset
        Example output grid definition. both the bounding box and resolution in lat/lon directions will be used.
    buffer_size : int or float
        Buffer size in unit of degree. for each subdomain, how much extra area to run for each subdomain
    region_def : str
        Subregion definition name. Options are `'ar6'` or `'srex'`. See the docs https://regionmask.readthedocs.io/en/stable/defined_scientific.html for more details.

    Returns
    -------
    subdomains : dict
        Dictionary mapping subdomain code to bounding boxes ([min_lon, min_lat, max_lon, max_lat]) for each subdomain
    mask : xarray.DataArray
        Mask of which subdomain code to use for each grid cell
    """
    if region_def == 'ar6':
        # if we want to include oceans, do regionmask.defined_regions.ar6.all
        regions = regionmask.defined_regions.ar6.land
    elif region_def == 'srex':
        regions = regionmask.defined_regions.srex
    else:
        raise NotImplementedError('region_def must be eitehr ar6 or srex')

    mask = regions.mask(ex_output_grid)
    region_codes = np.unique(mask.values)
    region_codes = region_codes[np.isfinite(region_codes)]

    subdomains = {}
    for n, bound in zip(regions.numbers, regions.bounds):
        if n in region_codes:
            # max(low, min(high, value))
            min_lon = max(min(bound[0] - buffer_size, 180), -180)
            min_lat = max(min(bound[1] - buffer_size, 90), -90)
            max_lon = max(min(bound[2] + buffer_size, 180), -180)
            max_lat = max(min(bound[3] + buffer_size, 90), -90)
            # there is a small region of eastern siberia that is part of region 28 of AR6
            # but since it's difficult to get a subdomain crossing the -180 longitude line, add this region into region 1 instead
            if n == 1 and region_def == 'ar6':
                min_lon = -180
            elif n == 28 and region_def == 'ar6':
                min_lon = 40.0 - buffer_size
            mask = xr.where((mask == 28) & (mask.lon > 180), x=1, y=mask)

            subdomains[n] = (min(min_lon, max_lon), min_lat, max(min_lon, max_lon), max_lat)
    return subdomains, mask


def get_region(id):
    return xr.open_zarr(f'{prefix}/id')


def _merge(mask, varname, times):
    components = pd.unique(mask.values.ravel())
    components = components[~np.isnan(components)]
    
    if components.size > 0:
        merged = (
            xr.merge(
                (
                    xr.open_zarr(f'az://scratch/regions/{ind}.zarr')
                    .where(mask.isin(ind))
                    .sortby(["lon", "lat"])
                    for ind in components
                )
            )
            .reindex_like(mask)
            .sortby("lat", ascending=False)
        )
    else:
        merged = mask.expand_dims(time=times).to_dataset(name=varname)
    return merged.load()


@task(log_stdout=True)
def combine_regions(region_paths: list[str], mask_path: UPath, run_parameters: RunParameters):

    # Todo:
    # - should we pass all paths to each block task where the list of paths can be filtered or should we just pass a template path?
    # - deci

    ds_hash = str_to_hash(str(region_paths) + str(mask_path))
    target = intermediate_dir / 'combine_regions' / ds_hash

    if use_cache and zmetadata_exists(target):
        print(f"found existing target: {target}")
        return target

    mask = xr.open_zarr(mask_path)

    combined_ds = mask.map_blocks(_merge, run_parameters.variable, times=template.time, template=template)

    combined_ds.attrs.update({'title': 'combine_regions'}, **get_cf_global_attrs(version=version))
    combined_ds.to_zarr(target, mode='w')



def combine_outputs(
    ds_dict: dict[float | int, xr.Dataset],
    mask: xr.DataArray,
) -> xr.Dataset:
    """
    Combines values in ds_dict according to mask. Mask should be a 2D dataarray with lat/lon as the dimensions. The values in mask should
    correspond to the keys in ds_dict.

    Parameters
    ----------
    ds_dict : dict
        Dictionary mapping subdomain code to output (subdomain code : xr.Dataset)
    mask : xarray.DataArray
        Mask of which subdomain code to use for each grid cell (2D, lat/lon)

    Returns
    --------
    ds : xarray.Dataset
        The combined output where values come from the respective ds in ds_dict according to mask
    """
    # compare the region codes in ds_dict and in mask to make sure that they match with each other
    region_codes_available = list(ds_dict.keys())
    region_codes = np.unique(mask.values)
    region_codes = region_codes[np.isfinite(region_codes)]
    for code in region_codes:
        assert code in region_codes_available

    # construct the output
    out = xr.Dataset()
    template = ds_dict[region_codes_available[0]]
    for v in template.data_vars:
        # TODO: this might be prohibitive in terms of memory usage
        out[v] = xr.DataArray(
            np.nan,
            dims=template.dims,
            coords={'time': template.time, 'lat': mask.lat, 'lon': mask.lon},
        )
        for code in region_codes:
            # this is the values in ds_dict, which contains the information for each region separately
            single_region_output = ds_dict[code][v]
            single_region_output = single_region_output.reindex({'lat': mask.lat, 'lon': mask.lon})
            out[v] = xr.where(mask == code, single_region_output, out[v])

    return out
