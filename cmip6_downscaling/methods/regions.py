from typing import Any, Dict, Tuple, Union

import numpy as np
import regionmask
import xarray as xr
from shapely.geometry import Polygon


def generate_subdomains(
    ex_output_grid: Union[xr.Dataset, xr.DataArray],
    buffer_size: Union[float, int],
    region_def: str = 'ar6',
) -> Tuple[Dict[Union[int, float], Any], xr.DataArray]:
    """
    Given an example output grid, determine all subdomains that need to be process in order to generate the final output.
    Outputs the list of bounding boxes for each subdomain considering the buffer size, as well as a mask in the resolution of the example output grid specifying
    which subdomain's value to use for each grid cell.

    Parameters
    ----------
    ex_output_grid : xarray dataarray or dataset. example output grid definition. both the bounding box and resolution in lat/lon directions will be used.
    buffer_size    : int/float. for each subdomain, how much buffer area to run.
    region_def     : str. whether to use SREX region definition or ar6 region definition to define subdomain.
                     see the docs https://regionmask.readthedocs.io/en/stable/defined_scientific.html for more details.

    Returns
    -------
    subdomains : dictionary mapping subdomain code to bounding boxes ([min_lon, min_lat, max_lon, max_lat]) for each subdomain
    mask       : mask of which subdomain code to use for each grid cell
    """
    if region_def == 'ar6':
        regions = regionmask.defined_regions.ar6.land
    elif region_def == 'srex':
        regions = regionmask.defined_regions.srex
    else:
        raise NotImplementedError('region_def must be eitehr ar6 or srex')

    mask = regions.mask(ex_output_grid)
    region_codes = np.unique(mask.values)
    region_codes = region_codes[~np.isnan(region_codes)]

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
            subdomains[n] = (min_lon, min_lat, max_lon, max_lat)

    return subdomains, mask


def combine_outputs(
    ds_dict: Dict[Union[float, int], xr.Dataset],
    mask: xr.DataArray,
) -> xr.Dataset:
    """
    Combines values in ds_dict according to mask. Mask should be a 2D dataarray with lat/lon as the dimensions. The values in mask should
    correspond to the keys in ds_dict.

    Parameters
    ----------
    ds_dict : dictionary mapping subdomain code to output
    mask    : mask of which subdomain code to use for each grid cell (2D, lat/lon)

    Returns
    --------
    ds      : the combined output where values come from the respective ds in ds_dict according to mask
    """
    region_codes_available = list(ds_dict.keys())
    region_codes = np.unique(mask.values)
    region_codes = region_codes[~np.isnan(region_codes)]
    for code in region_codes:
        assert code in region_codes_available

    out = xr.Dataset()
    template = ds_dict[region_codes_available[0]]
    for v in template.data_vars:
        out[v] = xr.DataArray(
            np.nan,
            dims=template.dims,
            coords={'time': template.time, 'lat': mask.lat, 'lon': mask.lon},
        )
        for code in region_codes:
            downscale_output = ds_dict[code][v]
            downscale_output = downscale_output.reindex({'lat': mask.lat, 'lon': mask.lon})
            out[v] = xr.where(mask == code, downscale_output, out[v])

    return out
