from typing import List, Optional, Union

import xarray as xr

from cmip6_downscaling.workflows.paths import make_rechunked_obs_path
from cmip6_downscaling.workflows.utils import rechunk_zarr_array_with_caching

from . import cat

variable_name_dict = {
    "tasmax": "air_temperature_at_2_metres_1hour_Maximum",
    "tasmin": "air_temperature_at_2_metres_1hour_Minimum",
    "pr": "precipitation_amount_1hour_Accumulation",
}


def open_era5(variables: Union[str, List[str]], time_period=slice) -> xr.Dataset:
    """Open ERA5 daily data for one or more variables for period 1979-2021

    Parameters
    ----------
    variables : str or list of string
        The variable(s) you want to grab from the ERA5 dataset.
    time_period: slice
        Start and end year slice. Ex: slice('2020','2020')

    Returns
    -------
    xarray.Dataset
        A daily dataset for one variable.
    """
    if isinstance(variables, str):
        variables = [variables]

    years = range(int(time_period.start), int(time_period.stop) + 1)

    ds = xr.concat([cat.era5(year=year).to_dask()[variables] for year in years], dim='time')

    return ds


def get_obs(
    obs: str,
    train_period_start: str,
    train_period_end: str,
    variables: Union[str, List[str]],
    chunking_approach: Optional[str] = None,
    cache_within_rechunk: Optional[bool] = True,
) -> xr.Dataset:
    if obs == 'ERA5':
        ds_obs = open_era5(
            variables=variables, start_year=train_period_start, end_year=train_period_end
        )
    else:
        raise NotImplementedError('only ERA5 is available as observation dataset right now')

    if chunking_approach is None:
        return ds_obs

    if cache_within_rechunk:
        path_dict = {
            'obs': obs,
            'train_period_start': train_period_start,
            'train_period_end': train_period_end,
            'variables': variables,
        }
        rechunked_path = make_rechunked_obs_path(
            chunking_approach=chunking_approach,
            **path_dict,
        )
    else:
        rechunked_path = None
    ds_obs_rechunked = rechunk_zarr_array_with_caching(
        zarr_array=ds_obs,
        chunking_approach=chunking_approach,
        output_path=rechunked_path,
    )

    return ds_obs_rechunked
