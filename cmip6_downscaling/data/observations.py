from __future__ import annotations

import xarray as xr

from . import cat
from .utils import lon_to_180

variable_name_dict = {
    "tasmax": "air_temperature_at_2_metres_1hour_Maximum",
    "tasmin": "air_temperature_at_2_metres_1hour_Minimum",
    "pr": "precipitation_amount_1hour_Accumulation",
}


def open_era5(variables: str | list[str], time_period: slice) -> xr.Dataset:
    """Open ERA5 daily data for one or more variables for period 1979-2021

    Parameters
    ----------
    variables : str or list of string
        The variable(s) you want to grab from the ERA5 dataset.
    time_period : slice
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

    if 'pr' in variables:
        # convert to mm/day - helpful to prevent rounding errors from very tiny numbers
        ds['pr'] *= 86400
        ds['pr'] = ds['pr'].astype('float32')
        ds['pr'].attrs = {
            'least_significant_digit': 4,
            'standard_name': 'precipitation_amount',
            'units': 'mm',
            'long_name': 'Total precipitation',
            'nameECMWF': 'Total precipitation',
            'shortNameECMWF': 'tp',
            'product_type': 'forecast'
    }

    # correct error in raw data attributes for tasmin/tasmax
    if 'tasmin' in variables:
        ds['tasmin'].attrs = {
            'least_significant_digit': 1,
            'standard_name': 'air_temperature',
            'units': 'K',
            'long_name': 'Minimum temperature at 2 metres since previous post-processing',
            'nameECMWF': 'Minimum temperature at 2 metres since previous post-processing',
            'shortNameECMWF': 'mn2t',
            'nameCDM': 'Minimum_temperature_at_2_metres_since_previous_post-processing_surface_1_Hour_2',
            'product_type': 'forecast'
        }
    if 'tasmax' in variables:
        ds['tasmax'].attrs = {
            'least_significant_digit': 1,
            'standard_name': 'air_temperature',
            'units': 'K',
            'long_name': 'Maximum temperature at 2 metres since previous post-processing',
            'nameECMWF': 'Maximum temperature at 2 metres since previous post-processing',
            'shortNameECMWF': 'mx2t',
            'nameCDM': 'Maximum_temperature_at_2_metres_since_previous_post-processing_surface_1_Hour_2',
            'product_type': 'forecast'
        }

    ds = lon_to_180(ds)

    # Reorders latitudes to [-90, 90]
    if ds.lat[0] > ds.lat[-1]:
        ds = ds.reindex({"lat": ds.lat[::-1]})

    return ds
