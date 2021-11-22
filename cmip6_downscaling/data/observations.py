import os

os.environ["PREFECT__FLOWS__CHECKPOINTING"] = "True"
import intake
import numpy as np
import xarray as xr
import zarr
from xarray_schema import DataArraySchema

from cmip6_downscaling.workflows.utils import load_paths, regrid_dataset

connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

schema_map_chunks = DataArraySchema(chunks={'lat': -1, 'lon': -1})
schema_timeseries_chunks = DataArraySchema(chunks={'time': -1})
variable_name_dict = {
    "tasmax": "air_temperature_at_2_metres_1hour_Maximum",
    "tasmin": "air_temperature_at_2_metres_1hour_Minimum",
    "pr": "precipitation_amount_1hour_Accumulation",
}


def get_store(bucket, prefix, account_key=None):
    """helper function to create a zarr store"""

    if account_key is None:
        account_key = os.environ.get("AccountKey", None)

    store = zarr.storage.ABSStore(
        bucket, prefix=prefix, account_name="cmip6downscaling", account_key=account_key
    )
    return store


def open_era5(var: str, start_year: str, end_year: str) -> xr.Dataset:
    """Open ERA5 daily data for a single variable for period 1979-2021

    Parameters
    ----------
    var : str
        The variable you want to grab from the ERA5 dataset.

    start_year : str
        The first year of the time period you want to grab from ERA5 dataset.

    end_year : str
        The last year of the time period you want to grab from ERA5 dataset.

    Returns
    -------
    xarray.Dataset
        A daily dataset for one variable.
    """
    stores = [
        f'https://cmip6downscaling.blob.core.windows.net/cmip6/ERA5_daily/{y}'
        for y in range(int(start_year), int(end_year) + 1)
    ]
    ds = xr.open_mfdataset(stores, engine='zarr', consolidated=True)
    return ds[[var]]


def get_spatial_anomolies(
    coarse_obs, fine_obs_rechunked_path, variable, connection_string
) -> xr.Dataset:
    """Calculate the seasonal cycle (12 timesteps) spatial anomaly associated
    with aggregating the fine_obs to a given coarsened scale and then reinterpolating
    it back to the original spatial resolution. The outputs of this function are
    dependent on three parameters:
    * a grid (as opposed to a specific GCM since some GCMs run on the same grid)
    * the time period which fine_obs (and by construct coarse_obs) cover
    * the variable

    Parameters
    ----------
    coarse_obs : xr.Dataset
        Coarsened to a GCM resolution. Chunked along time.
    fine_obs_rechunked_path : xr.Dataset
        Original observationa spatial resolution. Chunked along time.
    variable: str
        The variable included in the dataset.

    Returns
    -------
    seasonal_cycle_spatial_anomolies : xr.Dataset
        Spatial anomaly for each month (i.e. of shape (nlat, nlon, 12))
    """
    # interpolate coarse_obs back to the original scale
    [fine_obs_rechunked] = load_paths([fine_obs_rechunked_path])

    obs_interpolated = regrid_dataset(
        coarse_obs,
        fine_obs_rechunked.isel(time=0),
        variable=variable,
        connection_string=connection_string,
    )
    # use rechunked fine_obs from coarsening step above because that is in map chunks so it
    # will play nice with the interpolated obs

    fine_obs_rechunked = schema_map_chunks.validate(fine_obs_rechunked)

    # calculate difference between interpolated obs and the original obs
    spatial_anomolies = obs_interpolated - fine_obs_rechunked

    # calculate seasonal cycle (12 time points)
    seasonal_cycle_spatial_anomolies = spatial_anomolies.groupby("time.month").mean()
    return seasonal_cycle_spatial_anomolies
