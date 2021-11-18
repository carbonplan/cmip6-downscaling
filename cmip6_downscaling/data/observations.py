import os

os.environ["PREFECT__FLOWS__CHECKPOINTING"] = "True"

import fsspec
import intake
import numpy as np
import xarray as xr
import xesmf as xe
import zarr
from cmip6_downscaling.workflows.utils import regrid_dataset, rechunk_zarr_array

connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")


def get_store(bucket, prefix, account_key=None):
    """helper function to create a zarr store"""

    if account_key is None:
        account_key = os.environ.get("AccountKey", None)

    store = zarr.storage.ABSStore(
        bucket, prefix=prefix, account_name="cmip6downscaling", account_key=account_key
    )
    return store


def open_era5(var):
    """[summary]

    Parameters
    ----------
    var : str
        The variable you want to grab from the ERA5 dataset.

    Returns
    -------
    xarray.Dataset
        An hourly dataset for one variable.
    """
    stores = [f'https://cmip6downscaling.blob.core.windows.net/cmip6/ERA5_daily/{y}' for y in range(1979, 2021)]
    ds = xr.open_mfdataset(stores, engine='zarr', consolidated=True)
    return ds[[var]]


def load_obs(obs_id, variable, time_period):
    """
    Parameters
    ----------
    obs_id : [type]
        [description]
    variable : [type]
        [description]
    time_period : [type]
        [description]
    domain : [type]
        [description]

    Returns
    -------
    [xarray dataset]
        [Chunked {time:-1,lat=10,lon=10}]
    """
    ## most of this can be deleted once new ERA5 dataset complete
    ## we'll instead just want to have something like
    ## open_era5(chunking_method='space', variables=['tasmax', 'pr'], time_period=slice(start, end), domain=slice())
    if obs_id == "ERA5":
        return open_era5(variable).sel(time=time_period)

def get_spatial_anomolies(coarse_obs, fine_obs, variable, connection_string):
    # check if this has been done, if do the math
    # if it has been done, just read them in
    """[summary]

    Parameters
    ----------
    coarse_obs : [type]
        [chunked in space (lat=-1,lon=-1,time=1)]
    fine_obs : [type]
        [chunked in space (lat=-1,lon=-1,time=1)]

    Returns
    -------
    [type]
        [description]
    """
    print(coarse_obs.chunks)
    obs_interpolated = regrid_dataset(coarse_obs, 
                                        fine_obs.isel(time=0), 
                                        variable=variable,
                                        connection_string=connection_string)
    # get fine_obs into map chunks so it plays nice with the interpolated obs
    fine_obs_rechunked, fine_obs_rechunked_path = rechunk_zarr_array(fine_obs, connection_string, variable, chunk_dims=('time',), max_mem='1GB')
    print(fine_obs.chunks)
    print(fine_obs_rechunked.chunks)
    print(obs_interpolated.chunks)
    spatial_anomolies = obs_interpolated - fine_obs_rechunked
    seasonal_cycle_spatial_anomolies = spatial_anomolies.groupby("time.month").mean()
    return seasonal_cycle_spatial_anomolies
