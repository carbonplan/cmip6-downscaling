import os

os.environ["PREFECT__FLOWS__CHECKPOINTING"] = "True"
import xarray as xr
import zarr

from cmip6_downscaling.workflows.utils import rechunk_zarr_array, regrid_dataset

connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")


def get_store(bucket, prefix, account_key=None):
    """helper function to create a zarr store"""

    if account_key is None:
        account_key = os.environ.get("AccountKey", None)

    store = zarr.storage.ABSStore(
        bucket, prefix=prefix, account_name="cmip6downscaling", account_key=account_key
    )
    return store


def open_era5(var: str):
    """Open ERA5 daily data for a single variable for period 1979-2021

    Parameters
    ----------
    var : str
        The variable you want to grab from the ERA5 dataset.

    Returns
    -------
    xarray.Dataset
        A daily dataset for one variable.
    """
    stores = [
        f'https://cmip6downscaling.blob.core.windows.net/cmip6/ERA5_daily/{y}'
        for y in range(1979, 2021)
    ]
    ds = xr.open_mfdataset(stores, engine='zarr', consolidated=True)
    return ds[[var]]


def load_obs(obs_id: str, variable: str, time_period: slice) -> xr.Dataset:
    """Load a temporal subset of an observational dataset for one variable
    Parameters
    ----------
    obs_id : str
        name of observation dataset. currently only "ERA5" is supported
    variable : str
        name of variable following CMIP conventions (e.g. tasmax, tasmin, pr)
    time_period : slice
        time period you want to subset. e.g. slice('1985', '2015'). using
        full years is recommended if you are going to be doing the spatial
        anomaly calculations.

    Returns
    -------
    open_era5(variable).sel(time=time_period) : xr.Dataset
        Temporal subset of observational dataset
    """
    ## most of this can be deleted once new ERA5 dataset complete
    ## we'll instead just want to have something like
    ## open_era5(chunking_method='space', variables=['tasmax', 'pr'], time_period=slice(start, end), domain=slice())
    if obs_id == "ERA5":
        return open_era5(variable).sel(time=time_period)


def get_spatial_anomolies(coarse_obs, fine_obs, variable, connection_string) -> xr.Dataset:
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
    fine_obs : xr.Dataset
        Original observationa spatial resolution. Chunked along time.
    variable: str
        The variable included in the dataset.

    Returns
    -------
    seasonal_cycle_spatial_anomolies : xr.Dataset
        Spatial anomaly for each month (i.e. of shape (nlat, nlon, 12))
    """
    # interpolate coarse_obs back to the original scale
    obs_interpolated = regrid_dataset(
        coarse_obs, fine_obs.isel(time=0), variable=variable, connection_string=connection_string
    )
    # get fine_obs into map chunks so it plays nice with the interpolated obs
    fine_obs_rechunked, fine_obs_rechunked_path = rechunk_zarr_array(
        fine_obs, connection_string, variable, chunk_dims=('time',), max_mem='1GB'
    )

    # calculate difference between interpolated obs and the original obs
    spatial_anomolies = obs_interpolated - fine_obs_rechunked

    # calculate seasonal cycle (12 time points)
    seasonal_cycle_spatial_anomolies = spatial_anomolies.groupby("time.month").mean()
    return seasonal_cycle_spatial_anomolies
