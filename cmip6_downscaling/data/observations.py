import os

os.environ["PREFECT__FLOWS__CHECKPOINTING"] = "True"

import fsspec
import intake
import xarray as xr
<<<<<<< HEAD

=======
import xesmf as xe
>>>>>>> origin
import zarr

connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

<<<<<<< HEAD
=======

>>>>>>> origin
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
    print("getting stores")
    col = intake.open_esm_datastore(
        "https://cmip6downscaling.blob.core.windows.net/cmip6/ERA5_catalog.json"
    )
    stores = col.df.zstore
    era5_var = variable_name_dict[var]
    store_list = stores[stores.str.contains(era5_var)].to_list()
<<<<<<< HEAD
    # store_list[:10]
=======
>>>>>>> origin
    ds = xr.open_mfdataset(
        store_list,
        engine="zarr",  # these options set the inputs and how to read them
        consolidated=True,
        parallel=True,  # these options speed up the reading of individual datasets (before they are combined)
        combine="by_coords",  # these options tell xarray how to combine the data
        # data_vars=['air_temperature_at_2_metres_1hour_Maximum']  # these options limit the amount of data that is read to only variables of interest
    ).drop("time1_bounds")
    print("return mfdataset")
    return ds

<<<<<<< HEAD
def load_obs(obs_id, variable, time_period, domain):
    """

=======

def load_obs(obs_id, variable, time_period, domain, agg_func=np.max):
    """
>>>>>>> origin
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
        print("open era5")
        full_obs = open_era5(variable)
        print("resample era5")
        obs = (
            full_obs[variable_name_dict[variable]]
            .sel(time=time_period)
            .resample(time="1D")
<<<<<<< HEAD
            .max()
=======
            .reduce(agg_func)
>>>>>>> origin
            .rename(variable)
            # .load(scheduler="threads")  # GOAL! REMOVE THE `LOAD`!
        )
    return obs

<<<<<<< HEAD
=======

>>>>>>> origin
def get_coarse_obs(
    obs,
    gcm_ds_single_time_slice,
):
    """[summary]

    Parameters
    ----------
    obs : xarray dataset
        chunked in space (lat=-1, lon=-1, time=1)
    gcm_ds_single_time_slice : [type]
        Chunked in space

    Returns
    -------
    [type]
        [Chunked in space (lat=-1, lon=-1, time=1)]
    """
    # TEST TODO: Check that obs is chunked appropriately and throw error if not
    # Like: assert obs.chunks == (lat=-1, lon=-1, time=1) - then eventually we can move the rechunker in as an `else`
    regridder = xe.Regridder(obs, gcm_ds_single_time_slice, "bilinear")

    obs_coarse = regridder(obs)
    return obs_coarse

<<<<<<< HEAD
=======

>>>>>>> origin
def get_spatial_anomolies(coarse_obs, fine_obs):
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
    # check chunks specs & run regridder if needed
    regridder = xe.Regridder(
        coarse_obs, fine_obs.isel(time=0), "bilinear", extrap_method="nearest_s2d"
    )

    obs_interpolated = regridder(coarse_obs)
    spatial_anomolies = obs_interpolated - fine_obs
    seasonal_cycle_spatial_anomolies = spatial_anomolies.groupby("time.month").mean()
<<<<<<< HEAD
    return seasonal_cycle_spatial_anomolies
=======
    return seasonal_cycle_spatial_anomolies
>>>>>>> origin
