import os

import fsspec
import xarray as xr
import zarr
from skdownscale.pointwise_models import BcAbsolute, PointWiseDownscaler

from cmip6_downscaling.data.cmip import load_cmip
from cmip6_downscaling.data.observations import get_spatial_anomolies, load_obs
from cmip6_downscaling.workflows.utils import (
    calc_auspicious_chunks_dict,
    delete_chunks_encoding,
    load_paths,
    rechunk_zarr_array,
    regrid_dataset,
    write_dataset,
)

connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

account_key = os.environ.get("account_key")

# converts cmip standard names to ERA5 names
# can be deleted once new ERA5 dataset complete
variable_name_dict = {
    "tasmax": "air_temperature_at_2_metres_1hour_Maximum",
    "tasmin": "air_temperature_at_2_metres_1hour_Minimum",
    "pr": "precipitation_amount_1hour_Accumulation",
}

# TODO: add templates for paths, maybe using prefect context


def preprocess_bcsd(
    gcm,
    obs_id,
    train_period_start,
    train_period_end,
    variable,
    coarse_obs_path,
    spatial_anomolies_path,
    connection_string,
    rerun=True,
):

    """
    take experiment id and return the gcm, obs at the coarse scale to match that
    gcm's grid, and the spatial anomolies, all chunked in a performant way
    create checkpoint when this has run
    TODO: do we want to assign this to a class? write out? probably
    """
    # check which components you'll need to create in order to
    # run the flow for this experiment
    # inputs_complete = check_preparation(experiment)
    # TODO: this part (working with grid and getting weight file) will just help us not repeat
    # the regridding step by grabbing the grid and checking for an existing weights file, but
    # for now we'll just not try to do that checking and we'll just regrid regardless (optimize later)
    ## grid_name_gcm = get_grid(gcm)
    ## grid_name_obs = get_grid(obs)
    ## get_weight_file_task = task(get_weight_file,
    ##             checkpoint=True,
    ##             result=AzureResult('cmip6'),
    ##             target='grid.zarr')
    ## weight_file = get_weight_file_task(grid_name_gcm, grid_name_obs)
    if rerun:
        obs_ds = load_obs(
            obs_id,
            variable,
            time_period=slice(train_period_start, train_period_end))
        # )  # We want it chunked in space. (time=1,lat=-1, lon=-1)
        
        # find a good chunking scheme then chunk the obs appropriately, might be able to
        # delete this once era5 is chunked well
        gcm_one_slice = load_cmip(return_type='xr', variable_ids=[variable]).isel(time=0) # This comes chunked in space (time~600,lat-1,lon-1), which is good.
        coarse_obs = regrid_dataset(obs_ds.to_dataset(name=variable), 
                                        gcm_one_slice, 
                                        variable='tasmax', 
                                        connection_string=connection_string)
        # TODO : make function call with parameters as opposed to dataset for caching
        # save the coarse obs because it might be used by another gcm

        write_dataset(coarse_obs, coarse_obs_path)

        spatial_anomolies = get_spatial_anomolies(coarse_obs, rechunked_obs, variable, connection_string)
        write_dataset(spatial_anomolies, spatial_anomolies_path)

        return coarse_obs_path, spatial_anomolies_path


def prep_bcsd_inputs(
    coarse_obs_path,
    gcm,
    obs_id,
    train_period_start,
    train_period_end,
    predict_period_start,
    predict_period_end,
    variable,
):
    """[summary]

    Parameters
    ----------
    gcm : [type]
        [description]
    obs_id : [type]
        [description]
    train_period_start : [type]
        [description]
    train_period_end : [type]
        [description]
    predict_period_start : [type]
        [description]
    predict_period_end : [type]
        [description]
    variable : [type]
        [description]
    out_bucket : [type]
        [description]
    domain : [type]
        [description]

    Returns
    -------
    xarray datasets
        Chunked in lat=10,lon=10,time=-1
    """
    # load in coarse obs as xarray ds to get chunk sizes
    [y] = load_paths([coarse_obs_path])
    # TODO: add xarray_schema test here for chunking - if the chunking fails then try rechunking
    # if passes skip
    y_rechunked, y_rechunked_path = rechunk_zarr_array(
        y,
        chunk_dims=('lat', 'lon'),
        variable=variable,
        connection_string=connection_string,
        max_mem='1GB',
    )

    X_train = load_cmip(return_type='xr').sel(time=slice(train_period_start, train_period_end))
    X_train['time'] = y.time.values

    X_train_rechunked, X_train_rechunked_path = rechunk_zarr_array(
        X_train,
        variable=variable,
        chunk_dims=('lat', 'lon'),
        connection_string=connection_string,
        max_mem='1GB',
    )

    X_predict = load_cmip(
        activity_ids=['ScenarioMIP'], experiment_ids=['ssp370'], return_type='xr'
    ).sel(time=slice(predict_period_start, predict_period_end))

    X_predict_rechunked, X_predict_rechunked_path = rechunk_zarr_array(
        X_predict,
        variable=variable,
        chunk_dims=('lat', 'lon'),
        connection_string=connection_string,
        max_mem='1GB',
    )

    return y_rechunked_path, X_train_rechunked_path, X_predict_rechunked_path


def fit_and_predict(
    X_train_path, y_path, X_predict_path, bias_corrected_path, dim="time", variable="tasmax"
):
    """expects X,y, X_predict to be chunked in (lat=10,lon=10,time=-1)

    Parameters
    ----------
    X_train : [type]
        Chunked to preserve full timeseries
    y : [type]
        Chunked to preserve full timeseries
    X_predict : [type]
        Chunked to preserve full timeseries
    dim : str, optional
        [description], by default "time"
    feature_list : list, optional
        [description], by default ["tasmax"]

    Returns
    -------
    xarray dataset
        Unknown chunking.  Probably (lat=10,lon=10,time=-1)
    """
    y, X_train, X_predict = load_paths([y_path, X_train_path, X_predict_path])
    bcsd_model = BcAbsolute(return_anoms=False)
    pointwise_model = PointWiseDownscaler(model=bcsd_model, dim=dim)
    pointwise_model.fit(X_train[variable], y[variable])
    bias_corrected = pointwise_model.predict(X_predict[variable])
    write_dataset(bias_corrected.to_dataset(name=variable), bias_corrected_path)
    return bias_corrected_path


def postprocess_bcsd(
    y_predict_path: str,
    spatial_anomalies_path: str,
    final_out_path: str,
    variable: str,
    connection_string: str,
):
    """[summary]

    Parameters
    ----------
    y_predict : xarray dataset
        (lat=10,lon=10,time=-1)
    gcm : [type]
        [description]
    obs_id : [type]
        [description]
    train_period_start : [type]
        [description]
    train_period_end : [type]
        [description]
    variable : [type]
        [description]
    predict_period_start : [type]
        [description]
    predict_period_end : [type]
        [description]
    """
    # TODO: make spatial_anomalies_store as input from preprocess
    y_predict, spatial_anomalies = load_paths([y_predict_path, spatial_anomalies_path])

    # TODO: test - create sample input, run it through rechunk/regridder and then
    # assert that it looks like i want it to
    y_predict_fine = regrid_dataset(y_predict, spatial_anomalies, variable, connection_string)
    bcsd_results = y_predict_fine.groupby("time.month") + spatial_anomalies
    write_dataset(bcsd_results, final_out_path)

    # TODO: string version
    return final_out_path
