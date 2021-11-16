# from cmip6_downscaling.workflows.utils import rechunk_dataset
import os
import random
import string

import fsspec
import intake
import xarray as xr
import xesmf as xe
import zarr
from dask.distributed import Client, LocalCluster
from dask_kubernetes import KubeCluster, make_pod_spec
from prefect import Flow, Parameter, task
from prefect.executors import DaskExecutor
from prefect.run_configs import KubernetesRun
from prefect.storage import Azure
from rechunker import api
from skdownscale.pointwise_models import BcAbsolute, PointWiseDownscaler

from cmip6_downscaling.workflows.utils import rechunk_zarr_array, calc_auspicious_chunks_dict, delete_chunks_encoding
import numpy as np
from xarray_schema import DataArraySchema, DatasetSchema
from cmip6_downscaling.data.observations import load_obs, get_coarse_obs, get_spatial_anomolies
from cmip6_downscaling.data.cmip import load_cmip_dictionary, gcm_munge, convert_to_360


chunks = {"lat": 10, "lon": 10, "time": -1}
connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
test_specs = {
    "domain": {
        "lat": slice(50, 45),
        "lon": slice(convert_to_360(-124.8), convert_to_360(-120.0)),
    }
}  # bounding box of downscaling region

account_key = os.environ.get("account_key")
run_hyperparameters = {
    "FLOW_NAME": "BCSD_testing",  # want to populate with unique string name?
    # "MODEL": "BCSD",
    # "VARIABLES": ["tasmax"],
    # "INTERPOLATION": "bilinear",
    "GCMS": ["MIROC6"],
    # "EXPERIMENT_ID": 'CMIP.MIROC.MIROC6.historical.day.gn',
    # "SCENARIOS": ["ssp370"],
    "TRAIN_PERIOD_START": "1990",
    "TRAIN_PERIOD_END": "1990",
    "PREDICT_PERIOD_START": "2090",
    "PREDICT_PERIOD_END": "2090",
    # "DOMAIN": {'lat': slice(50, 45),
    #             'lon': slice(235.2, 240.0)},
    "VARIABLE": "tasmax",
    # "GRID_NAME": None,
    # "INTERMEDIATE_STORE": "scratch/cmip6",
    # "FINAL_STORE": "cmip6/downscaled",
    # "SAVE_MODEL": False,
    "OBS": "ERA5",
}
flow_name = run_hyperparameters.pop("FLOW_NAME")  # pop it out because if you leave it in the dict
# but don't call it as a parameter it'll complain

# converts cmip standard names to ERA5 names
# can be deleted once new ERA5 dataset complete
variable_name_dict = {
    "tasmax": "air_temperature_at_2_metres_1hour_Maximum",
    "tasmin": "air_temperature_at_2_metres_1hour_Minimum",
    "pr": "precipitation_amount_1hour_Accumulation",
}

@task(  # target="{flow_name}.txt", checkpoint=True,
    # result=LocalResult(dir="~/.prefect"),
    # cache_for=datetime.timedelta(hours=1),
    log_stdout=True
)
def preprocess_bcsd(
    gcm,
    obs_id,
    train_period_start,
    train_period_end,
    variable,
    out_bucket,
    domain,
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
    coarse_obs_store = fsspec.get_mapper(
        f"az://cmip6/intermediates/{obs_id}_{gcm[0]}_{train_period_start}_{train_period_end}_{variable}.zarr",
        connection_string=connection_string,
    )
    spatial_anomolies_store = fsspec.get_mapper(
        f"az://cmip6/intermediates/anomalies_{obs_id}_{gcm[0]}_{train_period_start}_{train_period_end}_{variable}.zarr",
        connection_string=connection_string,
    )
    if rerun:
        # obs_ds = load_obs(
        #     obs_id,
        #     variable,
        #     time_period=slice(train_period_start, train_period_end),
        #     domain=domain
        # )  # We want it chunked in space. (time=1,lat=-1, lon=-1)
        # for testing we'll use the lines below but eventually comment out above lines
        local_obs_path_maps = 'era5_tasmax_1990_maps.zarr'
        obs_ds = xr.open_zarr(local_obs_path_maps)
        # find a good chunking scheme then chunk the obs appropriately, might be able to 
        # delete this once era5 is chunked well
        target_chunks={variable: calc_auspicious_chunks_dict(obs_ds[variable], chunk_dims=('time',)), 
               'time': None, # don't rechunk this array
                'lon': None,
                'lat': None}
        obs_ds = zarr.open_consolidated(local_obs_path_maps, mode='r')
        rechunked_obs, path_tgt = rechunk_zarr_array(obs_ds, 
                                                    target_chunks, 
                                                    connection_string, 
                                                    max_mem="100MB")
        gcm_ds = load_cmip_dictionary(source_ids=gcm, variable_ids=[variable])[
            "CMIP.MIROC.MIROC6.historical.day.gn"
        ]  # This comes chunked in space (time~600,lat-1,lon-1), which is good.

        # Check whether gcm latitudes go from low to high, if so, swap them to match ERA5 which goes from high to low
        # after we create era5 daily processed product we should still leave this in but should switch < to > in if statement
        gcm_ds = gcm_munge(gcm_ds)
        delete_chunks_encoding(gcm_ds)

        gcm_ds_single_time_slice = gcm_ds.isel(
            time=0
        )  # .load() #TODO: check whether we need the load here
        chunks_dict_obs_maps = calc_auspicious_chunks_dict(obs_ds, chunk_dims=('time',)) # you'll need to put comma after the one element tuple
        # we want to pass rechunked obs to both get_coarse_obs and get_spatial_anomalies
        # since they both work in map space instead of time space
        coarse_obs = get_coarse_obs(
            rechunked_obs,
            gcm_ds_single_time_slice, # write_cache=True, read_cache=True
        )
        spatial_anomolies = get_spatial_anomolies(coarse_obs, rechunked_obs)

        # save the coarse obs because it might be used by another gcm
        coarse_obs.to_zarr(coarse_obs_store, mode="w", consolidated=True)

        spatial_anomolies.to_zarr(spatial_anomolies_store, mode="w", consolidated=True)
        return coarse_obs_store, spatial_anomolies_store


@task(log_stdout=True, nout=3)
def prep_bcsd_inputs(
    coarse_obs_store,
    gcm,
    obs_id,
    train_period_start,
    train_period_end,
    predict_period_start,
    predict_period_end,
    variable,
    out_bucket,
    domain,
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
    y = xr.open_zarr(coarse_obs_store, consolidated=True)
    coarse_time_chunks = {"tasmax": calc_auspicious_chunks_dict(coarse_obs.tasmax,chunk_dims=('lat','lon')), 
                      'time': None, # don't rechunk this array
                        'lon': None,
                        'lat': None}
    y_rechunked, path_tgt_y = rechunk_zarr_array(coarse_obs, 
                                                chunks_dict=coarse_time_chunks, 
                                                connection_string=connection_string, 
                                                max_mem='1G')

    X_train = load_cmip_dictionary(return_type='xr').sel(time=slice(train_period_start, 
                                                                    train_period_end))
    X_train = gcm_munge(X_train)

    X_train['time'] = y.time.values
    delete_chunks_encoding(X_train)

    X_train_rechunked, path_tgt_X_train = rechunk_zarr_array(X_train.chunk(coarse_time_chunks['tasmax']), 
                                                         chunks_dict=coarse_time_chunks, connection_string=connection_string, max_mem='1G')

    X_predict = load_cmip_dictionary(activity_ids=['ScenarioMIP'],
                                experiment_ids=['ssp370'],
                             return_type='xr').sel(time=slice(predict_period_start, predict_period_end))
    X_predict = gcm_munge(X_predict)
    delete_chunks_encoding(X_predict)
    X_predict_rechunked, path_tgt_X_predict = rechunk_zarr_array(X_predict.chunk(coarse_time_chunks['tasmax']), 
                                             coarse_time_chunks, connection_string, max_mem="1GB")
    delete_chunks_encoding(X_predict_rechunked)
    # maybe change to stores
    return y_rechunked, X_train_rechunked, X_predict_rechunked

@task(log_stdout=True)
def fit_and_predict(X_train, y, X_predict, dim="time", feature_list=["tasmax"], write=False):
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
    bcsd_model = BcAbsolute(return_anoms=False)
    pointwise_model = PointWiseDownscaler(model=bcsd_model, dim=dim)
    pointwise_model.fit(X_train, y)
    bias_corrected = pointwise_model.predict(X_predict)
    if write:
        bias_corrected_store = fsspec.get_mapper(
        f"az://cmip6/intermediates/bc_{obs_id}_{gcm[0]}_{train_period_start}_{train_period_end}_{variable}.zarr",
        connection_string=connection_string,
        )
        bias_corrected.to_zarr(bias_corrected_store)

    return bias_corrected


@task(log_stdout=True)
def postprocess_bcsd(
    y_predict,
    gcm,
    obs_id,
    train_period_start,
    train_period_end,
    variable,
    predict_period_start,
    predict_period_end,
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
    spatial_anomalies_store = fsspec.get_mapper(
        f"az://cmip6/intermediates/anomalies_{obs_id}_{gcm[0]}_{train_period_start}_{train_period_end}_{variable}.zarr",
        connection_string=connection_string,
    )
    # spatial anomalies is chunked in (lat=-1,lon=-1,time=1)
    spatial_anomalies = xr.open_zarr(spatial_anomalies_store, consolidated=True)
    regridder = xe.Regridder(y_predict, spatial_anomalies, "bilinear", extrap_method="nearest_s2d")
    # Rechunk y_predict to (lat=-1,lon=-1,time=1)
    rechunked_y_predict, rechunked_y_predict_path = rechunk_dataset(
        y_predict,
        chunks_dict={"tasmax": (-1, -1, 1)},
        connection_string=connection_string,
        max_mem="1GB",
    )
    y_predict_fine = regridder(y_predict)
    # This lat=-1,lon=-1,time=1) chunking might be best? Check to make sure that is what is returned from regridder.
    bcsd_results = y_predict_fine.groupby("time.month") + spatial_anomalies
    bcsd_store = fsspec.get_mapper(
        f"az://cmip6/intermediates/bcsd_{obs_id}_{gcm[0]}_{predict_period_start}_{predict_period_end}_{variable}.zarr",
        connection_string=connection_string,
    )
    bcsd_results.to_zarr(bcsd_store, mode="w", consolidated=True)


# # Prefect cloud config settings -----------------------------------------------------------

# run_config = KubernetesRun(
#     cpu_request=2,
#     memory_request="2Gi",
#     image="gcr.io/carbonplan/hub-notebook:7252fc3", #CHANGE ASK JOE FOR NEW IMAGE
#     labels=["az-eu-west"],
#     env=run_hyperparameters
# )
# Prefect Flow -----------------------------------------------------------
# put the experiment_ids outside of this loop?
with Flow(name="bcsd_flow", storage=storage, run_config=run_config, executor=executor) as flow:

    # run preprocess and create dependency/checkpoint to show it's done
    obs = run_hyperparameters["OBS"]  # Parameter("OBS")
    gcm = run_hyperparameters["GCMS"]  # Parameter("GCMS")
    train_period_start = run_hyperparameters[
        "TRAIN_PERIOD_START"
    ]  # Parameter("TRAIN_PERIOD_START")
    train_period_end = run_hyperparameters["TRAIN_PERIOD_END"]  # Parameter("TRAIN_PERIOD_END")
    predict_period_start = run_hyperparameters[
        "PREDICT_PERIOD_START"
    ]  # Parameter("PREDICT_PERIOD_START")
    predict_period_end = run_hyperparameters[
        "PREDICT_PERIOD_END"
    ]  # Parameter("PREDICT_PERIOD_END")
    domain = test_specs["domain"]
    variable = run_hyperparameters["VARIABLE"]  # Parameter("VARIABLE")
    # `preprocess` will create the necessary coarsened input files and write them out
    # then we'll read them below

    coarse_obs = preprocess_bcsd(
        gcm,
        obs_id=obs,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        variable=variable,
        out_bucket="cmip6",
        domain=domain,
        rerun=True,
    )  # can remove this once we have caching working

    X, y, X_predict = prep_bcsd_inputs(
        coarse_obs,
        gcm,
        obs_id=obs,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        predict_period_start=predict_period_start,
        predict_period_end=predict_period_end,
        variable=variable,
        out_bucket="cmip6",
        domain=domain,
    )

    y_predict = fit_and_predict(X, y, X_predict)
    postprocess_bcsd(
        y_predict,
        gcm,
        obs,
        train_period_start,
        train_period_end,
        variable,
        predict_period_start,
        predict_period_end,
    )
flow.run(parameters=run_hyperparameters)
