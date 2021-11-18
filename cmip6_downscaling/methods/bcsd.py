import os

import fsspec
import intake
import xarray as xr
import xesmf as xe
import zarr
from skdownscale.pointwise_models import BcAbsolute, PointWiseDownscaler

from xarray_schema import DataArraySchema, DatasetSchema
from cmip6_downscaling.data.observations import load_obs, get_coarse_obs, get_spatial_anomolies
from cmip6_downscaling.data.cmip import load_cmip, convert_to_360
from cmip6_downscaling.workflows.utils import rechunk_zarr_array, calc_auspicious_chunks_dict, delete_chunks_encoding, load_paths, write_dataset, regrid_dataset

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
        # TODO: add cleanup to rechunk_zarr_array 
        # TODO: open_subsetted_obs_chunked_in_time('ERA5', time_period, chunks_dict) as task and cached
        rechunked_obs, path_tgt = rechunk_zarr_array(obs_ds, 
                                                    target_chunks, 
                                                    connection_string, 
                                                    max_mem="100MB")
        gcm_ds = load_cmip(source_ids=gcm, variable_ids=[variable])[
            "CMIP.MIROC.MIROC6.historical.day.gn"
        ]  # This comes chunked in space (time~600,lat-1,lon-1), which is good.

        delete_chunks_encoding(gcm_ds)

        gcm_ds_single_time_slice = gcm_ds.isel(
            time=0
        )  # .load() #TODO: check whether we need the load here
        chunks_dict_obs_maps = calc_auspicious_chunks_dict(obs_ds, chunk_dims=('time',)) # you'll need to put comma after the one element tuple
        # we want to pass rechunked obs to both get_coarse_obs and get_spatial_anomalies
        # since they both work in map space instead of time space
        # TODO: currently the caching would be on gcm name as opposed to grid name
        # TODO : make function call with parameters as opposed to dataset for caching
        coarse_obs = get_coarse_obs(
            rechunked_obs,
            gcm_ds_single_time_slice, # write_cache=True, read_cache=True
        )
        # TODO : make function call with parameters as opposed to dataset for caching

        spatial_anomolies = get_spatial_anomolies(coarse_obs, rechunked_obs)

        # save the coarse obs because it might be used by another gcm
        coarse_obs.to_zarr(coarse_obs_store, mode="w", consolidated=True)

        spatial_anomolies.to_zarr(spatial_anomolies_store, mode="w", consolidated=True)
        return coarse_obs_store, spatial_anomolies_store


def prep_bcsd_inputs(
    coarse_obs_path,
    gcm,
    obs_id,
    train_period_start,
    train_period_end,
    predict_period_start,
    predict_period_end,
    variable
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
    y_rechunked, y_rechunked_path = rechunk_zarr_array(y, chunk_dims=('lat', 'lon'), 
                                        variable=variable, 
                                        connection_string=connection_string,
                                        max_mem='1GB')

    X_train = load_cmip(return_type='xr').sel(time=slice(train_period_start, 
                                                                    train_period_end))
    X_train['time'] = y.time.values

    X_train_rechunked, X_train_rechunked_path = rechunk_zarr_array(X_train, 
                                                        variable=variable,
                                                         chunk_dims=('lat', 'lon'), 
                                                         connection_string=connection_string, 
                                                         max_mem='1GB')

    X_predict = load_cmip(activity_ids=['ScenarioMIP'],
                                experiment_ids=['ssp370'],
                             return_type='xr').sel(time=slice(predict_period_start, predict_period_end))
                              
    X_predict_rechunked, X_predict_rechunked_path = rechunk_zarr_array(X_predict, 
                                                        variable=variable,
                                                         chunk_dims=('lat', 'lon'), 
                                                         connection_string=connection_string, 
                                                         max_mem='1GB')

    return y_rechunked_path, X_train_rechunked_path, X_predict_rechunked_path


def fit_and_predict(X_train_path, y_path, X_predict_path, bias_corrected_path, dim="time", variable="tasmax"):
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
     # TODO: change to reading stores
    bias_corrected_store = fsspec.get_mapper(bias_corrected_path, connection_string=connection_string)
    bcsd_model = BcAbsolute(return_anoms=False)
    pointwise_model = PointWiseDownscaler(model=bcsd_model, dim=dim)
    pointwise_model.fit(X_train[variable], y[variable])
    bias_corrected = pointwise_model.predict(X_predict[variable])
    print('here')
    write_dataset(bias_corrected.to_dataset(name=variable), bias_corrected_path)
    # GLOBAL TODO: swith all stores to the strings
    return bias_corrected_path


def postprocess_bcsd(
    y_predict_path : str,
    spatial_anomalies_path : str,
    final_out_path : str,
    variable : str,
    connection_string : str
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
         # TODO: change to reading stores
    # TODO: clean up all the store/path creation
    # TODO: make spatial_anomalies_store as input from preprocess
    y_predict, spatial_anomalies = load_paths([y_predict_path, spatial_anomalies_path])

    # TODO: test - create sample input, run it through rechunk/regridder and then 
    # assert that it looks like i want it to
    y_predict_fine = regrid_dataset(y_predict, spatial_anomalies, variable, connection_string)
    # This lat=-1,lon=-1,time=1) chunking might be best? Check to make sure that is what is returned from regridder.
    # comment
    bcsd_results = y_predict_fine.groupby("time.month") + spatial_anomalies
    # TODO: write_bcsd()
    write_dataset(bcsd_results, final_out_path)

    # TODO: string version
    return final_out_path