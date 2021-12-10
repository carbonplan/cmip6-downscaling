import os

os.environ['PREFECT__FLOWS__CHECKPOINTING'] = 'true'
from typing import Tuple

import dask
import fsspec
import xarray
import xarray as xr
from skdownscale.pointwise_models import BcAbsolute, BcRelative, PointWiseDownscaler
from xarray_schema import DataArraySchema

from cmip6_downscaling.constants import ABSOLUTE_VARS, RELATIVE_VARS
from cmip6_downscaling.data.cmip import load_cmip
from cmip6_downscaling.data.observations import open_era5
from cmip6_downscaling.workflows.utils import (
    load_paths,
    rechunk_zarr_array,
    regrid_dataset,
    write_dataset,
)

connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
fs = fsspec.filesystem('az')


def make_flow_paths(
    GCM: str,
    SCENARIO: str,
    TRAIN_PERIOD_START: str,
    TRAIN_PERIOD_END: str,
    PREDICT_PERIOD_START: str,
    PREDICT_PERIOD_END: str,
    VARIABLE: str,
    workdir: str = "az://cmip6",
    outdir: str = "az://cmip6/results",
) -> Tuple[str, str, str, str]:
    """Build the paths where your outputs (both intermediate and final) will go

    Parameters
    ----------
    GCM : str
        From run hyperparameters
    SCENARIO : str
        From run hyperparameters
    TRAIN_PERIOD_START : str
        From run hyperparameters
    TRAIN_PERIOD_END : str
        From run hyperparameters
    PREDICT_PERIOD_START : str
        From run hyperparameters
    PREDICT_PERIOD_END : str
        From run hyperparameters
    VARIABLE : str
        From run hyperparameters
    workdir : str, optional
        Intermediate files for caching (and might be used by other gcms), by default "az://cmip6"
    outdir : str, optional
        Final result space, by default "az://cmip6/results"

    Returns
    -------
    tuple[str, str, str, str]
        From run hyperparameters
    """
    coarse_obs_path = f"{workdir}/intermediates/ERA5_{GCM}_{TRAIN_PERIOD_START}_{TRAIN_PERIOD_END}_{VARIABLE}.zarr"
    spatial_anomalies_path = f"{workdir}/intermediates/anomalies_{GCM}_{TRAIN_PERIOD_START}_{TRAIN_PERIOD_END}_{VARIABLE}.zarr"
    bias_corrected_path = f"{workdir}/intermediates/bc_{SCENARIO}_{GCM}_{TRAIN_PERIOD_START}_{TRAIN_PERIOD_END}_{VARIABLE}.zarr"
    final_out_path = f"{outdir}/bcsd_{SCENARIO}_{GCM}_{PREDICT_PERIOD_START}_{PREDICT_PERIOD_END}_{VARIABLE}.zarr"
    return coarse_obs_path, spatial_anomalies_path, bias_corrected_path, final_out_path


def get_transformed_data():
    ds = xr.tutorial.open_dataset('air_temperature').chunk({'lat': 50, 'lon': 50})
    ds['air'] = ds['air'] - 273.13
    ds['air'].attrs['units'] = 'degC'
    # compute a dask array to confirm that dask tasks are executed on the executor's client
    dask.array.zeros((10000, 10000), chunks=(100, 100)).mean().compute()
    return ds


def return_obs(train_period_start: str, train_period_end: str, variable: str) -> xarray.Dataset:
    obs_ds = open_era5(variable, start_year=train_period_start, end_year=train_period_end)
    obs_ds = obs_ds.chunk({'time': 365})
    return obs_ds


def get_coarse_obs(
    obs_ds: xarray.Dataset,
    variable: str,
    connection_string: str,
) -> xarray.Dataset:

    """Part 1. of preprocess_bcsd with caching. Returns coarse_obs"""

    gcm_one_slice = load_cmip(return_type='xr', variable_ids=[variable]).isel(time=0)
    # # calculate and write out the coarsened version of obs dataset to match the gcm
    # # (will be used in training)
    coarse_obs_ds, fine_obs_rechunked_path = regrid_dataset(
        ds=obs_ds,
        ds_path=None,
        target_grid_ds=gcm_one_slice,
        variable=variable,
        connection_string=connection_string,
    )
    return coarse_obs_ds


def get_spatial_anomalies(
    coarse_obs: xarray.Dataset,
    obs_ds: xarray.Dataset,
    variable: str,
    connection_string: str,
) -> xarray.Dataset:

    """Part 2. of preprocess_bcsd with caching. Returns spatial anomalies
    Calculate the seasonal cycle (12 timesteps) spatial anomaly associated
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
        Original observation spatial resolution. Chunked along time.
    variable: str
        The variable included in the dataset.

    Returns
    -------
    seasonal_cycle_spatial_anomalies : xr.Dataset
        Spatial anomaly for each month (i.e. of shape (nlat, nlon, 12))
    """
    print(obs_ds)
    obs_interpolated, _ = regrid_dataset(
        ds=coarse_obs,
        ds_path=None,
        target_grid_ds=obs_ds.isel(time=0),
        variable=variable,
        connection_string=connection_string,
    )
    print(obs_interpolated)

    # # use rechunked fine_obs from coarsening step above because that is in map chunks so it
    # # will play nice with the interpolated obs
    schema_maps_chunks = DataArraySchema(chunks={"lat": -1, "lon": -1})

    schema_maps_chunks.validate(obs_ds[variable])

    # calculate difference between interpolated obs and the original obs
    spatial_anomalies = obs_interpolated - obs_ds

    # calculate seasonal cycle (12 time points)
    seasonal_cycle_spatial_anomalies = spatial_anomalies.groupby("time.month").mean()
    return seasonal_cycle_spatial_anomalies


def return_y_rechunked(coarse_obs_ds: xarray.Dataset, variable: str) -> xarray.Dataset:
    y_rechunked_ds, y_rechunked_ds_path = rechunk_zarr_array(
        coarse_obs_ds,
        None,
        chunk_dims=('lat', 'lon'),
        variable=variable,
        connection_string=connection_string,
        max_mem='1GB',
    )
    print('y_rechunked ds: ', y_rechunked_ds)

    return y_rechunked_ds


def return_x_train_rechunked(
    gcm: str,
    variable: str,
    train_period_start: str,
    train_period_end: str,
    y_rechunked_ds: xarray.Dataset,
) -> xarray.Dataset:
    X_train = load_cmip(source_ids=gcm, variable_ids=[variable], return_type='xr').sel(
        time=slice(train_period_start, train_period_end)
    )
    X_train['time'] = y_rechunked_ds.time.values

    X_train_rechunked_ds, X_train_rechunked_path = rechunk_zarr_array(
        X_train,
        zarr_array_location=None,
        variable=variable,
        chunk_dims=('lat', 'lon'),
        connection_string=connection_string,
        max_mem='1GB',
    )
    return X_train_rechunked_ds


def return_x_predict_rechunked(
    gcm: str,
    scenario: str,
    variable: str,
    predict_period_start: str,
    predict_period_end: str,
    X_train_rechunked_ds: xarray.Dataset,
) -> xarray.Dataset:
    X_predict = load_cmip(
        source_ids=gcm,
        activity_ids='ScenarioMIP',
        experiment_ids=scenario,
        variable_ids=[variable],
        return_type='xr',
    ).sel(time=slice(predict_period_start, predict_period_end))
    # validate that X_predict spatial chunks match those of X_train since the spatial chunks of predict data need
    # to match when they get passed to the fit_and_predict utility
    # if they are not, rechunk X_predict to match those spatial chunks specifically (don't just pass lat/lon as the chunking dims)
    matching_chunks_dict = {
        variable: {
            'time': X_train_rechunked_ds.chunks['time'][0],
            'lat': X_train_rechunked_ds.chunks['lat'][0],
            'lon': X_train_rechunked_ds.chunks['lon'][0],
        }
    }
    x_predict_rechunked_ds, X_predict_rechunked_path = rechunk_zarr_array(
        X_predict,
        zarr_array_location=None,
        variable=variable,
        chunk_dims=matching_chunks_dict,
        connection_string=connection_string,
        max_mem='1GB',
    )
    return x_predict_rechunked_ds


def preprocess_bcsd(
    gcm: str,
    train_period_start: str,
    train_period_end: str,
    variable: str,
    coarse_obs_path: str,
    spatial_anomalies_path: str,
    connection_string: str,
    rerun: bool = True,
) -> tuple[str, str]:
    """Given GCM and selected obs dataset for a given variable and a given time period, at the coarse scale to match that
    write out the coarsened version of the obs to match that GCM's grid and the spatial anomalies
    for the obs over that time period associated with interpolating the coarse obs back to the original fine
    obs resolution.

    Parameters
    ----------
    gcm : str
        Name of GCM
    train_period_start : str
        Date for training period start (e.g. '1985')
    train_period_end : str
        Date for training period end (e.g. '2015')
    variable : str
        Variable of interest in CMIP conventions (e.g. 'tasmax')
    coarse_obs_path : str
        Path to write coarsened obs zarr dataset (e.g. 'az://bucket/out.zarr')
    spatial_anomalies_path : str
        Path to write coarsened obs zarr dataset (e.g. 'az://bucket/out.zarr')
    connection_string : str
        Connection string to give you read/write access to the out buckets specified above
    rerun : bool, optional
        [description], by default True

    Returns
    -------
    coarse_obs_path, spatial_anomalies_path : str, str
        Paths to where the outputs were written. These will be cached by prefect.
    """
    # TODO: add in functionality to label the obs/spatial anomalies by grid name as opposed to GCM
    #  since it will just help us not repeat the coarsening/spatial anomalies step
    if (not (fs.exists(coarse_obs_path) and fs.exists(spatial_anomalies_path))) or rerun:
        obs_ds = open_era5(variable, start_year=train_period_start, end_year=train_period_end)
        obs_ds = obs_ds.chunk({'time': 365})

        gcm_one_slice = load_cmip(return_type='xr', variable_ids=[variable]).isel(time=0)
        # # calculate and write out the coarsened version of obs dataset to match the gcm
        # # (will be used in training)
        coarse_obs, fine_obs_rechunked_path = regrid_dataset(
            ds=obs_ds,
            ds_path=None,
            target_grid_ds=gcm_one_slice,
            variable=variable,
            connection_string=connection_string,
        )
        write_dataset(coarse_obs, coarse_obs_path)
        # calculate the seasonal cycle (ntime = 12) of spatial anomalies due to interpolating
        # the coarsened obs back to its original resolution and write it out (will be used in postprocess)
        spatial_anomalies = get_spatial_anomalies(
            coarse_obs_path, fine_obs_rechunked_path, variable, connection_string
        )
        write_dataset(spatial_anomalies, spatial_anomalies_path, chunks_dims=('month',))

    return coarse_obs_path, spatial_anomalies_path


def prep_bcsd_inputs(
    coarse_obs_path: str,
    gcm: str,
    scenario: str,
    train_period_start: str,
    train_period_end: str,
    predict_period_start: str,
    predict_period_end: str,
    variable: str,
) -> tuple[str, str, str]:
    """Prepare the inputs to be fed into the training and fitting of the model. Largely
    this converts any datasets chunked in time into ones chunked in space since
    the models are pointwise and need the full timeseries to be performant.

    Parameters
    ----------
    coarse_obs_path : str
        Path to read coarsened obs zarr dataset (e.g. 'az://bucket/out.zarr')
    gcm : str
        Name of GCM
    scenario : str
        Name of the scenario (e.g. ssp370)
    train_period_start : str
        Date for training period start (e.g. '1985')
    train_period_end : str
        Date for training period end (e.g. '2015')
    predict_period_start : str
        Date for prediction period start (e.g. '2070')
    predict_period_end : str
        Date for prediction period end (e.g. '2099')
    variable : str
        Variable of interest in CMIP conventions (e.g. 'tasmax')

    Returns
    -------
    y_rechunked_path: str,
    X_train_rechunked_path: str,
    X_predict_rechunked_path: str
        Paths to to write y, X_train, and X_predict ready for eventual use by model
    """
    # load in coarse obs as xarray ds
    [y] = load_paths([coarse_obs_path])
    # The rechunking will only run if y isn't chunked in the way specified, if it already looks
    # like that, then it will pass
    _, y_rechunked_path = rechunk_zarr_array(
        y,
        coarse_obs_path,
        chunk_dims=('lat', 'lon'),
        variable=variable,
        connection_string=connection_string,
        max_mem='1GB',
    )

    X_train = load_cmip(source_ids=gcm, variable_ids=[variable], return_type='xr').sel(
        time=slice(train_period_start, train_period_end)
    )
    X_train['time'] = y.time.values

    X_train_rechunked, X_train_rechunked_path = rechunk_zarr_array(
        X_train,
        zarr_array_location=None,
        variable=variable,
        chunk_dims=('lat', 'lon'),
        connection_string=connection_string,
        max_mem='1GB',
    )

    X_predict = load_cmip(
        source_ids=gcm,
        activity_ids='ScenarioMIP',
        experiment_ids=scenario,
        variable_ids=[variable],
        return_type='xr',
    ).sel(time=slice(predict_period_start, predict_period_end))
    # validate that X_predict spatial chunks match those of X_train since the spatial chunks of predict data need
    # to match when they get passed to the fit_and_predict utility
    # if they are not, rechunk X_predict to match those spatial chunks specifically (don't just pass lat/lon as the chunking dims)
    matching_chunks_dict = {
        variable: {
            'time': X_train_rechunked.chunks['time'][0],
            'lat': X_train_rechunked.chunks['lat'][0],
            'lon': X_train_rechunked.chunks['lon'][0],
        }
    }
    _, X_predict_rechunked_path = rechunk_zarr_array(
        X_predict,
        zarr_array_location=None,
        variable=variable,
        chunk_dims=matching_chunks_dict,
        connection_string=connection_string,
        max_mem='1GB',
    )

    return y_rechunked_path, X_train_rechunked_path, X_predict_rechunked_path


def fit_and_predict(
    x_train_rechunked_ds: xarray.Dataset,
    y_rechunked_ds: xarray.Dataset,
    x_predict_rechunked_ds: xarray.Dataset,
    variable: str = "tasmax",
    dim: str = "time",
) -> xarray.Dataset:
    """Fit bcsd model on prepared CMIP data with obs at corresponding spatial scale.
    Then predict for a set of CMIP data (likely future).

    Parameters
    ----------
    x_train_rechunked_ds : xarray.Dataset
        GCM training dataset chunked along space
    y_rechunked_ds : xarray.Dataset
        Obs training dataset chunked along space
    x_predict_rechunked_ds : xarray.Dataset
        GCM prediction dataset chunked along space.
    variable : str, optional
        variable you're modelling, by default "tasmax"
    dim : str, optional
        dimension on which you want to do the modelling, by default "time"


    Returns
    -------
    bias_corrected_ds : xarray.Dataset
        Bias-corrected dataset
    """
    if variable in ABSOLUTE_VARS:
        bcsd_model = BcAbsolute(return_anoms=False)
    elif variable in RELATIVE_VARS:
        bcsd_model = BcRelative(return_anoms=False)
    pointwise_model = PointWiseDownscaler(model=bcsd_model, dim=dim)
    pointwise_model.fit(x_train_rechunked_ds[variable], y_rechunked_ds[variable])
    bias_corrected_da = pointwise_model.predict(x_predict_rechunked_ds[variable])
    bias_corrected_ds = bias_corrected_da.to_dataset(name=variable)
    return bias_corrected_ds


def postprocess_bcsd(
    bias_corrected_ds: xarray.Dataset, spatial_anomalies_ds: xarray.Dataset, variable: str
) -> xarray.Dataset:
    """Downscale the bias-corrected data by interpolating and then
    adding the spatial anomalies back in.

    Parameters
    ----------
    bias_corrected_ds : xarray.Dataset
        bias-corrected dataset
    spatial_anomalies_ds : xarray.Dataset
        spatial anomalies dataset
    variable : str
        Input variable

    Returns
    -------
    bcsd_results_ds : xarray.Dataset
        Final BCSD dataset
    """

    y_predict_fine, _ = regrid_dataset(
        ds=bias_corrected_ds,
        ds_path=None,
        target_grid_ds=spatial_anomalies_ds,
        variable=variable,
        connection_string=connection_string,
    )
    bcsd_results_ds = y_predict_fine.groupby("time.month") + spatial_anomalies_ds

    return bcsd_results_ds


def write_bcsd_results(bcsd_results_ds: xarray.Dataset, output_path: str):
    with dask.config.set(scheduler="single-threaded"):
        write_dataset(bcsd_results_ds, output_path)
