import os
from typing import Tuple

import fsspec
import xarray as xr
from skdownscale.pointwise_models import PointWiseDownscaler
from skdownscale.pointwise_models.bcsd import BcsdPrecipitation, BcsdTemperature

from cmip6_downscaling.constants import ABSOLUTE_VARS, RELATIVE_VARS
from cmip6_downscaling.data.cmip import load_cmip
from cmip6_downscaling.data.observations import open_era5
from cmip6_downscaling.workflows.utils import (
    delete_chunks_encoding,
    rechunk_zarr_array,
    regrid_dataset,
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


def return_obs(train_period_start: str, train_period_end: str, variable: str) -> xr.Dataset:
    """Loads ERA5 observation data for given time bounds and variable

    Parameters
    ----------
    train_period_start : str
        Starting time bounds
    train_period_end : str
        Ending time bounds
    variable : str
        ERA5 variable. ex: 'tasmax'

    Returns
    -------
    xr.Dataset
        Loaded xarray dataset of ERA5 observation data. Chunked in time: 365
    """
    obs_ds = open_era5(variable, start_year=train_period_start, end_year=train_period_end)
    # Chunking the observation dataset by 'time':365 fixes irregular zarr chunking issues caused by leap-years.
    obs_ds = obs_ds.chunk({'time': 365})
    return obs_ds


def get_coarse_obs(obs_ds: xr.Dataset, variable: str) -> xr.Dataset:
    """Regrids the observation dataset to match the GCM resolution

    Parameters
    ----------
    obs_ds : xr.Dataset
        Observation dataset
    variable : str
        Input variable. ex. 'tasmax'
    connection_string : str
        Azure storage connection string.

    Returns
    -------
    xr.Dataset
        observation dataset at coarse resolution
    """
    # Load single slice of target cmip6 dataset for target grid dimensions
    gcm_one_slice = load_cmip(return_type='xr', variable_ids=[variable]).isel(time=0)

    # rechunk and regrid observation dataset to target gcm resolution
    coarse_obs_ds, fine_obs_rechunked_path = regrid_dataset(
        ds=obs_ds,
        ds_path=None,
        target_grid_ds=gcm_one_slice,
        variable=variable,
        connection_string=connection_string,
    )
    return coarse_obs_ds


def get_spatial_anomalies(
    coarse_obs: xr.Dataset,
    obs_ds: xr.Dataset,
    gcm: str,
    scenario: str,
    train_period_start: str,
    train_period_end: str,
    predict_period_start: str,
    predict_period_end: str,
    variable: str,
) -> xr.Dataset:

    """Returns spatial anomalies
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
    obs_ds : xr.Dataset
        Input observation dataset.
    gcm : str
        Input GCM
    scenario: str
        Input GCM scenario
    train_period_start : str
        Date for training period start (e.g. '1985')
    train_period_end : str
        Date for training period end (e.g. '2015')
    predict_period_start : str
        Date for prediction period start (e.g. '2090')
    predict_period_end : str
        Date for prediction period end (e.g. '2090')
    variable: str
        The variable included in the dataset.

    Returns
    -------
    seasonal_cycle_spatial_anomalies : xr.Dataset
        Spatial anomaly for each month (i.e. of shape (nlat, nlon, 12))
    """
    # Regrid coarse observation dataset
    obs_interpolated, _ = regrid_dataset(
        ds=coarse_obs,
        ds_path=None,
        target_grid_ds=obs_ds.isel(time=0),
        variable=variable,
        connection_string=connection_string,
    )
    obs_rechunked, _ = rechunk_zarr_array(
        obs_ds,
        None,
        chunk_dims=('time',),
        variable=variable,
        connection_string=connection_string,
        max_mem='1GB',
    )
    spatial_anomalies = obs_interpolated - obs_rechunked
    seasonal_cycle_spatial_anomalies = spatial_anomalies.groupby("time.month").mean()
    return seasonal_cycle_spatial_anomalies


def return_y_full_time(
    coarse_obs_ds: xr.Dataset,
    gcm: str,
    scenario: str,
    train_period_start: str,
    train_period_end: str,
    predict_period_start: str,
    predict_period_end: str,
    variable: str,
) -> xr.Dataset:
    """Return y full time rechunked dataset from coarse observation dataset

    Parameters
    ----------
    coarse_obs_ds : xr.Dataset
        Input coarse observation dataset
    gcm : str
        Input GCM
    scenario: str
        Input GCM scenario
    train_period_start : str
        Date for training period start (e.g. '1985')
    train_period_end : str
        Date for training period end (e.g. '2015')
    predict_period_start : str
        Date for prediction period start (e.g. '2090')
    predict_period_end : str
        Date for prediction period end (e.g. '2090')
    variable: str
        The variable included in the dataset.

    Returns
    -------
    xr.Dataset
        y_full_time rechunked dataset
    """
    y_full_time_ds, y_full_time_path = rechunk_zarr_array(
        coarse_obs_ds,
        None,
        chunk_dims=('lat', 'lon'),
        variable=variable,
        connection_string=connection_string,
        max_mem='1GB',
    )
    return y_full_time_ds


def return_x_train_full_time(
    y_full_time_ds: xr.Dataset,
    gcm: str,
    scenario: str,
    train_period_start: str,
    train_period_end: str,
    predict_period_start: str,
    predict_period_end: str,
    variable: str,
) -> xr.Dataset:
    """Returns x training rechunked dataset in full time.

    Parameters
    ----------
    y_full_time_ds : xr.Dataset
        Output y rechunked dataset in full_time
    gcm : str
        Input GCM
    scenario: str
        Input GCM scenario
    train_period_start : str
        Date for training period start (e.g. '1985')
    train_period_end : str
        Date for training period end (e.g. '2015')
    predict_period_start : str
        Date for prediction period start (e.g. '2090')
    predict_period_end : str
        Date for prediction period end (e.g. '2090')
    variable: str
        The variable included in the dataset.

    Returns
    -------
    xr.Dataset
        x_train rechunked dataset in full time.
    """
    X_train = load_cmip(source_ids=gcm, variable_ids=[variable], return_type='xr').sel(
        time=slice(train_period_start, train_period_end)
    )
    X_train['time'] = y_full_time_ds.time.values

    X_train_full_time_ds, X_train_full_time_path = rechunk_zarr_array(
        X_train,
        zarr_array_location=None,
        variable=variable,
        chunk_dims=('lat', 'lon'),
        connection_string=connection_string,
        max_mem='1GB',
    )
    return X_train_full_time_ds


def return_x_predict_rechunked(
    X_train_rechunked_ds: xr.Dataset,
    gcm: str,
    scenario: str,
    train_period_start: str,
    train_period_end: str,
    predict_period_start: str,
    predict_period_end: str,
    variable: str,
) -> xr.Dataset:
    """Return x training rechunked dataset. Chunks are matched to chunks of X train. In the current use case, this means in full_time.

    Parameters
    ----------
    X_train_rechunked_ds : xr.Dataset
        Input x training rechunked dataset
    gcm : str
        Input GCM
    scenario: str
        Input GCM scenario
    train_period_start : str
        Date for training period start (e.g. '1985')
    train_period_end : str
        Date for training period end (e.g. '2015')
    predict_period_start : str
        Date for prediction period start (e.g. '2090')
    predict_period_end : str
        Date for prediction period end (e.g. '2090')
    variable: str
        The variable included in the dataset.

    Returns
    -------
    xr.Dataset
        x prediction rechunked dataset
    """
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


def fit_and_predict(
    x_train_rechunked_ds: xr.Dataset,
    y_rechunked_ds: xr.Dataset,
    x_predict_rechunked_ds: xr.Dataset,
    gcm: str,
    scenario: str,
    train_period_start: str,
    train_period_end: str,
    predict_period_start: str,
    predict_period_end: str,
    variable: str,
    dim: str = "time",
) -> xr.Dataset:
    """Fit bcsd model on prepared CMIP data with obs at corresponding spatial scale.
    Then predict for a set of CMIP data (likely future).

    Parameters
    ----------
    x_train_rechunked_ds : xr.Dataset
        GCM training dataset chunked along space
    y_rechunked_ds : xr.Dataset
        Obs training dataset chunked along space
    x_predict_rechunked_ds : xr.Dataset
        GCM prediction dataset chunked along space.
    gcm : str
        Input GCM
    scenario: str
        Input GCM scenario
    train_period_start : str
        Date for training period start (e.g. '1985')
    train_period_end : str
        Date for training period end (e.g. '2015')
    predict_period_start : str
        Date for prediction period start (e.g. '2090')
    predict_period_end : str
        Date for prediction period end (e.g. '2090')
    variable: str
        The variable included in the dataset.
    dim : str, optional
        dimension on which you want to do the modelling, by default "time"

    Returns
    -------
    bias_corrected_ds : xr.Dataset
        Bias-corrected dataset
    """
    if variable in ABSOLUTE_VARS:
        bcsd_model = BcsdTemperature(return_anoms=False)
    elif variable in RELATIVE_VARS:
        bcsd_model = BcsdPrecipitation(return_anoms=False)
    pointwise_model = PointWiseDownscaler(model=bcsd_model, dim=dim)
    pointwise_model.fit(x_train_rechunked_ds[variable], y_rechunked_ds[variable])
    bias_corrected_da = pointwise_model.predict(x_predict_rechunked_ds[variable])
    bias_corrected_ds = bias_corrected_da.to_dataset(name=variable)
    return bias_corrected_ds


def postprocess_bcsd(
    bias_corrected_ds: xr.Dataset,
    spatial_anomalies_ds: xr.Dataset,
    gcm,
    scenario,
    train_period_start,
    train_period_end,
    predict_period_start,
    predict_period_end,
    variable,
) -> xr.Dataset:
    """Downscale the bias-corrected data by interpolating and then
    adding the spatial anomalies back in.

    Parameters
    ----------
    bias_corrected_ds : xr.Dataset
        bias-corrected dataset
    spatial_anomalies_ds : xr.Dataset
        spatial anomalies dataset
    gcm : str
        Input GCM
    scenario: str
        Input GCM scenario
    train_period_start : str
        Date for training period start (e.g. '1985')
    train_period_end : str
        Date for training period end (e.g. '2015')
    predict_period_start : str
        Date for prediction period start (e.g. '2090')
    predict_period_end : str
        Date for prediction period end (e.g. '2090')
    variable: str
        The variable included in the dataset.
    Returns
    -------
    bcsd_results_ds : xr.Dataset
        Final BCSD dataset
    """

    y_predict_fine, _ = regrid_dataset(
        ds=bias_corrected_ds,
        ds_path=None,
        target_grid_ds=spatial_anomalies_ds,
        variable=variable,
        connection_string=connection_string,
    )
    print(y_predict_fine)
    bcsd_results_ds = y_predict_fine.groupby("time.month") + spatial_anomalies_ds
    delete_chunks_encoding(bcsd_results_ds)
    bcsd_results_ds = bcsd_results_ds.chunk({'time': 30})
    return bcsd_results_ds
