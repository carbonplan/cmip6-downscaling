import os
from typing import Tuple

import fsspec
from skdownscale.pointwise_models import BcAbsolute, BcRelative, PointWiseDownscaler

from cmip6_downscaling.constants import ABSOLUTE_VARS, RELATIVE_VARS
from cmip6_downscaling.data.cmip import load_cmip
from cmip6_downscaling.data.observations import open_era5
from cmip6_downscaling.workflows.utils import (
    get_spatial_anomalies,
    load_paths,
    rechunk_zarr_array,
    regrid_dataset,
    write_dataset,
)

connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
fs = fsspec.filesystem('az')
# TODO: add templates for paths, maybe using prefect context. Do this once functions ported to prefect world in bcsd_flow.py


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
    X_train_path: str,
    y_path: str,
    X_predict_path: str,
    bias_corrected_path: str,
    dim: str = "time",
    variable: str = "tasmax",
) -> str:
    """Fit bcsd model on prepared CMIP data with obs at corresponding spatial scale.
    Then predict for a set of CMIP data (likely future).

    Parameters
    ----------
    X_train_path : str
        Path to GCM training data chunked along space
    y_path : str
        Path to obs training data chunked along space
    X_predict_path : str
        Path to GCM prediction data chunked along space.
    bias_corrected_path : str
        Path to final bias corrected data (matches dims of X_predict).
    dim : str, optional
        dimension on which you want to do the modelling, by default "time"
    variable : str, optional
        variable you're modelling, by default "tasmax"

    Returns
    -------
    bias_corrected_path : str
        Path to where coarse bias-corrected data is
    """
    y, X_train, X_predict = load_paths([y_path, X_train_path, X_predict_path])

    if variable in ABSOLUTE_VARS:
        bcsd_model = BcAbsolute(return_anoms=False)
    elif variable in RELATIVE_VARS:
        bcsd_model = BcRelative(return_anoms=False)
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
) -> str:
    """Downscale the bias-corrected data by interpolating and then
    adding the spatial anomalies back in.

    Parameters
    ----------
    y_predict_path : str
        location where bias-corrected data is.
    spatial_anomalies_path : str
        location where spatial anomalies are
    final_out_path : str
        location to write final BCSD data
    variable : str
        variable you're working on
    connection_string : str
        Connection string to give you read/write access to the out buckets specified above

    Returns
    -------
    final_out_path : str
        location where final BCSD data was written
    """
    y_predict, spatial_anomalies = load_paths([y_predict_path, spatial_anomalies_path])

    # TODO: test - create sample input, run it through rechunk/regridder and then
    # assert that it looks like i want it to
    y_predict_fine, _ = regrid_dataset(
        ds=y_predict,
        ds_path=y_predict_path,
        target_grid_ds=spatial_anomalies,
        variable=variable,
        connection_string=connection_string,
    )
    bcsd_results = y_predict_fine.groupby("time.month") + spatial_anomalies
    write_dataset(bcsd_results, final_out_path)

    return final_out_path
