from __future__ import annotations

import warnings
from datetime import timedelta

import dask
import xarray as xr
from carbonplan_data.metadata import get_cf_global_attrs
from prefect import task
from skdownscale.pointwise_models import PointWiseDownscaler
from skdownscale.pointwise_models.bcsd import BcsdPrecipitation, BcsdTemperature
from upath import UPath

from cmip6_downscaling import __version__ as version, config
from cmip6_downscaling.constants import ABSOLUTE_VARS, RELATIVE_VARS
from cmip6_downscaling.methods.bcsd.utils import reconstruct_finescale
from cmip6_downscaling.methods.common.containers import RunParameters
from cmip6_downscaling.methods.common.utils import apply_land_mask, zmetadata_exists
from cmip6_downscaling.utils import str_to_hash

warnings.filterwarnings(
    "ignore",
    "(.*) filesystem path not explicitly implemented. falling back to default implementation. This filesystem may not be tested",
    category=UserWarning,
)


intermediate_dir = UPath(config.get("storage.intermediate.uri")) / version
results_dir = UPath(config.get("storage.results.uri")) / version
use_cache = config.get('run_options.use_cache')


@task(log_stdout=True, max_retries=3, retry_delay=timedelta(seconds=5))
def spatial_anomalies(obs_full_time_path: UPath, interpolated_obs_full_time_path: UPath) -> UPath:
    """Returns spatial anomalies
    Calculate the seasonal cycle (12 timesteps) spatial anomaly associated
    with aggregating the fine_obs to a given coarsened scale and then reinterpolating
    it back to the original spatial resolution. The outputs of this function are
    dependent on three parameters:
    * a grid (as opposed to a specific GCM since some GCMs run on the same grid)
    * the time period which fine_obs (and by construct coarse_obs) cover
    * the variable
    We will save these anomalies to use them in the post-processing. We will add them to the
    spatially-interpolated coarse predictions to add the spatial heterogeneity back in.
    Conceptually, this step figures out, for example, how much colder a finer-scale pixel
    containing Mt. Rainier is compared to the coarse pixel where it exists. By saving those anomalies,
    we can then preserve the fact that "Mt Rainier is x degrees colder than the pixels around it"
    for the prediction. It is important to note that that spatial anomaly is the same for every month of the
    year and the same for every day. So, if in January a finescale pixel was on average 4 degrees colder than
    the neighboring pixel to the west, in every day in the prediction (historic or future) that pixel
    will also be 4 degrees colder.

    Parameters
    ----------
    obs_full_time_path : UPath
        UPath to observation dataset chunked in full_time.
    interpolated_obs_full_time_path : UPath
        UPath to observation dataset interpolated to gcm grid and chunked in full time.

    Returns
    -------
    UPath
        Path to spatial anomalies dataset.  (shape (nlat, nlon, 12))
    """

    ds_hash = str_to_hash(str(obs_full_time_path) + str(interpolated_obs_full_time_path))
    target = intermediate_dir / 'bcsd_spatial_anomalies' / ds_hash

    if use_cache and zmetadata_exists(target):
        print(f"found existing target: {target}")
        return target
    interpolated_obs_full_time_ds = xr.open_zarr(interpolated_obs_full_time_path)
    obs_full_time_ds = xr.open_zarr(obs_full_time_path)

    # calculate the difference between the actual obs (with finer spatial heterogeneity)
    # and the interpolated coarse obs this will be saved and added to the
    # spatially-interpolated coarse predictions to add the spatial heterogeneity back in.

    spatial_anomalies = obs_full_time_ds - interpolated_obs_full_time_ds
    seasonal_cycle_spatial_anomalies = spatial_anomalies.groupby("time.month").mean()
    seasonal_cycle_spatial_anomalies.attrs.update(
        {'title': 'bcsd_spatial_anomalies'}, **get_cf_global_attrs(version=version)
    )
    seasonal_cycle_spatial_anomalies.to_zarr(target, mode='w')

    return target


def _fit_and_predict_wrapper(xtrain, ytrain, xpred, run_parameters, dim='time'):
    """Wrapper for map_blocks for fit and predict task

    Parameters
    ----------
    xtrain : xr.Dataset
        Experiment training dataset
    ytrain : xr.Dataset
        Observation training dataset
    xpred : xr.Dataset
        Experiment prediction dataset
    run_parameters : RunParameters
        Prefect run parameters
    dim : str, optional
        dimension, by default 'time'

    Returns
    -------
    xr.Dataset
        Output bias corrected dataset

    Raises
    ------
    ValueError
        raise ValueError if the given variable is not implimented.
    """

    xpred = xpred.rename({'t2': 'time'})
    if run_parameters.variable in ABSOLUTE_VARS:
        model = BcsdTemperature(return_anoms=False)
    elif run_parameters.variable in RELATIVE_VARS:
        model = BcsdPrecipitation(return_anoms=False)
    else:
        raise ValueError('run_parameters.variable not found in ABSOLUTE_VARS OR RELATIVE_VARS.')

    pointwise_model = PointWiseDownscaler(model=model, dim=dim)

    pointwise_model.fit(xtrain[run_parameters.variable], ytrain[run_parameters.variable])

    bias_corrected_da = pointwise_model.predict(xpred[run_parameters.variable])
    bias_corrected_ds = bias_corrected_da.astype('float32').to_dataset(name=run_parameters.variable)

    return bias_corrected_ds


@task(log_stdout=True, max_retries=3, retry_delay=timedelta(seconds=5))
def fit_and_predict(
    experiment_train_full_time_path: UPath,
    experiment_predict_full_time_path: UPath,
    coarse_obs_full_time_path: UPath,
    run_parameters: RunParameters,
) -> UPath:
    """Fit bcsd model on prepared CMIP data with obs at corresponding spatial scale.
    Then predict for a set of CMIP data (likely future).

    Parameters
    ----------
    experiment_train_full_time_path : UPath
        UPath to experiment training dataset chunked in full time
    experiment_predict_full_time_path : UPath
        UPath to experiment prediction dataset chunked in full time
    coarse_obs_full_time_path : UPath
        UPath to coarse observation dataset chunked in full time
    run_parameters : RunParameters
        Prefect run parameters

    Returns
    -------
    UPath
        UPath to prediction results dataset.

    Raises
    ------
    ValueError
        ValueError checking validity of input variables.
    """

    title = "bcsd_predictions"
    ds_hash = str_to_hash(
        str(experiment_train_full_time_path)
        + str(experiment_predict_full_time_path)
        + str(coarse_obs_full_time_path)
    )

    target = intermediate_dir / 'bcsd_fit_and_predict' / ds_hash

    if use_cache and zmetadata_exists(target):
        print(f"found existing target: {target}")
        return target

    xtrain = xr.open_zarr(coarse_obs_full_time_path)
    ytrain = xr.open_zarr(experiment_train_full_time_path)
    xpred = xr.open_zarr(experiment_predict_full_time_path)

    # Create a template dataset for map blocks
    # This feals a bit fragile.
    template_var = list(xpred.data_vars.keys())[0]
    template = (
        xpred[[template_var]].astype('float32').rename({template_var: run_parameters.variable})
    )

    out = xr.map_blocks(
        _fit_and_predict_wrapper,
        xtrain,
        args=(ytrain, xpred.rename({'time': 't2'}), run_parameters),
        kwargs={'dim': 'time'},
        template=template,
    )
    out = dask.optimize(out)[0]
    out.attrs.update({'title': title}, **get_cf_global_attrs(version=version))

    out.to_zarr(target, mode='w')

    return target


@task(log_stdout=True, max_retries=3, retry_delay=timedelta(seconds=5))
def postprocess_bcsd(
    bias_corrected_fine_full_time_path: UPath, spatial_anomalies_path: UPath
) -> UPath:
    """Downscale the bias-corrected data (fit_and_predict results) by interpolating and then
    adding the spatial anomalies back in.

    Parameters
    ----------
    bias_corrected_fine_full_time_path : UPath
        UPath to output dataset from the fit_and_predict task.
    spatial_anomalies_path : UPath
        UPath to the output of the spatial_anomalies task.

    Returns
    -------
    UPath
        UPath to post-processed dataset.
    """

    title = "bcsd_postprocess"

    ds_hash = str_to_hash(str(bias_corrected_fine_full_time_path) + str(spatial_anomalies_path))
    target = results_dir / title / ds_hash
    print(target)
    if use_cache and zmetadata_exists(target):
        print(f"found existing target: {target}")
        return target

    bias_corrected_fine_full_time_ds = xr.open_zarr(bias_corrected_fine_full_time_path)
    # hint for mapblocks about which month each day corresponds to
    bias_corrected_fine_full_time_ds = bias_corrected_fine_full_time_ds.assign_coords(
        {'month': bias_corrected_fine_full_time_ds['time.month']}
    )
    spatial_anomalies_ds = xr.open_zarr(spatial_anomalies_path)
    # make all spatial anomalies into one chunk so that map_blocks has access to every month.
    # otherwise it will only have access to one chunk and will only grab the last chunk (december)
    # and result in nans in all months except for december
    spatial_anomalies_ds = spatial_anomalies_ds.chunk({'month': -1}).persist()
    bcsd_results_ds = xr.map_blocks(
        reconstruct_finescale,
        bias_corrected_fine_full_time_ds,
        args=[spatial_anomalies_ds],
        template=bias_corrected_fine_full_time_ds,
    )
    # masking out ocean regions
    bcsd_results_ds = apply_land_mask(bcsd_results_ds)

    bcsd_results_ds.attrs.update({'title': title}, **get_cf_global_attrs(version=version))
    bcsd_results_ds.to_zarr(target, mode='w')

    return target
