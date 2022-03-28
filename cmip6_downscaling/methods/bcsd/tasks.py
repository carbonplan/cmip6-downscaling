import warnings
from dataclasses import asdict

import xarray as xr
from prefect import task
from skdownscale.pointwise_models import PointWiseDownscaler
from skdownscale.pointwise_models.bcsd import BcsdPrecipitation, BcsdTemperature
from upath import UPath

import cmip6_downscaling

from . import config
from .constants import ABSOLUTE_VARS, RELATIVE_VARS
from .methods.common.containers import RunParameters
from .methods.common.utils import zmetadata_exists

warnings.filterwarnings(
    "ignore",
    "(.*) filesystem path not explicitly implemented. falling back to default implementation. This filesystem may not be tested",
    category=UserWarning,
)


code_version = cmip6_downscaling.__version__
scratch_dir = UPath(config.get("storage.scratch.uri"))
intermediate_dir = UPath(config.get("storage.intermediate.uri")) / cmip6_downscaling.__version__
results_dir = UPath(config.get("storage.results.uri")) / cmip6_downscaling.__version__
use_cache = config.get('run_options.use_cache')


@task
def spatial_anomalies(
    obs_full_time_path: UPath, interpolated_obs_full_time_path: UPath, run_parameters: RunParameters
) -> UPath:
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
    run_parameters : RunParameters
        Prefect run parameters

    Returns
    -------
    UPath
        Path to spatial anomalies dataset.  (shape (nlat, nlon, 12))
    """
    target = (
        intermediate_dir
        / "spatial_anomalies"
        / "{obs}_{model}_{variable}_{latmin}_{latmax}_{lonmin}_{lonmax}_{train_dates[0]}_{train_dates[1]}".format(
            **asdict(run_parameters)
        )
    )
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

    seasonal_cycle_spatial_anomalies.to_zarr(target, mode="w")

    return target


@task(log_stdout=True)
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

    target = intermediate_dir / "fit_and_predict" / run_parameters.run_id
    if use_cache and zmetadata_exists(target):
        print(f"found existing target: {target}")
        return target

    if run_parameters.variable in ABSOLUTE_VARS:
        bcsd_model = BcsdTemperature(return_anoms=False)
    elif run_parameters.variable in RELATIVE_VARS:
        bcsd_model = BcsdPrecipitation(return_anoms=False)
    else:
        raise ValueError('variable not found in ABSOLUTE_VARS OR RELATIVE_VARS.')

    pointwise_model = PointWiseDownscaler(model=bcsd_model, dim="time")

    coarse_obs_full_time_ds = xr.open_zarr(coarse_obs_full_time_path)
    experiment_train_full_time_ds = xr.open_zarr(experiment_train_full_time_path)
    experiment_predict_full_time_ds = xr.open_zarr(experiment_predict_full_time_path)

    pointwise_model.fit(
        experiment_train_full_time_ds[run_parameters.variable],
        coarse_obs_full_time_ds[run_parameters.variable],
    )
    bias_corrected_da = pointwise_model.predict(
        experiment_predict_full_time_ds[run_parameters.variable]
    )

    bias_corrected_ds = bias_corrected_da.astype('float32').to_dataset(name=run_parameters.variable)
    bias_corrected_ds.to_zarr(target, mode='w')
    return target


@task
def postprocess_bcsd(
    bias_corrected_fine_full_time_path: UPath,
    spatial_anomalies_path: UPath,
    run_parameters: RunParameters,
) -> UPath:
    """Downscale the bias-corrected data (fit_and_predict results) by interpolating and then
    adding the spatial anomalies back in.

    Parameters
    ----------
    bias_corrected_fine_full_time_path : UPath
        UPath to output dataset from the fit_and_predict task.
    spatial_anomalies_path : UPath
        UPath to the output of the spatial_anomalies task.
    run_parameters : RunParameters
        Prefect run parameters.

    Returns
    -------
    UPath
        UPath to post-processed dataset.
    """

    target = results_dir / "daily" / run_parameters.run_id
    if use_cache and zmetadata_exists(target):
        print(f"found existing target: {target}")
        return target

    bias_corrected_fine_full_time_ds = xr.open_zarr(bias_corrected_fine_full_time_path)
    spatial_anomalies_ds = xr.open_zarr(spatial_anomalies_path)
    bcsd_results_ds = bias_corrected_fine_full_time_ds.groupby("time.month") + spatial_anomalies_ds
    del bcsd_results_ds['month'].encoding['chunks']

    # The groupby operation above results in inconsistent chunking (# of days per month)
    # This manual chunk step returns the chunking scheme to that of the input dataset
    rechunked_bcsd_results_ds = bcsd_results_ds.chunk(bias_corrected_fine_full_time_ds.chunks)

    rechunked_bcsd_results_ds.to_zarr(target, mode='w')
    return target
