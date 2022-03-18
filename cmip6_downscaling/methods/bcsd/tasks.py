import warnings
from dataclasses import asdict

import xarray as xr
from prefect import task
from skdownscale.pointwise_models import PointWiseDownscaler
from skdownscale.pointwise_models.bcsd import BcsdPrecipitation, BcsdTemperature
from upath import UPath

from cmip6_downscaling import config
from cmip6_downscaling.constants import ABSOLUTE_VARS, RELATIVE_VARS
from cmip6_downscaling.methods.common.containers import RunParameters

warnings.filterwarnings(
    "ignore",
    "(.*) filesystem path not explicitly implemented. falling back to default implementation. This filesystem may not be tested",
    category=UserWarning,
)


intermediate_dir = UPath(config.get("storage.intermediate.uri"))
results_dir = UPath(config.get("storage.results.uri"))

use_cache = config.get('run_options.use_cache')


@task
def spatial_anomalies(
    obs_full_time_path: UPath, interpolated_obs_full_time_path: UPath, run_parameters: RunParameters
) -> UPath:
    target = (
        intermediate_dir
        / "spatial_anomalies"
        / "{obs}_{model}_{variable}_{latmin}_{latmax}_{lonmin}_{lonmax}_{train_dates[0]}_{train_dates[1]}".format(
            **asdict(run_parameters)
        )
    )
    if use_cache and (target / ".zmetadata").exists():
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


@task
def fit_and_predict(
    experiment_train_full_time_path: UPath,
    experiment_predict_full_time_path: UPath,
    coarse_obs_full_time_path: UPath,
    run_parameters: RunParameters,
) -> UPath:

    target = intermediate_dir / "fit_and_predict" / run_parameters.run_id
    if use_cache and (target / ".zmetadata").exists():
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

    target = results_dir / "interpolate_prediction" / run_parameters.run_id
    if use_cache and (target / ".zmetadata").exists():
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
