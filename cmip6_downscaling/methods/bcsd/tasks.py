from dataclasses import asdict

import xarray as xr
from prefect import task
from upath import UPath

from cmip6_downscaling import config
from cmip6_downscaling.methods.common.containers import RunParameters

intermediate_dir = UPath(config.get("storage.intermediate.uri"))

use_cache = config.get('run_options.use_cache')


@task
def coarsen_obs(obs_path: UPath, experiment_path: UPath, run_parameters: RunParameters) -> UPath:
    # Q: for coarsen_obs and interpolate_obs, should we split the task into a separate tasks for regridding and rechunking?
    target = (
        intermediate_dir
        / "coarsen_obs"
        / "{obs}_{model}_{variable}_{latmin}_{latmax}_{lonmin}_{lonmax}_{train_dates[0]}_{train_dates[1]}".format(
            **asdict(run_parameters)
        )
    )
    if use_cache and (target / ".zmetadata").exists():
        print(f"found existing target: {target}")
        return target

    # experiment_ds = xr.open_zarr(experiment_path)

    # coarse_obs_ds = regrid_ds(ds_path=obs_path, target_grid_ds=experiment_ds)

    # coarse_obs_ds.to_zarr(target, mode='w')
    return target


@task

def interpolate_obs(
    obs_path: UPath, coarse_obs_path: UPath, run_parameters: RunParameters
) -> UPath:
    target = (
        intermediate_dir
        / "interpolate_obs"
        / "{obs}_{model}_{variable}_{latmin}_{latmax}_{lonmin}_{lonmax}_{train_dates[0]}_{train_dates[1]}".format(
            **asdict(run_parameters)
        )
    )
    if use_cache and (target / ".zmetadata").exists():
        print(f"found existing target: {target}")
        return target

    # TODO

    # interpolated_obs_ds.to_zarr(target, mode='w')
    return target


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
    experiment_full_time_path: UPath,
    coarse_obs_full_time_path: UPath,
    run_parameters: RunParameters,
) -> UPath:

    target = intermediate_dir / "fit_and_predict" / run_parameters.run_id
    if use_cache and (target / ".zmetadata").exists():
        print(f"found existing target: {target}")
        return target

    # TODO

    # bias_corrected.to_zarr(target, mode='w')
    return target


@task
def interpolate_prediction(
    bias_corrected_path: UPath, obs_path: UPath, run_parameters: RunParameters
) -> UPath:

    target = intermediate_dir / "interpolate_prediction" / run_parameters.run_id
    if use_cache and (target / ".zmetadata").exists():
        print(f"found existing target: {target}")
        return target

    # TODO

    # bias_corrected_fine_full_space.to_zarr(target, mode='w')
    return target


@task
def postprocess_bcsd(
    bias_corrected_fine_full_time_path: UPath,
    spatial_anomalies_path: UPath,
    run_parameters: RunParameters,
) -> UPath:

    target = intermediate_dir / "interpolate_prediction" / run_parameters.run_id
    if use_cache and (target / ".zmetadata").exists():
        print(f"found existing target: {target}")
        return target

    # TODO

    # results.to_zarr(target, mode='w')
    return target
