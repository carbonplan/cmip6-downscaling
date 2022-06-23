from __future__ import annotations

from datetime import timedelta

import dask
import regionmask
import xarray as xr
from carbonplan_data.metadata import get_cf_global_attrs
from prefect import task
from upath import UPath

from cmip6_downscaling import __version__ as version, config
from cmip6_downscaling.methods.common.containers import RunParameters
from cmip6_downscaling.methods.common.utils import zmetadata_exists
from cmip6_downscaling.methods.maca import core as maca_core
from cmip6_downscaling.utils import str_to_hash

intermediate_dir = UPath(config.get("storage.intermediate.uri")) / version
results_dir = UPath(config.get("storage.results.uri")) / version
use_cache = config.get('run_options.use_cache')


@task(log_stdout=True)
def bias_correction(x_path: UPath, y_path: UPath, run_parameters: RunParameters) -> UPath:

    ds_hash = str_to_hash(str(x_path) + str(y_path))
    target = intermediate_dir / 'bias_correction' / ds_hash

    if use_cache and zmetadata_exists(target):
        print(f"found existing target: {target}")
        return target

    print('x_path', x_path)

    x_ds = xr.open_zarr(x_path).chunk({'time': -1, 'lat': 48, 'lon': 48})
    y_ds = xr.open_zarr(y_path).chunk({'time': -1, 'lat': 48, 'lon': 48})

    print('x_ds', x_ds)
    print('y_ds', y_ds)

    bc_ds = maca_core.bias_correction(x_ds, y_ds, variables=[run_parameters.variable])

    print('bc_ds', bc_ds)

    bc_ds.attrs.update({'title': 'bias_correction'}, **get_cf_global_attrs(version=version))
    bc_ds.to_zarr(target, mode='w')

    return target


@task(log_stdout=True, nout=2)
def epoch_trend(
    data_path: UPath,
    run_parameters: RunParameters,
) -> tuple[UPath, UPath]:
    """
    Task to calculate the epoch trends in MACA.

    The epoch trend is a long term rolling average, and thus the first and last few years of the
    output suffers from edge effects. Thus, this task gets additional years for calculating the
    rolling averages.

    Parameters
    ----------
    data_path : UPath
    run_parameters : RunParameters

    Returns
    -------
    trend : xr.Dataset
        The long term average trend
    """

    # TODO: figure out how this task should differ for ['pr', 'huss', 'vas', 'uas']

    ds_hash = str_to_hash(str(data_path) + run_parameters.run_id_hash)
    trend_target = intermediate_dir / 'epoch_trend' / ds_hash
    detrend_target = intermediate_dir / 'epoch_detrend' / ds_hash

    print(trend_target)
    if use_cache and zmetadata_exists(trend_target) and zmetadata_exists(detrend_target):
        print(f'found existing targets: {trend_target} and {detrend_target}')
        return trend_target, detrend_target

    # obtain a buffer period for both training and prediction period equal to half of the year_rolling_window
    y_offset = int((run_parameters.year_rolling_window - 1) / 2)

    train_end = int(run_parameters.train_period.stop) + y_offset
    predict_start = int(run_parameters.predict_period.start) - y_offset
    # TODO: why do we do this step?
    # make sure there are no overlapping years
    if train_end > int(run_parameters.train_period.start):
        train_end = int(run_parameters.train_period.start) - 1
        predict_start = int(run_parameters.train_period.start)
    elif train_end > predict_start:
        predict_start = train_end + 1

    ds_gcm_full_time = xr.open_zarr(data_path).load()

    # note that this is the non-buffered slice
    historical_period = run_parameters.train_period.time_slice
    trend = maca_core.epoch_trend(
        data=ds_gcm_full_time,
        historical_period=historical_period,
        day_rolling_window=run_parameters.day_rolling_window,
        year_rolling_window=run_parameters.year_rolling_window,
    )

    trend = trend.chunk({'lat': 48, 'lon': 48})
    trend.attrs.update({'title': 'epoch_trend'}, **get_cf_global_attrs(version=version))
    trend.to_zarr(trend_target, mode='w')

    detrended_data = ds_gcm_full_time - trend
    detrended_data = detrended_data.chunk({'time': -1, 'lat': 48, 'lon': 48})

    detrended_data.attrs.update(
        {'title': 'epoch_trend - detrended'}, **get_cf_global_attrs(version=version)
    )
    detrended_data.to_zarr(detrend_target, mode='w')

    return trend_target, detrend_target


@task(log_stdout=True, max_retries=5, retry_delay=timedelta(seconds=1))
def construct_analogs(
    gcm_path: UPath,
    coarse_obs_path: UPath,
    fine_obs_path: UPath,
    fine_trend_path: UPath,
    run_parameters: RunParameters,
) -> UPath:

    print('construct_analogs')
    print('gcm_path', gcm_path)
    print('coarse_obs_path', coarse_obs_path)
    print('fine_obs_path', fine_obs_path)
    print('fine_trend_path', fine_trend_path)

    ds_hash = str_to_hash(
        str(gcm_path)
        + str(coarse_obs_path)
        + str(fine_obs_path)
        + str(fine_trend_path)
        + run_parameters.run_id_hash
    )
    target = intermediate_dir / 'construct_analogs' / ds_hash

    if use_cache and zmetadata_exists(target):
        print(f"found existing target: {target}")
        return target

    # Q: should we just do the region subsetting here?
    ds_gcm = xr.open_zarr(gcm_path)
    ds_obs_coarse = xr.open_zarr(coarse_obs_path)
    ds_obs_fine = xr.open_zarr(fine_obs_path)
    ds_fine_trend = xr.open_zarr(fine_trend_path)
    print('ds_fine_trend')
    print(ds_fine_trend)

    with dask.config.set(scheduler='threads'):
        print('constructing analogs now')
        analogs = maca_core.construct_analogs(
            ds_gcm, ds_obs_coarse, ds_obs_fine, run_parameters.variable
        )

    print('analogs')
    print(analogs)
    # replace epoch trend
    # TODO: figure out how this should differ for ['pr', 'huss', 'vas', 'uas']
    downscaled = analogs + ds_fine_trend  # .drop('dayofyear')

    downscaled = downscaled.chunk({'lat': -1, 'lon': -1, 'time': 1000})
    downscaled.attrs.update({'title': 'construct_analogs'}, **get_cf_global_attrs(version=version))
    downscaled.to_zarr(target, mode='w')

    return target


def _get_regions(region_def):

    if region_def == 'ar6.land':
        regions = regionmask.defined_regions.ar6.land
    else:
        regions = getattr(regionmask.defined_regions, region_def)

    return regions


@task(log_stdout=True)
def split_by_region(data_path: UPath, region_def: str = 'ar6.land') -> list[UPath]:

    regions = _get_regions(region_def)
    region_codes = regions.numbers

    target_paths = []
    for i, region in enumerate(region_codes):
        if i > 3:
            break  # for debugging only
        trend_target = (
            intermediate_dir
            / 'split_by_region'
            / str_to_hash(str(data_path) + region_def + str(region))
        )
        target_paths.append(trend_target)

    if use_cache and all(zmetadata_exists(p) for p in target_paths):
        print('found that all targets exist')
        return target_paths

    ds = xr.open_zarr(data_path)

    # can we move this to the epoch task?
    if 'dayofyear' in ds.variables:
        ds = ds.drop('dayofyear')

    print('splitting regions')
    mask = regions.mask(ds)
    for target, (key, group) in zip(target_paths, ds.groupby(mask)):
        ds = group.unstack('stacked_lat_lon').sortby(['lon', 'lat'])
        ds = ds.chunk({'lat': -1, 'lon': -1, 'time': 1000})
        ds.attrs.update({'title': f'region {key}'}, **get_cf_global_attrs(version=version))
        ds.to_zarr(target, mode='w')

    return target_paths


@task(log_stdout=True)
def combine_regions(
    region_paths: list[str], obs_path: UPath, region_def: str = 'ar6.land'
) -> UPath:

    ds_hash = str_to_hash(str(region_paths) + str(obs_path))
    target = intermediate_dir / 'combine_regions' / ds_hash

    if use_cache and zmetadata_exists(target):
        print(f"found existing target: {target}")
        return target

    obs_ds = xr.open_zarr(obs_path)

    regions = _get_regions(region_def)

    # TODO: update with logic being worked out now
    mask = regions.mask(obs_ds)
    combined_ds = xr.Dataset()
    combined_ds['mask'] = mask

    combined_ds.attrs.update({'title': 'combine_regions'}, **get_cf_global_attrs(version=version))
    combined_ds.to_zarr(target, mode='w')

    return target
