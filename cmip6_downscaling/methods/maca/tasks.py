from __future__ import annotations

from datetime import timedelta

import dask
import regionmask
import xarray as xr
import zarr
from carbonplan_data.metadata import get_cf_global_attrs
from prefect import task
from upath import UPath

from cmip6_downscaling import __version__ as version, config
from cmip6_downscaling.methods.common.containers import RunParameters
from cmip6_downscaling.methods.common.utils import blocking_to_zarr, is_cached
from cmip6_downscaling.methods.maca import core as maca_core
from cmip6_downscaling.methods.maca.utils import (
    initialize_out_store,
    make_regions_mask,
    merge_block_to_zarr,
)
from cmip6_downscaling.utils import str_to_hash

intermediate_dir = UPath(config.get("storage.intermediate.uri")) / version
results_dir = UPath(config.get("storage.results.uri")) / version
use_cache = config.get('run_options.use_cache')


@task(log_stdout=True)
def bias_correction(x_path: UPath, y_path: UPath, run_parameters: RunParameters) -> UPath:
    """
    MACA bias correction task

    Parameters
    ----------
    x_path : UPath
        Path to dataset to be bias corrected
    y_path : UPath
        Path to target dataset
    run_parameters : RunParameters
        Downscaling run parameter container

    Returns
    -------
    target : UPath
    """

    ds_hash = str_to_hash(str(x_path) + str(y_path))
    target = intermediate_dir / 'bias_correction' / ds_hash

    if use_cache and is_cached(target):
        print(f"found existing target: {target}")
        return target

    x_ds = xr.open_zarr(x_path).chunk({'time': -1})
    y_ds = xr.open_zarr(y_path).chunk({'time': -1})

    bc_ds = xr.map_blocks(
        maca_core.bias_correction,
        x_ds,
        args=(y_ds.rename({'time': 't2'}),),
        kwargs=dict(variables=[run_parameters.variable]),
        template=x_ds,
    )

    bc_ds.attrs.update({'title': 'bias_correction'}, **get_cf_global_attrs(version=version))

    blocking_to_zarr(ds=bc_ds, target=target, validate=True, write_empty_chunks=True)

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
        Input data path
    run_parameters : RunParameters
        Downscaling run parameter container

    Returns
    -------
    trend_target : UPath
        Path to the trend
    detrend_target : UPath
        Path to the detrended data

    See also
    --------
    replace_epoch_trend
    """

    # TODO: figure out how this task should differ for ['pr', 'huss', 'vas', 'uas']

    ds_hash = str_to_hash(str(data_path) + run_parameters.run_id_hash)
    trend_target = intermediate_dir / 'epoch_trend' / ds_hash
    detrend_target = intermediate_dir / 'epoch_detrend' / ds_hash

    if use_cache and is_cached(trend_target) and is_cached(detrend_target):
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
    # TODO: this could probably be sped up with map_blocks
    historical_period = run_parameters.train_period.time_slice
    trend = maca_core.epoch_trend(
        data=ds_gcm_full_time,
        historical_period=historical_period,
        day_rolling_window=run_parameters.day_rolling_window,
        year_rolling_window=run_parameters.year_rolling_window,
    )

    if 'dayofyear' in trend.variables:
        trend = trend.drop('dayofyear')

    trend = trend.chunk({'lat': 48, 'lon': 48})
    trend.attrs.update({'title': 'epoch_trend'}, **get_cf_global_attrs(version=version))

    blocking_to_zarr(ds=trend, target=trend_target, validate=True, write_empty_chunks=True)

    detrended_data = ds_gcm_full_time - trend
    detrended_data = detrended_data.chunk({'time': -1, 'lat': 48, 'lon': 48})

    detrended_data.attrs.update(
        {'title': 'epoch_trend - detrended'}, **get_cf_global_attrs(version=version)
    )
    blocking_to_zarr(
        ds=detrended_data, target=detrend_target, validate=True, write_empty_chunks=True
    )

    return trend_target, detrend_target


@task(log_stdout=True, max_retries=5, retry_delay=timedelta(seconds=1))
def construct_analogs(
    gcm_path: UPath,
    coarse_obs_path: UPath,
    fine_obs_path: UPath,
    run_parameters: RunParameters,
) -> UPath:
    """
    MACA analog construction

    Parameters
    ----------
    gcm_path : UPath
        Path to GCM dataset
    coarse_obs_path : UPath
        Path to coarse obs dataset
    fine_obs_path : UPath
        Path to fine obs dataset
    run_parameters : RunParameters
        Downscaling run parameter container

    Returns
    -------
    target : UPath
    """

    ds_hash = str_to_hash(
        str(gcm_path) + str(coarse_obs_path) + str(fine_obs_path) + run_parameters.run_id_hash
    )
    target = intermediate_dir / 'construct_analogs' / ds_hash

    if use_cache and is_cached(target):
        print(f"found existing target: {target}")
        return target

    # TODO: see if this can be removed
    # Originally put here to avoid killed-worker / memory errors
    with dask.config.set(scheduler='threads'):
        ds_gcm = xr.open_zarr(gcm_path)
        ds_obs_coarse = xr.open_zarr(coarse_obs_path)
        ds_obs_fine = xr.open_zarr(fine_obs_path)

        analogs = maca_core.construct_analogs(
            ds_gcm, ds_obs_coarse, ds_obs_fine, run_parameters.variable
        )

        analogs = analogs.chunk({'lat': -1, 'lon': -1, 'time': 365})
        analogs.attrs.update(
            {'title': 'construct_analogs ' + ds_gcm.attrs['title']},
            **get_cf_global_attrs(version=version),
        )
        blocking_to_zarr(ds=analogs, target=target, validate=True, write_empty_chunks=True)

    return target


def _get_regions(region_def):
    """helper function to extract regionmaks regions"""

    if region_def == 'ar6.land':
        regions = regionmask.defined_regions.ar6.land
    else:
        regions = getattr(regionmask.defined_regions, region_def)

    return regions


@task(log_stdout=True)
def get_region_numbers(region_def: str = 'ar6.land'):
    regions = _get_regions(region_def)
    return regions.numbers[:-2]  # drop antarctica


@task(log_stdout=True)
def split_by_region(region: int, data_path: UPath, region_def: str = 'ar6.land') -> list[UPath]:
    """
    Split dataset into separate regions

    Parameters
    ----------
    region : int
        Region key
    data_path : UPath
        Path to dataset
    region_def : str, optional
        Regionmask key, by default `ar6.land`

    Returns
    -------
    target : UPath

    See also
    --------
    combine_regions
    """

    regions = _get_regions(region_def)

    target = (
        intermediate_dir
        / 'split_by_region'
        / str_to_hash(str(data_path) + region_def + str(region))
    )

    if use_cache and is_cached(target):
        print(f"found existing target: {target}")
        return target

    ds = xr.open_zarr(data_path)

    with dask.config.set(**{'array.slicing.split_large_chunks': False}):
        mask = regions.mask(ds)
        ds_region = ds.where(mask == region, drop=True)
        ds_region = ds_region.chunk({'lat': -1, 'lon': -1, 'time': 365})
    ds_region.attrs.update({'title': f'region {region}'}, **get_cf_global_attrs(version=version))
    blocking_to_zarr(ds=ds_region, target=target, validate=True, write_empty_chunks=True)

    return target


@task(log_stdout=True)
def combine_regions(
    regions: list,
    region_paths: list[UPath],
    template_path: UPath,
) -> UPath:
    """
    Combine regions

    Parameters
    ----------
    regions : list of ints
        Region keys
    region_paths : list of UPath
        Paths to region datasets
    template_path : UPath
        Path to template dataset, defines output schema

    Returns
    -------
    target : UPath

    See also
    --------
    split_by_region
    """

    target = (
        results_dir
        / 'combine_regions'
        / str_to_hash(str(regions) + str(region_paths) + str(template_path))
    )

    if use_cache and is_cached(target):
        print(f"found existing target: {target}")
        return target

    n_chunk_per_block = 4
    chunk_size = 48

    out_time_index = xr.open_zarr(region_paths[0]).time.values
    template = initialize_out_store(template_path, target, out_time_index, chunk_size=chunk_size)

    mask = make_regions_mask(template.isel(time=0), chunk_size=chunk_size)

    region_paths_dict = dict(zip(regions, region_paths))

    block_size = chunk_size * n_chunk_per_block
    # Iterate over chunks and merge pieces within each chunk
    total = []
    for ilon in range(0, mask.sizes['lon'], block_size):
        if ilon <= mask.sizes['lon'] - block_size:
            xslice = slice(ilon, ilon + block_size)
        else:
            xslice = slice(ilon, mask.sizes['lon'])
        for ilat in range(0, mask.sizes['lat'], block_size):
            if ilat <= mask.sizes['lat'] - block_size:
                yslice = slice(ilat, ilat + block_size)
            else:
                yslice = slice(ilat, mask.sizes['lat'])
            total.append(
                merge_block_to_zarr(
                    mask.isel(lon=xslice, lat=yslice),
                    template.isel(lon=xslice, lat=yslice),
                    region_paths_dict,
                    target,
                    xslice=xslice,
                    yslice=yslice,
                )
            )
    # TODO: merge_block_to_zarr was originally design to be a dask.delayed function
    # But this wasn't playing nice with dask+prefect.
    # result = dask.compute(*total, scheduler='threads')
    zarr.consolidate_metadata(target)
    return target


@task(log_stdout=True)
def replace_epoch_trend(analogs_path: UPath, trend_path: UPath) -> UPath:
    """
    Replace epoch trend

    Parameters
    ----------
    analogs_path : UPath
        Path to constructed analogs
    trend_path : UPath
        Path to trend dataset

    Returns
    -------
    target : UPath

    See also
    --------
    epoch_trend
    """

    target = results_dir / 'replace_epoch_trend' / str_to_hash(str(analogs_path) + str(trend_path))

    if use_cache and is_cached(target):
        print(f"found existing target: {target}")
        return target

    analogs_ds = xr.open_zarr(analogs_path)
    trend_ds = xr.open_zarr(trend_path)

    downscaled = analogs_ds + trend_ds

    downscaled.attrs.update(
        {'title': 'replace_epoch_trend'}, **get_cf_global_attrs(version=version)
    )
    blocking_to_zarr(ds=downscaled, target=target, validate=True, write_empty_chunks=True)

    return target
