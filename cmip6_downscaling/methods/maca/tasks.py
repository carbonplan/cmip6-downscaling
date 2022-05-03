from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr
import xesmf as xe
from carbonplan_data.metadata import get_cf_global_attrs
from prefect import task

from upath import UPath

from cmip6_downscaling import __version__ as version, config
from cmip6_downscaling.methods.common.containers import RunParameters
from . import core
from . import detrend

from cmip6_downscaling.methods.common.utils import zmetadata_exists
from cmip6_downscaling.utils import str_to_hash


intermediate_dir = UPath(config.get("storage.intermediate.uri")) / version
results_dir = UPath(config.get("storage.results.uri")) / version
use_cache = config.get('run_options.use_cache')


@task(log_stdout=True)
def bias_correction(gcm_full_time_path: UPath, obs_full_time_path: UPath, run_parameters):

    ds_hash = str_to_hash(str(gcm_full_time_path) + str(obs_full_time_path))
    target = intermediate_dir / 'bias_correction' / ds_hash

    if use_cache and zmetadata_exists(target):
        print(f"found existing target: {target}")
        return target

    gcm_ds = xr.open_zarr(gcm_full_time_path)
    obs_ds = xr.open_zarr(obs_full_time_path)

    bc_ds = core.maca_bias_correction(gcm_ds, obs_ds, variables=[run_parameters.variable])

    bc_ds.attrs.update({'title': 'bias_correction'}, **get_cf_global_attrs(version=version))
    bc_ds.to_zarr(target, mode='w')

    return target


@task(log_stdout=True, nout=2)
def calc_epoch_trend(
    data_path: UPath,
    run_parameters: RunParameters,
):
    """
    Task to calculate the epoch trends in MACA. 
    
    The epoch trend is a long term rolling average, and thus the first and last few years of the
    output suffers from edge effects. Thus, this task gets additional years for calculating the
    rolling averages.

    Parameters
    ----------
    data_path : UPath
    run_parameters: RunParameters

    Returns
    -------
    trend : xr.Dataset
        The long term average trend
    """

    ds_hash = str_to_hash(str(data_path)+run_parameters.run_id_hash)
    trend_target = intermediate_dir / 'epoch_trend' / ds_hash
    detrend_target = intermediate_dir / 'epoch_detrend' / ds_hash

    print(trend_target)
    if use_cache and zmetadata_exists(trend_target) and zmetadata_exists(detrend_target):
        print(f'found existing targets: {trend_target} and {detrend_target}')
        return trend_target, detrend_target

    # obtain a buffer period for both training and prediction period equal to half of the year_rolling_window
    y_offset = int((run_parameters.year_rolling_window - 1) / 2)

    train_start = int(run_parameters.train_period.start) - y_offset
    train_end = int(run_parameters.train_period.end) + y_offset
    predict_start = int(run_parameters.predict_period.start) - y_offset
    predict_end = int(run_parameters.predict_period.end) + y_offset
    # TODO: why do we do this step?
    # make sure there are no overlapping years
    if train_end > int(run_parameters.train_period.start):
        train_end = int(run_parameters.train_period.start) - 1
        predict_start = int(run_parameters.train_period.start)
    elif train_end > predict_start:
        predict_start = train_end + 1

    ds_gcm_full_time = xr.open_zarr(data_path)

    # note that this is the non-buffered slice
    historical_period = run_parameters.train_dates.time_slice
    predict_period = run_parameters.predict_dates.time_slice
    trend = detrend.calc_epoch_trend(
        data=ds_gcm_full_time,
        historical_period=historical_period,
        day_rolling_window=run_parameters.day_rolling_window,
        year_rolling_window=run_parameters.year_rolling_window,
    )

    hist_trend = trend.sel(time=historical_period)
    pred_trend = trend.sel(time=predict_period)

    trend = xr.combine_by_coords([hist_trend, pred_trend], combine_attrs='drop_conflicts')
    trend.attrs.update({'title': 'calc_epoch_trend'}, **get_cf_global_attrs(version=version))
    trend.to_zarr(trend_target, mode='w')

    detrended_data = ds_gcm_full_time - trend
    detrended_data.attrs.update({'title': 'calc_epoch_trend - detrended'}, **get_cf_global_attrs(version=version))
    trend.to_zarr(detrended_data, mode='w')

    return trend_target, detrend_target


@task(nout=4)
def get_subdomains(
    ds_obs: xr.Dataset, buffer_size: Union[float, int] = 5, region_def: str = 'ar6'
):
    """
    Get the definition of subdomains according to region_def specified.

    Parameters
    ----------
    ds_obs : xr.Dataset
        Observation dataset
    buffer_size : int or float
        Buffer size in unit of degree. for each subdomain, how much extra area to run for each subdomain
    region_def : str
        Subregion definition name. Options are `'ar6'` or `'srex'`. See the docs https://regionmask.readthedocs.io/en/stable/defined_scientific.html for more details.

    Returns
    -------
    subdomains_list: List
        List of all subdomain boundaries sorted by the region code
    subdomains_dict : dict
        Dictionary mapping subdomain code to bounding boxes ([min_lon, min_lat, max_lon, max_lat]) for each subdomain
    mask : xarray.DataArray
        Mask of which subdomain code to use for each grid cell
    n_subdomains: int
        The number of subdomains that are included
    """
    subdomains_dict, mask = generate_subdomains(
        ex_output_grid=ds_obs.isel(time=0),
        buffer_size=buffer_size,
        region_def=region_def,
    )

    subdomains_list = [subdomains_dict[k] for k in sorted(subdomains_dict.keys())]
    return subdomains_list, subdomains_dict, mask, len(subdomains_list)


@task(nout=3)
def subset(
    ds_gcm: xr.Dataset,
    ds_obs_coarse: xr.Dataset,
    ds_obs_fine: xr.Dataset,
    subdomains_list: List[Tuple[float, float, float, float]],
):
    """
    Subset each dataset spatially into areas within each subdomain bound.

    Parameters
    ----------
    ds_gcm : xr.Dataset
        GCM dataset, original/coarse resolution
    ds_obs_coarse : xr.Dataset
        Observation dataset coarsened to the GCM resolution
    ds_obs_fine : xr.Dataset
        Observation dataset, original/fine resolution
    subdomains_list : List
        List of all subdomain boundaries sorted by the region code

    Returns
    -------
    ds_gcm_list : List
        List of subsetted GCM datasets in the same order of subdomains_list
    ds_obs_coarse_list : List
        List of subsetted coarened obs datasets in the same order of subdomains_list
    ds_obs_fine_list : List
        List of subsetted fine obs datasets in the same order of subdomains_list
    """
    ds_gcm_list, ds_obs_coarse_list, ds_obs_fine_list = [], [], []
    for (min_lon, min_lat, max_lon, max_lat) in subdomains_list:
        lat_slice = slice(max_lat, min_lat)
        lon_slice = slice(min_lon, max_lon)
        ds_gcm_list.append(ds_gcm.sel(lat=lat_slice, lon=lon_slice))
        ds_obs_coarse_list.append(ds_obs_coarse.sel(lat=lat_slice, lon=lon_slice))
        ds_obs_fine_list.append(ds_obs_fine.sel(lat=lat_slice, lon=lon_slice))

    return ds_gcm_list, ds_obs_coarse_list, ds_obs_fine_list


@task(log_stdout=True)
def combine_outputs(
    ds_list: List[xr.Dataset],
    subdomains_dict: Dict[Union[int, float], Any],
    mask: xr.DataArray,
    **kwargs,
):
    """
    Combine a list of datasets spatially according to the subdomain list and mask.

    Parameters
    ----------
    ds_list: List[xr.Dataset]
        List of datasets to be combined
    subdomains_dict : dict
        Dictionary mapping subdomain code to bounding boxes ([min_lon, min_lat, max_lon, max_lat]) for each subdomain
    mask : xarray.DataArray
        Mask of which subdomain code to use for each grid cell

    Returns
    -------
    combined_output: xr.Dataset
        The combined output
    """
    ds_dict = {k: ds_list.pop(0) for k in sorted(subdomains_dict.keys())}
    return combine_outputs(ds_dict=ds_dict, mask=mask)


@task(log_stdout=True)
def maca_epoch_replacement(
    ds_gcm_fine: xr.Dataset,
    trend_coarse: xr.Dataset,
    **kwargs,
) -> xr.Dataset:
    """
    Replace the epoch trend. The trend was calculated on coarse scale GCM, so the trend is first interpolated
    into the finer grid before being added back into the downscaled GCM.

    Parameters
    ----------
    ds_gcm_fine: xr.Dataset
        Downscaled GCM, fine/observation resolution
    trend_coarse: xr.Dataset
        The epoch trend, coarse/original GCM resolution

    Returns
    -------
    epoch_replaced_gcm: xr.Dataset
        The downscaled GCM dataset with the epoch trend replaced back
    """
    trend_fine = regrid_ds(
        ds=trend_coarse,
        target_grid_ds=ds_gcm_fine.isel(time=0).chunk({'lat': -1, 'lon': -1}),
    )

    return ds_gcm_fine + trend_fine


@task(log_stdout=True)
def maca_fine_bias_correction(
    ds_gcm: xr.Dataset,
    ds_obs: xr.Dataset,
    train_period_start: str,
    train_period_end: str,
    label: str,
    batch_size: Optional[int] = 15,
    buffer_size: Optional[int] = 15,
    **kwargs,
):
    """
    Task that implements the fine scale bias correction in MACA. The historical GCM is mapped to historical
    coarsened observation in the bias correction. Rechunks the GCM data to match observation data because
    the bias correction model in skdownscale requires these datasets to have the same chunks/blocks.

    ds_gcm: xr.Dataset
        GCM dataset
    ds_obs: xr.Dataset
        Observation dataset
    train_period_start: str
        Start year of training/historical period
    train_period_end: str
        End year of training/historical period
    variables: List[str]
        Names of the variables used in obs and gcm dataset (including features and label)
    chunking_approach: str
        'full_space', 'full_time', 'matched' or None
    batch_size: Optional[int]
        The batch size in terms of day of year to bias correct together
    buffer_size: Optional[int]
        The buffer size in terms of day of year to include in the bias correction

    Returns
    -------
    bias_corrected: xr.Dataset
        Bias corrected GCM dataset
    """
    ds_gcm_rechunked = rechunk_zarr_array_with_caching(
        zarr_array=ds_gcm, template_chunk_array=ds_obs
    )

    historical_period = slice(train_period_start, train_period_end)
    bias_corrected = maca_bias_correction(
        ds_gcm=ds_gcm_rechunked,
        ds_obs=ds_obs,
        historical_period=historical_period,
        variables=[label],
        batch_size=batch_size,
        buffer_size=buffer_size,
    )

    return bias_corrected
