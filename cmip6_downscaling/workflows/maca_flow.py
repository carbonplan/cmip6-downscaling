from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import xarray as xr
import xesmf as xe
from prefect import task

from cmip6_downscaling.methods.detrend import calc_epoch_trend, remove_epoch_trend
from cmip6_downscaling.methods.maca import maca_bias_correction, maca_construct_analogs
from cmip6_downscaling.workflows.utils import generate_batches, rechunk_zarr_array_with_caching
from cmip6_downscaling.methods.regions import generate_subdomains, combine_outputs
from xpersist.prefect.result import XpersistResult

from cmip6_downscaling.config.config import intermediate_cache_store, serializer
from cmip6_downscaling.workflows.utils import regrid_ds
from cmip6_downscaling.workflows.paths import (
    make_epoch_trend_path,
    make_epoch_adjusted_gcm_path,
    make_bias_corrected_gcm_path,
    make_epoch_adjusted_downscaled_gcm_path,
    make_epoch_replaced_downscaled_gcm_path,
    make_maca_output_path,
)
from cmip6_downscaling.tasks.common_tasks import (
    path_builder_task,
    get_obs_task,
    get_coarse_obs_task,
    get_gcm_task,
    rechunker_task,
)


@task(
    checkpoint=True,
    result=XpersistResult(intermediate_cache_store, serializer=serializer),
    target=make_epoch_trend_path,
)
def calc_epoch_trend_task(
    data: xr.Dataset,
    train_period_start: str,
    train_period_end: str,
    day_rolling_window: int = 21, 
    year_rolling_window: int = 31,
    **kwargs,
):
    """
    Input data should be in full time chunks 
    """
    historical_period = slice(train_period_start, train_period_end)
    trend = calc_epoch_trend(
        data=data, 
        historical_period=historical_period, 
        day_rolling_window=day_rolling_window, 
        year_rolling_window=year_rolling_window
    )
    return trend 


remove_epoch_trend_task = task(
    remove_epoch_trend,
    result=XpersistResult(intermediate_cache_store, serializer=serializer),
    target=make_epoch_adjusted_gcm_path,    
)


@task(
    checkpoint=True,
    result=XpersistResult(intermediate_cache_store, serializer=serializer),
    target=make_bias_corrected_gcm_path,    
)
def maca_coarse_bias_correction_task(
    ds_gcm: xr.Dataset,
    ds_obs: xr.Dataset,
    train_period_start: str,
    train_period_end: str,
    variables: Union[str, List[str]],
    chunking_approach: str,
    batch_size: Optional[int] = 15,
    buffer_size: Optional[int] = 15,
    **kwargs,
):
    """
    """
    # TODO: test if this is needed if both ds_gcm and ds_obs are chunked in full time 
    ds_gcm_rechunked = rechunk_zarr_array_with_caching(
        zarr_array=ds_gcm, template_chunk_array=ds_obs, output_path='maca_test_bias_correction_rechunk.zarr'  # TODO: remove this 
    )

    historical_period = slice(train_period_start, train_period_end)
    bias_corrected = maca_bias_correction(
        ds_gcm=ds_gcm_rechunked,
        ds_obs=ds_obs,
        historical_period=historical_period,
        variables=variables,
        batch_size=batch_size,
        buffer_size=buffer_size,
    )

    return bias_corrected_rechunked


@task(nout=4)
def get_subdomains_task(
    ds_obs: xr.Dataset,
    buffer_size: Union[float, int] = 5,
    region_def: str = 'ar6'
):
    subdomains_dict, mask = generate_subdomains(
        ex_output_grid=ds_obs.isel(time=0),
        buffer_size=buffer_size,
        region_def=region_def,
    )

    subdomains_list = []
    for k in sorted(subdomains_dict.keys()):
        subdomains_list.append(subdomains_dict[k])

    return subdomains_list, subdomains_dict, mask, len(subdomains_list)


@task(nout=3)
def subset_task(
    ds_gcm: xr.Dataset,
    ds_obs_coarse: xr.Dataset,
    ds_obs_fine: xr.Dataset,
    subdomains_list: List[Tuple[float, float, float, float]],
):
    ds_gcm_list, ds_obs_coarse_list, ds_obs_fine_list = [], [], []
    for (min_lon, min_lat, max_lon, max_lat) in subdomains_list:
        lat_slice = slice(max_lat, min_lat)
        lon_slice = slice(min_lon, max_lon)
        ds_gcm_list.append(ds_gcm.sel(lat=lat_slice, lon=lon_slice))
        ds_obs_coarse_list.append(ds_obs_coarse.sel(lat=lat_slice, lon=lon_slice))
        ds_obs_fine_list.append(ds_obs_fine.sel(lat=lat_slice, lon=lon_slice))

    return ds_gcm_list, ds_obs_coarse_list, ds_obs_fine_list


maca_construct_analogs_task = task(
    maca_construct_analogs,
    result=XpersistResult(intermediate_cache_store, serializer=serializer),
    target=make_epoch_adjusted_downscaled_gcm_path,   
)


@task(
    checkpoint=True,
    result=XpersistResult(intermediate_cache_store, serializer=serializer),
    target=make_epoch_adjusted_downscaled_gcm_path,
)
def combine_outputs_task(
    ds_list: List[xr.Dataset],
    subdomains_dict: Dict[Union[int, float], Any], 
    mask: xr.DataArray, 
    **kwargs,
):  
    ds_dict = {}
    for k in sorted(subdomains_dict.keys()):
        ds_dict[k] = ds_list.pop(0)

    combined_output = combine_outputs(
        ds_dict=ds_dict,
        mask=mask
    )

    return combined_output


@task(
    checkpoint=True,
    result=XpersistResult(intermediate_cache_store, serializer=serializer),
    target=make_epoch_replaced_downscaled_gcm_path,
)
def maca_epoch_replacement_task(
    ds_gcm_fine: xr.Dataset,
    trend_coarse: xr.Dataset,
    **kwargs,
) -> xr.Dataset:

    trend_fine = regrid_ds(
        ds=trend_coarse,
        target_grid_ds=ds_gcm_fine.isel(time=0).chunk({'lat': -1, 'lon': -1}),
        rechunked_ds_path='maca_test_epoch_replacement_rechunk.zarr'  # TODO: remove this
    )

    return ds_gcm_fine + trend_fine


@task(
    checkpoint=True,
    result=XpersistResult(intermediate_cache_store, serializer=serializer),
    target=make_maca_output_path,    
)
def maca_fine_bias_correction_task(
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
    """
    ds_gcm_rechunked = rechunk_zarr_array_with_caching(
        zarr_array=ds_gcm, template_chunk_array=ds_obs, output_path='maca_test_fine_bias_correction_rechunk.zarr'  # TODO: remove this 
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


with Flow(name='maca-flow') as maca_flow:
    obs = Parameter("OBS")
    gcm = Parameter("GCM")
    scenario = Parameter("SCENARIO")
    label = Parameter("LABEL")

    train_period_start = Parameter("TRAIN_PERIOD_START")
    train_period_end = Parameter("TRAIN_PERIOD_END")
    predict_period_start = Parameter("PREDICT_PERIOD_START")
    predict_period_end = Parameter("PREDICT_PERIOD_END")

    epoch_adjustment_day_rolling_window = Parameter("EPOCH_ADJUSTMENT_DAY_ROLLING_WINDOW")
    epoch_adjustment_year_rolling_window = Parameter("EPOCH_ADJUSTMENT_YEAR_ROLLING_WINDOW")
    bias_correction_batch_size = Parameter("BIAS_CORRECTION_BATCH_SIZE")
    bias_correction_buffer_size = Parameter("BIAS_CORRECTION_BUFFER_SIZE")
    constructed_analog_n_analogs = Parameter("CONSTRUCTED_ANALOG_N_ANALOGS")
    constructed_analog_doy_range = Parameter("CONSTRUCTED_ANALOG_DOY_RANGE")

    # dictionary with information to build appropriate paths for caching
    gcm_grid_spec, obs_identifier, gcm_identifier = path_builder_task(
        obs=obs,
        gcm=gcm,
        scenario=scenario,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        predict_period_start=predict_period_start,
        predict_period_end=predict_period_end,
        variables=[label],
    )
    
    # get original resolution observations 
    ds_obs_full_space = get_obs_task(
        obs=obs,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        variables=[label],
        chunking_approach='full_space',
        cache_within_rechunk=True,
    )

    # get coarsened resolution observations 
    # this coarse obs is going to be used in bias correction next, so rechunk into full time first 
    ds_obs_coarse_full_time = get_coarse_obs_task(
        ds_obs=ds_obs_full_space, 
        gcm=gcm, 
        chunking_approach='full_time', 
        gcm_grid_spec=gcm_grid_spec,
        obs_identifier=obs_identifier,
    )
    
    # get gcm 
    ds_gcm_full_time = get_gcm_task(
        gcm=gcm,
        scenario=scenario,
        variables=[label],
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        predict_period_start=predict_period_start,
        predict_period_end=predict_period_end,
        chunking_approach='full_time',
        cache_within_rechunk=True,
    )
    
    # epoch adjustment #1 -- all variables undergo this epoch adjustment 
    coarse_epoch_trend = calc_epoch_trend_task(
        data=ds_gcm_full_time,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        day_rolling_window=epoch_adjustment_day_rolling_window,
        year_rolling_window=epoch_adjustment_year_rolling_window,
        gcm_identifier=gcm_identifier,
    )
    
    epoch_adjusted_gcm = remove_epoch_trend_task(
        data=ds_gcm_full_time,
        trend=coarse_epoch_trend,
        day_rolling_window=epoch_adjustment_day_rolling_window,
        year_rolling_window=epoch_adjustment_year_rolling_window,
        gcm_identifier=gcm_identifier,
    )
    
    # coarse scale bias correction 
    bias_corrected_gcm = maca_coarse_bias_correction_task(
        ds_gcm=epoch_adjusted_gcm,
        ds_obs=ds_obs_coarse_full_time,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        variables=[label],
        batch_size=bias_correction_batch_size,
        buffer_size=bias_correction_buffer_size,
        method='maca_edcdfm',
        gcm_identifier=gcm_identifier,
        chunking_approach='matched',
    )
    
    # do epoch adjustment again for multiplicative variables 
    if label in ['pr', 'huss', 'vas', 'uas']:
        coarse_epoch_trend_2 = calc_epoch_trend_task(
            data=bias_corrected_gcm,
            train_period_start=train_period_start,
            train_period_end=train_period_end,
            day_rolling_window=epoch_adjustment_day_rolling_window,
            year_rolling_window=epoch_adjustment_year_rolling_window,
            gcm_identifier=gcm_identifier+'_2',
        )

        bias_corrected_gcm = remove_epoch_trend_task(
            data=bias_corrected_gcm,
            trend=coarse_epoch_trend_2,
            day_rolling_window=epoch_adjustment_day_rolling_window,
            year_rolling_window=epoch_adjustment_year_rolling_window,
            gcm_identifier=gcm_identifier+'_2',
        )    
        
    # rechunk into full space and cache the output 
    bias_corrected_gcm_full_space = rechunker_task(
        zarr_array=bias_corrected, 
        chunking_approach='full_space',
        naming_func=make_bias_corrected_gcm_path,
        gcm_identifier=gcm_identifier,
    )

    # subset into regions 
    subdomains_list, subdomains_dict, mask, n_subdomains = get_subdomains_task(
        ds_obs=ds_obs_full_space
    )
    
    # everything should be rechunked to full space and then subset 
    ds_obs_coarse_full_space = get_coarse_obs_task(
        ds_obs=ds_obs_full_space, 
        gcm=gcm, 
        chunking_approach='full_space', 
        gcm_grid_spec=gcm_grid_spec,
        obs_identifier=obs_identifier,
    )
    # all inputs into the map function needs to be a list 
    ds_gcm_list, ds_obs_coarse_list, ds_obs_fine_list = subset_task(
        ds_gcm=bias_corrected_gcm_full_space,
        ds_obs_coarse=ds_obs_coarse_full_space,
        ds_obs_fine=ds_obs_full_space,
        subdomains_list=subdomains_list,
    )
    
    # downscaling by constructing analogs 
    downscaled_gcm_list = maca_construct_analogs_task.map(
        ds_gcm=ds_gcm_list,
        ds_obs_coarse=ds_obs_coarse_list,
        ds_obs_fine=ds_obs_fine_list,
        subdomain_bound=subdomains_list,
        n_analogs=[constructed_analog_n_analogs] * n_subdomains,
        doy_range=[constructed_analog_doy_range] * n_subdomains,
        gcm_identifier=[gcm_identifier] * n_subdomains,
        label=[label] * n_subdomains,
    )
    
    # combine back into full domain 
    combined_downscaled_output = combine_outputs_task(
        ds_list=downscaled_gcm_list,
        subdomains_dict=subdomains_dict, 
        mask=mask,
        gcm_identifier=gcm_identifier,
        label=label
    )
    
    # replace epoch 
    if label in ['pr', 'huss', 'vas', 'uas']:
        combined_downscaled_output = maca_epoch_replacement_task(
            ds_gcm_fine=combined_downscaled_output,
            trend_coarse=coarse_epoch_trend_2,
            day_rolling_window=epoch_adjustment_day_rolling_window,
            year_rolling_window=epoch_adjustment_year_rolling_window,
            gcm_identifier=gcm_identifier+'_2',
        )
        
    epoch_replaced_gcm = maca_epoch_replacement_task(
        ds_gcm_fine=combined_downscaled_output,
        trend_coarse=coarse_epoch_trend,
        day_rolling_window=epoch_adjustment_day_rolling_window,
        year_rolling_window=epoch_adjustment_year_rolling_window,
        gcm_identifier=gcm_identifier,
    )
    
    ds_obs_full_time = get_obs_task(
        obs=obs,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        variables=[label],
        chunking_approach='full_time',
        cache_within_rechunk=True,
    )
    # fine scale bias correction 
    final_output = maca_fine_bias_correction_task(
        ds_gcm=epoch_replaced_gcm,
        ds_obs=ds_obs_full_time,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        label=label,
        batch_size=bias_correction_batch_size,
        buffer_size=bias_correction_buffer_size,
        gcm_identifier=gcm_identifier,
    )
