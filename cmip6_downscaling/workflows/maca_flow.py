from typing import Any, Dict, List, Optional, Union

import numpy as np
import xarray as xr
import xesmf as xe
from prefect import task
from skdownscale.pointwise_models import EquidistantCdfMatcher, PointWiseDownscaler
from sklearn.linear_model import LinearRegression

from cmip6_downscaling.methods.detrend import calc_epoch_trend, remove_epoch_trend
from cmip6_downscaling.methods.maca import maca_bias_correction
from cmip6_downscaling.workflows.utils import generate_batches, rechunk_zarr_array_with_caching
from xpersist.prefect.result import XpersistResult

from cmip6_downscaling.config.config import intermediate_cache_store, serializer
from cmip6_downscaling.workflows.paths import (
    make_epoch_trend_path,
    make_epoch_adjusted_gcm_path,
    make_bias_corrected_gcm_path,
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
def maca_bias_correction_task(
    ds_gcm: xr.Dataset,
    ds_obs: xr.Dataset,
    train_period_start: str,
    train_period_end: str,
    variables: Union[str, List[str]],
    batch_size: Optional[int] = 15,
    buffer_size: Optional[int] = 15,
    **kwargs,
):
    """
    """
    ds_gcm_rechunked = rechunk_zarr_array_with_caching(
        zarr_array=ds_gcm, template_chunk_array=ds_obs
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

    return bias_corrected


def get_doy_mask(
    source_doy: xr.DataArray,
    target_doy: xr.DataArray,
    doy_range: int = 45,
) -> xr.DataArray:
    """
    source_doy and target_doy are 1D xr data arrays with day of year information
    return a mask in the shape of len(target_doy) x len(source_doy),
    where cell (i, j) is True if the source doy j is within doy_range days of the target doy i,
    and False otherwise
    """
    # get the min and max doy within doy_range days to target doy
    target_doy_min = target_doy - doy_range
    target_doy_max = target_doy + doy_range
    # make sure the range is within 0-365
    # TODO: what to do with leap years???
    target_doy_min[target_doy_min <= 0] += 365
    target_doy_max[target_doy_max > 365] -= 365

    # if the min is larger than max, the target doy is at the edge of a year, and we can accept the
    # source if any one of the condition is True
    one_sided = target_doy_min > target_doy_max
    edges = ((source_doy >= target_doy_min) | (source_doy <= target_doy_max)) & (one_sided)
    # otherwise the source doy needs to be within min and max
    within = (source_doy >= target_doy_min) & (source_doy <= target_doy_max) & (~one_sided)

    # mask is true if either one of the condition is satisfied
    mask = edges | within

    return mask


def maca_constructed_analogs(
    ds_gcm: xr.DataArray,
    ds_obs_coarse: xr.DataArray,
    ds_obs_fine: xr.DataArray,
    n_analogs: int = 10,
    doy_range: int = 45,
) -> xr.DataArray:
    """
    it is assumed that ds_obs and ds_gcm have dimensions time, lat, lon
    """
    for dim in ['time', 'lat', 'lon']:
        for ds in [ds_gcm, ds_obs_coarse, ds_obs_fine]:
            assert dim in ds.dims

    assert len(ds_obs_coarse.time) == len(ds_obs_fine.time)

    # get dimension sizes from input data
    ndays_in_obs = len(ds_obs_coarse.time)
    ndays_in_gcm = len(ds_gcm.time)
    domain_shape_coarse = (len(ds_obs_coarse.lat), len(ds_obs_coarse.lon))
    n_pixel_coarse = domain_shape_coarse[0] * domain_shape_coarse[1]
    domain_shape_fine = (len(ds_obs_fine.lat), len(ds_obs_fine.lon))
    n_pixel_fine = domain_shape_fine[0] * domain_shape_fine[1]

    # rename the time dimension in order to get cross products
    X = ds_obs_coarse.rename({'time': 'ndays_in_obs'})  # coarse obs
    y = ds_gcm.rename({'time': 'ndays_in_gcm'})  # coarse gcm

    # get rmse between each GCM slices to be downscaled and each observation slices
    # will have the shape ndays_in_gcm x ndays_in_obs
    rmse = np.sqrt(((X - y) ** 2).sum(dim=['lat', 'lon'])) / n_pixel_coarse

    # get a day of year mask in the same shape of rmse according to the day range input
    mask = get_doy_mask(
        source_doy=X.ndays_in_obs.dt.dayofyear,
        target_doy=y.ndays_in_gcm.dt.dayofyear,
        doy_range=doy_range,
    )

    # find the analogs with the lowest rmse within the day of year constraint
    dim_order = ['ndays_in_gcm', 'ndays_in_obs']

    inds = (
        xr.apply_ufunc(
            np.argsort,
            rmse.where(mask),
            vectorize=True,
            input_core_dims=[['ndays_in_obs']],
            output_core_dims=[['ndays_in_obs']],
            dask='parallelized',
            output_dtypes=['int'],
            dask_gufunc_kwargs={'allow_rechunk': True},
        )
        .isel(ndays_in_obs=slice(0, n_analogs))
        .transpose(*dim_order)
        .compute()
    )

    # train one linear regression model per gcm slice
    X = X.stack(pixel_coarse=['lat', 'lon'])
    y = y.stack(pixel_coarse=['lat', 'lon'])

    lr_model = LinearRegression()
    # double check whether this is actually created (??)
    regridder = xe.Regridder(
        ds_obs_coarse.isel(time=0),
        ds_obs_fine.isel(time=0),
        "bilinear",
        extrap_method="nearest_s2d",
    )
    downscaled = []

    # make sure X and y are spatially contiguous
    for i in range(len(y)):
        # get input data
        ind = inds.isel(ndays_in_gcm=i).values
        xi = X.isel(ndays_in_obs=ind).transpose('pixel_coarse', 'ndays_in_obs')
        yi = y.isel(ndays_in_gcm=i)

        # fit model
        lr_model.fit(xi, yi)

        # construct prediction at the fine resolution
        residual = yi - lr_model.predict(xi)

        # check if residual is spatially contiguous
        interpolated_residual = regridder(residual.unstack('pixel_coarse'))
        fine_pred = (
            (ds_obs_fine.isel(time=ind).transpose('lat', 'lon', 'time') * lr_model.coef_).sum(
                dim='time'
            )
            + lr_model.intercept_
            + interpolated_residual
        )
        downscaled.append(fine_pred)

    downscaled = xr.concat(downscaled, dim='time').sortby('time')
    return downscaled


def maca_epoch_replacement(
    ds_gcm_fine: xr.Dataset,
    trend_coarse: xr.Dataset,
) -> xr.Dataset:

    regridder = xe.Regridder(
        trend_coarse.isel(time=0), ds_gcm_fine.isel(time=0), "bilinear", extrap_method="nearest_s2d"
    )

    trend_fine = regridder(trend_coarse)

    return ds_gcm_fine + trend_fine


def maca_flow(
    gcm: str,
    obs: str,
    scenario: str,
    variables: Union[List[str], str],
    historical_start: str,
    historical_end: str,
    future_start: str,
    future_end: str,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    member: Optional[str] = "r1i1p1f1",
    epoch_adjustment_kwargs: Optional[Dict[str, Any]] = None,
    bias_correction_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """
    gcm     : string. name of gcm model to use.
    obs     : string. name of observation data to use.
    scenario: string. name of the emission scenario to use (e.g. ssp370)
    """
    # load data: historical gcm, future gcm, obs

    # get params
    historical_period = slice(historical_start, historical_end)
    future_period = slice(future_start, future_end)

    # bounding box of downscaling region
    full_gcm, coarse_obs = maca_preprocess(
        historical_gcm=historical_gcm.sel(time=historical_period),
        future_gcm=future_gcm.sel(time=future_period),
        obs=obs,
        min_lon=min_lon,
        max_lon=max_lon,
        min_lat=min_lat,
        max_lat=max_lat,
    )

    epoch_adjustment_kws = {'day_rolling_window': 21, 'year_rolling_window': 31}
    epoch_adjustment_kws.update({} if not epoch_adjustment_kwargs else epoch_adjustment_kwargs)

    # here, the time dimension of ea_gcm needs to be in 1 chunk
    ea_gcm, trend = epoch_adjustment(
        data=full_gcm, historical_period=historical_period, **epoch_adjustment_kws
    )

    # also need to be time: -1 chunked
    bias_correction_kws = {'batch_size': 15, 'buffer_size': 15}
    bias_correction_kws.update({} if not bias_correction_kwargs else bias_correction_kwargs)
    bc_ea_gcm = maca_bias_correction(
        ds_gcm=ea_gcm,
        ds_obs=coarse_obs,
        historical_period=historical_period,
        variables=variables,
        **bias_correction_kws
    )

    # rechunk into time 
    # split into sub region 
    # rechunk into space 
    constructed_analogs_kws = {'n_analogs': 10, 'doy_range': 45}
    constructed_analogs_kws.update(
        {} if not constructed_analogs_kwargs else constructed_analogs_kwargs
    )

    downscaled_bc_ea_gcm = maca_constructed_analogs(
        ds_gcm=bc_ea_gcm[variables],
        ds_obs_coarse=coarse_obs[variables],
        ds_obs_fine=obs[variables],
        **constructed_analogs_kws
    )

    # epoch replacement
    downscaled_bc_gcm = maca_epoch_replacement(
        ds_gcm_fine=downscaled_bc_ea_gcm,
        trend_coarse=trend,
    )

    # fine scale bias correction
    final_gcm = maca_bias_correction(
        ds_gcm=downscaled_bc_gcm,
        ds_obs=obs,
        historical_period=historical_period,
        variables=variables,
        **bias_correction_kws
    )

    return final_gcm
