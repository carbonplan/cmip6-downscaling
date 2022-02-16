from typing import Any, Dict, Union

import xarray as xr
from skdownscale.pointwise_models import (
    PointWiseDownscaler,
    QuantileMappingReressor,
    TrendAwareQuantileMappingRegressor,
)
from sklearn.preprocessing import QuantileTransformer, StandardScaler

VALID_CORRECTIONS = ['absolute', 'relative']


def bias_correct_obs_by_method(
    da_obs: Union[xr.DataArray, xr.Dataset],
    method: str,
    bc_kwargs: Dict[str, Any],
) -> xr.DataArray:
    if method == 'quantile_transform':
        if 'n_quantiles' not in bc_kwargs:
            bc_kwargs['n_quantiles'] = len(da_obs.time)
        qt = PointWiseDownscaler(model=QuantileTransformer(**bc_kwargs))
        qt.fit(da_obs)
        return qt.transform(da_obs)

    elif method == 'z_score':
        # transform obs
        sc = PointWiseDownscaler(model=StandardScaler(**bc_kwargs))
        sc.fit(da_obs)
        return sc.transform(da_obs)

    elif method in ['quantile_map', 'detrended_quantile_map', 'none']:
        return da_obs

    else:
        availalbe_methods = [
            'quantile_transform',
            'z_score',
            'quantile_map',
            'detrended_quantile_map',
            'none',
        ]
        raise NotImplementedError(f'bias correction method must be one of {availalbe_methods}')


def bias_correct_gcm_by_method(
    da_gcm: Union[xr.DataArray, xr.Dataset],
    da_obs: Union[xr.DataArray, xr.Dataset],
    historical_period: slice,
    method: str,
    bc_kwargs: Dict[str, Any],
):
    if method == 'quantile_transform':
        # transform gcm
        if 'n_quantiles' not in bc_kwargs:
            bc_kwargs['n_quantiles'] = len(da_gcm.sel(time=historical_period).time)
        qt = PointWiseDownscaler(model=QuantileTransformer(**bc_kwargs))
        qt.fit(da_gcm.sel(time=historical_period))
        return qt.transform(da_gcm)

    elif method == 'z_score':
        # transform gcm
        sc = PointWiseDownscaler(model=StandardScaler(**bc_kwargs))
        sc.fit(da_gcm.sel(time=historical_period))
        return sc.transform(da_gcm)

    # TODO: test to see QuantileMappingReressor and TrendAwareQuantileMappingRegressor
    # can handle multiple variables at once
    elif method == 'quantile_map':
        qm = PointWiseDownscaler(model=QuantileMappingReressor(**bc_kwargs), dim='time')
        qm.fit(da_gcm.sel(time=historical_period), da_obs)
        return qm.predict(da_gcm)

    elif method == 'detrended_quantile_map':
        qm = PointWiseDownscaler(
            TrendAwareQuantileMappingRegressor(QuantileMappingReressor(**bc_kwargs))
        )
        qm.fit(da_gcm.sel(time=historical_period), da_obs)
        return qm.predict(da_gcm)

    elif method == 'none':
        return da_gcm

    else:
        availalbe_methods = [
            'quantile_transform',
            'z_score',
            'quantile_map',
            'detrended_quantile_map',
            'none',
        ]
        raise NotImplementedError(f'bias correction method must be one of {availalbe_methods}')
