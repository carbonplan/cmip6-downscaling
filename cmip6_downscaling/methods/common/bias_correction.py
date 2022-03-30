from __future__ import annotations

from typing import Any

import xarray as xr
from skdownscale.pointwise_models import (
    PointWiseDownscaler,
    QuantileMappingReressor,
    TrendAwareQuantileMappingRegressor,
)
from sklearn.preprocessing import QuantileTransformer, StandardScaler

VALID_CORRECTIONS = ['absolute', 'relative']


def bias_correct_obs_by_method(
    da_obs: xr.DataArray | xr.Dataset,
    method: str,
    bc_kwargs: dict[str, Any],
) -> xr.DataArray:
    if method == 'quantile_transform':
        if 'n_quantiles' not in bc_kwargs:
            bc_kwargs['n_quantiles'] = len(da_obs)
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
        available_methods = [
            'quantile_transform',
            'z_score',
            'quantile_map',
            'detrended_quantile_map',
            'none',
        ]
        raise NotImplementedError(f'bias correction method must be one of {available_methods}')


def bias_correct_gcm_by_method(
    gcm_train: xr.DataArray | xr.Dataset,
    obs_train: xr.DataArray | xr.Dataset,
    gcm_predict: xr.DataArray | xr.Dataset,
    method: str,
    bc_kwargs: dict[str, Any],
):
    if method == 'quantile_transform':
        # transform gcm
        if 'n_quantiles' not in bc_kwargs:
            bc_kwargs['n_quantiles'] = len(gcm_train)
        qt = PointWiseDownscaler(model=QuantileTransformer(**bc_kwargs))
        qt.fit(gcm_train)
        return qt.transform(gcm_predict)

    elif method == 'z_score':
        # transform gcm
        sc = PointWiseDownscaler(model=StandardScaler(**bc_kwargs))
        sc.fit(gcm_train)
        return sc.transform(gcm_predict)

    # TODO: test to see QuantileMappingReressor and TrendAwareQuantileMappingRegressor
    # can handle multiple variables at once
    elif method == 'quantile_map':
        qm = PointWiseDownscaler(model=QuantileMappingReressor(**bc_kwargs), dim='time')
        qm.fit(gcm_train, obs_train)
        return qm.predict(gcm_predict)

    elif method == 'detrended_quantile_map':
        qm = PointWiseDownscaler(
            TrendAwareQuantileMappingRegressor(QuantileMappingReressor(**bc_kwargs))
        )
        qm.fit(gcm_train, obs_train)
        return qm.predict(gcm_predict)

    elif method == 'none':
        return gcm_predict

    else:
        available_methods = [
            'quantile_transform',
            'z_score',
            'quantile_map',
            'detrended_quantile_map',
            'none',
        ]
        raise NotImplementedError(f'bias correction method must be one of {available_methods}')
