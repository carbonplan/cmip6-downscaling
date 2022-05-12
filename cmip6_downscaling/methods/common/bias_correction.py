from __future__ import annotations

from typing import Any

import xarray as xr
from skdownscale.pointwise_models import (
    PointWiseDownscaler,
    QuantileMapper,
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
            # Note: scikit-learn's quantile transform is very slow for n_quantiles > 2000.
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
        available_methods = [
            'quantile_transform',
            'z_score',
            'quantile_map',
            'detrended_quantile_map',
            'none',
        ]
        raise NotImplementedError(f'bias correction method must be one of {available_methods}')


def bias_correct_gcm_by_method(
    gcm_pred: xr.DataArray | xr.Dataset,
    method: str,
    bc_kwargs: dict[str, Any],
    obs: xr.DataArray | xr.Dataset = None,
    gcm_hist: xr.DataArray | xr.Dataset = None,
):
    if method == 'quantile_transform':
        # transform gcm
        if 'n_quantiles' not in bc_kwargs:
            bc_kwargs['n_quantiles'] = len(obs.time)

        qt = PointWiseDownscaler(model=QuantileTransformer(**bc_kwargs))
        qt.fit(gcm_hist)
        return qt.transform(gcm_pred)
    elif method == 'cunnane_transform':
        return bc_kwargs['transformer_interp'].transform(gcm_pred)

    elif method == 'z_score':
        # transform gcm
        sc = PointWiseDownscaler(model=StandardScaler(**bc_kwargs))
        sc.fit(gcm_hist)
        return sc.transform(gcm_pred)
    elif method == 'quantile_mapper':
        qm = PointWiseDownscaler(model=QuantileMapper(detrend=True), dim='time')
        qm.fit(obs)
        return qm.transform(gcm_pred)

    # TODO: test to see QuantileMappingReressor and TrendAwareQuantileMappingRegressor
    # can handle multiple variables at once
    elif method == 'quantile_map':
        qm = PointWiseDownscaler(model=QuantileMappingReressor(**bc_kwargs), dim='time')
        qm.fit(gcm_hist, obs)
        return qm.predict(gcm_pred)

    elif method == 'detrended_quantile_map':
        qm = PointWiseDownscaler(
            TrendAwareQuantileMappingRegressor(QuantileMappingReressor(**bc_kwargs))
        )
        qm.fit(gcm_hist, obs)
        return qm.predict(gcm_pred)

    elif method == 'none':
        return gcm_pred

    else:
        available_methods = [
            'quantile_transform',
            'z_score',
            'quantile_map',
            'detrended_quantile_map',
            'none',
            'quantile_mapper',
            'cunnane_transform',
        ]
        raise NotImplementedError(f'bias correction method must be one of {available_methods}')
