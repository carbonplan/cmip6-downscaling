from typing import Any, Dict, List, Optional, Union

import gstools as gs
from scipy.stats import norm as norm
from skdownscale.pointwise_models import (
    AnalogRegression,
    PointWiseDownscaler,
    PureAnalog,
    PureRegression,
    QuantileMappingReressor,
    TrendAwareQuantileMappingRegressor,
)
from skdownscale.pointwise_models.core import xenumerate
from skdownscale.pointwise_models.utils import default_none_kwargs
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from cmip6_downscaling.data.observations import open_era5
from cmip6_downscaling.data.cmip import get_gcm_grid_spec
import xarray as xr


def get_coarse_obs(
    ds_obs: xr.Dataset, 
    gcm: str,
    connection_string: str,
    **kwargs
) -> xr.Dataset:
    """
    **kwargs are used to construct target file path
    """
    # Load single slice of target cmip6 dataset for target grid dimensions
    gcm_grid = load_cmip(
        source_ids=gcm,
        return_type='xr',
    ).isel(time=0)

    # rechunk and regrid observation dataset to target gcm resolution
    ds_obs_coarse = regrid_ds(
        ds=ds_obs,
        target_grid_ds=gcm_grid,
        connection_string=connection_string,
    )
    return ds_obs_coarse


def bias_correct_obs(
    da_obs: xr.DataArray,
    methods: str,
    bc_kwargs: Dict[str, Any],
) -> xr.DataArray:

    if method == 'quantile_transform':
        if 'n_quantiles' not in bc_kwargs:
            bc_kwargs['n_quantiles'] = len(da_obs)
        qt = PointWiseDownscaler(model=QuantileTransformer(**bc_kwargs))
        qt.fit(da_obs)
        obs_out = qt.transform(da_obs)

    elif method == 'z_score':
        # transform obs
        sc = PointWiseDownscaler(model=StandardScaler(**bc_kwargs))
        sc.fit(da_obs)
        obs_out = sc.transform(da_obs)

    elif method in ['quantile_map', 'detrended_quantile_map', 'none']:
        obs_out = da_obs

    else:
        availalbe_methods = [
            'quantile_transform',
            'z_score',
            'quantile_map',
            'detrended_quantile_map',
            'none',
        ]
        raise NotImplementedError(f'bias correction method must be one of {availalbe_methods}')

    return obs_out


def bias_correct_gcm(
    da_gcm: xr.DataArray,
    da_obs: xr.DataArray,
    historical_period_start: str,
    historical_period_end: str,
    method: str,
    bc_kwargs: Dict[str, Any],
) -> xr.DataArray:
    historical_period = slice(historical_period_start, historical_period_end)

    if method == 'quantile_transform':
        # transform gcm
        if 'n_quantiles' not in bc_kws:
            bc_kws['n_quantiles'] = len(da_gcm.sel(time=historical_period))
        qt = PointWiseDownscaler(model=QuantileTransformer(**bc_kwargs))
        qt.fit(da_gcm.sel(time=historical_period))
        gcm_out = qt.transform(da_gcm)

    elif method == 'z_score':
        # transform gcm
        sc = PointWiseDownscaler(model=StandardScaler(**bc_kwargs))
        sc.fit(da_gcm.sel(time=historical_period))
        gcm_out = sc.transform(da_gcm)

    elif method == 'quantile_map':
        qm = PointWiseDownscaler(model=QuantileMappingReressor(**bc_kwargs), dim='time')
        qm.fit(da_gcm.sel(time=historical_period), da_obs)
        gcm_out = qm.predict(da_gcm)

    elif self.bias_correction_method == 'detrended_quantile_map':
        qm = PointWiseDownscaler(
            TrendAwareQuantileMappingRegressor(QuantileMappingReressor(**bc_kwargs))
        )
        qm.fit(da_gcm.sel(time=historical_period), da_obs)
        gcm_out = qm.predict(da_gcm)

    elif method == 'none':
        gcm_out = da_gcm

    else:
        availalbe_methods = [
            'quantile_transform',
            'z_score',
            'quantile_map',
            'detrended_quantile_map',
            'none',
        ]
        raise NotImplementedError(f'bias correction method must be one of {availalbe_methods}')

    return gcm_out


def gard_bias_correction(
    ds_obs: xr.Dataset,
    ds_gcm: xr.Dataset,
    train_period_start: str,
    train_period_end: str,
    methods: Union[Dict[str, str], str],
    bias_correction_kwargs: Dict[str, Dict[str, Any]],
) -> xr.Dataset:
    """
    Bias correct or otherwise preprocess input gcm and obs data
    """
    ds_gcm_out, ds_obs_out = xr.Dataset(), xr.Dataset()
    for v in variables:
        bc_kws = default_none_kwargs(bias_correction_kwargs.get(v, {}), copy=True)
        ds_gcm_out[v], ds_obs_out[v] = bias_correction_by_var(
            da_gcm=ds_gcm[v],
            da_obs=ds_obs[v],
            historical_period=historical_period,
            method=methods.get(v, 'none'),
            bc_kwargs=bc_kws,
        )
    return ds_gcm_out, ds_obs_out


def get_gard_model(
    model_type: str,
    model_params: Dict[str, Any],
) -> Union[AnalogRegression, PureAnalog, PureRegression]:
    if model_type == 'AnalogRegression':
        return AnalogRegression(**model_params)
    elif model_type == 'PureAnalog':
        return PureAnalog(**model_params)
    elif model_type == 'PureRegression':
        return PureRegression(**model_params)
    else:
        raise NotImplementedError(
            'model_type must be AnalogRegression, PureAnalog, or PureRegression'
        )


def gard_fit_and_predict(
    ds_obs_interpolated: xr.Dataset,
    ds_obs: xr.Dataset,
    ds_gcm_interpolated: xr.Dataset,
    variable: str,
    model_type: str,
    model_params: Dict[str, Any],
    dim: str = 'time',
) -> xr.Dataset:

    # point wise downscaling
    model = PointWiseDownscaler(model=get_gard_model(model_type, model_params), dim=dim)
    model.fit(ds_obs_interpolated, ds_obs[variable])

    out = xr.Dataset()
    out['pred'] = model.predict(ds_gcm_interpolated)
    # TODO: fix syntax here
    out['error'] = model.predict(ds_gcm_interpolated, return_errors=True)
    out['exceedance_prob'] = model.predict(ds_gcm_interpolated, return_exceedance_prob=True)

    return out


def calc_correlation_length_scale(
    da: xr.DataArray,
    seasonality_period: int = 31,
    temporal_scaler: float = 1000.0,
) -> Dict[str, float]:
    """
    find the correlation length for a dataarray with dimensions lat, lon and time
    it is assumed that the correlation length in lat and lon directions are the same due to the implementation in gstools
    """
    for dim in ['lat', 'lon', 'time']:
        assert dim in da

    # remove seasonality before finding correlation length, otherwise the seasonality correlation dominates
    seasonality = (
        data.rolling({'time': seasonality_period}, center=True, min_periods=1)
        .mean()
        .groupby('time.dayofyear')
        .mean()
    )
    detrended = data.groupby("time.dayofyear") - seasonality

    # find spatial length scale
    bin_center, gamma = gs.vario_estimate(
        # TODO: need to figure whether we need to do .values for this function
        pos=(detrended.lon.values, detrended.lat.values),
        field=detrended.values,
        latlon=True,
        mesh_type='structured',
    )
    spatial = gs.Gaussian(dim=2, latlon=True, rescale=gs.EARTH_RADIUS)
    spatial.fit_variogram(bin_center, gamma, sill=np.mean(np.var(fields, axis=(1, 2))))

    # find temporal length scale
    # break the time series into fields of 1 year length, since otherwise the algorithm struggles to find the correct length scale
    fields = []
    day_in_year = 365
    for yr, group in detrended.groupby('time.year'):
        # TODO: this is a very long list for a large domain, perhaps need random sampling
        v = (
            group.isel(time=slice(0, day_in_year))
            .stack(point=['lat', 'lon'])
            .transpose('point', 'time')
            .values
        )
        fields.extend(list(v))
    t = np.arange(day_in_year) / temporal_scaler
    bin_center, gamma = gs.vario_estimate(pos=t, field=fields, mesh_type='structured')
    temporal = gs.Gaussian(dim=1)
    temporal.fit_variogram(bin_center, gamma, sill=np.mean(np.var(fields, axis=1)))

    return {'temporal': temporal.len_scale, 'spatial': spatial.len_scale}


def generate_scrf(
    source_da: xr.DataArray,
    output_template: xr.DataArray,
    seasonality_period: int = 31,
    seed: int = 0,
    temporal_scaler: float = 1000.0,
    crs: str = 'ESRI:54008',
) -> xr.DataArray:
    # find correlation length from source data
    length_scale = calc_correlation_length_scale(
        da=source_da,
        seasonality_period=seasonality_period,
        temporal_scaler=temporal_scaler,
    )
    ss = length_scale['spatial']
    ts = length_scale['temporal']

    # reproject template into Sinusoidal projection
    # TODO: any better ones for preserving distance between two arbitrary points on the map?
    template = output_template.isel(time=0)
    if 'x' not in output_template:
        template = template.rename({'lon': 'x', 'lat': y})
    projected = template.isel(time=0).rio.write_crs('EPSG:4326').rio.reproject(crs)
    x = projected.x
    y = projected.y
    t = np.arange(len(output_template.time)) / temporal_scaler

    # model is specified as spatial_dim1, spatial_dim2, temporal_scale
    model = gs.Gaussian(dim=3, var=1.0, len_scale=[ss, ss, ts])
    srf = gs.SRF(model, seed=seed)

    # TODO: figure out how to chunk this
    field = xr.DataArray(
        srf.structured((x, y, t)), dims=['lon', 'lat', 'time'], coords=[x, y, output_template.time]
    ).rio.write_crs(crs)

    field = field.rio.reproject('EPSG:4326')
    return field


def gard_postprocess(
    ds_obs: xr.Dataset,
    variable: str,
    model_output: xr.Dataset,
    thresh: Union[float, None],
    seasonality_period: int = 31,
    seed: int = 0,
    temporal_scaler: float = 1000.0,
    crs: str = 'ESRI:54008',
) -> xr.DataArray:
    # generate spatio-tempprally correlated random fields based on observation data
    scrf = generate_scrf(data=ds_obs[variable], template=model_output['pred'], seed=seed)

    if thresh is not None:
        # convert scrf from a normal distribution to a uniform distribution
        scrf_uniform = xr.apply_ufunc(
            norm.cdf, scrf, dask='parallelized', output_dtypes=[scrf.dtype]
        )

        # find where exceedance prob is exceeded
        mask = scrf_uniform > (1 - model_output['exceedance_prob'])

        # Rescale the uniform distribution
        new_uniform = (scrf_uniform - (1 - model_output['exceedance_prob'])) / model_output[
            'exceedance_prob'
        ]

        # Get the normal distribution equivalent of new_uniform
        r_normal = xr.apply_ufunc(
            norm.ppf, new_uniform, dask='parallelized', output_dtypes=[new_uniform.dtype]
        )

        downscaled = model_output['pred'] + r_normal * model_output['error']

        # what do we do for thresholds like heat wave?
        valids = xr.ufuncs.logical_or(mask, downscaled >= 0)
        downscaled = downscaled.where(valids, 0)
    else:
        downscaled = model_output['pred'] + scrf * model_output['error']

    return downscaled
