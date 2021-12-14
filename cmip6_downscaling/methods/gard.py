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
