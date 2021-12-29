from typing import Any, Dict, Optional, Union

import gstools as gs
import numpy as np
import xarray as xr
from scipy.stats import norm as norm
from skdownscale.pointwise_models import (
    AnalogRegression,
    PointWiseDownscaler,
    PureAnalog,
    PureRegression,
)


def get_gard_model(
    model_type: str,
    model_params: Dict[str, Any],
) -> Union[AnalogRegression, PureAnalog, PureRegression]:
    """
    Based on input, return the corresponding GARD model instance

    Parameters
    ----------
    model_type : str
        Name of the GARD model type to be used, should be one of AnalogRegression, PureAnalog, or PureRegression
    model_params : Dict
        Model parameter dictionary

    Returns
    -------
    model : AnalogRegression, PureAnalog, or PureRegression model instance
        skdownscale GARD model instance
    """
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
    X_train: xr.Dataset,
    y_train: xr.Dataset,
    X_pred: xr.Dataset,
    label: str,
    model_type: str,
    model_params: Dict[str, Any],
    dim: str = 'time',
    **kwargs,
) -> xr.Dataset:
    """
    Fit a GARD model for each point in the input training data then return the prediction using predict input

    Parameters
    ----------
    X_train : xr.Dataset
        Training features dataset
    y_train : xr.Dataset
        Training label dataset
    X_pred : xr.Dataset
        Prediction feature dataset
    label : str
        Name of the variable to be predicted
    model_type : str
        Name of the GARD model type to be used, should be one of AnalogRegression, PureAnalog, or PureRegression
    model_params : Dict
        Model parameter dictionary
    dim : str, optional
        Dimension to apply the model along. Default is ``time``.

    Returns
    -------
    output : xr.Dataset
        GARD model prediction output. Should contain three variables: pred (predicted mean), prediction_error
        (prediction error in fit), and exceedance_prob (probability of exceedance for threshold)
    """
    # point wise downscaling
    model = PointWiseDownscaler(model=get_gard_model(model_type, model_params), dim=dim)
    model.fit(X_train, y_train[label])

    out = xr.Dataset()
    out['pred'] = model.predict(X_pred)
    out['prediction_error'] = model.get_attr(
        'prediction_error_', dtype='float64', template_output=out['pred']
    )
    out['exceedance_prob'] = model.get_attr(
        'exceedance_prob_', dtype='float64', template_output=out['pred']
    )

    return out


def calc_correlation_length_scale(
    data: xr.DataArray,
    seasonality_period: int = 31,
    temporal_scaler: float = 1000.0,
) -> Dict[str, float]:
    """
    find the correlation length for a dataarray with dimensions lat, lon and time
    it is assumed that the correlation length in lat and lon directions are the same due to the implementation in gstools
    """
    for dim in ['lat', 'lon', 'time']:
        assert dim in data.dims

    # remove seasonality before finding correlation length, otherwise the seasonality correlation dominates
    seasonality = (
        data.rolling({'time': seasonality_period}, center=True, min_periods=1)
        .mean()
        .groupby('time.dayofyear')
        .mean()
    )
    detrended = data.groupby("time.dayofyear") - seasonality
    detrended = detrended.transpose('time', 'lon', 'lat')

    # find spatial length scale
    fields = detrended.values
    bin_center, gamma = gs.vario_estimate(
        # TODO: need to figure whether we need to do .values for this function
        pos=(detrended.lon.values, detrended.lat.values),
        field=fields,
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
    data: xr.Dataset,
    label: str,
    n_timepoints: int = 365,
    seasonality_period: int = 31,
    seed: int = 0,
    temporal_scaler: float = 1000.0,
    crs: str = 'ESRI:54008',
    **kwargs,
) -> xr.DataArray:
    """
    Generate spatio-temporally correlated random fields (SCRF) based on the input data.

    Parameters
    ----------
    data : xr.Dataset
        Data used to define the spatial and temporal correlation lengths that are later used to generate the random fields
    label : str
        Name of the variable to be predicted
    n_timepoints : int, optional
        Number of timepoints to return in generated random fields
    seasonality_period : int, optional
        The period to get rolling average from in order to remove seasonality
    seed : int, optional
        Random seed
    temporal_scaler : float, optional
        Value to be used to scale the temporal aspect of the data. Used because it's more difficult for the variogram estimate
        to converge when the time series is too long and thus the scale is too large
    crs : str, optional
        The projection in which the SCRF will be first generated before being projected back to lat/lon space

    Returns
    -------
    scrf : xr.Dataset
        Spatio-temporally correlated random fields (SCRF) based on the input data.
    """
    # find correlation length from source data
    length_scale = calc_correlation_length_scale(
        data=data[label].isel(lat=slice(100, 110), lon=slice(100, 110)),
        seasonality_period=seasonality_period,
        temporal_scaler=temporal_scaler,
    )
    ss = length_scale['spatial']
    ts = length_scale['temporal']

    # reproject template into Sinusoidal projection
    # TODO: any better ones for preserving distance between two arbitrary points on the map?
    if 'x' not in data[label].dims:
        template = data[label].rename({'lon': 'x', 'lat': 'y'})
    projected = template.isel(time=0).rio.write_crs('EPSG:4326').rio.reproject(crs)
    x = projected.x
    y = projected.y
    t = np.arange(n_timepoints) / temporal_scaler

    # model is specified as spatial_dim1, spatial_dim2, temporal_scale
    model = gs.Gaussian(dim=3, var=1.0, len_scale=[ss, ss, ts])
    srf = gs.SRF(model, seed=seed)

    print('putting into dataarray')
    # TODO: figure out how to chunk this
    field = xr.DataArray(
        srf.structured((x, y, t)),
        dims=['lon', 'lat', 'time'],
        coords=[x, y, np.arange(n_timepoints)],
    ).rio.write_crs(crs)

    print('reprojecting')
    field = field.transpose('time', 'lat', 'lon').rio.reproject('EPSG:4326')
    return field.to_dataset(name='scrf')


def gard_postprocess(
    model_output: xr.Dataset,
    scrf: xr.Dataset,
    model_params: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> xr.Dataset:
    """
    Add perturbation to the mean prediction of GARD to more accurately represent extreme events. The perturbation is
    generated with the prediction error during model fit scaled with a spatio-temporally correlated random field.

    Parameters
    ----------
    model_output : xr.Dataset
        GARD model prediction output. Should contain three variables: pred (predicted mean), prediction_error
        (prediction error in fit), and exceedance_prob (probability of exceedance for threshold)
    scrf : xr.Dataset
        Spatio-temporally correlated random fields (SCRF)
    model_params : Dict
        Model parameter dictionary

    Returns
    -------
    downscaled : xr.Dataset
        Final downscaled output
    """
    if model_params is not None:
        thresh = model_params.get('thresh')
    else:
        thresh = None

    # trim the scrf dataset to the length of the model output
    n_timepoints = len(model_output.time)
    scrf = scrf.isel(time=slice(0, n_timepoints))

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

        downscaled = model_output['pred'] + r_normal * model_output['prediction_error']

        # what do we do for thresholds like heat wave?
        valids = xr.ufuncs.logical_or(mask, downscaled >= 0)
        downscaled = downscaled.where(valids, 0)
    else:
        downscaled = model_output['pred'] + scrf * model_output['prediction_error']

    return downscaled.to_dataset(name='downscaled')
