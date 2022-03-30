from __future__ import annotations

import math
from typing import Any

import fsspec
import numpy as np
import xarray as xr
from scipy.stats import norm as norm
from skdownscale.pointwise_models import AnalogRegression, PureAnalog, PureRegression

from .containers import RunParameters


def get_gard_model(
    model_type: str,
    model_params: dict[str, Any],
) -> AnalogRegression | PureAnalog | PureRegression:
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


def make_scrf_path(
    obs: str,
    label: str,
    start_year: str,
    end_year: str,
    **kwargs,
):
    """
    Path where spatially-temporally correlated random fields (SCRF) are saved.

    Parameters
    ----------
    obs : str
        Name of observation dataset
    label: str
        The variable being predicted
    start_year: str
        Start year of the dataset
    end_year: str
        End year of the dataset

    Returns
    -------
    scrf_path : str
        Path of SCRF
    """
    return f"scrf/{obs}_{label}_{start_year}_{end_year}.zarr"


def read_scrf(run_parameters: RunParameters):
    """
    Read spatial-temporally correlated random fields on file and subset into the correct spatial/temporal domain according to model_output.
    The random fields are stored in decade (10 year) long time series for the global domain and pre-generated using `scrf.ipynb`.

    Parameters
    ----------
    run_parameters : RunParameters


    Returns
    -------
    scrf : xr.DataArray
        Spatio-temporally correlated random fields (SCRF)
    """

    def get_decade_start_year(year):
        # decades start in xxx1 and end in xxx0 (e.g. 1991-2000)
        return math.floor((int(year) - 1) / 10.0) * 10 + 1

    # TODO: put this somewhere else - run_config maybe? or run_parameters?
    scrf_storage = 'az://flow-outputs/intermediate/'

    train_period_start = run_parameters.train_period.time_slice.start
    train_period_end = run_parameters.train_period.time_slice.stop
    predict_period_start = run_parameters.predict_period.time_slice.start
    predict_period_end = run_parameters.predict_period.time_slice.stop

    # first find out which decades of random fields we'd need to load
    train_start_decade = get_decade_start_year(train_period_start)
    train_end_decade = get_decade_start_year(train_period_end)
    predict_start_decade = get_decade_start_year(predict_period_start)
    predict_end_decade = get_decade_start_year(predict_period_end)

    training_decades = np.arange(train_start_decade, train_end_decade + 1, 10)
    predict_decades = np.arange(predict_start_decade, predict_end_decade + 1, 10)
    all_decades = list(set(list(training_decades) + list(predict_decades)))

    # load all the random field data
    scrf = []
    for decade_start in sorted(all_decades):
        start_year = str(int(decade_start))
        end_year = str(int(decade_start + 9))
        scrf_path = make_scrf_path(
            obs=run_parameters.obs,
            label=run_parameters.variable,
            start_year=start_year,
            end_year=end_year,
        )
        mapper = fsspec.get_mapper(scrf_storage + scrf_path)
        scrf.append(xr.open_zarr(mapper))
    # TODO: confirm whether this breaks the distributed fashion?
    # TODO: check how are the scrfs chunked??
    scrf = xr.combine_by_coords(scrf, combine_attrs='drop_conflicts')

    # subset into the spatial domain
    scrf = scrf.sel(
        lon=run_parameters.bbox.lon_slice,
        lat=run_parameters.bbox.lat_slice,
    )

    # subset into the temporal period
    # TODO: check whether this works when we change the prediction period
    # we have two main use cases - the prediction period being 1950-2014 and 2015-2099, both
    # with train_period as 1981-2010 - this feels like a spot where there could be inefficiency
    historical = scrf.sel(time=run_parameters.train_period)
    future = scrf.sel(time=run_parameters.predict_period)
    scrf = xr.combine_by_coords([historical, future], combine_attrs='drop_conflicts')
    # TODO: check here in particular for inefficiency
    scrf = scrf.reindex(time=sorted(scrf.time.values))
    # TODO: what? why do we only return the future scrf after doing the whole scrf step above this line?
    # confirm that we don't want the full scrf
    return future.scrf


def gard_postprocess(
    model_output: xr.Dataset,
    scrf: xr.DataArray,
    label: str,
    model_params: dict[str, Any] | None = None,
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
    scrf : xr.DataArray
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

    ## CURRENTLY needs calendar to be gregorian
    ## TODO: merge in the calendar conversion for GCMs and this should work great!
    assert len(scrf.time) == len(model_output.time)
    assert len(scrf.lat) == len(model_output.lat)
    assert len(scrf.lon) == len(model_output.lon)

    scrf = scrf.assign_coords(
        {'lat': model_output.lat, 'lon': model_output.lon, 'time': model_output.time}
    )

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
    downscaled = downscaled.chunk({'time': 365, 'lat': 150, 'lon': 150})
    return downscaled.to_dataset(name=label)
