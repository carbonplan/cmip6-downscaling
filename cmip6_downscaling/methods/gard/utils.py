from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from scipy.special import cbrt
from scipy.stats import norm as norm
from skdownscale.pointwise_models import AnalogRegression, PureAnalog, PureRegression

from ..common.containers import RunParameters

xr.set_options(keep_attrs=True)


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


def add_random_effects(
    model_output: xr.Dataset, scrf: xr.DataArray, run_parameters: RunParameters
) -> xr.Dataset:
    if run_parameters.model_params is not None:
        thresh = run_parameters.model_params.get('thresh')
    else:
        thresh = None

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
        if run_parameters.variable == 'pr':
            downscaled = (
                cbrt(model_output['pred']) + (model_output['prediction_error'] * r_normal)
            ) ** 3
        else:
            downscaled = model_output['pred'] + r_normal * model_output['prediction_error']

        # what do we do for thresholds like heat wave?
        valids = np.logical_or(mask, downscaled >= 0)
        downscaled = downscaled.where(valids, 0)
        downscaled = downscaled.where(downscaled >= 0, 0)

    else:
        downscaled = model_output['pred'] + scrf * model_output['prediction_error']

    return downscaled.to_dataset(name=run_parameters.variable)
