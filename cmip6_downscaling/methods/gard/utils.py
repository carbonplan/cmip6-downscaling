from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from skdownscale.pointwise_models import AnalogRegression, PureAnalog, PureRegression

from ..common.containers import RunParameters


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
    # TODO: this is a temporary creation of random fields. ultimately we probably want to have
    # ~150 years of random fields, but this is fine.
    # DEFINITELY want to check whether this should be chunked differently - want it to align
    # with the chunks of the prediction
    scrf_ten_years = xr.open_zarr('az://static/scrf/ERA5_tasmax_1981_1990.zarr')
    scrf_list = []
    for year in np.arange(1981, 2110, 10):
        scrf_list.append(scrf_ten_years.drop('time'))
    scrf = xr.concat(scrf_list, dim='time')
    scrf['time'] = pd.date_range(start='1981-01-01', periods=scrf.dims['time'])
    scrf = scrf.sel(time=run_parameters.predict_period.time_slice)
    # TODO: confirm whether this breaks the distributed fashion?
    # TODO: check how are the scrfs chunked??
    # subset into the spatial domain
    scrf = scrf.sel(
        lon=run_parameters.bbox.lon_slice,
        lat=run_parameters.bbox.lat_slice,
    )
    scrf = scrf.drop('spatial_ref').astype('float32').chunk({'time': -1})
    return scrf
