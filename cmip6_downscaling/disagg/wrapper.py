from __future__ import annotations

import os
from typing import Iterable

import dask
import numba
import numpy as np
import pandas as pd
import xarray as xr

from ..disagg import derived_variables, terraclimate

# minimum set of input variables
input_vars = ['ppt', 'tmax', 'tmin', 'ws', 'srad']

# variables required by terraclimate
force_vars = input_vars + ['tmean', 'tdew']

# derived variables to be included in the output
derived_vars = ['rh', 'tdew', 'tmean', 'vap', 'vpd']

# variables calculated by terraclimate
model_vars = ['aet', 'def', 'pdsi', 'pet', 'q', 'soil', 'swe']

# aux variables required by wrapper (and/or terraclimate)
aux_vars = ['awc', 'elevation', 'mask']

# variables returned by the wrapper function
wrapper_vars = derived_vars + model_vars


# set threading options
def _set_thread_settings():
    """helper function to disable numba and openmp multi-threading"""
    numba.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'


_set_thread_settings()


def create_template(
    like_da: xr.DataArray, var_names: Iterable[str], fill_val: float = np.nan
) -> xr.Dataset:
    """Create an empty dataset with the given variables in it

    Parameters
    ----------
    like_da : xr.DataArray
        Template variable.
    var_names : list-like
        List of variable names

    Returns
    -------
    ds : xr.Dataset
        Template dataset
    """

    ds = xr.Dataset()
    for v in var_names:
        ds[v] = xr.full_like(like_da, fill_val, dtype=np.float32)
    return ds


def run_terraclimate_model(ds_in: xr.Dataset) -> xr.Dataset:
    """Run the terraclimate model over all x/y locations in ds

    Parameters
    ----------
    ds_in : xr.Dataset
        Input dataset. Must include the following variables: {awc, elevation, lat, mask, ppt, tdew, tmax, tmean, tmin, and ws}

    Returns
    -------
    ds_out : xr.Dataset
        Output dataset, includes the follwoing variables: {aet, def, pdsi, pet, q, soil, swe}
    """
    _set_thread_settings()

    ds_in = calc_valid_mask(ds_in)

    # derive physical quantities
    ds_in = derived_variables.process(ds_in)

    ds_out = create_template(ds_in['ppt'], model_vars)

    df_point = pd.DataFrame(
        index=ds_in.indexes['time'], columns=force_vars + model_vars, dtype=np.float32
    )

    with dask.config.set(scheduler='single-threaded'):

        for index, mask_val in np.ndenumerate(ds_in['mask'].values):
            if not mask_val:
                # skip values outside the mask
                continue
            y, x = index

            # extract aux vars
            awc = ds_in['awc'].values[y, x]
            elev = ds_in['elevation'].values[y, x]
            lat = ds_in['lat'].values[y, x]

            # extract forcing variables
            for v in force_vars:
                df_point[v] = ds_in[v].values[:, y, x]

            # run terraclimate model
            df_point = terraclimate.model(df_point, awc, lat, elev)

            # copy results to dataset
            for v in model_vars:
                ds_out[v].values[:, y, x] = df_point[v].to_numpy()

            df_point[model_vars] = np.nan

    for v in wrapper_vars:
        if v not in ds_out:
            ds_out[v] = ds_in[v]

    return ds_out


def calc_valid_mask(ds: xr.Dataset) -> xr.Dataset:
    """helper function to calculate a valid mask for given input variables"""
    # Temporary fix to correct for mismatched masks (along coasts)
    ds['mask'] = (
        ds['mask'].astype(bool)
        * ds['awc'].notnull()
        * ds['elevation'].notnull()
        * ds['ppt'].isel(time=-1).notnull()
    )
    return ds


def disagg(ds: xr.Dataset) -> xr.Dataset:
    """Execute the disaggregation routines

    Parameters
    ----------
    ds_in : xr.Dataset
        Input dataset. Must include the following variables: {awc, elevation, lat, mask, ppt, tdew, tmax, tmean, tmin, and ws}

    Returns
    -------
    ds_disagg_out : xr.Dataset
        Output dataset, includes the follwoing variables: {aet, def, pdsi, pet, q, soil, swe}
    """

    # create a template dataset that we can pass to map blocks
    template = create_template(ds['ppt'], wrapper_vars)

    # run the model using map_blocks
    ds_disagg_out = ds.map_blocks(run_terraclimate_model, template=template)

    # make sure the output dataset has all the input and aux variables in it
    for v in set(aux_vars + list(ds.data_vars)):
        if v not in ds_disagg_out:
            ds_disagg_out[v] = ds[v]

    return ds_disagg_out
