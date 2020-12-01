import os
from typing import Iterable

import dask
import numba
import numpy as np
import pandas as pd
import xarray as xr

from cmip6_downscaling.disagg import derived_variables, terraclimate

force_vars = ['tmean', 'ppt', 'tmax', 'tmin', 'ws', 'tdew', 'srad']
extra_vars = ['vap', 'vpd', 'rh']
model_vars = ['aet', 'def', 'pdsi', 'pet', 'q', 'soil', 'swe']
aux_vars = ['awc', 'elevation', 'mask']


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
    ds_in = derived_variables.process(ds_in).load()

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

    return ds_out


def calc_valid_mask(ds: xr.Dataset) -> xr.Dataset:
    """ helper function to calculate a valid mask for given input variables """
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
    template = create_template(ds['ppt'], model_vars)

    # run the model using map_blocks
    ds_disagg_out = ds.map_blocks(run_terraclimate_model, template=template)

    # copy vars from input dataset to output dataset
    # for v in in_vars + extra_vars:
    #     ds_disagg_out[v] = ds_in[v]

    return ds_disagg_out
