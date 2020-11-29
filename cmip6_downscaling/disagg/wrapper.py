import os
from typing import Iterable

import numba
import numpy as np
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


def create_template(like_da: xr.DataArray, var_names: Iterable[str]) -> xr.Dataset:
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
        ds[v] = xr.full_like(like_da, np.nan, dtype=np.float32)
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

    ds_out = create_template(ds_in['ppt'], model_vars)

    for index, mask_val in np.ndenumerate(ds_in['mask'].data):
        y, x = index
        if not mask_val:
            # skip values outside the mask
            continue

        # extract aux vars
        awc = ds_in['awc'][index].data
        elev = ds_in['elevation'][index].data
        lat = ds_in['lat'][index].data

        if awc <= 0 or np.isnan(awc):
            print(f'invalid awc value {awc} for this point {ds_in.isel(x=x, y=y)}')
            continue
        if elev <= -420 or np.isnan(elev):
            print(f'invalid elev value {elev} for this point {ds_in.isel(x=x, y=y)}')
            continue

        # run terraclimate model
        df_point = ds_in[force_vars].isel(y=y, x=x).to_dataframe()
        df_out = terraclimate.model(df_point, awc, lat, elev)

        # copy results to dataset
        for v in model_vars:
            ds_out[v].data[:, y, x] = df_out[v].values

    return ds_out


def preprocess(ds: xr.Dataset) -> xr.Dataset:
    """ helper function to preprocess input dataset """
    ds_in = ds.copy()

    # make sure coords are all pre-loaded
    ds_in['mask'] = ds_in['mask'].load()
    ds_in['lon'] = ds_in['lon'].load()
    ds_in['lat'] = ds_in['lat'].load()
    for v in ['lon', 'lat']:
        if 'chunks' in ds_in[v].encoding:
            del ds_in[v].encoding['chunks']
    return ds_in


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
    # cleanup dataset and load coordinates
    ds_in = preprocess(ds)

    # derive physical quantities
    ds_in = derived_variables.process(ds_in)

    # create a template dataset that we can pass to map blocks
    template = create_template(ds_in['ppt'], model_vars)

    # run the model using map_blocks
    in_vars = force_vars + aux_vars
    ds_disagg_out = ds_in[in_vars].map_blocks(run_terraclimate_model, template=template)

    # copy vars from input dataset to output dataset
    for v in in_vars + extra_vars:
        ds_disagg_out[v] = ds_in[v]

    return ds_disagg_out
