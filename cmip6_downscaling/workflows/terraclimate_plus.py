#!/usr/bin/env python

import os

import numba
import numpy as np
import xarray as xr
import zarr
from carbonplan.data import cat

from cmip6_downscaling.disagg import derived_variables, terraclimate
from cmip6_downscaling.workflows.utils import get_store

# from skdownscale.pointwise_models.core import xenumerate


target = 'obs/conus/monthly/4000m/terraclimate_plus.zarr'

xy_region = {'x': slice(200, 210), 'y': slice(200, 210)}
force_vars = ['tmean', 'ppt', 'tmax', 'tmin', 'ws', 'tdew', 'srad']
extra_vars = ['vap', 'vpd', 'rh']
model_vars = ['aet', 'def', 'pdsi', 'pet', 'q', 'soil', 'swe']
aux_vars = ['awc', 'elevation', 'mask']
out_vars = force_vars + extra_vars + model_vars


# set threading options
def _set_thread_settings():
    numba.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'


_set_thread_settings()


def create_template(like_da, var_names):
    ds = xr.Dataset()
    for v in var_names:
        ds[v] = xr.full_like(like_da, np.nan)
    return ds


def run_terraclimate_model(ds_in):
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

        # run terraclimate model
        df_point = ds_in[force_vars].isel(y=y, x=x).to_dataframe()
        df_out = terraclimate.model(df_point, awc, lat, elev)

        # copy results to dataset
        for v in model_vars:
            ds_out[v].data[:, y, x] = df_out[v].values

    return ds_out


def preprocess(ds):
    ds_in = ds.copy()

    # make sure coords are all pre-loaded
    ds_in['mask'] = ds_in['mask'].load()
    ds_in['lon'] = ds_in['lon'].load()
    ds_in['lat'] = ds_in['lat'].load()
    for v in ['lon', 'lat']:
        if 'chunks' in ds_in[v].encoding:
            del ds_in[v].encoding['chunks']
    return ds_in


def disagg(ds):
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

    return ds_disagg_out[out_vars]


if __name__ == '__main__':
    # client = Client(n_workers=4, threads_per_worker=1)
    # print(client)
    # print(client.dashboard_link)

    # open terraclimate data
    # rechunked version
    mapper = zarr.storage.ABSStore(
        'carbonplan-scratch',
        prefix='rechunker/terraclimate/target.zarr/',
        account_name="carbonplan",
        account_key=os.environ["BLOB_ACCOUNT_KEY"],
    )

    # open grid dataset
    ds_grid = cat.grids.conus4k.to_dask()

    # open terraclimate data
    # todo: pull aux fields from rechunker version
    ds_aux = cat.terraclimate.terraclimate.to_dask()
    ds_in = xr.open_zarr(mapper, consolidated=True)

    ds_in['mask'] = ds_grid['mask']
    ds_in['awc'] = ds_aux['awc']
    ds_in['elevation'] = ds_aux['elevation']

    if xy_region:
        ds_in = ds_in.isel(**xy_region)

    # do the disaggregation
    ds_out = disagg(ds_in)

    print(ds_out.load())

    store = get_store(target)
    store.clear()

    write = ds_out.to_zarr(store, compute=False, mode='w')
    write.compute(retries=1)
    zarr.consolidate_metadata(store)
