#!/usr/bin/env python

import os
from typing import Dict

import dask
import numpy as np
import xarray as xr
import zarr
from carbonplan.data import cat

from cmip6_downscaling.disagg.wrapper import create_template, run_terraclimate_model
from cmip6_downscaling.workflows.share import (
    awc_fill,
    chunks,
    finish_store,
    get_regions,
    load_coords,
    maybe_slice_region,
)
from cmip6_downscaling.workflows.utils import get_store

# from dask_gateway import Gateway


out_vars = [
    'aet',
    'pdsi',
    'pet',
]
force_vars = ['tmax', 'tmin', 'srad', 'ppt', 'vap', 'ws']
aux_vars = ['mask', 'awc', 'elevation']
in_vars = force_vars + aux_vars

target = 'obs/conus/4000m/monthly/terraclimate_plus.zarr'


def get_out_mapper(account_key: str) -> zarr.storage.ABSStore:
    """Get output dataset mapper to Azure Blob store

    Parameters
    ----------
    account_key : str
        Secret key giving Zarr access to Azure store

    Returns
    -------
    mapper : zarr.storage.ABSStore
    """
    return get_store(target, account_key=account_key)


@dask.delayed(pure=False, traverse=False)
def block_wrapper(region: Dict, account_key: str):
    """Delayed wrapper function to run Terraclimate model over a single x/y chunk

    Parameters
    ----------
    region : dict
        Dictionary of slices defining a single chunk, e.g. {'x': slice(1150, 1200), 'y': slice(700, 750), 'time': slice(None)}
    account_key : str
        Secret key giving Zarr access to Azure store
    """

    print(region)
    with dask.config.set(scheduler='single-threaded'):
        ds_in = get_obs(region)
        ds_in = ds_in[in_vars].load()
        ds_out = run_terraclimate_model(ds_in)[out_vars]
        out_mapper = get_out_mapper(account_key)
        ds_out.to_zarr(out_mapper, mode='a', region=region)


def get_obs(region=None):

    mapper = zarr.storage.ABSStore(
        'carbonplan-scratch',
        prefix='rechunker/terraclimate/target.zarr',
        account_name="carbonplan",
        account_key=os.environ["BLOB_ACCOUNT_KEY"],
    )

    ds_in = xr.open_zarr(mapper, consolidated=True).pipe(load_coords)
    ds_aux = cat.terraclimate.terraclimate.to_dask().pipe(load_coords)
    ds_grid = cat.grids.conus4k.to_dask().pipe(load_coords)

    ds_in['mask'] = ds_grid['mask'].load()
    ds_in['awc'] = ds_aux['awc'].load()
    ds_in['elevation'] = ds_aux['elevation'].load()

    # Temporary fix to correct awc data
    max_soil = ds_in['soil'].max('time').load()
    ds_in['awc'] = ds_in['awc'].where(ds_in['awc'] > 0).fillna(awc_fill)
    ds_in['awc'] = np.maximum(ds_in['awc'], max_soil)

    for v in force_vars:
        ds_in[v] = ds_in[v].astype(np.float32)

    for v in ds_in.variables:
        if 'chunks' in ds_in[v].encoding:
            del ds_in[v].encoding['chunks']

    return ds_in.pipe(maybe_slice_region, region)


def main():

    ds_in = get_obs()
    print('ds_in size: ', ds_in[in_vars].nbytes / 1e9)

    full_template = create_template(ds_in['ppt'], out_vars)
    full_template = full_template.chunk(chunks)

    out_mapper = get_out_mapper(os.environ["BLOB_ACCOUNT_KEY"])
    print('clearing existing store')
    out_mapper.clear()

    full_template.to_zarr(out_mapper, mode='w', compute=False)

    regions = get_regions(ds_in)
    reg_tasks = []
    for region in regions:
        reg_tasks.append(block_wrapper(region, os.environ['BLOB_ACCOUNT_KEY']))

    return finish_store(out_mapper, reg_tasks)


if __name__ == '__main__':
    from dask.distributed import Client

    with Client(threads_per_worker=1, memory_limit='6 G') as client:
        # gateway = Gateway()
        # with gateway.new_cluster(worker_cores=1, worker_memory=6) as cluster:
        #     client = cluster.get_client()
        #     cluster.adapt(minimum=5, maximum=375)
        print(client)
        print(client.dashboard_link)

        task = main()
        dask.compute(task, retries=10)
