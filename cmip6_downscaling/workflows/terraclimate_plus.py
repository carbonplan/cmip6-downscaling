#!/usr/bin/env python

import os

import numpy as np
import xarray as xr
import zarr
from carbonplan.data import cat
from dask_gateway import Gateway

from cmip6_downscaling.disagg.wrapper import disagg
from cmip6_downscaling.workflows.share import xy_region
from cmip6_downscaling.workflows.utils import get_store

out_vars = [
    'aet',
    'pdsi',
    'pet',
]
force_vars = ['tmax', 'tmin', 'srad', 'ppt', 'vap', 'ws']
aux_vars = ['mask', 'awc', 'elevation']
in_vars = force_vars + aux_vars

target = 'obs/conus/4000m/monthly/terraclimate_plus.zarr'


def main():

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

    ds_in['mask'] = ds_grid['mask'].load()
    ds_in['awc'] = ds_aux['awc'].load()
    ds_in['elevation'] = ds_aux['elevation'].load()
    ds_in['lon'] = ds_in['lon'].load()
    ds_in['lat'] = ds_in['lat'].load()

    # Temporary fix to correct awc data
    ds_in['awc'] = np.maximum(ds_in['awc'], ds_in['soil'].max('time')).load()

    if xy_region:
        ds_in = ds_in.isel(**xy_region)

    ds_in = ds_in.unify_chunks()

    for v in force_vars:
        ds_in[v] = ds_in[v].astype(np.float32)

    print('ds_in size: ', ds_in[in_vars].nbytes / 1e9)

    # do the disaggregation
    ds_out = disagg(ds_in[in_vars])

    print(ds_out[out_vars])
    print('ds_out size: ', ds_out[out_vars].nbytes / 1e9)

    for v in ['lon', 'lat']:
        if 'chunks' in ds_out[v].encoding:
            del ds_out[v].encoding['chunks']

    store = get_store(target)
    store.clear()

    write = ds_out[out_vars].to_zarr(store, compute=False, mode='w')
    write.compute(retries=1)
    zarr.consolidate_metadata(store)


if __name__ == '__main__':
    # from dask.distributed import Client
    # with Client(threads_per_worker=1, memory_limit='10 G') as client:

    gateway = Gateway()
    with gateway.new_cluster(worker_cores=1, worker_memory=10) as cluster:
        client = cluster.get_client()
        cluster.adapt(minimum=5, maximum=375)
        print(client)
        print(client.dashboard_link)

        main()
