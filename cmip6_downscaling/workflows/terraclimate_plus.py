#!/usr/bin/env python

import os

import numpy as np
import xarray as xr
import zarr
from carbonplan.data import cat
from dask_gateway import Gateway

from cmip6_downscaling.disagg.wrapper import disagg
from cmip6_downscaling.workflows.share import chunks, xy_region
from cmip6_downscaling.workflows.utils import get_store

target = 'obs/conus/monthly/4000m/terraclimate_plus.zarr'


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

    ds_in = ds_in.unify_chunks().chunk(chunks)

    # do the disaggregation
    ds_out = disagg(ds_in)

    print(ds_out)

    for v in ['lon', 'lat']:
        if 'chunks' in ds_out[v].encoding:
            del ds_out[v].encoding['chunks']

    store = get_store(target)
    store.clear()

    write = ds_out.to_zarr(store, compute=False, mode='w')
    write.compute(retries=1)
    zarr.consolidate_metadata(store)


if __name__ == '__main__':
    gateway = Gateway()
    with gateway.new_cluster(worker_cores=1, worker_memory=12) as cluster:
        client = cluster.get_client()
        cluster.adapt(minimum=5, maximum=100)

        print(client)
        print(client.dashboard_link)

        main()
