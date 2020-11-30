#!/usr/bin/env python

import os

import numpy as np
import xarray as xr
import zarr
from carbonplan.data import cat
from dask.distributed import Client

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

    ds_in['mask'] = ds_grid['mask']
    ds_in['awc'] = ds_aux['awc']
    ds_in['elevation'] = ds_aux['elevation']

    # Temporary fix to correct awc data
    ds_in['awc'] = np.maximum(ds_in['awc'], ds_in['soil'].max('time')).persist()

    # Temporary fix to correct for mismatched masks (along coasts)
    ds_in['mask'] = (
        ds_in['mask'].astype(bool)
        * ds_in['awc'].notnull()
        * ds_in['elevation'].notnull()
        * ds_in['ppt'].isel(time=-1).notnull()
    ).persist()

    # cast input data to float32
    for v in ds_in.data_vars:
        if v not in ['mask']:
            ds_in[v] = ds_in[v].astype(np.float32)

    if xy_region:
        ds_in = ds_in.isel(**xy_region)

    ds_in = ds_in.chunk(chunks)

    # do the disaggregation
    ds_out = disagg(ds_in)

    print(ds_out)

    store = get_store(target)
    store.clear()

    write = ds_out.to_zarr(store, compute=False, mode='w')
    write.compute(retries=1)
    zarr.consolidate_metadata(store)


if __name__ == '__main__':
    with Client(n_workers=12, threads_per_worker=1) as client:
        print(client)
        print(client.dashboard_link)

        main()
