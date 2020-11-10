#!/usr/bin/env python

import dask
import xesmf
from carbonplan.data import cat
from dask.distributed import Client

from cmip6_downscaling.data.cmip import cmip
from cmip6_downscaling.workflows.utils import get_store

target = 'cmip6/regridded/conus/monthly/4000m/{key}.zarr'
update_vars = ['area', 'crs', 'mask']
trange = slice('1950', '2120')


def regrid_one_model(source_ds, target_grid, method='bilinear', reuse_weights=True):
    ''' simple wrapper around xesmf '''
    with dask.config.set(scheduler='threads'):
        regridder = xesmf.Regridder(
            source_ds, target_grid, method=method, reuse_weights=reuse_weights
        )
        out = regridder(source_ds)
    return out


def slim_cmip_key(key, member_id):
    _, _, source_id, experiment_id, _, _ = key.split('.')
    out_key = f'{source_id}.{experiment_id}.{member_id}'
    return out_key


if __name__ == '__main__':
    # client = Client(n_workers=16, threads_per_worker=1)
    # print(client)
    # print(client.dashboard_link)
    skip_existing = True
    model_dict, data = cmip()

    # target grid
    grid_ds = cat.grids.conus4k.to_dask().load()

    # collect all keys to regrid
    keys_to_regrid = []
    for hkey, fkeys in model_dict.items():
        keys_to_regrid.append(hkey)
        keys_to_regrid.extend(fkeys)
    print(f'regridding {len(keys_to_regrid)} keys')

    failed = {}
    for key in keys_to_regrid:

        # get source dataset
        source_ds = data[key].sel(time=trange)
        print(source_ds)
        for member_id in source_ds['member_id'].data:

            # create output store
            out_key = slim_cmip_key(key, member_id)
            store = get_store(target.format(key=out_key))

            # skip if existing
            if skip_existing and '.zmetadata' in store:
                print(f'{out_key} in store, skipping...')
                continue
            store.clear()

            # perform the regridding
            print(f'regridding {out_key}')
            ds = regrid_one_model(source_ds.sel(member_id=member_id), grid_ds).chunk(
                {'time': 198, 'x': 50, 'y': 50}
            )

            # write output dataset to store
            ds.update(grid_ds[update_vars])
            try:
                ds.to_zarr(store, mode='w', consolidated=True)
            except Exception as e:
                print(key, e)
                failed[key] = e
print(failed)
