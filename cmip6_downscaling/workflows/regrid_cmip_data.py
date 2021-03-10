#!/usr/bin/env python

import dask
import fsspec
import pandas as pd
import xarray as xr
import xesmf
from carbonplan.data import cat

from cmip6_downscaling.workflows.share import skip_unmatched
from cmip6_downscaling.workflows.utils import get_store

target = 'cmip6/regridded/{region}/4000m/monthly/{key}.zarr'
update_vars = ['area', 'crs', 'mask']
skip_existing = True


def regrid_one_model(source_ds, target_grid, method='bilinear', reuse_weights=False):
    ''' simple wrapper around xesmf '''
    with dask.config.set(scheduler='threads'):
        regridder = xesmf.Regridder(
            source_ds, target_grid, method=method, reuse_weights=reuse_weights
        )
        out = regridder(source_ds)
    return out


if __name__ == '__main__':

    with fsspec.open(
        'az://carbonplan-downscaling/cmip6/ssps_with_matching_historical_members.csv',
        'r',
        account_name='carbonplan',
    ) as f:
        df = pd.read_csv(f)

    # target grid
    for grid in ['conus', 'ak']:

        grid_ds = cat.grids.albers4k(region=grid).read()

        for i, row in df.iterrows():

            if skip_unmatched and not row.has_match:
                continue

            target_key = f'{row.model}.{row.scenario}.{row.member}'
            target_path = target.format(region=grid, key=target_key)
            target_store = get_store(target_path)

            # skip if existing
            if skip_existing and '.zmetadata' in target_store:
                print(f'{target_key} in store, skipping...')
                continue

            source_store = get_store(row.path)
            source_ds = xr.open_zarr(source_store, consolidated=True)

            # perform the regridding
            print(f'regridding {target_path}')
            ds = regrid_one_model(source_ds, grid_ds).chunk({'time': 198, 'x': 50, 'y': 50})

            # write output dataset to store
            ds.update(grid_ds[update_vars])

            ds.to_zarr(target_store, mode='w', consolidated=True)
