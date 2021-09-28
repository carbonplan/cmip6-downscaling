#!/usr/bin/env python

import os

import numpy as np
import xarray as xr
import zarr
from dask.distributed import Client
from skdownscale.pointwise_models import (
    PointWiseDownscaler,
    QuantileMappingReressor,
    TrendAwareQuantileMappingRegressor,
)

from cmip6_downscaling import CLIMATE_NORMAL_PERIOD
from cmip6_downscaling.constants import KELVIN, PERCENT, SEC_PER_DAY
from cmip6_downscaling.workflows.share import (
    chunks,
    future_time,
    get_cmip_runs,
    hist_time,
    xy_region,
)
from cmip6_downscaling.workflows.utils import get_store

# warning settings
np.seterr(invalid='ignore')

# time slices
train_time = slice(str(CLIMATE_NORMAL_PERIOD[0]), str(CLIMATE_NORMAL_PERIOD[1]))

# variable names
bc_vars = ['tmin', 'tmax', 'ppt', 'srad', 'rh']
update_vars = ['area', 'crs', 'mask']

# output chunks (for dask/zarr)
rename_dict = {'pr': 'ppt', 'tasmax': 'tmax', 'tasmin': 'tmin', 'rsds': 'srad', 'hurs': 'rh'}
skip_existing = False
dry_run = False
version = 'v3'

extrapolate = {'tmin': '1to1', 'tmax': '1to1', 'ppt': '1to1', 'srad': '1to1', 'rh': '1to1'}

clip = {
    'tmin': None,
    'tmax': None,
    'ppt': {'min': 0.0, 'max': None},
    'srad': {'min': 0.0, 'max': None},
    'rh': {'min': 0.0, 'max': 1.0},
}

# target
source = 'cmip6/regridded/conus/4000m/monthly/{key}.zarr'
target = f'cmip6/quantile-mapping-{version}/conus/4000m/monthly/{{key}}.zarr'


def load_coords(ds: xr.Dataset) -> xr.Dataset:
    '''helper function to pre-load coordinates'''
    ds = ds.update(ds[list(ds.coords)].load())
    for v in ds.variables:
        if 'chunks' in ds[v].encoding:
            del ds[v].encoding['chunks']

    return ds


def process_cmip(ds):

    ds = ds.rename(rename_dict)

    ds['tmax'] -= KELVIN
    ds['tmin'] -= KELVIN
    ds['rh'] /= PERCENT
    ds['ppt'] *= xr.Variable('time', ds.indexes['time'].days_in_month * SEC_PER_DAY)

    # ds = derived_variables.process(ds)

    ds = load_coords(ds)

    return ds[bc_vars].drop(['member_id', 'height']).chunk(chunks)


def open_single(model, scenario, member):
    uri = f'cmip6/regridded/conus/4000m/monthly/{model}.{scenario}.{member}.zarr'
    store = get_store(uri)
    return xr.open_zarr(store, consolidated=True)


def get_obs():

    mapper = zarr.storage.ABSStore(
        'carbonplan-downscaling',
        prefix='obs/conus/4000m/monthly/terraclimate_plus_v3.zarr',
        account_name="carbonplan",
        account_key=os.environ["BLOB_ACCOUNT_KEY"],
    )
    ds = xr.open_zarr(mapper, consolidated=True)

    return ds[bc_vars]


def main(model, scenario, member):
    print('---------->', model, scenario, member)

    # get the output store
    key = f'{model}.{scenario}.{member}'

    target_uri = target.format(key=key)
    print(target_uri)
    store = get_store(target_uri)

    if skip_existing and '.zmetadata' in store:
        print(f'{key} in store, skipping...')
        return 'skipped'

    y_hist = get_obs().pipe(load_coords)

    if xy_region:
        y_hist = y_hist.isel(**xy_region)

    print('y_hist:\n', y_hist)

    x_hist = open_single(model, 'historical', member).pipe(process_cmip)

    if xy_region:
        x_hist = x_hist.isel(**xy_region)

    print('x_hist:\n', x_hist)

    x_scen = open_single(model, scenario, member).pipe(process_cmip)

    if xy_region:
        x_scen = x_scen.isel(**xy_region)
    if 'hist' in scenario:
        x_scen = x_scen.sel(time=hist_time)
    else:
        x_scen = x_scen.sel(time=future_time)
    print('x_scen:\n', x_scen)

    print('fitting models')

    models = {}
    y_scen = xr.Dataset()

    for v in bc_vars:
        print(v)
        models[v] = PointWiseDownscaler(
            TrendAwareQuantileMappingRegressor(QuantileMappingReressor(extrapolate=extrapolate[v]))
        )

        # train models with historical data
        models[v].fit(x_hist[v].sel(time=train_time), y_hist[v].sel(time=train_time))

        # predict this ensemble member
        y_scen[v] = models[v].predict(x_scen[v])

        # post process predicted data
        if clip[v] is not None:
            y_scen[v] = y_scen[v].clip(**clip[v])

    y_scen = y_scen.chunk(chunks)
    print('y_scen:\n', y_scen)

    if dry_run:
        print('skipping write of ... dry_run=True')
        return 'skipped'
    else:
        store.clear()
        write = y_scen.to_zarr(store, compute=False, mode='w')
        write.compute(retries=3)
        zarr.consolidate_metadata(store)
        return 'done'


if __name__ == '__main__':

    with Client(threads_per_worker=1, memory_limit='14 G') as client:
        print(client)
        print(client.dashboard_link)

        df = get_cmip_runs(comp=True, unique=True).reset_index()
        print(df)

        for i, row in df.iterrows():
            print(f'quantile mapping {i+1} of {len(df)}')
            print(row)

            result = main(row.model, row.scenario, row.member)
            if result == 'done':
                client.restart()
