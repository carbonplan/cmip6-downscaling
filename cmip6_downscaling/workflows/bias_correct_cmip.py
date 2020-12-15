#!/usr/bin/env python

import os
from collections import defaultdict

import dask
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from adlfs import AzureBlobFileSystem
from dask.distributed import Client

from cmip6_downscaling import CLIMATE_NORMAL_PERIOD
from cmip6_downscaling.constants import KELVIN, PERCENT, SEC_PER_DAY
from cmip6_downscaling.methods.bias_correction import MontlyBiasCorrection
from cmip6_downscaling.workflows.share import chunks, future_time, hist_time, xy_region
from cmip6_downscaling.workflows.utils import get_store

# warning settings
np.seterr(invalid='ignore')

# time slices
train_time = slice(str(CLIMATE_NORMAL_PERIOD[0]), str(CLIMATE_NORMAL_PERIOD[1]))

# variable names
absolute_vars = ['tmin', 'tmax']
relative_vars = ['ppt', 'srad', 'rh']
bc_vars = absolute_vars + relative_vars
update_vars = ['area', 'crs', 'mask']

# output chunks (for dask/zarr)
rename_dict = {'pr': 'ppt', 'tasmax': 'tmax', 'tasmin': 'tmin', 'rsds': 'srad', 'hurs': 'rh'}

skip_existing = False
dry_run = False

dask.config.set(**{'array.slicing.split_large_chunks': False})

# file system
fs = AzureBlobFileSystem(account_name='carbonplan', account_key=os.environ['BLOB_ACCOUNT_KEY'])

# target
target = 'cmip6/bias-corrected/conus/4000m/monthly/{key}.zarr'


def load_coords(ds: xr.Dataset) -> xr.Dataset:
    ''' helper function to pre-load coordinates '''
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

    keys = absolute_vars + relative_vars

    ds = load_coords(ds)

    return ds[keys].drop(['member_id', 'height']).chunk(chunks)


def parse_store_uri(uri):
    model, scenario, member, _ = uri.split('/')[-2].split('.')
    return (model, scenario, member)


def open_single(model, scenario, member):
    uri = f'cmip6/regridded/conus/monthly/4000m/{model}.{scenario}.{member}.zarr'
    store = get_store(uri)
    return xr.open_zarr(store, consolidated=True)


def open_group(model, scenario, members):
    dss = [open_single(model, scenario, member) for member in members]
    return xr.concat(dss, dim=xr.Variable('member', members))


def get_regridded_stores():
    regridded_stores = fs.ls('carbonplan-downscaling/cmip6/regridded/conus/monthly/4000m/')

    d = defaultdict(list)
    d['uri'] = regridded_stores
    for r in regridded_stores:
        model, scenario, member = parse_store_uri(r)
        d['model'].append(model)
        d['scenario'].append(scenario)
        d['member'].append(member)

    return pd.DataFrame(d)


def get_obs():

    mapper = zarr.storage.ABSStore(
        'carbonplan-downscaling',
        prefix='obs/conus/4000m/monthly/terraclimate_plus.zarr',
        account_name="carbonplan",
        account_key=os.environ["BLOB_ACCOUNT_KEY"],
    )
    ds = xr.open_zarr(mapper, consolidated=True)

    return ds[bc_vars]


def maybe_ensemble_mean(ds):
    if 'member' in ds.dims:
        return ds.mean('member')
    return ds


def main(model, scenario, member, members):
    print('---------->', model, scenario, member, members)

    # get the output store
    key = f'{model}.{scenario}.{member}'

    uri = target.format(key=key)
    print(uri)
    store = get_store(uri)

    if skip_existing and '.zmetadata' in store:
        print(f'{key} in store, skipping...')
        return

    absolute_model = MontlyBiasCorrection(correction='absolute')
    relative_model = MontlyBiasCorrection(correction='relative')

    y_hist = get_obs().pipe(load_coords)

    if xy_region:
        y_hist = y_hist.isel(**xy_region)

    print('y_hist:\n', y_hist)

    # open all ensemble members
    x_hist = open_group(model, 'historical', members).pipe(process_cmip)

    if xy_region:
        x_hist = x_hist.isel(**xy_region)

    print('x_hist:\n', x_hist)
    print('fitting models')

    # train models with historical data
    absolute_model.fit(
        x_hist[absolute_vars].sel(time=train_time), y_hist[absolute_vars].sel(time=train_time)
    )
    relative_model.fit(
        x_hist[relative_vars].sel(time=train_time), y_hist[relative_vars].sel(time=train_time)
    )
    print('absolute_model:\n', absolute_model.correction_)

    absolute_model = absolute_model.compute()
    relative_model = relative_model.compute()

    x_scen = open_single(model, scenario, member).pipe(process_cmip)

    if xy_region:
        x_scen = x_scen.isel(**xy_region)

    print('x_scen:\n', x_scen)

    if 'hist' in scenario:
        x_scen = x_scen.sel(time=hist_time)
    else:
        x_scen = x_scen.sel(time=future_time)

    # predict this emsemble member
    y_scen = absolute_model.predict(x_scen[absolute_vars])
    y_scen.update(relative_model.predict(x_scen[relative_vars]))

    y_scen = y_scen.chunk(chunks)
    print('y_scen:\n', y_scen)

    if dry_run:
        print('skipping write of ... dry_run=True')
        return
    else:
        store.clear()
        write = y_scen.to_zarr(store, compute=False, mode='w')
        write.compute(retries=3)
        zarr.consolidate_metadata(store)


if __name__ == '__main__':

    with Client(threads_per_worker=1, memory_limit='12 G') as client:
        print(client)
        print(client.dashboard_link)

        df = get_regridded_stores()

        df2 = pd.read_csv('../../notebooks/ssps_with_matching_historical_members.csv')

        for model, dfgroup in df2.groupby('model'):
            hist_members = []
            print(model)

            for i, row in dfgroup.iterrows():

                # get list of historical members
                members = df[(df.model == model) & (df.scenario == 'historical')].member.tolist()

                # maybe run the historical case -- if it hasn't already been run
                if row.member not in hist_members:
                    # run historical member
                    main(model, 'historical', row.member, members)

                    # store this key so we don't run it again
                    hist_members.append(row.member)

                # now run the scenario member
                main(model, row.scenario, row.member, members)
