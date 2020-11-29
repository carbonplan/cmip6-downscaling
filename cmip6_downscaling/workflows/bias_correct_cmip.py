#!/usr/bin/env python

import os
from collections import defaultdict

import dask
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from adlfs import AzureBlobFileSystem

from cmip6_downscaling import CLIMATE_NORMAL_PERIOD
from cmip6_downscaling.constants import KELVIN, SEC_PER_DAY
from cmip6_downscaling.disagg import derived_variables
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

skip_existing = True
dry_run = False

# file system
fs = AzureBlobFileSystem(account_name='carbonplan', account_key=os.environ['BLOB_ACCOUNT_KEY'])

# target
target = 'cmip6/bias-corrected/conus/monthly/4000m/{key}.zarr'


def load_coords(ds):
    for v in ds.coords:
        if isinstance(ds[v].data, dask.array.Array):
            ds[v] = ds[v].load()
    for v in ds.variables:
        if 'chunks' in ds[v].encoding:
            del ds[v].encoding['chunks']

    return ds


def process_cmip(ds):

    ds = ds.rename(rename_dict)

    ds['tmax'] -= KELVIN
    ds['tmin'] -= KELVIN
    ds['rh'] /= 100
    ds['ppt'] *= xr.Variable('time', ds.indexes['time'].days_in_month * SEC_PER_DAY)

    keys = absolute_vars + relative_vars

    ds = load_coords(ds)

    return ds[keys].drop(['member_id', 'height']).chunk(chunks)


def parse_store_uri(uri):
    model, scenario, member, _ = uri.split('/')[-2].split('.')
    return (model, scenario, member)


def open_single(uri):
    store = get_store(uri.replace('carbonplan-downscaling/', ''))
    return xr.open_zarr(store, consolidated=True)


def open_group(gdf):
    dss = []
    for i, row in gdf.iterrows():
        dss.append(open_single(row.uri))
    return xr.concat(dss, dim=xr.Variable('member', gdf.member))


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
        prefix='obs/conus/monthly/4000m/terraclimate_plus.zarr',
        account_name="carbonplan",
        account_key=os.environ["BLOB_ACCOUNT_KEY"],
    )
    ds = xr.open_zarr(mapper, consolidated=True)

    # temporary fix
    ds = derived_variables.process(ds)
    return ds[bc_vars].chunk(chunks)


if __name__ == '__main__':
    from dask.distributed import Client

    client = Client(n_workers=36, threads_per_worker=1)
    print(client)
    print(client.dashboard_link)

    # dict of cmip data (raw) - just used for the simulation keys
    df = get_regridded_stores()

    absolute_model = MontlyBiasCorrection(correction='absolute')
    relative_model = MontlyBiasCorrection(correction='relative')

    # grid_ds = cat.grids.conus4k.to_dask().load()
    y_hist = get_obs().pipe(load_coords)  # .update(grid_ds[update_vars])

    if xy_region:
        y_hist = y_hist.isel(**xy_region)

    print('y_hist:\n', y_hist)

    for model, gdf in df[df.scenario == 'historical'].groupby('model'):

        # open all ensemble members
        x_hist = open_group(gdf).pipe(process_cmip)

        if xy_region:
            x_hist = x_hist.isel(**xy_region)

        print('fitting models')
        # train models with historical data
        absolute_model.fit(
            x_hist[absolute_vars].sel(time=train_time), y_hist[absolute_vars].sel(time=train_time)
        )
        relative_model.fit(
            x_hist[relative_vars].sel(time=train_time), y_hist[relative_vars].sel(time=train_time)
        )
        print('absolute_model:\n', absolute_model.correction_)

        # try pre-loading the corrections
        absolute_model.persist()
        relative_model.persist()

        print('x_hist:\n', x_hist)

        for scen in df[df.model == model].scenario.unique():

            # for now, select only the first member
            ens_members = df[(df.model == model) & (df.scenario == scen)]

            for i, row in ens_members.iterrows():

                # get the output store
                key = f'{model}.{scen}.{row.member}'
                store = get_store(target.format(key=key))

                print(target.format(key=key))

                if skip_existing and '.zmetadata' in store:
                    print(f'{key} in store, skipping...')
                    continue
                else:
                    # write data to the store
                    print(f'writing {key}')

                x_scen = open_single(row.uri).pipe(process_cmip)

                if xy_region:
                    x_scen = x_scen.isel(**xy_region)

                print('x_scen:\n', x_scen)

                if 'hist' in scen:
                    x_scen = x_scen.sel(time=hist_time)
                else:
                    x_scen = x_scen.sel(time=future_time)

                # predict this emsemble member
                y_scen = absolute_model.predict(x_scen[absolute_vars])
                y_scen.update(relative_model.predict(x_scen[relative_vars]))
                print('y_scen:\n', y_scen)

                if dry_run:
                    print('skipping write of ... dry_run=True')
                    continue
                else:
                    store.clear()
                    y_scen.to_zarr(store, consolidated=True, mode='w')

                break  # for now, only bias correct the first member
