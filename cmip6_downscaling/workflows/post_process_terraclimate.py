#!/usr/bin/env python

import os

import dask
import xarray as xr
import zarr
from dask.diagnostics import ProgressBar
from dask.distributed import Client

from cmip6_downscaling.workflows.share import get_cmip_runs

account_key = os.environ.get('BLOB_ACCOUNT_KEY', None)
chunks = {'x': 50, 'y': 50, 'time': -1}
mean_vars = ['tmin', 'tmax', 'srad', 'rh', 'tmean', 'tdew', 'vap', 'vpd', 'pdsi', 'soil', 'swe']
sum_vars = ['ppt', 'aet', 'pet', 'def', 'q']
skip_existing = True


def get_scratch_ds(model, scenario, member, method):

    mapper = zarr.storage.ABSStore(
        'carbonplan-scratch',
        prefix=f'cmip6/{method}/conus/4000m/monthly/{model}.{scenario}.{member}.zarr',
        account_name="carbonplan",
        account_key=account_key,
    )
    ds = xr.open_zarr(mapper, consolidated=True)
    return ds


def split_and_write(model, scenario, member, method):

    ds = get_scratch_ds(model, scenario, member, method)

    scen_mapper = zarr.storage.ABSStore(
        'carbonplan-downscaling',
        prefix=f'cmip6/{method}/conus/4000m/monthly/{model}.{scenario}.{member}.zarr',
        account_name="carbonplan",
        account_key=account_key,
    )

    if not (skip_existing and 'pdsi/.zarray' in scen_mapper):
        print('writing scen')
        ds.sel(time=slice('2015-01', None)).to_zarr(scen_mapper, mode='a')
        zarr.consolidate_metadata(scen_mapper)

    hist_mapper = zarr.storage.ABSStore(
        'carbonplan-downscaling',
        prefix=f'cmip6/{method}/conus/4000m/monthly/{model}.historical.{member}.zarr',
        account_name="carbonplan",
        account_key=account_key,
    )
    if not (skip_existing and 'pdsi/.zarray' in hist_mapper):
        print('writing hist')
        ds.sel(time=slice(None, '2014-12')).to_zarr(hist_mapper, mode='a')
        zarr.consolidate_metadata(hist_mapper)


def weighted_mean(ds, *args, **kwargs):
    weights = ds.time.dt.days_in_month
    return ds.weighted(weights).mean(dim='time')


def _annual(ds_monthly, compute=True):
    with dask.config.set(scheduler='single-threaded'):

        if compute:
            ds_monthly = ds_monthly.load()

        ds_annual = ds_monthly[mean_vars].resample(time='AS').map(weighted_mean, dim='time')
        ds_annual = ds_annual.update(ds_monthly[sum_vars].resample(time='AS').sum())
    return ds_annual


def make_annual(model, scenario, member, method):

    if 'hist' in scenario:
        tslice = slice(None, '2014-12')
    else:
        tslice = slice('2015', '2100')
    monthly_mapper = zarr.storage.ABSStore(
        'carbonplan-downscaling',
        prefix=f'cmip6/{method}/conus/4000m/monthly/{model}.{scenario}.{member}.zarr',
        account_name="carbonplan",
        account_key=account_key,
    )

    annual_mapper = zarr.storage.ABSStore(
        'carbonplan-downscaling',
        prefix=f'cmip6/{method}/conus/4000m/annual/{model}.{scenario}.{member}.zarr',
        account_name="carbonplan",
        account_key=account_key,
    )

    if skip_existing and '.zmetadata' in annual_mapper:
        return 'skipped'

    ds_monthly = xr.open_zarr(monthly_mapper, consolidated=True).sel(time=tslice).chunk(chunks)
    template = _annual(ds_monthly, compute=False).chunk(chunks)
    ds_annual = ds_monthly.map_blocks(_annual, template=template)
    annual_mapper.clear()
    task = ds_annual.to_zarr(annual_mapper, mode='w', compute=False)
    dask.compute(task, retries=4)
    zarr.consolidate_metadata(annual_mapper)
    return 'done'


if __name__ == '__main__':

    df = get_cmip_runs(comp=False, unique=True)
    print(df)

    method = 'quantile-mapping'

    split_df = df[df.scenario.str.contains('ssp')].reset_index()

    with dask.config.set(scheduler='processes'):
        with ProgressBar():
            for i, row in split_df.iterrows():
                print(f'processing {i+1} of {len(df)}')
                print(row)

                split_and_write(row.model, row.scenario, row.member, method)

    with Client(threads_per_worker=1, memory_limit='22 G') as client:
        print(client)
        print(client.dashboard_link)
        for i, row in df.iterrows():
            print(f'processing {i+1} of {len(df)}')
            print(row)

            result = make_annual(row.model, row.scenario, row.member, method)
            if result == 'done':
                client.restart()
