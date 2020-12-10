#!/usr/bin/env python
import itertools
import os

import dask
import numpy as np
import xarray as xr
import zarr
from dask_gateway import Gateway

from cmip6_downscaling.disagg.wrapper import create_template, run_terraclimate_model
from cmip6_downscaling.workflows.share import chunks
from cmip6_downscaling.workflows.utils import get_store

skip_existing = False
dry_run = False

out_vars = [
    'aet',
    'def',
    'pdsi',
    'pet',
    'q',
    'soil',
    'swe',
    'tmean',
    'tdew',
    'vap',
    'vpd',
]
force_vars = ['tmax', 'tmin', 'srad', 'ppt', 'rh']
aux_vars = ['mask', 'awc', 'elevation']

space_chunks = {**chunks}
del space_chunks['time']


def preprocess(ds):
    if 'month' in ds:
        ds = ds.drop('month')
    return ds


def load_coords(ds):
    for v in ds.coords:
        ds[v] = ds[v].load()
    return ds


def maybe_slice_region(ds, region):
    if region:
        return ds.isel(**region)
    return ds


def get_obs(region=None, account_key=None):
    # open the obs dataset
    # we'll use this for the wind climatology (temporary) and the aux vars (awc, elevation, mask)
    obs_mapper = get_store(
        'obs/conus/4000m/monthly/terraclimate_plus.zarr', account_key=account_key
    )
    obs = xr.open_zarr(obs_mapper, consolidated=True).pipe(load_coords)
    obs = maybe_slice_region(obs, region)
    return obs


def get_cmip(model, scenario, member, region=None, account_key=None):
    # open the historical simulation dataset
    hist_mapper = get_store(
        f'cmip6/bias-corrected/conus/4000m/monthly/{model}.historical.{member}.zarr',
        account_key=account_key,
    )
    ds_hist = xr.open_zarr(hist_mapper, consolidated=True)[force_vars]
    ds_hist = maybe_slice_region(ds_hist, region)
    ds_hist = ds_hist.pipe(preprocess).pipe(load_coords)

    # open the future simulation dataset
    scen_mapper = get_store(
        f'cmip6/bias-corrected/conus/4000m/monthly/{model}.{scenario}.{member}.zarr'
    )
    ds_scen = xr.open_zarr(scen_mapper, consolidated=True)[force_vars]
    ds_scen = maybe_slice_region(ds_scen, region)
    ds_scen = ds_scen.pipe(preprocess).pipe(load_coords)

    # combine the historical and future simulation datasets together
    ds_in = xr.concat(
        [ds_hist, ds_scen], dim='time', data_vars=force_vars, coords='minimal', compat='override'
    )

    for v in force_vars:
        ds_in[v] = ds_in[v].astype(np.float32)

    return ds_in


def get_out_mapper(model, scenario, member, account_key):
    out_mapper = zarr.storage.ABSStore(
        'carbonplan-scratch',
        prefix=f'cmip6/bias-corrected/conus/4000m/monthly/{model}.{scenario}.{member}.zarr',
        account_name="carbonplan",
        account_key=account_key,
    )
    return out_mapper


@dask.delayed(pure=True, traverse=False)
def block_wrapper(model, scenario, member, region, account_key):

    with dask.config.set(scheduler='single-threaded'):
        obs = get_obs(region=region)
        ds_in = get_cmip(model, scenario, member, region=region)
        ds_in['ws'] = ds_in['ppt'] * 0.0 + 2.0
        for v in aux_vars:
            ds_in[v] = obs[v]

        ds_out = run_terraclimate_model(ds_in)[out_vars]

        out_mapper = get_out_mapper(model, scenario, member, account_key)
        ds_out.to_zarr(out_mapper, mode='a', region=region)


def get_slices(length, chunk_size):
    xi = range(0, length, chunk_size)

    slices = [slice(left, right) for left, right in zip(xi, xi[1:])] + [slice(xi[-1], length + 1)]
    return slices


def get_regions(ds):
    x_slices = get_slices(ds.dims['x'], chunks['x'])
    y_slices = get_slices(ds.dims['y'], chunks['y'])
    t_slices = [slice(None)]
    keys = ['x', 'y', 'time']
    return [dict(zip(keys, s)) for s in itertools.product(x_slices, y_slices, t_slices)]


def main(model, scenario, member):
    print('---------->', model, scenario, member)

    ds_in = get_cmip(model, scenario, member)

    full_template = create_template(ds_in['ppt'], out_vars)
    full_template = full_template.chunk(chunks)

    out_mapper = get_out_mapper(model, scenario, member, os.environ['BLOB_ACCOUNT_KEY'])
    full_template.to_zarr(out_mapper, mode='w', compute=False)

    regions = get_regions(ds_in)
    tasks = []
    for i, region in enumerate(regions):
        print(region)
        tasks.append(block_wrapper(model, scenario, member, region, os.environ['BLOB_ACCOUNT_KEY']))

        # if i >= 4:
        #     break
    print('computing...')
    dask.compute(*tasks)


if __name__ == '__main__':
    # from dask.distributed import Client
    # with Client(threads_per_worker=1) as client:  # , n_workers=12, memory_limit='18 G'
    gateway = Gateway()
    with gateway.new_cluster(worker_cores=1, worker_memory=6) as cluster:
        client = cluster.get_client()

        cluster.adapt(minimum=1, maximum=375)
        print(client)
        print(client.dashboard_link)

        main('CanESM5', 'ssp370', 'r10i1p1f1')
