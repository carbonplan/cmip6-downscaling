#!/usr/bin/env python

import pandas as pd
import xarray as xr
import zarr
from dask_gateway import Gateway

from cmip6_downscaling.disagg.wrapper import disagg
from cmip6_downscaling.workflows.share import chunks, future_time, hist_time
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
    'awc',
    'elevation',
    'mask',
    'vap',
    'vpd',
    'ws',
]
force_vars = ['tmax', 'tmin', 'srad', 'ppt', 'rh']
aux_vars = ['mask', 'awc', 'elevation']


def preprocess(ds):
    if 'month' in ds:
        ds = ds.drop('month')
    return ds


def load_coords(ds):
    for v in ds.coords:
        ds[v] = ds[v].load()
    return ds


def main(model, scenario, member, write_hist=True):
    print('---------->', model, scenario, member)

    # open the obs dataset
    # we'll use this for the wind climatology (temporary) and the aux vars (awc, elevation, mask)
    obs_mapper = get_store('obs/conus/4000m/monthly/terraclimate_plus.zarr')
    obs = xr.open_zarr(obs_mapper, consolidated=True).pipe(load_coords)
    for v in ['mask', 'awc', 'elevation', 'lon', 'lat']:
        obs[v] = obs[v].load()

    # open the historical simulation dataset
    hist_mapper = get_store(
        f'cmip6/bias-corrected/conus/4000m/monthly/{model}.historical.{member}.zarr'
    )
    ds_hist = (
        xr.open_zarr(hist_mapper, consolidated=True)[force_vars].pipe(preprocess).pipe(load_coords)
    )

    # open the future simulation dataset
    scen_mapper = get_store(
        f'cmip6/bias-corrected/conus/4000m/monthly/{model}.{scenario}.{member}.zarr'
    )
    ds_scen = (
        xr.open_zarr(scen_mapper, consolidated=True)[force_vars].pipe(preprocess).pipe(load_coords)
    )

    # combine the historical and future simulation datasets together
    ds_in = xr.concat(
        [ds_hist, ds_scen], dim='time', data_vars=force_vars, coords='minimal', compat='override'
    )
    print(obs)

    # copy the aux vars over
    for v in aux_vars:
        ds_in[v] = obs[v]

    # try ws of 2m/s
    ds_in['ws'] = ds_in['ppt'] * 0.0 + 2
    # (temporary) wind speed climatology
    # xr.zeros_like(ds_in['ppt']).groupby('time.month') + obs['ws'].groupby(
    #     'time.month'
    # ).mean('time')

    # rechunk
    ds_in = ds_in.chunk(chunks).persist()
    print('ds_in', ds_in)

    # do the disaggregation
    ds_out = disagg(ds_in)
    print(ds_out)

    print('testing load of the first 50 timesteps')
    ds_out.isel(time=slice(0, 50)).load()
    return

    # write datasets
    tasks = []

    # write hist
    if write_hist:
        out_hist = ds_out.sel(time=hist_time)[out_vars]
        tasks.append(out_hist.to_zarr(hist_mapper, mode='a', compute=False))

    # write scenario
    out_scen = ds_out.sel(time=future_time)[out_vars]
    tasks.append(out_scen.to_zarr(scen_mapper, mode='a', compute=False))

    # compute all tasks simultaneously since they share common nodes in the compute graph
    client.compute(tasks, retries=3)

    # now that we've added variables to the dataset, we need to re-consolidate the metadata
    if write_hist:
        zarr.consolidate_metadata(hist_mapper)
    zarr.consolidate_metadata(scen_mapper)


if __name__ == '__main__':
    gateway = Gateway()
    with gateway.new_cluster(worker_cores=1, worker_memory=14) as cluster:
        client = cluster.get_client()
        cluster.adapt(minimum=5, maximum=375)
        print(client)
        print(client.dashboard_link)

        df = pd.read_csv('../../notebooks/ssps_with_matching_historical_members.csv')

        for model, dfgroup in df.groupby('model'):

            for i, row in dfgroup.iterrows():
                first = i == 0
                main(model, row.scenario, row.member, write_hist=first)
                client.restart()
                break  # for testing: remove me!!!!
            break
