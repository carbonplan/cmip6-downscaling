#!/usr/bin/env python

import xarray as xr
import zarr

from cmip6_downscaling.disagg.wrapper import disagg
from cmip6_downscaling.workflows.share import chunks, future_time, hist_time
from cmip6_downscaling.workflows.utils import get_store

skip_existing = True
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
]  # TODO: add `ws`
force_vars = ['tmax', 'tmin', 'srad', 'ppt', 'rh']
aux_vars = ['mask', 'awc', 'elevation']


def preprocess(ds):
    if 'month' in ds:
        ds = ds.drop('month')
    return ds


if __name__ == '__main__':
    from dask.distributed import Client

    client = Client(n_workers=24, threads_per_worker=1)
    print(client)
    print(client.dashboard_link)

    # open the obs dataset
    # we'll use this for the wind climatology (temporary) and the aux vars (awc, elevation, mask)
    obs_mapper = get_store('obs/conus/monthly/4000m/terraclimate_plus.zarr')
    obs = xr.open_zarr(obs_mapper, consolidated=True)

    # open the historical simulation dataset
    hist_mapper = get_store(
        'cmip6/bias-corrected/conus/monthly/4000m/MRI-ESM2-0.historical.r1i1p1f1.zarr'
    )
    ds_hist = xr.open_zarr(hist_mapper, consolidated=True)[force_vars].pipe(preprocess)

    # open the future simulation dataset
    scen_mapper = get_store(
        'cmip6/bias-corrected/conus/monthly/4000m/MRI-ESM2-0.ssp585.r1i1p1f1.zarr'
    )
    ds_scen = xr.open_zarr(scen_mapper, consolidated=True)[force_vars].pipe(preprocess)

    # combine the historical and future simulation datasets together
    ds_in = xr.concat([ds_hist, ds_scen], dim='time')
    print(obs)
    print(ds_in)

    # copy the aux vars over
    for v in aux_vars:
        ds_in[v] = obs[v]

    # (temporary) wind speed climatology
    ds_in['ws'] = xr.zeros_like(ds_in['ppt']).groupby('time.month') + obs['ws'].groupby(
        'time.month'
    ).mean('time')

    # rechunk
    ds_in = ds_in.chunk(chunks)

    # do the disaggregation
    ds_out = disagg(ds_in)
    print(ds_out)

    # write datasets
    tasks = []

    # write hist
    write_hist = False
    if write_hist:
        out_hist = ds_out.sel(time=hist_time)[out_vars]
        tasks.append(out_hist.to_zarr(hist_mapper, mode='a', compute=False))

    # write scenario
    out_scen = ds_out.sel(time=future_time)[out_vars]
    tasks.append(out_scen.to_zarr(scen_mapper, mode='a', compute=False))

    # compute all tasks simultaneously since they share common nodes in the compute graph
    client.compute(tasks)

    # now that we've added variables to the dataset, we need to re-consolidate the metadata
    if write_hist:
        zarr.consolidate_metadata(hist_mapper)
    zarr.consolidate_metadata(scen_mapper)
