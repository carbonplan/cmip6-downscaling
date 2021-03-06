#!/usr/bin/env python
import os
from typing import Dict, List

import dask
import numpy as np
import xarray as xr
import zarr
from dask.distributed import Client

from cmip6_downscaling import CLIMATE_NORMAL_PERIOD
from cmip6_downscaling.disagg.wrapper import create_template, run_terraclimate_model
from cmip6_downscaling.workflows.share import (
    chunks,
    dummy_store,
    finish_store,
    get_cmip_runs,
    get_regions,
    load_coords,
    maybe_slice_region,
    preprocess,
)
from cmip6_downscaling.workflows.utils import get_store

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
downscale_method = 'quantile-mapping-v3'
force_vars = [
    'tmax',
    'tmin',
    'srad',
    'ppt',
    'rh',
]
aux_vars = ['mask', 'awc', 'elevation']
in_vars = force_vars + aux_vars + ['ws']
skip_existing = False


def get_obs(region: dict = None, account_key: str = None) -> xr.Dataset:
    """Load observed climate data and auxillary variables

    Parameters
    ----------
    region : dict
        Dictionary of slices defining a single chunk, e.g. {'x': slice(1150, 1200), 'y': slice(700, 750), 'time': slice(None)}
    account_key : str
        Secret key giving Zarr access to Azure store
    """

    obs_mapper = get_store(
        'obs/conus/4000m/monthly/terraclimate_plus_v3.zarr', account_key=account_key
    )
    obs = xr.open_zarr(obs_mapper, consolidated=True).pipe(load_coords)
    obs = maybe_slice_region(obs, region)
    return obs


def get_wind_climatology(ws, time_index) -> xr.Dataset:
    """Load wind climatology

    Parameters
    ----------
    ws : xr.DataArray
        Wind speed timeseries
    time_index : pd.Index
        Time index to map repeated climatology to
    """
    new_shape = list(ws.shape)
    new_shape[ws.get_axis_num('time')] = len(time_index)
    coords = dict(ws.coords)
    coords['time'] = time_index

    new_wind = xr.DataArray(np.zeros(new_shape), dims=ws.dims, coords=coords)
    climo = ws.sel(time=slice(*map(str, CLIMATE_NORMAL_PERIOD))).groupby('time.month').mean().load()
    new_wind = new_wind.groupby('time.month') + climo
    return new_wind


def get_cmip(
    model: str, scenario: str, member: str, region: Dict = None, account_key: str = None
) -> xr.Dataset:
    """Load downscaled CMIP data, concatenating historical and future scenarios along their time dimension

    Parameters
    ----------
    model : str
        CMIP model_id
    scenario : str
        CMIP scenario
    member : str
        CMIP member_id
    region : dict
        Dictionary of slices defining a single chunk, e.g. {'x': slice(1150, 1200), 'y': slice(700, 750), 'time': slice(None)}
    account_key : str
        Secret key giving Zarr access to Azure store
    """
    # open the historical simulation dataset
    hist_mapper = get_store(
        f'cmip6/{downscale_method}/conus/4000m/monthly/{model}.historical.{member}.zarr',
        account_key=account_key,
    )
    ds_hist = xr.open_zarr(hist_mapper, consolidated=True)[force_vars]
    ds_hist = maybe_slice_region(ds_hist, region)
    ds_hist = ds_hist.pipe(preprocess).pipe(load_coords)

    # open the future simulation dataset
    scen_mapper = get_store(
        f'cmip6/{downscale_method}/conus/4000m/monthly/{model}.{scenario}.{member}.zarr'
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


def get_out_mapper(
    model: str, scenario: str, member: str, account_key: str
) -> zarr.storage.ABSStore:
    """Get output dataset mapper to Azure Blob store

    Parameters
    ----------
    model : str
        CMIP model_id
    scenario : str
        CMIP scenario
    member : str
        CMIP member_id
    account_key : str
        Secret key giving Zarr access to Azure store

    Returns
    -------
    mapper : zarr.storage.ABSStore
    """
    out_mapper = zarr.storage.ABSStore(
        'carbonplan-scratch',
        prefix=f'cmip6/{downscale_method}/conus/4000m/monthly/{model}.{scenario}.{member}.zarr',
        account_name="carbonplan",
        account_key=account_key,
    )
    return out_mapper


@dask.delayed(pure=False, traverse=False)
def block_wrapper(model: str, scenario: str, member: str, region: Dict, account_key: str):
    """Delayed wrapper function to run Terraclimate model over a single x/y chunk

    Parameters
    ----------
    model : str
        CMIP model_id
    scenario : str
        CMIP scenario
    member : str
        CMIP member_id
    region : dict
        Dictionary of slices defining a single chunk, e.g. {'x': slice(1150, 1200), 'y': slice(700, 750), 'time': slice(None)}
    account_key : str
        Secret key giving Zarr access to Azure store
    """
    print(region)
    with dask.config.set(scheduler='single-threaded'):
        obs = get_obs(region=region)
        ds_in = get_cmip(model, scenario, member, region=region)
        ds_in['ws'] = get_wind_climatology(obs['ws'], ds_in['time'])
        for v in aux_vars:
            ds_in[v] = obs[v]
        ds_in = ds_in[in_vars].load()

        ds_out = run_terraclimate_model(ds_in)[out_vars]

        out_mapper = get_out_mapper(model, scenario, member, account_key)
        ds_out.to_zarr(out_mapper, mode='a', region=region)


def main(model: str, scenario: str, member: str, compute: bool = False) -> List:
    """main run function

    Parameters
    ----------
    model : str
        CMIP model_id
    scenario : str
        CMIP scenario
    member : str
        CMIP member_id
    """
    print('---------->', model, scenario, member)

    out_mapper = get_out_mapper(model, scenario, member, os.environ['BLOB_ACCOUNT_KEY'])

    if skip_existing and '.zmetadata' in out_mapper:
        print('skipping')
        return dummy_store(out_mapper)
    else:
        print('clearing existing store')
        out_mapper.clear()

    ds_in = get_cmip(model, scenario, member)

    full_template = create_template(ds_in['ppt'], out_vars)
    full_template = full_template.chunk(chunks)
    full_template.to_zarr(out_mapper, mode='w', compute=False)

    regions = get_regions(ds_in)
    reg_tasks = []
    for i, region in enumerate(regions):
        reg_tasks.append(
            block_wrapper(model, scenario, member, region, os.environ['BLOB_ACCOUNT_KEY'])
        )

    return finish_store(out_mapper, reg_tasks)


if __name__ == '__main__':
    df = get_cmip_runs(comp=True, unique=True).reset_index()
    df = df[df.scenario.str.contains('ssp')].reset_index()

    print(df)

    with Client(threads_per_worker=1, memory_limit='6 G') as client:
        print(client)
        print(client.dashboard_link)
        for i, row in df.iterrows():
            print(f'running terraclimate on {i + 1} of {len(df)}')
            print(row)
            task = main(row.model, row.scenario, row.member, compute=False)
            dask.compute(task, retries=2)
