#!/usr/bin/env python

import os
import warnings

import numba
import numpy as np
import xarray as xr
import zarr
from climate_indices import indices
from skdownscale.pointwise_models.core import xenumerate

from cmip6_downscaling.workflows.utils import get_store

target = 'obs/conus/monthly/4000m/terraclimate_plus.zarr'

xy_region = {'x': slice(200, 210), 'y': slice(200, 210)}
index_vars = ['pdsi', 'pet']
INCH_TO_MM = 0.0393701
WM2_TO_MGM2D = 86400 / 1e6
MISSING = -9999


# set threading options
def _set_thread_settings():
    numba.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'


_set_thread_settings()


def create_template(like_da, var_names):

    ds = xr.Dataset()
    for v in var_names:
        ds[v] = xr.full_like(like_da, np.nan)
    return ds


def calc_indices(ds_small, calib_start=1960, calib_end=2009):
    _set_thread_settings()

    for v in ['mask', 'awc', 'tavg', 'ppt']:
        assert v in ds_small, ds_small

    ds = create_template(ds_small['ppt'], index_vars)

    start_year = ds_small['time'].dt.year[0].data.item()

    for index, mask_val in xenumerate(ds_small['mask']):
        if not mask_val.data.item():
            # skip values outside the mask
            continue

        ds_point = ds_small[['tavg', 'ppt', 'awc']].loc[index]

        try:
            pet = indices.pet(ds_point['tavg'].data, ds_point['lat'].data, start_year)
            pdsi = indices.pdsi(
                ds_point['ppt'].data * INCH_TO_MM,
                pet * INCH_TO_MM,
                ds_point['awc'].data * INCH_TO_MM,
                start_year,
                calib_start,
                calib_end,
            )[0]
        except Exception as e:
            # TODO: figure out exactly why a very few grid cells are failing with a divide by zero error.
            warnings.warn(
                f'PET/PDSI calculation raised an exception, inserting missing values ({MISSING}) to keep going. Error: \n{e}'
            )
            pet = MISSING
            pdsi = MISSING

        ds['pet'].loc[index] = pet
        ds['pdsi'].loc[index] = pdsi

    return ds


def disagg(ds):

    ds_in = ds.copy()

    # make sure coords are all pre-loaded
    ds_in['lon'] = ds_in['lon'].load()
    ds_in['lat'] = ds_in['lat'].load()
    for v in ['lon', 'lat']:
        if 'chunks' in ds_in[v].encoding:
            del ds_in[v].encoding['chunks']

    if 'tavg' not in ds_in:
        ds_in['tavg'] = (ds_in['tmax'] + ds_in['tmin']) / 2

    template = create_template(ds_in['tmax'], index_vars)

    index_in_vars = ['tavg', 'ppt', 'mask', 'awc']
    ds_disagg_out = ds_in[index_in_vars].map_blocks(calc_indices, template=template)

    return ds_disagg_out


if __name__ == '__main__':
    # client = Client(n_workers=4, threads_per_worker=1)
    # print(client)
    # print(client.dashboard_link)

    # open terraclimate data
    # rechunked version
    mapper = zarr.storage.ABSStore(
        'carbonplan-scratch',
        prefix='rechunker/terraclimate/target.zarr/',
        account_name="carbonplan",
        account_key=os.environ["BLOB_ACCOUNT_KEY"],
    )
    ds_conus = xr.open_zarr(mapper, consolidated=True)

    # open awc raster
    awc = xr.open_rasterio('/home/jovyan/awc_4000m.tif').load().drop(['x', 'y']).squeeze(drop=True)

    # combine and mask
    ds_conus['awc'] = awc
    ds_conus['mask'] = xr.where(awc < 255, 1, 0)  # we should probably use the grid file for this
    ds_conus = ds_conus.where(ds_conus['mask'])

    if xy_region:
        ds_conus = ds_conus.isel(**xy_region)

    # do the disaggregation
    ds_conus = ds_conus[['ppt', 'tmax', 'tmin', 'awc', 'mask']]
    ds_out = disagg(ds_conus)
    # print(ds_out.load())

    store = get_store(target)
    store.clear()

    # write = ds_out.to_zarr(store, compute=False, mode='w')
    # write.compute(retries=1)
    # zarr.consolidate_metadata(store)
