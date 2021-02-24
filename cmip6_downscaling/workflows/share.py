import itertools
from typing import Dict, List

import dask
import xarray as xr
import zarr

awc_fill = 50  # mm
hist_time = slice('1950', '2014')
future_time = slice('2015', '2120')
chunks = {'time': -1, 'x': 50, 'y': 50}
skip_unmatched = True
# xy_region = {'x': slice(0, 100), 'y': slice(0, 100)}
xy_region = None

models = [
    ['CanESM5', 'historical', 'r10i1p1f1'],
    ['CanESM5', 'ssp245', 'r10i1p1f1'],
    ['CanESM5', 'ssp370', 'r10i1p1f1'],
    ['CanESM5', 'ssp585', 'r10i1p1f1'],
    ['FGOALS-g3', 'historical', 'r1i1p1f1'],
    ['FGOALS-g3', 'ssp245', 'r1i1p1f1'],
    ['FGOALS-g3', 'ssp370', 'r1i1p1f1'],
    ['FGOALS-g3', 'ssp585', 'r1i1p1f1'],
    ['HadGEM3-GC31-LL', 'historical', 'r1i1p1f3'],
    ['HadGEM3-GC31-LL', 'ssp245', 'r1i1p1f3'],
    ['MIROC-ES2L', 'historical', 'r1i1p1f2'],
    ['MIROC-ES2L', 'ssp245', 'r1i1p1f2'],
    ['MIROC-ES2L', 'ssp370', 'r1i1p1f2'],
    ['MIROC-ES2L', 'ssp585', 'r1i1p1f2'],
    ['MIROC6', 'historical', 'r10i1p1f1'],
    ['MIROC6', 'ssp245', 'r10i1p1f1'],
    ['MIROC6', 'ssp585', 'r10i1p1f1'],
    ['MRI-ESM2-0', 'historical', 'r1i1p1f1'],
    ['MRI-ESM2-0', 'ssp245', 'r1i1p1f1'],
    ['MRI-ESM2-0', 'ssp370', 'r1i1p1f1'],
    ['MRI-ESM2-0', 'ssp585', 'r1i1p1f1'],
    ['UKESM1-0-LL', 'historical', 'r10i1p1f2'],
    ['UKESM1-0-LL', 'ssp245', 'r10i1p1f2'],
    ['UKESM1-0-LL', 'ssp370', 'r10i1p1f2'],
]


@dask.delayed(pure=True, traverse=False)
def finish_store(store, regions):
    zarr.consolidate_metadata(store)
    return store


@dask.delayed(pure=True, traverse=False)
def dummy_store(store):
    print(store)
    return store


def preprocess(ds: xr.Dataset) -> xr.Dataset:
    ''' preprocess datasets after loading them '''
    if 'month' in ds:
        ds = ds.drop('month')
    return ds


def load_coords(ds: xr.Dataset) -> xr.Dataset:
    ''' helper function to pre-load coordinates '''
    return ds.update(ds[list(ds.coords)].load())


def maybe_slice_region(ds: xr.Dataset, region: Dict) -> xr.Dataset:
    """helper function to pull out region of dataset"""
    if region:
        return ds.isel(**region)
    return ds


def get_slices(length: int, chunk_size: int) -> List:
    '''helper function to create a list of slices along one axis'''
    xi = range(0, length, chunk_size)

    slices = [slice(left, right) for left, right in zip(xi, xi[1:])] + [slice(xi[-1], length + 1)]
    return slices


def get_regions(ds: xr.Dataset) -> xr.Dataset:
    ''' create a list of regions (dict of slices) '''
    x_slices = get_slices(ds.dims['x'], chunks['x'])
    y_slices = get_slices(ds.dims['y'], chunks['y'])
    t_slices = [slice(None)]
    keys = ['x', 'y', 'time']
    return [dict(zip(keys, s)) for s in itertools.product(x_slices, y_slices, t_slices)]
