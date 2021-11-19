import itertools
import os
from typing import Dict, List

import dask
import fsspec
import pandas as pd
import xarray as xr
import zarr

awc_fill = 50  # mm
hist_time = slice("1950", "2014")
future_time = slice("2015", "2120")
chunks = {"time": -1, "x": 50, "y": 50}
skip_unmatched = True
# xy_region = {'x': slice(0, 100), 'y': slice(0, 100)}
xy_region = None


def get_cmip_runs(comp=True, unique=True, has_match=True):

    with fsspec.open(
        "az://carbonplan-downscaling/cmip6/ssps_with_matching_historical_members.csv",
        mode="r",
        account_name="carbonplan",
    ) as f:
        df = pd.read_csv(f).drop(columns=["Unnamed: 0", "path"])

    if has_match:
        df = df[df.has_match]

    df["comp"] = [
        len(set(df[(df["model"] == d[1]["model"]) & (df["member"] == d[1]["member"])]["scenario"]))
        == 4
        for d in df.iterrows()
    ]
    df["unique"] = [
        d[1]["member"] == df[(df["model"] == d[1]["model"])]["member"].values[0]
        for d in df.iterrows()
    ]

    if comp and unique:
        df = df[df.comp & df.unique]
    elif comp and not unique:
        df = df[df.comp]
    elif unique and not comp:
        df = df[df.unique]

    return df[["model", "scenario", "member"]]


@dask.delayed(pure=True, traverse=False)
def finish_store(store, regions):
    zarr.consolidate_metadata(store)
    return store


@dask.delayed(pure=True, traverse=False)
def dummy_store(store):
    print(store)
    return store


def preprocess(ds: xr.Dataset) -> xr.Dataset:
    """preprocess datasets after loading them"""
    if "month" in ds:
        ds = ds.drop("month")
    return ds


def load_coords(ds: xr.Dataset) -> xr.Dataset:
    """helper function to pre-load coordinates"""
    return ds.update(ds[list(ds.coords)].load())


def maybe_slice_region(ds: xr.Dataset, region: Dict) -> xr.Dataset:
    """helper function to pull out region of dataset"""
    if region:
        return ds.isel(**region)
    return ds


def get_slices(length: int, chunk_size: int) -> List:
    """helper function to create a list of slices along one axis"""
    xi = range(0, length, chunk_size)

    slices = [slice(left, right) for left, right in zip(xi, xi[1:])] + [slice(xi[-1], length + 1)]
    return slices


def get_regions(ds: xr.Dataset) -> xr.Dataset:
    """create a list of regions (dict of slices)"""
    x_slices = get_slices(ds.dims["x"], chunks["x"])
    y_slices = get_slices(ds.dims["y"], chunks["y"])
    t_slices = [slice(None)]
    keys = ["x", "y", "time"]
    return [dict(zip(keys, s)) for s in itertools.product(x_slices, y_slices, t_slices)]


def get_store(bucket, prefix, account_key=None):
    """helper function to create a zarr store"""
    if account_key is None:
        account_key = os.environ.get("AccountKey", None)

    store = zarr.storage.ABSStore(
        bucket, prefix=prefix, account_name="cmip6downscaling", account_key=account_key
    )
    return store
