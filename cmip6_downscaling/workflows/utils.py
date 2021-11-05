import os
import random
import string

import fsspec
import xarray as xr
import zarr
from rechunker import api


def get_store(prefix, account_key=None):
    """helper function to create a zarr store"""

    if account_key is None:
        account_key = os.environ.get("BLOB_ACCOUNT_KEY", None)

    store = zarr.storage.ABSStore(
        "carbonplan-downscaling",
        prefix=prefix,
        account_name="carbonplan",
        account_key=account_key,
    )
    return store


def temp_file_name():
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(10))


def rechunk_dataset(ds, chunks_dict, connection_string, max_mem="500MB"):
    """[summary]

    Parameters
    ----------
    ds : [xarray dataset]
        [description]
    chunks_dict : dict
        Desired chunks sizes for each variable. They can either be specified in tuple or dict form.
        But dict is probably safer! When working in space you proabably want somehting like
        (1, -1, -1) where dims are of form (time, lat, lon). In time you probably want
        (-1, 10, 10). You likely want the same chunk sizes for each variable.
    connection_string : str
        [description]
    max_mem : str
        Likely can go higher than 500MB!

    Returns
    -------
    [type]
        [description]
    """
    path_tmp, path_tgt = temp_file_name(), temp_file_name()

    store_tmp = fsspec.get_mapper(
        "az://cmip6/temp/{}.zarr".format(path_tmp), connection_string=connection_string
    )
    store_tgt = fsspec.get_mapper(
        "az://cmip6/temp/{}.zarr".format(path_tgt), connection_string=connection_string
    )

    if "chunks" in ds["tasmax"].encoding:
        del ds["tasmax"].encoding["chunks"]

    api.rechunk(
        ds,
        target_chunks=chunks_dict,
        max_mem=max_mem,
        target_store=store_tgt,
        temp_store=store_tmp,
    ).execute()
    print("done with rechunk")
    rechunked_ds = xr.open_zarr(store_tgt)  # ideally we want consolidated=True but
    # it isn't working for some reason
    print("done with open_zarr")
    print(rechunked_ds["tasmax"].data.chunks)
    return rechunked_ds, path_tgt
