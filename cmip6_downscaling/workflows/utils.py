import os
import random
import re
import string

import fsspec
import numpy as np
import xarray as xr
import zarr
from rechunker import rechunk
import xesmf as xe
from typing import Union



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

def load_paths(paths): # What type do i use here since paths is of unknown length? : list[str]):
    ds_list = [xr.open_zarr(path) for path in paths]
    return ds_list

def temp_file_name():
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(10))


def delete_chunks_encoding(ds : Union[xr.Dataset, xr.DataArray]):
    for data_var in ds.data_vars:
        if 'chunks' in ds[data_var].encoding:
            del ds[data_var].encoding['chunks']
    for coord in ds.coords:
        if 'chunks' in ds[coord].encoding:
            del ds[coord].encoding['chunks']


def rechunk_zarr_array(zarr_array, connection_string, variable, chunk_dims : tuple = ('time',), max_mem="10MB"):
    """[summary]

    Parameters
    ----------
    ds : [zarr array]
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
    chunks_dict = {variable: calc_auspicious_chunks_dict(zarr_array, chunk_dims=chunk_dims), 
                      'time': None, # write None here because you don't want to rechunk this array
                        'lon': None,
                        'lat': None}
    
    delete_chunks_encoding(zarr_array)
    path_tmp = "az://cmip6/temp/{}.zarr".format(temp_file_name())
    path_tgt =  "az://cmip6/temp/{}.zarr".format(temp_file_name())
    temp_store = fsspec.get_mapper(path_tmp, connection_string=connection_string)
    target_store = fsspec.get_mapper(path_tgt, connection_string=connection_string)
    # delete_chunks_encoding(ds) # need to do this before since it wont work on zarr array
    # for some reason doing this on zarr arrays is faster than on xr.open_zarr - it calls `copy_chunk` less.
    # TODO: could switch this to a validation with xarray schema - confirm that the chunks are all uniform and 
    # if not, chunk them according to the spec provided by `calc_auspicious_chunks_dict`
    try:
        rechunk_plan = rechunk(zarr_array, chunks_dict, max_mem, target_store, temp_store=temp_store)
        rechunk_plan.execute()
    except ValueError:
        print('WARNING: Failed to write zarr store, perhaps because of variable chunk sizes, trying to rechunk it')
        rechunk_plan = rechunk(zarr_array.chunk(calc_auspicious_chunks_dict(zarr_array, 
                                chunk_dims=chunk_dims)), 
                                hunks_dict, 
                                max_mem, 
                                target_store, 
                                temp_store=temp_store)
    rechunked_ds = xr.open_zarr(
        target_store
    )  # ideally we want consolidated=True but it seems that functionality isn't offered in rechunker right now
    # we can just add a consolidate_metadata step here to do it after the fact (once rechunker is done) but only
    # necessary if we'll reopen this rechukned_ds multiple times
    return rechunked_ds, path_tgt



def calc_auspicious_chunks_dict(da : Union[xr.DataArray, xr.Dataset], target_size : str = '100mb', chunk_dims : tuple = ('lat', 'lon')):
    assert target_size == '100mb', "Apologies, but not implemented for anything but 100m right now!"
    assert (
        type(chunk_dims) == tuple
    ), "Your chunk_dims likely includes one string but needs a comma after it! to be a tuple!"
    if type(da) == xr.Dataset:
        da = da.to_array().squeeze()
    target_size_bytes = 100e6
    array_dims = dict(zip(da.dims, da.shape))
    chunks_dict = {}
    # dims not in chunk_dims should be one chunk (length -1)
    for dim in array_dims.keys():
        if dim not in chunk_dims:
            # rechunker doesn't like the the shorthand of -1 meaning the full length
            #  so we'll always just give it the full length of the dimension
            chunks_dict[dim] = array_dims[dim]
    # calculate the bytesize given the dtype
    data_bytesize = int(re.findall(r'\d+', str(da.dtype))[0])
    # calculate single non_chunked_size based upon dtype
    smallest_size_one_chunk = data_bytesize * np.prod(
        [array_dims[dim] for dim in chunks_dict.keys()]
    )
    # the dims in chunk_dims should be of a square size that creates ~100 mb
    perfect_chunk = target_size_bytes / smallest_size_one_chunk
    # then make reasonable chunk size by rounding up (avoids corner case of it rounding down to 0...)
    perfect_chunk_length = int(np.ceil(perfect_chunk ** (1 / len(chunk_dims))))
    for dim in chunk_dims:
        chunks_dict[dim] = perfect_chunk_length

    return chunks_dict

def regrid_dataset(ds : xr.Dataset, target_grid_ds : xr.Dataset, variable : str, connection_string : str):
    ds_rechunked, ds_rechunked_path = rechunk_zarr_array(ds, connection_string, variable, chunk_dims=('time',), max_mem="1GB")
    regridder = xe.Regridder(
        ds_rechunked, target_grid_ds, "bilinear", extrap_method="nearest_s2d"
    )
    ds_regridded = regridder(ds_rechunked)
    return ds_regridded
    
def write_dataset(ds, path):
    store = fsspec.get_mapper(path)
    try:
        ds.to_zarr(store, mode='w', consolidated=True)
    except ValueError:
        # if your chunk size isn't uniform you'll probably get a value error so
        # you can try doing this you can rechunk it
        print('WARNING: Failed to write zarr store, perhaps because of variable chunk sizes, trying to rechunk it')
        chunks_dict = calc_auspicious_chunks_dict(ds, chunk_dims=('time',))
        print(chunks_dict)
        delete_chunks_encoding(ds)
        ds.chunk(chunks_dict).to_zarr(store, mode='w', consolidated=True)
