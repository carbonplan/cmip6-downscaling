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

def delete_chunks_encoding(ds):
    for data_var in ds.data_vars:
        if 'chunks' in ds[data_var].encoding:
            del ds[data_var].encoding['chunks']
    for coord in ds.coords:
        if 'chunks' in ds[coords].encoding:
            del ds[coords].encoding['chunks']
            
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
    print(path_tmp)
    print(path_tgt)
    delete_chunks_encoding(ds)

    api.rechunk(
        ds,
        target_chunks=chunks_dict,
        max_mem=max_mem,
        target_store=store_tgt,
        temp_store=store_tmp,
    ).execute()
    rechunked_ds = xr.open_zarr(store_tgt)  # ideally we want consolidated=True but it seems that functionality isn't offered in rechunker right now
    return rechunked_ds, path_tgt


def calc_auspicious_chunks_dict(ds,
                    target_size='100mb', 
                     chunk_dims=('lat', 'lon')):
    assert target_size=='100mb',"Apologies, but not implemented for anything but 100m right now!"
    assert type(chunk_dims)==tuple,"Your chunk_dims likely includes one string but needs a comma after it! to be a tuple!"
    target_size_bytes=100e6
    array_dims = dict(zip(ds.dims, ds.shape))
    chunks_dict = {}
    # dims not in chunk_dims should be one chunk (length -1)
    for dim in array_dims.keys():
        if dim not in chunk_dims:
            chunks_dict[dim] = -1
    # calculate the bytesize given the dtype
    data_bytesize = int(re.findall(r'\d+', str(ds.dtype))[0])
    # calculate single non_chunked_size based upon dtype
    smallest_size_one_chunk = data_bytesize * np.prod([array_dims[dim] for dim in chunks_dict.keys()])
    # the dims in chunk_dims should be of a square size that creates ~100 mb
    perfect_chunk = target_size_bytes/smallest_size_one_chunk
    # then make reasonable chunk size by rounding up (avoids corner case of it rounding down to 0...)
    perfect_chunk_length = int(np.ceil(perfect_chunk ** (1/len(chunk_dims))))
    for dim in chunk_dims:
        chunks_dict[dim] = perfect_chunk_length
    
    return chunks_dict