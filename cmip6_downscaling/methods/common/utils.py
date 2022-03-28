import pathlib
import re
from typing import Tuple

import numpy as np
import xarray as xr
from upath import UPath
from xarray_schema import DataArraySchema
from xarray_schema.base import SchemaError

from .containers import BBox


def zmetadata_exists(path: UPath):
    '''temporary workaround until path.exists() works'''

    if isinstance(path, pathlib.PosixPath):
        return (path / '.zmetadata').exists()
    else:
        return path.fs.exists(str(path / '.zmetadata'))


def subset_dataset(
    ds: xr.Dataset,
    variable: str,
    time_period: slice,
    bbox: BBox,
    chunking_schema: dict = None,
) -> xr.Dataset:
    """Uses Xarray slicing to spatially subset a dataset based on input params.

    Parameters
    ----------
    ds : xarray.Dataset
         Input Xarray dataset
    time_period : slice
        Start and end year slice. Ex: slice('2020','2020')
    bbox : dataclass
        dataclass containing the latmin,latmax,lonmin,lonmax. Class can be found in utils.
    chunking_schema : str, optional
        Desired chunking schema. ex: {'time': 365, 'lat': 150, 'lon': 150}

    Returns
    -------
    Xarray Dataset
        Spatially subsetted Xarray dataset.
    """

    subset_ds = ds.sel(
        time=time_period,
        lon=bbox.lon_slice,
        lat=bbox.lat_slice,
    )
    if chunking_schema is not None:
        target_schema = DataArraySchema(chunks=chunking_schema)
        try:
            target_schema.validate(subset_ds[variable])
        except SchemaError:
            subset_ds = subset_ds.chunk(chunking_schema)

    return subset_ds


def calc_auspicious_chunks_dict(
    da: xr.DataArray,
    chunk_dims: Tuple = ("lat", "lon"),
) -> dict:
    """Figure out a chunk size that, given the size of the dataset, the dimension(s) you want to chunk on
    and the data type, will fit under the target_size. Currently only works for 100mb which
    is the recommended chunk size for dask processing.

    Parameters
    ----------
    da : xr.DataArray
        Dataset or data array you're wanting to chunk
    chunk_dims : tuple, optional
        Dimension(s) you want to chunk along, by default ('lat', 'lon')

    Returns
    -------
    chunks_dict : dict
        Dictionary of chunk sizes with the dimensions not listed in `chunk_dims` being the
        length of that dimension (avoiding the shorthand -1 in order to play nice
        with rechunker)
    """
    if not isinstance(chunk_dims, tuple):
        raise TypeError(
            "Your chunk_dims likely includes one string but needs a comma after it! to be a tuple!"
        )
    # setting target_size_bytes to the 100mb chunksize recommended by dask. could modify in future.
    target_size_bytes = 100e6
    dim_sizes = dict(zip(da.dims, da.shape))

    # initialize chunks_dict
    chunks_dict = {}

    # dims not in chunk_dims should be one chunk (length -1), since these ones are going to
    # be contiguous while the dims in chunk_dims will be chunked
    for dim in dim_sizes.keys():
        if dim not in chunk_dims:
            # we'll only add the unchunked dimensions to chunks_dict right now
            # rechunker doesn't like the the shorthand of -1 meaning the full length
            # so we'll always just give it the full length of the dimension
            chunks_dict[dim] = dim_sizes[dim]
    # calculate the bytesize given the dtype
    data_bytesize = int(re.findall(r"\d+", str(da.dtype))[0])
    # calculate the size of the smallest minimum chunk based upon dtype and the
    # length of the unchunked dim(s). chunks_dict currently only has unchunked dims right now
    smallest_size_one_chunk = data_bytesize * np.prod(
        [dim_sizes[dim] for dim in chunks_dict.keys()]
    )
    # the dims in chunk_dims should be of a square size that creates ~100 mb
    perfect_chunk = target_size_bytes / smallest_size_one_chunk
    # then make reasonable chunk size by rounding up (avoids corner case of it rounding down to 0...)
    perfect_chunk_length = int(np.ceil(perfect_chunk ** (1 / len(chunk_dims))))

    for dim in chunk_dims:
        # check that the rounding up as part of the `perfect_chunk_length` calculation
        # didn't make the chunk sizes bigger than the array itself, and if so
        # clip it to that size
        chunks_dict[dim] = min(perfect_chunk_length, dim_sizes[dim])

    return chunks_dict
