from __future__ import annotations

import pathlib
import re

import numpy as np
import regionmask
import xarray as xr
import zarr
from upath import UPath
from xarray_schema import DataArraySchema
from xarray_schema.base import SchemaError

import cmip6_downscaling.methods.common.containers as containers


def zmetadata_exists(path: UPath):
    '''temporary workaround until path.exists() works'''

    if isinstance(path, pathlib.PosixPath):
        return (path / '.zmetadata').exists()
    elif isinstance(path, UPath):
        return path.fs.exists(str(path / '.zmetadata'))
    else:
        return (UPath(path) / '.zmetadata').exists()


def blocking_to_zarr(ds: xr.Dataset, target):
    '''helper function to write a xarray Dataset to a zarr store.

    The function blocks until the write is complete then writes Zarr's consolidated metadata
    '''

    # testing a workaround
    if str(target).startswith('/'):
        print('fell back to using a string target')
        target = 'az://flow-outputs' + str(target)

    t = ds.to_zarr(target, mode='w', compute=False)
    t.compute(retries=5)
    zarr.consolidate_metadata(target)


def subset_dataset(
    ds: xr.Dataset,
    variable: str,
    time_period: slice,
    bbox: containers.BBox,
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


def apply_land_mask(ds: xr.Dataset) -> xr.Dataset:
    """
    Apply a land mask to a dataset with lat/lon coordinates

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset to mask.

    Returns
    -------
    xr.Dataset
    """

    mask = regionmask.defined_regions.natural_earth_v5_0_0.land_10.mask(ds)
    return ds.where(mask == 0)


def calc_auspicious_chunks_dict(
    da: xr.DataArray,
    chunk_dims: tuple = ("lat", "lon"),
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
    # calculate the bytesize given the dtype bitsize and divide by 8
    data_bytesize = int(re.findall(r"\d+", str(da.dtype))[0]) / 8
    # calculate the size of the smallest minimum chunk based upon dtype and the
    # length of the unchunked dim(s). chunks_dict currently only has unchunked dims right now
    smallest_size_one_chunk = data_bytesize * np.prod(
        [dim_sizes[dim] for dim in chunks_dict.keys()]
    )

    # the dims in chunk_dims should be of an array size (as in number of elements in the array)
    # that creates ~100 mb. `perfect_chunk` is the how many of the smallest_size_chunks you can
    # handle at once while still staying below the `target_size_bytes`
    perfect_chunk = target_size_bytes / smallest_size_one_chunk

    # then make reasonable chunk size by rounding up (avoids corner case of it rounding down to 0...)
    # but if the array is oblong it might get big (? is logic right there- might it get small??)
    perfect_chunk_length = int(np.ceil(perfect_chunk ** (1 / len(chunk_dims))))
    for dim in chunk_dims:
        # check that the rounding up as part of the `perfect_chunk_length` calculation
        # didn't make the chunk sizes bigger than the array itself, and if so
        # clip it to that size
        chunks_dict[dim] = min(perfect_chunk_length, dim_sizes[dim])
    return chunks_dict


def _resample_func(ds, freq='1MS'):
    """Helper function to apply resampling."""
    out_ds = xr.Dataset(attrs=ds.attrs)

    for v in ds.data_vars:
        if v in ['tasmax', 'tasmin']:
            out_ds[v] = ds[v].resample(time=freq).mean(dim='time')
        elif v in ['pr']:
            out_ds[v] = ds[v].resample(time=freq).sum(dim='time')
        else:
            print(f'{v} not implemented')

    return out_ds


def resample_wrapper(ds, freq='1MS'):
    """Wrapper function for resampling.

    Parameters
    ----------
    ds : xarray.Dataset
        Input xarray dataset.
    freq : str, optional
        resample frequency, by default '1MS'.

    Returns
    -------
    xr.Dataset
        xarray dataset resampled to freq
    """

    # Use _resample_func() to make template dataset
    template = _resample_func(ds, freq=freq).chunk({'time': -1})
    # Apply map_blocks input dataset
    out_ds = xr.map_blocks(_resample_func, ds, kwargs={'freq': freq}, template=template)

    return out_ds
