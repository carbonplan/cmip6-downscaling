import re
from typing import Tuple, Union

import numpy as np
import xarray as xr
from xarray_schema import DataArraySchema
from xarray_schema.base import SchemaError

from .containers import BBox


def lon_to_180(ds):
    '''Converts longitude values to (-180, 180)

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with `lon` coordinate

    Returns
    -------
    xr.Dataset
        Copy of `ds` with updated coordinates

    See also
    --------
    cmip6_preprocessing.preprocessing.correct_lon
    '''

    ds = ds.copy()

    lon = ds["lon"].where(ds["lon"] < 180, ds["lon"] - 360)
    ds = ds.assign_coords(lon=lon)

    if not (ds["lon"].diff(dim="lon") > 0).all():
        ds = ds.reindex(lon=np.sort(ds["lon"].data))

    if "lon_bounds" in ds.variables:
        lon_b = ds["lon_bounds"].where(ds["lon_bounds"] < 180, ds["lon_bounds"] - 360)
        ds = ds.assign_coords(lon_bounds=lon_b)

    return ds


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
    da: Union[xr.DataArray, xr.Dataset],
    target_size: str = "100mb",
    chunk_dims: Tuple = ("lat", "lon"),
) -> dict:
    """Figure out a chunk size that, given the size of the dataset, the dimension(s) you want to chunk on
    and the data type, will fit under the target_size. Currently only works for 100mb which
    is the recommended chunk size for dask processing.
    Parameters
    ----------
    da : Union[xr.DataArray, xr.Dataset]
        Dataset or data array you're wanting to chunk
    target_size : str, optional
        Target size for chunk- dask recommends 100mb, by default '100mb'
    chunk_dims : tuple, optional
        Dimension(s) you want to chunk along, by default ('lat', 'lon')
    Returns
    -------
    chunks_dict : dict
        Dictionary of chunk sizes
    """
    assert target_size == "100mb", "Apologies, but not implemented for anything but 100m right now!"
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
    data_bytesize = int(re.findall(r"\d+", str(da.dtype))[0])
    # calculate single non_chunked_size based upon dtype
    smallest_size_one_chunk = data_bytesize * np.prod(
        [array_dims[dim] for dim in chunks_dict.keys()]
    )
    # the dims in chunk_dims should be of a square size that creates ~100 mb
    perfect_chunk = target_size_bytes / smallest_size_one_chunk
    # then make reasonable chunk size by rounding up (avoids corner case of it rounding down to 0...)
    perfect_chunk_length = int(np.ceil(perfect_chunk ** (1 / len(chunk_dims))))
    for dim in chunk_dims:
        chunks_dict[dim] = min(perfect_chunk_length, array_dims[dim])
