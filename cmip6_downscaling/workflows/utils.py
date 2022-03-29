import random
import re
import string
from typing import Optional, Tuple, Union

import fsspec
import numpy as np
import xarray as xr
import xesmf as xe
import zarr
from rechunker import rechunk
from xarray_schema import DataArraySchema, DatasetSchema
from xarray_schema.base import SchemaError

from cmip6_downscaling import config

schema_maps_chunks = DataArraySchema(chunks={'lat': -1, 'lon': -1})

from dataclasses import dataclass


def subset_dataset(
    ds: xr.Dataset,
    variable: str,
    time_period: slice,
    bbox: dataclass,
    chunking_schema: Optional[dict] = None,
) -> xr.Dataset:
    """Uses Xarray slicing to spatially subset a dataset based on input params.

    Parameters
    ----------
    ds : xarray.Dataset
         Input Xarray dataset
    time_period: slice
        Start and end year slice. Ex: slice('2020','2020')
    bbox: dataclass
        dataclass containing the latmin,latmax,lonmin,lonmax. Class can be found in utils.
    chunking_schema : str, optional
        Desired chunking schema. ex: {'time': 365, 'lat': 150, 'lon': 150}

    Returns
    -------
    Xarray Dataset
        Spatially subsetted Xarray dataset.
    """

    """
    lon=slice(float(lonmin), float(lonmax)),
    lat=slice(float(latmax), float(latmin)),"""

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


def generate_batches(n, batch_size, buffer_size, one_indexed=False):
    """
    Given the max value n, batch_size, and buffer_size, returns batches (include the buffer) and
    cores (exclude the buffer). For the smallest numbers, the largest values would be included in the buffer, and
    vice versa. For example, with n=10, batch_size=5, buffer_size=3, one_indexed=False. The `cores` output will contain
    [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], and `batches` output will contain [[7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2]].

    Parameters
    ----------
    ds : xarray.Dataset
         Input Xarray dataset
    start_time : str
        Starting Time
    end_time : str
        Ending Time
    latmin : str
        Latitude Minimum
    latmax : str
        Latitude Maximum
    lonmin : str
        Longitude Minimum
    lonmax : str
        Longitude Maximum

    Returns
    -------
    Xarray Dataset
        Spatially subsetted Xarray dataset.
    """

    cores = []
    batches = []
    if one_indexed:
        xmin = 1
        xmax = n + 1
    else:
        xmin = 0
        xmax = n
    for start in range(xmin, xmax, batch_size):
        end = min(start + batch_size, xmax)
        cores.append(np.arange(start, end))

        # add buffer
        end = end + buffer_size
        start = start - buffer_size
        batch = np.arange(start, end)
        batch[batch < xmin] += n
        batch[batch > xmax - 1] -= n
        batches.append(batch)

    return batches, cores


def load_paths(
    paths,
):  # What type do i use here since paths is of unknown length? : list[str]):
    ds_list = [xr.open_zarr(path) for path in paths]
    return ds_list


def temp_file_name():
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(10))


def delete_chunks_encoding(ds: Union[xr.Dataset, xr.DataArray]):
    for data_var in ds.data_vars:
        if "chunks" in ds[data_var].encoding:
            del ds[data_var].encoding["chunks"]
    for coord in ds.coords:
        if "chunks" in ds[coord].encoding:
            del ds[coord].encoding["chunks"]


# def lon_to_180(ds):
#     '''Converts longitude values to (-180, 180)

#     Parameters
#     ----------
#     ds : xr.Dataset
#         Input dataset with `lon` coordinate

#     Returns
#     -------
#     xr.Dataset
#         Copy of `ds` with updated coordinates

#     See also
#     --------
#     cmip6_preprocessing.preprocessing.correct_lon
#     '''

#     ds = ds.copy()

#     lon = ds["lon"].where(ds["lon"] < 180, ds["lon"] - 360)
#     ds = ds.assign_coords(lon=lon)

#     if not (ds["lon"].diff(dim="lon") > 0).all():
#         ds = ds.reindex(lon=np.sort(ds["lon"].data))

#     if "lon_bounds" in ds.variables:
#         lon_b = ds["lon_bounds"].where(ds["lon_bounds"] < 180, ds["lon_bounds"] - 360)
#         ds = ds.assign_coords(lon_bounds=lon_b)

#     return ds


def make_rechunker_stores(
    output_path: Optional[str] = None,
) -> Tuple[fsspec.FSMap, fsspec.FSMap, str]:
    """Initialize two stores for rechunker to use as temporary and final rechunked locations
    Parameters
    ----------
    output_path : str, optional
        Output path for rechunker stores
    Returns
    -------
    temp_store, target_store, path_tgt : tuple[fsspec.mapping.FSmap, fsspec.mapping.FSmap, string]
        Stores where rechunker will write and the path to the target store
    """
    storage_options = config.get('storage.temporary.storage_options')
    path_tmp = config.get('storage.temporary.uri') + f"/{temp_file_name()}.zarr"

    temp_store = fsspec.get_mapper(path_tmp, **storage_options)

    if output_path is None:
        output_path = config.get('storage.temporary.uri') + f"/{temp_file_name()}.zarr"
    target_store = fsspec.get_mapper(output_path, **storage_options)
    print(f'this is temp_path: {path_tmp}. \n this is target_path: {output_path}')
    return temp_store, target_store, output_path


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

    return chunks_dict


def rechunk_zarr_array_with_caching(
    zarr_path: str,
    chunking_approach: Optional[str] = None,
    template_chunk_array: Optional[xr.Dataset] = None,
    output_path: Optional[str] = None,
    max_mem: str = "2GB",
    overwrite: bool = False,
    **kwargs,
) -> str:
    """Use `rechunker` package to adjust chunks of dataset to a form
    conducive for your processing.
    Parameters
    ----------
    zarr_path : str
        path to zarr store
    output_path: str
        Path to where the output data is saved. If output path is not empty, the content would be loaded and the schema checked. If the schema check passed,
        the content will be returned without rechunking again (i.e. caching); else, the content can be overwritten (see overwrite option).
    chunking_approach : str
        Has to be one of `full_space` or `full_time`. If `full_space`, the data will be rechunked such that the space dimensions are contiguous (i.e. each chunk
        will contain full maps). If `full_time`, the data will be rechunked such that the time dimension is contiguous (i.e. each chunk will contain full time
        series)
    max_mem : str
        The max memory you want to allow for a chunk. Probably want it to be around 100 MB, but that
        is also controlled by the `calc_auspicious_chunk_sizes` calls.
    overwrite : bool
        Whether to overwrite the content saved at output_path if the content did not pass schema check.
    Returns
    -------
    target_path : str
        Path to rechunked dataset
    """
    # step

    group = zarr.open_consolidated(zarr_path, mode='r')
    ds = xr.open_zarr(zarr_path)

    # determine the chunking schema
    if template_chunk_array is None:
        if chunking_approach == 'full_space':
            chunk_dims = ('time',)  # if we need full maps, chunk along the time dimension
        elif chunking_approach == 'full_time':
            chunk_dims = (
                'lat',
                'lon',
            )  # if we need full time series, chunk along the lat/lon dimensions
        else:
            raise NotImplementedError("chunking_approach must be in ['full_space', 'full_time']")
        example_var = list(ds.data_vars)[0]
        chunk_def = calc_auspicious_chunks_dict(ds[example_var], chunk_dims=chunk_dims)
    else:
        example_var = list(ds.data_vars)[0]
        chunk_def = {
            'time': min(template_chunk_array.chunks['time'][0], len(ds.time)),
            'lat': min(template_chunk_array.chunks['lat'][0], len(ds.lat)),
            'lon': min(template_chunk_array.chunks['lon'][0], len(ds.lon)),
        }
    chunks_dict = {
        'time': None,  # write None here because you don't want to rechunk this array
        'lon': None,
        'lat': None,
    }
    for var in ds.data_vars:
        chunks_dict[var] = chunk_def

    # make the schema for what you want the rechunking routine to produce
    # so that you can check whether what you passed in (zarr_array) already looks like that
    # if it does, you'll skip the rechunking!
    schema_dict = {}
    for var in ds.data_vars:
        schema_dict[var] = DataArraySchema(chunks=chunk_def)
    target_schema = DatasetSchema(schema_dict)
    # make storage patterns

    if output_path is not None:
        output_path = config.get('storage.intermediate.uri') + '/' + output_path
    temp_store, target_store, target_path = make_rechunker_stores(output_path)
    # check and see if the output is empty, if there is content, check that it's chunked correctly
    if len(target_store) > 0:
        output = xr.open_zarr(target_store)
        try:
            # if the content in target path is correctly chunked, return
            target_schema.validate(output)
            return output_path

        except SchemaError:
            if overwrite:
                target_store.clear()
            else:
                raise NotImplementedError(
                    'The content in the output path is incorrectly chunked, but overwrite is disabled.'
                    'Either clear the output or enable overwrite by setting overwrite=True'
                )

    try:
        # now check if the input is already correctly chunked. If so, save to the output location and return
        target_schema.validate(ds)
        if output_path is not None:
            ds.to_zarr(output_path)
            return output_path
        else:
            return zarr_path

    except SchemaError:
        rechunk_plan = rechunk(
            group,
            chunks_dict,
            max_mem,
            target_store,
            temp_store=temp_store,
        )

        rechunk_plan.execute()
        # remove any temp stores created by task
        # remove_stores([temp_store])
        # ideally we want consolidated=True but it seems that functionality isn't offered in rechunker right now
        # we can just add a consolidate_metadata step here to do it after the fact (once rechunker is done) but only
        # necessary if we'll reopen this rechukned_ds multiple times
        zarr.consolidate_metadata(target_store)
        return target_path


def regrid_ds(
    ds_path: str,
    target_grid_ds: xr.Dataset,
    rechunked_ds_path: Optional[str] = None,
    **kwargs,
) -> xr.Dataset:

    """Regrid a dataset to a target grid. For use in both coarsening or interpolating to finer resolution.
    The function will check whether the dataset is chunked along time (into spatially-contiguous maps)
    and if not it will rechunk it. **kwargs are used to construct target path
    Parameters
    ----------
    ds : xr.Dataset
        Dataset you want to regrid
    target_grid_ds : xr.Dataset
        Template dataset whose grid you'll match
    Returns
    -------
    ds_regridded : xr.Dataset
        Final regridded dataset
    """
    # regridding requires ds to be contiguous in lat/lon, check if the input matches the
    # target_schema.
    # import pdb
    # pdb.set_trace()
    ds = xr.open_zarr(ds_path)

    schema_dict = {}
    for var in ds.data_vars:
        schema_dict[var] = schema_maps_chunks
    target_schema = DatasetSchema(schema_dict)

    try:
        target_schema.validate(target_grid_ds)
    except SchemaError:
        target_grid_ds = target_grid_ds.chunk({'lat': -1, 'lon': -1}).load()

    try:
        target_schema.validate(ds)
        ds_rechunked = ds

    except SchemaError:
        ds_rechunked_path = rechunk_zarr_array_with_caching(
            zarr_path=ds_path, chunking_approach='full_space', max_mem='1GB'
        )
    ds_rechunked = xr.open_zarr(ds_rechunked_path)
    regridder = xe.Regridder(ds_rechunked, target_grid_ds, "bilinear", extrap_method="nearest_s2d")
    ds_regridded = regridder(ds_rechunked)

    return ds_regridded
