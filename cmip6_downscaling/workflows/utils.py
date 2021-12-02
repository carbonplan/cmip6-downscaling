import os
import random
import re
import string
from typing import Tuple, Union

import fsspec
import numpy as np
import xarray as xr
import xesmf as xe
import zarr
from rechunker import rechunk
from xarray_schema import DataArraySchema
from xarray_schema.base import SchemaError

schema_maps_chunks = DataArraySchema(chunks={'lat': -1, 'lon': -1})


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


def load_paths(paths):  # What type do i use here since paths is of unknown length? : list[str]):
    ds_list = [xr.open_zarr(path) for path in paths]
    return ds_list


def temp_file_name():
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(10))


def delete_chunks_encoding(ds: Union[xr.Dataset, xr.DataArray]):
    for data_var in ds.data_vars:
        if 'chunks' in ds[data_var].encoding:
            del ds[data_var].encoding['chunks']
    for coord in ds.coords:
        if 'chunks' in ds[coord].encoding:
            del ds[coord].encoding['chunks']


def make_rechunker_stores(
    connection_string: str,
):  # -> tuple[fsspec.mapping.FSmap, fsspec.mapping.FSmap]:
    """Initialize two stores for rechunker to use as temporary and final rechunked locations

    Parameters
    ----------
    connection_string : str
        Connection string to give you write access to the stores

    Returns
    -------
    temp_store, target_store, path_tgt : tuple[fsspec.mapping.FSmap, fsspec.mapping.FSmap, string]
        Stores where rechunker will write and the path to the target store
    """
    path_tmp = "az://cmip6/temp/{}.zarr".format(temp_file_name())
    path_tgt = "az://cmip6/temp/{}.zarr".format(temp_file_name())
    temp_store = fsspec.get_mapper(path_tmp, connection_string=connection_string)
    target_store = fsspec.get_mapper(path_tgt, connection_string=connection_string)
    return temp_store, target_store, path_tgt


def rechunk_zarr_array(
    zarr_array: xr.Dataset,
    zarr_array_location: str,
    connection_string: str,
    variable: str,
    chunk_dims: Union[Tuple, dict] = ('time',),
    max_mem: str = "200MB",
):
    """Use `rechunker` package to adjust chunks of dataset to a form
    conducive for your processing.

    Parameters
    ----------
    zarr_array : zarr or xarray dataset
        Dataset you want to rechunk.
    zarr_array_location: str
        Path to where the data is sitting.
    chunk_dims : Union[Tuple, dict]
        Information for chunking the ds. If a dict is passed, it will rechunk following sizes as specified. The dict should look like:
            {variable: {'lat': chunk_size_lat,
                        'lon': chunk_size_lon,
                        'time': chunk_size_lon}
            'lon': None,
            'lat': None,
            'time': None}.
        If a tuple is passed, it is the dimension(s) along which you want to chunk ds, and the optimal chunk sizes will get calculated internally.
    connection_string : str
        Connection string to give you write access
    max_mem : str
        The max memory you want to allow for a chunk. Probably want it to be around 100 MB, but that
        is also controlled by the `calc_auspicious_chunk_sizes` calls.

    Returns
    -------
    rechunked_ds, path_tgt : Tuple[xr.Dataset, str]
        Rechunked dataset as well as string of location where it's stored.
    """
    if type(chunk_dims) == tuple:
        chunks_dict = {
            variable: calc_auspicious_chunks_dict(zarr_array, chunk_dims=chunk_dims),
            'time': None,  # write None here because you don't want to rechunk this array
            'lon': None,
            'lat': None,
        }
    elif type(chunk_dims) == dict:
        chunks_dict = chunk_dims
        # ensure that the chunks_dict looks the way you want it to as {variable: {'lat': chunk_size_lat, 'lon': chunk_size_lon, 'time': chunk_size_lon}
        # 'lon': None, 'lat': None, 'time': none}
        assert variable in chunks_dict
        for dim in ['lat', 'lon', 'time']:
            chunks_dict[dim] = None
            assert dim in chunks_dict[variable]

    # make the schema for what you want the rechunking routine to produce
    # so that you can check whether what you passed in (zarr_array) already looks like that
    # if it does, you'll skip the rechunking!
    target_schema = DataArraySchema(chunks=chunks_dict[variable])
    try:
        # first confirm that you have a zarr_array_location
        assert zarr_array_location is not None
        target_schema.validate(zarr_array[variable])
        # return back the dataset you introduced, and the path is None since you haven't created a new dataset
        return zarr_array, zarr_array_location
    except (SchemaError, AssertionError):
        delete_chunks_encoding(zarr_array)
        temp_store, target_store, path_tgt = make_rechunker_stores(connection_string)
        # delete_chunks_encoding(ds) # need to do this before since it wont work on zarr array
        # for some reason doing this on zarr arrays is faster than on xr.open_zarr - it calls `copy_chunk` less.
        # TODO: could switch this to a validation with xarray schema - confirm that the chunks are all uniform and
        # if not, chunk them according to the spec provided by `calc_auspicious_chunks_dict`
        try:
            rechunk_plan = rechunk(
                zarr_array, chunks_dict, max_mem, target_store, temp_store=temp_store
            )
            rechunk_plan.execute(retries=5)
        except ValueError:
            print(
                'WARNING: Failed to write zarr store, perhaps because of variable chunk sizes, trying to rechunk it'
            )
            # make new stores in case it failed mid-write. alternatively could clean up that store but
            # we don't have delete permission currently
            temp_store, target_store, path_tgt = make_rechunker_stores(connection_string)
            print(chunks_dict[variable])
            delete_chunks_encoding(zarr_array)
            print(zarr_array.chunk(chunks_dict[variable]))
            rechunk_plan = rechunk(
                zarr_array.chunk(chunks_dict[variable]),
                chunks_dict,
                max_mem,
                target_store,
                temp_store=temp_store,
            )
            rechunk_plan.execute(retries=5)
        rechunked_ds = xr.open_zarr(
            target_store
        )  # ideally we want consolidated=True but it seems that functionality isn't offered in rechunker right now
        # we can just add a consolidate_metadata step here to do it after the fact (once rechunker is done) but only
        # necessary if we'll reopen this rechukned_ds multiple times
        return rechunked_ds, path_tgt


def calc_auspicious_chunks_dict(
    da: Union[xr.DataArray, xr.Dataset],
    target_size: str = '100mb',
    chunk_dims: Tuple = ('lat', 'lon'),
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


def regrid_dataset(
    ds: xr.Dataset,
    ds_path: Union[str, None],
    target_grid_ds: xr.Dataset,
    variable: str,
    connection_string: str,
) -> Tuple[xr.Dataset, str]:
    """Regrid a dataset to a target grid. For use in both coarsening or interpolating to finer resolution.
    The function will check whether the dataset is chunked along time (into spatially-contiguous maps)
    and if not it will rechunk it.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset you want to regrid
    ds_path: str
        Path to where the dataset is stored. If None
    target_grid_ds : xr.Dataset
        Template dataset whose grid you'll match
    variable : str
        variable you're working with
    connection_string : str
        Connection string to give you write access to the stores

    Returns
    -------
    ds_regridded : xr.Dataset
        Final regridded dataset
    ds_rechunked_path : str
        Path of original dataset rechunked along time (perhaps to be used by follow-on steps)
    """
    # use xarray_schema to check that the dataset is chunked into map space already
    # and if not rechunk it into map space (only do the rechunking if it's necessary)
    # we only have dataarray schema implemented now- can switch to datasets once that's done
    try:
        schema_maps_chunks.validate(ds[variable])
        ds_rechunked = ds
        ds_rechunked_path = ds_path
    except SchemaError:
        # assert ds_path is not None, 'Must pass path to dataset so that you can rechunk it'
        ds_rechunked, ds_rechunked_path = rechunk_zarr_array(
            ds, ds_path, connection_string, variable, chunk_dims=('time',), max_mem="1GB"
        )

    regridder = xe.Regridder(ds_rechunked, target_grid_ds, "bilinear", extrap_method="nearest_s2d")
    ds_regridded = regridder(ds_rechunked)
    return ds_regridded, ds_rechunked_path


def get_spatial_anomalies(
    coarse_obs_path, fine_obs_rechunked_path, variable, connection_string
) -> xr.Dataset:
    """Calculate the seasonal cycle (12 timesteps) spatial anomaly associated
    with aggregating the fine_obs to a given coarsened scale and then reinterpolating
    it back to the original spatial resolution. The outputs of this function are
    dependent on three parameters:
    * a grid (as opposed to a specific GCM since some GCMs run on the same grid)
    * the time period which fine_obs (and by construct coarse_obs) cover
    * the variable

    Parameters
    ----------
    coarse_obs : xr.Dataset
        Coarsened to a GCM resolution. Chunked along time.
    fine_obs_rechunked_path : xr.Dataset
        Original observation spatial resolution. Chunked along time.
    variable: str
        The variable included in the dataset.

    Returns
    -------
    seasonal_cycle_spatial_anomalies : xr.Dataset
        Spatial anomaly for each month (i.e. of shape (nlat, nlon, 12))
    """
    # interpolate coarse_obs back to the original scale
    [coarse_obs, fine_obs_rechunked] = load_paths([coarse_obs_path, fine_obs_rechunked_path])

    obs_interpolated, _ = regrid_dataset(
        ds=coarse_obs,
        ds_path=coarse_obs_path,
        target_grid_ds=fine_obs_rechunked.isel(time=0),
        variable=variable,
        connection_string=connection_string,
    )
    # use rechunked fine_obs from coarsening step above because that is in map chunks so it
    # will play nice with the interpolated obs

    schema_maps_chunks.validate(fine_obs_rechunked[variable])

    # calculate difference between interpolated obs and the original obs
    spatial_anomalies = obs_interpolated - fine_obs_rechunked

    # calculate seasonal cycle (12 time points)
    seasonal_cycle_spatial_anomalies = spatial_anomalies.groupby("time.month").mean()
    return seasonal_cycle_spatial_anomalies


def write_dataset(ds: xr.Dataset, path: str, chunks_dims: Tuple = ('time',)) -> None:
    """Write out a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset you want to write
    path : str
        Location where to write it
    chunks_dims : tuple, optional
        Dimension to chunk along if chunks are iffy, by default ('time',)
    """
    store = fsspec.get_mapper(path)
    try:
        ds.to_zarr(store, mode='w', consolidated=True)
    except ValueError:
        # if your chunk size isn't uniform you'll probably get a value error so
        # you can try doing this you can rechunk it
        print(
            'WARNING: Failed to write zarr store, perhaps because of variable chunk sizes, trying to rechunk it'
        )
        chunks_dict = calc_auspicious_chunks_dict(ds, chunk_dims=chunks_dims)
        delete_chunks_encoding(ds)
        ds.chunk(chunks_dict).to_zarr(store, mode='w', consolidated=True)
