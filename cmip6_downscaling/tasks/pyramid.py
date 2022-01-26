import dask
import datatree as dt
import fsspec
import xarray as xr
from carbonplan_data.metadata import get_cf_global_attrs
from carbonplan_data.utils import set_zarr_encoding
from ndpyramid import pyramid_regrid
from prefect import task

PIXELS_PER_TILE = 128


def _load_coords(ds: xr.Dataset) -> xr.Dataset:
    '''Helper function to explicitly load all dataset coordinates'''
    for var, da in ds.coords.items():
        ds[var] = da.load()
    return ds


def _postprocess(dt: dt.DataTree, levels: int, other_chunks: dict = None) -> dt.DataTree:
    '''Postprocess data pyramid

    Adds multiscales metadata and sets Zarr encoding

    Parameters
    ----------
    dt : dt.DataTree
        Input data pyramid
    levels : int
        Number of levels in pyramid
    other_chunks : dict
        Chunks for non-spatial dims

    Returns
    -------
    dt.DataTree
        Updated data pyramid with metadata / encoding set
    '''
    chunks = {"x": PIXELS_PER_TILE, "y": PIXELS_PER_TILE}
    if other_chunks is not None:
        chunks.update(other_chunks)

    for level in range(levels):
        slevel = str(level)
        dt.ds.attrs['multiscales'][0]['datasets'][level]['pixels_per_tile'] = PIXELS_PER_TILE

        # set dataset chunks
        dt[slevel].ds = dt[slevel].ds.chunk(chunks)
        if 'date_str' in dt[slevel].ds:
            dt[slevel].ds['date_str'] = dt[slevel].ds['date_str'].chunk(-1)

        # set dataset encoding
        dt[slevel].ds = set_zarr_encoding(
            dt[slevel].ds, codec_config={"id": "zlib", "level": 1}, float_dtype="float32"
        )
        for dim in ['time', 'time_bnds']:
            if dim in dt[slevel].ds:
                dt[slevel].ds[dim].encoding['dtype'] = 'int32'

    # set global metadata
    dt.ds.attrs.update(**get_cf_global_attrs())
    return dt


@task(log_stdout=True, tags=['dask-resource:TASKSLOTS=1'])
def regrid(ds: xr.Dataset, levels: int = 2, uri: str = None, other_chunks: dict = None) -> str:
    '''Task to create a data pyramid from an xarray Dataset

    Parameters
    ----------
    ds : xr.Dataset
        Input Dataset
    levels : int, optional
        Number of levels in pyramid, by default 2
    uri : str, optional
        Path to write output data pyamid to, by default None
    other_chunks : dict
        Chunks for non-spatial dims
    '''

    with dask.config.set(scheduler='threads'):

        ds.coords['date_str'] = ds['time'].dt.strftime('%Y-%m-%d').astype('S10')

        ds = _load_coords(ds)

        mapper = fsspec.get_mapper(uri)
        # Question: Is this needed or will prefect handle this for us?
        if '.zmetadata' in mapper:
            return

        # create
        dt = pyramid_regrid(ds, target_pyramid=None, levels=levels)

        # postprocess
        dt = _postprocess(dt, levels, other_chunks=other_chunks)

        # write to uri
        dt.to_zarr(mapper, mode='w')

    return uri
