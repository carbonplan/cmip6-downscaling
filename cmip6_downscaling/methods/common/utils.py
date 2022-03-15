import numpy as np
import xarray as xr
from xarray_schema import DataArraySchema
from xarray_schema.base import SchemaError

from .containers import BBox

schema_maps_chunks = DataArraySchema(chunks={'lat': -1, 'lon': -1})


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
