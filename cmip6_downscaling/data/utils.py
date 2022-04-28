import numpy as np
import xclim
from xarray.core.types import T_Xarray


def to_standard_calendar(obj: T_Xarray) -> T_Xarray:
    """Convert a Dataset's calendar to the "standard calendar"

    When necessary, "missing" time points are filled in using linear interpolation.

    Valid input dataset calendars include: `noleap`, `365_day`, `366_day`, and `all_leap`.

    Parameters
    ----------
    obj : xr.Dataset or xr.DataArray
        Xarray object with a `CFTimeIndex`.

    Returns
    -------
    obj_new : xr.Dataset or xr.DataArray
        Xarray object with standard calendar.

    Raises
    ------
    ValueError
        If an invalid calendar is supplied.
    """

    orig_calendar = getattr(obj.indexes["time"], "calendar", "standard")
    if orig_calendar == "standard":
        return obj
    if orig_calendar == "360_day":
        raise ValueError("360_day calendar is not supported")

    # reindex / interpolate -- Note: .chunk was added to fix dask error
    obj_new = (
        xclim.core.calendar.convert_calendar(obj, "standard", missing=np.nan)
        .chunk({'time': -1})
        .interpolate_na(dim="time", method="linear")
    )

    # reset encoding
    obj_new["time"].encoding["calendar"] = "standard"

    # sets time to datetimeindex
    obj_new['time'] = obj_new.indexes['time'].to_datetimeindex()

    return obj_new


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
