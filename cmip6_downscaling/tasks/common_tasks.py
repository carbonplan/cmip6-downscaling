import numpy as np
import xclim
from prefect import task
from xarray.core.types import T_Xarray


@task
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

    # reindex / interpolate
    obj_new = xclim.core.calendar.convert_calendar(obj, "standard", missing=np.nan).interpolate_na(
        dim="time", method="linear"
    )

    # reset encoding
    obj_new["time"].encoding["calendar"] = "standard"

    return obj_new
