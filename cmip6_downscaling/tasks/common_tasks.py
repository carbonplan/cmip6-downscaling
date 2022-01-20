import os

os.environ["PREFECT__FLOWS__CHECKPOINTING"] = "true"

from typing import Callable, List, Optional, Tuple

import numpy as np
import xarray as xr
import xclim
from prefect import task
from xarray.core.types import T_Xarray

from cmip6_downscaling.data.cmip import get_gcm_grid_spec
from cmip6_downscaling.workflows.paths import build_gcm_identifier, build_obs_identifier
from cmip6_downscaling.workflows.utils import rechunk_zarr_array_with_caching


@task
def rechunker_task(
    zarr_array: xr.Dataset,
    chunking_approach: Optional[str] = None,
    template_chunk_array: Optional[xr.Dataset] = None,
    naming_func: Optional[Callable] = None,
    **kwargs,
):
    """
    Task to rechunk a dataset

    Parameters
    ----------
    zarr_array : zarr or xarray dataset
        Dataset you want to rechunk.
    chunking_approach : str, optional
        Has to be one of `full_space` or `full_time`. If `full_space`, the data will be rechunked such that the space dimensions are contiguous (i.e. each chunk
        will contain full maps). If `full_time`, the data will be rechunked such that the time dimension is contiguous (i.e. each chunk will contain full time
        series). Either the chunking approach or the template chunk array must be provided.
    template_chunk_array: zarr or xarray dataset, optional
        A template dataset with the desired chunksizes. Either the chunking approach or the template chunk array must be provided.
    naming_func: callable, optional
        A function that returns a string that represents the output caching location that the rechunk task should save to.
        The input arguments of this naming func must be provided as kwargs to this method

    Returns
    -------
    rechunked_ds : xr.Dataset
        Rechunked dataset
    """
    if naming_func is not None:
        output_path = naming_func(chunking_approach=chunking_approach, **kwargs)
    else:
        output_path = None

    rechunked = rechunk_zarr_array_with_caching(
        zarr_array=zarr_array,
        chunking_approach=chunking_approach,
        template_chunk_array=template_chunk_array,
        output_path=output_path,
    )

    return rechunked


@task
def path_builder_task(
    obs: str,
    gcm: str,
    scenario: str,
    train_period_start: str,
    train_period_end: str,
    predict_period_start: str,
    predict_period_end: str,
    variables: List[str],
) -> Tuple[str, str, str]:
    """
    Take in input parameters and make string patterns that identifies the obs dataset, gcm dataset, and the gcm grid. These
    strings will then be used to identify cached files.

    Parameters
    ----------
    obs: str
        Name of obs dataset
    gcm: str
        Name of gcm model
    scenario: str
        Name of future emission scenario
    train_period_start: str
        Start year of training/historical period
    train_period_end: str
        End year of training/historical period
    predict_period_start: str
        Start year of predict/future period
    predict_period_end: str
        End year of predict/future period
    variables: List[str]
        Names of the variables used in obs and gcm dataset (including features and label)

    Returns
    -------
    gcm_grid_spec: str
        A string of parameters defining the grid of GCM, including number of lat/lon points, interval between points, lower left corner, etc.
    obs_identifier: str
        A string of parameters defining the obs dataset used, including variables, start/end year, etc
    gcm_identifier: str
        A string of parameters defining the GCM dataset used, including variables, start/end year for historical and future periods, etc
    """
    gcm_grid_spec = get_gcm_grid_spec(gcm_name=gcm)
    obs_identifier = build_obs_identifier(
        obs=obs,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        variables=variables,
    )
    gcm_identifier = build_gcm_identifier(
        gcm=gcm,
        scenario=scenario,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        predict_period_start=predict_period_start,
        predict_period_end=predict_period_end,
        variables=variables,
    )

    return gcm_grid_spec, obs_identifier, gcm_identifier


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
