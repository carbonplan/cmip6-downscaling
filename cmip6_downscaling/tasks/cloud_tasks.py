import os

os.environ["PREFECT__FLOWS__CHECKPOINTING"] = "true"

from typing import Any, Dict, List, Optional, Union

import xarray as xr
from prefect import task
from skdownscale.pointwise_models.utils import default_none_kwargs
from xpersist import CacheStore
from xpersist.prefect.result import XpersistResult

import cmip6_downscaling.config.config as config
from cmip6_downscaling.data.cmip import get_gcm, load_cmip
from cmip6_downscaling.data.observations import get_obs
from cmip6_downscaling.methods.bias_correction import (
    bias_correct_gcm_by_method,
    bias_correct_obs_by_method,
)
from cmip6_downscaling.workflows.paths import (
    make_bias_corrected_gcm_path,
    make_bias_corrected_obs_path,
    make_coarse_obs_path,
    make_interpolated_gcm_path,
    make_interpolated_obs_path,
)
from cmip6_downscaling.workflows.utils import rechunk_zarr_array_with_caching, regrid_ds

get_obs_task = task(get_obs)
get_gcm_task = task(get_gcm)


@task(
    checkpoint=True,
    result=XpersistResult(
        CacheStore(config.CloudConfig().intermediate_cache_path),
        serializer=config.CloudConfig().serializer,
    ),
    target=make_coarse_obs_path,
)
def get_coarse_obs_task(
    ds_obs: xr.Dataset, gcm: str, chunking_approach: str, **kwargs
) -> xr.Dataset:
    """
    Coarsen the observation dataset to the grid of the GCM model specified in inputs.

    Parameters
    ----------
    ds_obs: xr.Dataset
        Observation dataset to be coarsened
    gcm: str
        Name of the GCM model whose grid to coarsen to
    **kwargs: Dict
        Other arguments to be used in generating the target path

    Returns
    -------
    ds_obs_coarse: xr.Dataset
        Coarsened observation dataset
    """
    # Load single slice of target cmip6 dataset for target grid dimensions
    gcm_grid = load_cmip(
        source_ids=gcm,
        return_type="xr",
    ).isel(time=0)

    # rechunk and regrid observation dataset to target gcm resolution
    ds_obs_coarse = regrid_ds(
        ds=ds_obs,
        target_grid_ds=gcm_grid,
        connection_string=config.CloudConfig().connection_string,
    )

    if chunking_approach != 'full_space':
        ds_obs_coarse = rechunk_zarr_array_with_caching(
            zarr_array=ds_obs_coarse, chunking_approach=chunking_approach, output_path=None
        )

    return ds_obs_coarse


@task(
    checkpoint=True,
    result=XpersistResult(
        CacheStore(config.CloudConfig().intermediate_cache_path),
        serializer=config.CloudConfig().serializer,
    ),
    target=make_interpolated_obs_path,
)
def coarsen_and_interpolate_obs_task(
    obs, train_period_start, train_period_end, variables, gcm, chunking_approach, **kwargs
):
    """
    Coarsen the observation dataset to the grid of the GCM model specified in inputs then
    interpolate back into the observation grid. Rechunk the final output according to chunking approach.

    Parameters
    ----------
    obs: str
        Name of obs dataset
    gcm: str
        Name of GCM model
    training_period_start: str
        Start year of training/historical period
    training_period_end: str
        End year of training/historical period
    variables: List[str]
        List of variables to get in obs dataset
    chunking_approach: str
        'full_space', 'full_time', or None
    **kwargs: Dict
        Other arguments to be used in generating the target path

    Returns
    -------
    ds_obs_interpolated_rechunked: xr.Dataset
        An observation dataset that has been coarsened, interpolated back to original grid, and then rechunked.
    """
    # get obs in full space chunks
    ds_obs_full_space = get_obs(
        obs=obs,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        variables=variables,
        chunking_approach="full_space",
        cache_within_rechunk=True,
    )

    # regrid to coarse scale
    ds_obs_coarse = get_coarse_obs_task.run(
        ds_obs=ds_obs_full_space, gcm=gcm, chunking_approach="full_space", **kwargs
    )

    # interpolate to fine scale again
    ds_obs_interpolated = regrid_ds(
        ds=ds_obs_coarse,
        target_grid_ds=ds_obs_full_space.isel(time=0),
        chunking_approach="full_space",
    )

    # rechunked to final output chunking approach if needed
    ds_obs_interpolated_rechunked = rechunk_zarr_array_with_caching(
        zarr_array=ds_obs_interpolated,
        output_path=None,
        chunking_approach=chunking_approach,
    )

    return ds_obs_interpolated_rechunked


@task(
    checkpoint=True,
    result=XpersistResult(
        CacheStore(config.CloudConfig().intermediate_cache_path),
        serializer=config.CloudConfig().serializer,
    ),
    target=make_interpolated_gcm_path,
)
def interpolate_gcm_task(
    obs: str,
    gcm: str,
    scenario: str,
    train_period_start: str,
    train_period_end: str,
    predict_period_start: str,
    predict_period_end: str,
    variables: Union[str, List[str]],
    chunking_approach: str,
    **kwargs,
):
    """
    Interpolate the GCM dataset to the grid of the observation dataset.
    Rechunk the final output according to chunking approach.

    Parameters
    ----------
    obs: str
        Name of obs dataset
    gcm: str
        Name of the GCM model
    scenario: str
        Name of the emission scenario
    training_period_start: str
        Start year of training/historical period
    training_period_end: str
        End year of training/historical period
    predict_period_start: str
        Start year of predict/future period
    predict_period_end: str
        End year of predict/future period
    variables: List[str]
        List of variables to get in obs dataset
    chunking_approach: str
        'full_space', 'full_time', or None
    **kwargs: Dict
        Other arguments to be used in generating the target path

    Returns
    -------
    ds_gcm_interpolated_rechunked: xr.Dataset
        The GCM dataset that has been interpolated to the obs grid then rechunked.
    """
    # get gcm in full space chunks
    ds_gcm_full_space = get_gcm(
        gcm=gcm,
        scenario=scenario,
        variables=variables,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        predict_period_start=predict_period_start,
        predict_period_end=predict_period_end,
        chunking_approach="full_space",
        cache_within_rechunk=False,
    )

    # get obs as a template
    ds_obs_full_space = get_obs(
        obs=obs,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        variables=variables,
        chunking_approach=None,
        cache_within_rechunk=False,
    )

    # interpolate gcm to obs resolution
    ds_gcm_interpolated = regrid_ds(
        ds=ds_gcm_full_space,
        target_grid_ds=ds_obs_full_space.isel(time=0).load(),
        chunking_approach="full_space",
    )

    # rechunked to final output chunking approach if needed
    ds_gcm_interpolated_rechunked = rechunk_zarr_array_with_caching(
        zarr_array=ds_gcm_interpolated,
        output_path=None,
        chunking_approach=chunking_approach,
    )

    return ds_gcm_interpolated_rechunked


@task(
    log_stdout=True,
    result=XpersistResult(
        CacheStore(config.CloudConfig().intermediate_cache_path),
        serializer=config.CloudConfig().serializer,
    ),
    target=make_bias_corrected_obs_path,
)
def bias_correct_obs_task(
    ds_obs: xr.Dataset, method: str, bc_kwargs: Optional[Dict[str, Any]] = None, **kwargs
) -> xr.DataArray:
    """
    Bias correct observation data according to methods and kwargs.

    Parameters
    ----------
    ds_obs : xr.Dataset
        Observation dataset
    method : str
        Bias correction method to be used.
    bc_kwargs: dict or None
        Keyword arguments to be used with the bias correction method
    kwargs: dict
        Other arguments to be used in generating the target path

    Returns
    -------
    ds_obs_bias_corrected : xr.Dataset
        Bias corrected observation dataset
    """
    kws = default_none_kwargs(bc_kwargs, copy=True)
    bias_corrected = bias_correct_obs_by_method(
        da_obs=ds_obs, method=method, bc_kwargs=kws
    ).to_dataset(dim="variable")

    return bias_corrected


@task(
    result=XpersistResult(
        CacheStore(config.CloudConfig().intermediate_cache_path),
        serializer=config.CloudConfig().serializer,
    ),
    target=make_bias_corrected_gcm_path,
)
def bias_correct_gcm_task(
    ds_gcm: xr.Dataset,
    ds_obs: xr.Dataset,
    historical_period_start: str,
    historical_period_end: str,
    method: str,
    bc_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> xr.DataArray:
    """
    Bias correct gcm data to the provided observation data according to methods and kwargs.

    Parameters
    ----------
    ds_gcm : xr.Dataset
        GCM dataset to be bias corrected
    ds_obs : xr.Dataset
        Observation dataset to bias correct to
    historical_period_start : str
        Start year of the historical/training period
    historical_period_end : str
        End year of the historical/training period
    method : str
        Bias correction method to be used.
    bc_kwargs: dict or None
        Keyword arguments to be used with the bias correction method
    kwargs: dict
        Other arguments to be used in generating the target path

    Returns
    -------
    ds_gcm_bias_corrected : xr.Dataset
        Bias corrected GCM dataset
    """
    historical_period = slice(historical_period_start, historical_period_end)

    kws = default_none_kwargs(bc_kwargs, copy=True)

    for v in ds_gcm.data_vars:
        assert v in ds_obs.data_vars
    bias_corrected = bias_correct_gcm_by_method(
        da_gcm=ds_gcm,
        da_obs=ds_obs,
        historical_period=historical_period,
        method=method,
        bc_kwargs=kws,
    ).to_dataset(dim="variable")

    return bias_corrected