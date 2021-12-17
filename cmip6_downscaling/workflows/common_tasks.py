import os
os.environ['PREFECT__FLOWS__CHECKPOINTING'] = 'true'

import xarray as xr

from typing import Dict, Any, Optional, List, Tuple, Union
from xpersist.prefect.result import XpersistResult
from prefect import task
from skdownscale.pointwise_models.utils import default_none_kwargs


from cmip6_downscaling.config.config import CONNECTION_STRING, intermediate_cache_store, serializer
from cmip6_downscaling.data.observations import get_obs
from cmip6_downscaling.data.cmip import get_gcm, load_cmip, get_gcm_grid_spec
from cmip6_downscaling.methods.bias_correction import bias_correct_gcm_by_method, bias_correct_obs_by_method
from cmip6_downscaling.workflows.paths import (
    build_obs_identifier,
    build_gcm_identifier,
    make_rechunked_obs_path, 
    make_rechunked_gcm_path,
    make_coarse_obs_path, 
    make_interpolated_obs_path,
    make_interpolated_gcm_path,
    make_bias_corrected_obs_path, 
    make_bias_corrected_gcm_path,
)
from cmip6_downscaling.workflows.utils import (
    rechunk_zarr_array_with_caching, 
    regrid_ds
)


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
    gcm_grid_spec = get_gcm_grid_spec(gcm_name=gcm)
    obs_identifier = build_obs_identifier(obs=obs, train_period_start=train_period_start, train_period_end=train_period_end, variables=variables)
    gcm_identifier = build_gcm_identifier(
        gcm=gcm, 
        scenario=scenario, 
        train_period_start=train_period_start, 
        train_period_end=train_period_end, 
        predict_period_start=predict_period_start, 
        predict_period_end=predict_period_end, 
        variables=variables)

    return gcm_grid_spec, obs_identifier, gcm_identifier


@task(checkpoint=True, result=XpersistResult(intermediate_cache_store, serializer=serializer), target=make_coarse_obs_path)
def get_coarse_obs_task(
    ds_obs: xr.Dataset, 
    gcm: str,
    **kwargs
) -> xr.Dataset:
    """
    **kwargs are used to construct target file path
    """
    # Load single slice of target cmip6 dataset for target grid dimensions
    gcm_grid = load_cmip(
        source_ids=gcm,
        return_type='xr',
    ).isel(time=0)

    # rechunk and regrid observation dataset to target gcm resolution
    ds_obs_coarse = regrid_ds(
        ds=ds_obs,
        target_grid_ds=gcm_grid,
        connection_string=CONNECTION_STRING,
    )
    return ds_obs_coarse


@task(checkpoint=True, result=XpersistResult(intermediate_cache_store, serializer=serializer), target=make_interpolated_obs_path)
def coarsen_and_interpolate_obs_task(
    obs, 
    train_period_start,
    train_period_end,
    variables,
    gcm,
    chunking_approach,
    **kwargs
):
    """
    # goal here is to cache: 1) the rechunked fine obs, 2) the coarsened obs, and 3) the regridded obs
    """
    # get obs in full space chunks 
    ds_obs_full_space = get_obs(
        obs=obs,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        variables=variables,
        chunking_approach='full_space',
        cache_within_rechunk=True,
    )

    # regrid to coarse scale 
    ds_obs_coarse = get_coarse_obs_task.run(
        ds_obs=ds_obs_full_space, 
        gcm=gcm,
        chunking_approach='full_space',
        **kwargs
    )

    # interpolate to fine scale again 
    ds_obs_interpolated = regrid_ds(
        ds=ds_obs_coarse,
        target_grid_ds=ds_obs_full_space.isel(time=0),
        chunking_approach='full_space',
    )
    
    # rechunked to final output chunking approach if needed 
    ds_obs_interpolated_rechunked = rechunk_zarr_array_with_caching(
        zarr_array=ds_obs_interpolated,
        output_path=None,
        chunking_approach=chunking_approach
    )

    return ds_obs_interpolated_rechunked


@task(checkpoint=True, result=XpersistResult(intermediate_cache_store, serializer=serializer), target=make_interpolated_gcm_path)
def interpolate_gcm_task(
    obs: str,
    gcm: str,
    scenario: str,
    variables: Union[str, List[str]],
    train_period_start: str,
    train_period_end: str,
    predict_period_start: str,
    predict_period_end: str,
    chunking_approach: str,
    **kwargs
):
    """
    # goal here is to cache: 1) the rechunked fine obs, 2) the coarsened obs, and 3) the regridded obs
    """
    # get obs in full space chunks 
    ds_gcm_full_space = get_gcm(
        gcm=gcm,
        scenario=scenario,
        variables=variables,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        predict_period_start=predict_period_start,
        predict_period_end=predict_period_end,
        chunking_approach='full_space',
        cache_within_rechunk=False
    )

    # regrid to coarse scale 
    ds_obs_full_space = get_obs(
        obs=obs,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        variables=variables,
        chunking_approach=None,
        cache_within_rechunk=False,
    )

    # interpolate to fine scale again 
    ds_gcm_interpolated = regrid_ds(
        ds=ds_gcm_full_space,
        target_grid_ds=ds_obs_full_space.isel(time=0).load(),
        chunking_approach='full_space',
    )
    
    # rechunked to final output chunking approach if needed 
    ds_gcm_interpolated_rechunked = rechunk_zarr_array_with_caching(
        zarr_array=ds_gcm_interpolated,
        output_path=None,
        chunking_approach=chunking_approach
    )

    return ds_gcm_interpolated_rechunked


@task(log_stdout=True, result=XpersistResult(intermediate_cache_store, serializer=serializer), target=make_bias_corrected_obs_path)
def bias_correct_obs_task(
    ds_obs: xr.Dataset,
    method: str,
    bc_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs
) -> xr.DataArray:
    """
    Bias correct observation data according to methods and kwargs. 

    Parameters
    ----------
    ds_obs : xr.Dataset
        Observation dataset 
    methods : str
        Bias correction methods to be used.
    bc_kwargs: dict or None 
        Keyword arguments to be used with the bias correction method 
    kwargs: dict 
        Used to construct caching paths 

    Returns
    -------
    ds_obs_bias_corrected : xr.Dataset
        Bias corrected observation dataset 
    """
    kws = default_none_kwargs(bc_kwargs, copy=True)
    bias_corrected = bias_correct_obs_by_method(
        da_obs=ds_obs.isel(lat=slice(0, 100), lon=slice(0, 100)),
        method=method,
        bc_kwargs=kws
    ).to_dataset(dim='variable')

    return bias_corrected


@task(result=XpersistResult(intermediate_cache_store, serializer=serializer), target=make_bias_corrected_gcm_path)
def bias_correct_gcm_task(
    ds_gcm: xr.Dataset,
    ds_obs: xr.Dataset,
    historical_period_start: str,
    historical_period_end: str,
    method: str,
    bc_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs
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
        Used to construct caching paths 

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
        da_gcm=ds_gcm.isel(lat=slice(0, 100), lon=slice(0, 100)),
        da_obs=ds_obs.isel(lat=slice(0, 100), lon=slice(0, 100)),
        historical_period=historical_period,
        method=method,
        bc_kwargs=kws
    ).to_dataset(dim='variable')

    return bias_corrected
