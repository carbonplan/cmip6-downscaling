import os
os.environ['PREFECT__FLOWS__CHECKPOINTING'] = 'true'

from typing import Dict, Any, Optional, List

from dask_kubernetes import KubeCluster, make_pod_spec
from prefect import Flow, task
from prefect.executors import DaskExecutor
from prefect.run_configs import KubernetesRun
from prefect.storage import Azure
from funnel.prefect.result import FunnelResult
import xarray as xr

from cmip6_downscaling.config.config import CONNECTION_STRING, intermediate_cache_store, serializer
from cmip6_downscaling.data.cmip import get_gcm, get_gcm_grid_spec
from cmip6_downscaling.data.observations import get_obs

from cmip6_downscaling.methods.gard import gard_fit_and_predict
from cmip6_downscaling.workflows.common_tasks import (
    coarsen_and_interpolate_obs_task,
    bias_correct_obs_task,
    bias_correct_gcm_task
)
from cmip6_downscaling.workflows.utils import (
    rechunk_zarr_array_with_caching, 
    regrid_ds
)
from cmip6_downscaling.workflows.paths import (
    build_obs_identifier, 
    build_gcm_identifier, 
    make_interpolated_obs_path,
    make_gard_predict_output_path,
    make_rechunked_obs_path,
    make_bias_corrected_gcm_path
)


fit_and_predict_task = task(
    gard_fit_and_predict, 
    checkpoint=True,
    result=FunnelResult(intermediate_cache_store, serializer=serializer),
    target=make_gard_predict_output_path
)


@task(nout=3)
def prep_gard_input_task(
    obs: str,
    train_period_start: str,
    train_period_end: str,
    variables: List[str],
    X_train: xr.Dataset,
    X_pred: xr.Dataset,
    gcm_identifier: str,
    bias_correction_method: str,
):
    # get observation data in the same chunking scheme as 
    ds_obs = get_obs(
        obs=obs,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        variables=variables,
        chunking_approach=None,
        cache_within_rechunk=False
    )
    rechunked_obs_path = make_rechunked_obs_path(
        obs=obs,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        variables=variables,
        chunking_approach='matched'
    )
    y_train_rechunked = rechunk_zarr_array_with_caching(
        zarr_array=ds_obs,
        template_chunk_array=X_train,
        output_path=rechunked_obs_path
    )
    
    rechunked_gcm_path = make_bias_corrected_gcm_path(
        gcm_identifier=gcm_identifier, 
        method=bias_correction_method, 
        chunking_approach='matched'
    )
    X_pred_rechunked = rechunk_zarr_array_with_caching(
        zarr_array=X_pred,
        template_chunk_array=X_train,
        output_path=rechunked_gcm_path
    )

    return X_train, y_train_rechunked, X_pred_rechunked
