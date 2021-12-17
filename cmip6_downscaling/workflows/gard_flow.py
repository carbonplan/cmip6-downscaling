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

from cmip6_downscaling.config.config import CONNECTION_STRING, cache_store, serializer
from cmip6_downscaling.data.cmip import get_gcm, get_gcm_grid_spec
from cmip6_downscaling.data.observations import get_obs

from cmip6_downscaling.methods.gard import gard_fit_and_predict
from cmip6_downscaling.workflows.common_tasks import (
    get_obs_task,
    get_gcm_task,
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
    result=FunnelResult(cache_store, serializer=serializer),
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


# def gard_flow(
#     model,
#     label_name,
#     feature_list=None,
#     dim='time',
#     bias_correction_method='quantile_transform',
#     bc_kwargs=None,
#     generate_scrf=True,
# ):
#     """
#     Parameters
#     ----------
#     model                 : a GARD model instance to be fitted pointwise
#     feature_list          : a list of feature names to be used in predicting
#     dim                   : string. dimension to apply the model along. Default is ``time``.
#     bias_correction_method: string of the name of bias correction model
#     bc_kwargs             : kwargs dict. directly passed to the bias correction model
#     generate_scrf         : boolean. indicates whether a spatio-temporal correlated random field (scrf) will be
#                             generated based on the fine resolution data provided in .fit as y. if false, it is
#                             assumed that a pre-generated scrf will be passed into .predict as an argument that
#                             matches the prediction result dimensions.
#     spatial_feature       : (3, 3)
#     """
#     self._dim = dim
#     if not isinstance(model, (AnalogBase, PureRegression)):
#         raise TypeError('model must be part of the GARD family of pointwise models ')
#     self.features = feature_list
#     self.label_name = label_name
#     self._model = model
#     self.thresh = model.thresh

#     # shared between multiple method types but point wise
#     # TODO: spatial features
#     # TODO: extend this to include transforming the feature space into PCA, etc
#     # map + 1d component (pca) of ocean surface temperature for precip prediction
