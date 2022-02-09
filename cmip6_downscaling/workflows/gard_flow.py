import os

os.environ['PREFECT__FLOWS__CHECKPOINTING'] = 'true'

from typing import List

import xarray as xr
from prefect import Flow, Parameter, task
from xpersist import CacheStore
from xpersist.prefect.result import XpersistResult

from cmip6_downscaling import config, runtimes
from cmip6_downscaling.data.observations import get_obs
from cmip6_downscaling.methods.gard import gard_fit_and_predict, gard_postprocess, generate_scrf
from cmip6_downscaling.runtimes import get_runtime
from cmip6_downscaling.tasks.common_tasks import (
    build_bbox,
    build_time_period_slices,
    bias_correct_gcm_task,
    bias_correct_obs_task,
    coarsen_and_interpolate_obs_task,
    interpolate_gcm_task,
    path_builder_task,
)
from cmip6_downscaling.workflows.paths import (
    make_bias_corrected_gcm_path,
    make_gard_post_processed_output_path,
    make_gard_predict_output_path,
    make_rechunked_obs_path,
    make_scrf_path,
)
from cmip6_downscaling.workflows.utils import rechunk_zarr_array_with_caching

runtime = get_runtime()

intermediate_cache_store = CacheStore(
    config.get('storage.intermediate.uri'),
    storage_options=config.get('storage.intermediate.storage_options'),
)
results_cache_store = CacheStore(
    config.get('storage.results.uri'), storage_options=config.get('storage.results.storage_options')
)


fit_and_predict_task = task(
    gard_fit_and_predict,
    checkpoint=True,
    result=XpersistResult(intermediate_cache_store, serializer="xarray.zarr"),
    target=make_gard_predict_output_path,
)


generate_scrf_task = task(
    generate_scrf,
    result=XpersistResult(intermediate_cache_store, serializer="xarray.zarr"),
    target=make_scrf_path,
)


gard_postprocess_task = task(
    gard_postprocess,
    result=XpersistResult(intermediate_cache_store, serializer="xarray.zarr"),
    target=make_gard_post_processed_output_path,
)


@task(nout=3)
def prep_gard_input_task(
    obs: str,
    train_period_start: str,
    train_period_end: str,
    predict_period_start: str,
    predict_period_end: str,
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
        cache_within_rechunk=False,
    )
    rechunked_obs_path = make_rechunked_obs_path(
        obs=obs,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        variables=variables,
        chunking_approach='matched',
    )
    y_train_rechunked = rechunk_zarr_array_with_caching(
        zarr_array=ds_obs, template_chunk_array=X_train, output_path=rechunked_obs_path
    )

    rechunked_gcm_path = make_bias_corrected_gcm_path(
        gcm_identifier=gcm_identifier, method=bias_correction_method, chunking_approach='matched'
    )
    X_pred_rechunked = rechunk_zarr_array_with_caching(
        zarr_array=X_pred, template_chunk_array=X_train, output_path=rechunked_gcm_path
    )

    predict_period = slice(predict_period_start, predict_period_end)

    return X_train, y_train_rechunked, X_pred_rechunked.sel(time=predict_period)


with Flow(
    name='gard',
    storage=runtime.storage,
    run_config=runtime.run_config,
    executor=runtime.executor,
) as gard_flow:
    obs = Parameter("obs")
    gcm = Parameter("gcm")
    scenario = Parameter("scenario")
    variable = Parameter("variables")
    features = Parameter("features")
    # bbox and train and predict period had to be encapsulated into tasks to prevent prefect from complaining about unused parameters.
    bbox = build_bbox(
        latmin=Parameter("latmin"),
        latmax=Parameter("latmax"),
        lonmin=Parameter("lonmin"),
        lonmax=Parameter("lonmax"),
    )
    train_period = build_time_period_slices(Parameter('train_period'))
    predict_period = build_time_period_slices(Parameter('predict_period'))
    bias_correction_kwargs = Parameter("bias_correction_kwargs")
    bias_correction_method = Parameter("bias_correction_method")
    model_type = Parameter("model_type")
    model_params = Parameter("model_params")

    # dictionary with information to build appropriate paths for caching
    (
        gcm_grid_spec,
        obs_identifier,
        gcm_identifier,
        pyramid_path_daily,
        pyramid_path_monthly,
        pyramid_path_annual,
    ) = path_builder_task(
        obs=obs,
        gcm=gcm,
        scenario=scenario,
        variable=variable,
        train_period=train_period,
        predict_period=predict_period,
        bbox=bbox,
    )

    # get interpolated observation
    ds_obs_interpolated_full_time = coarsen_and_interpolate_obs_task(
        obs=obs,
        train_period=train_period,
        predict_period=predict_period,
        variable=variable,
        gcm=gcm,
        scenario=scenario,
        chunking_approach='full_time',
        bbox=bbox,
        gcm_grid_spec=gcm_grid_spec,
        obs_identifier=obs_identifier,
    )

    # get interpolated gcm
    ds_gcm_interpolated_full_time = interpolate_gcm_task(
        obs=obs,
        gcm=gcm,
        scenario=scenario,
        variable=variable,
        train_period=train_period,
        predict_period=predict_period,
        chunking_approach='full_time',
    )

    # bias correction and transformation
    ds_obs_bias_corrected = bias_correct_obs_task(
        ds_obs=ds_obs_interpolated_full_time,
        method=bias_correction_method,
        bc_kwargs=bias_correction_kwargs,
        chunking_approach='full_time',
        gcm_grid_spec=gcm_grid_spec,
        obs_identifier=obs_identifier,
    )

    # TODO: decide whether we are bias correcting to interpolated obs or actual obs
    ds_gcm_bias_corrected = bias_correct_gcm_task(
        ds_gcm=ds_gcm_interpolated_full_time,
        ds_obs=ds_obs_interpolated_full_time,
        historical_period_start=train_period_start,
        historical_period_end=train_period_end,
        method=bias_correction_method,
        bc_kwargs=bias_correction_kwargs,
        chunking_approach='full_time',
        gcm_identifier=gcm_identifier,
    )

    X_train, y_train, X_pred = prep_gard_input_task(
        obs=obs,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        predict_period_start=predict_period_start,
        predict_period_end=predict_period_end,
        variables=variables,
        X_train=ds_obs_bias_corrected,
        X_pred=ds_gcm_bias_corrected,
        gcm_identifier=gcm_identifier,
        bias_correction_method=bias_correction_method,
    )

    # fit and predict
    model_output = fit_and_predict_task(
        X_train=X_train,
        y_train=y_train,
        X_pred=X_pred,
        label=label,
        model_type=model_type,
        model_params=model_params,
        gcm_identifier=gcm_identifier,
        bias_correction_method=bias_correction_method,
    )

    # post process
    scrf = generate_scrf_task(data=y_train, obs_identifier=obs_identifier, label=label)

    final_output = gard_postprocess_task(
        model_output=model_output,
        scrf=scrf,
        model_params=model_params,
        gcm_identifier=gcm_identifier,
        bias_correction_method=bias_correction_method,
        model_type=model_type,
        label=label,
    )
