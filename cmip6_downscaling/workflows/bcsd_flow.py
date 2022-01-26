from prefect import Flow, Parameter, task
from xpersist import CacheStore
from xpersist.prefect.result import XpersistResult

from cmip6_downscaling import config, runtimes
from cmip6_downscaling.methods.bcsd import (
    fit_and_predict,
    get_coarse_obs,
    get_spatial_anomalies,
    postprocess_bcsd,
    return_coarse_obs_full_time,
    return_gcm_predict_rechunked,
    return_gcm_train_full_time,
    return_obs,
)
from cmip6_downscaling.tasks.common_tasks import path_builder_task
from cmip6_downscaling.workflows.paths import (
    make_bcsd_output_path,
    make_bias_corrected_path,
    make_coarse_obs_path,
    make_gcm_predict_path,
    make_rechunked_gcm_path,
    make_return_obs_path,
    make_spatial_anomalies_path,
)

runtime = runtimes.get_runtime()

config.set(
    {
        "storage.intermediate.uri": "az://flow-outputs/intermediates",
        "storage.results.uri": "az://flow-outputs/results",
        "storage.temporary.uri": "az://flow-outputs/temporary",
    }
)

intermediate_cache_store = CacheStore(
    config.get("storage.intermediate.uri"),
    storage_options=config.get("storage.intermediate.storage_options"),
)
results_cache_store = CacheStore(
    config.get("storage.results.uri"),
    storage_options=config.get("storage.results.storage_options"),
)


# Transform Functions into Tasks -----------------------------------------------------------


return_obs_task = task(
    return_obs,
    result=XpersistResult(intermediate_cache_store, serializer="xarray.zarr"),
    target=make_return_obs_path,
)
get_coarse_obs_task = task(
    get_coarse_obs,
    tags=['dask-resource:TASKSLOTS=1'],
    result=XpersistResult(intermediate_cache_store, serializer="xarray.zarr"),
    target=make_coarse_obs_path,
)
get_spatial_anomalies_task = task(
    get_spatial_anomalies,
    log_stdout=True,
    tags=['dask-resource:TASKSLOTS=1'],
    result=XpersistResult(intermediate_cache_store, serializer="xarray.zarr"),
    target=make_spatial_anomalies_path,
)

return_coarse_obs_full_time_task = task(
    return_coarse_obs_full_time,
    tags=['dask-resource:TASKSLOTS=1'],
    result=XpersistResult(intermediate_cache_store, serializer="xarray.zarr"),
    target=make_coarse_obs_path,
)

return_gcm_train_full_time_task = task(
    return_gcm_train_full_time,
    tags=['dask-resource:TASKSLOTS=1'],
    result=XpersistResult(intermediate_cache_store, serializer="xarray.zarr"),
    target=make_rechunked_gcm_path,
)

return_gcm_predict_rechunked_task = task(
    return_gcm_predict_rechunked,
    tags=['dask-resource:TASKSLOTS=1'],
    result=XpersistResult(intermediate_cache_store, serializer="xarray.zarr"),
    target=make_gcm_predict_path,
)

fit_and_predict_task = task(
    fit_and_predict,
    log_stdout=True,
    result=XpersistResult(intermediate_cache_store, serializer="xarray.zarr"),
    target=make_bias_corrected_path,
)

postprocess_bcsd_task = task(
    postprocess_bcsd,
    tags=['dask-resource:TASKSLOTS=1'],
    log_stdout=True,
    result=XpersistResult(results_cache_store, serializer="xarray.zarr"),
    target=make_bcsd_output_path,
)


with Flow(
    name="bcsd",
    storage=runtime.storage,
    run_config=runtime.run_config,
    executor=runtime.executor,
) as bcsd_flow:
    obs = Parameter("OBS")
    gcm = Parameter("GCM")
    scenario = Parameter("SCENARIO")
    train_period_start = Parameter("TRAIN_PERIOD_START")
    train_period_end = Parameter("TRAIN_PERIOD_END")
    predict_period_start = Parameter("PREDICT_PERIOD_START")
    predict_period_end = Parameter("PREDICT_PERIOD_END")
    latmin = Parameter("LATMIN")
    latmax = Parameter("LATMAX")
    lonmin = Parameter("LONMIN")
    lonmax = Parameter("LONMAX")
    variable = Parameter("VARIABLE")

    gcm_grid_spec, obs_identifier, gcm_identifier = path_builder_task(
        obs=obs,
        gcm=gcm,
        scenario=scenario,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        predict_period_start=predict_period_start,
        predict_period_end=predict_period_end,
        latmin=latmin,
        latmax=latmax,
        lonmin=lonmin,
        lonmax=lonmax,
        variable=variable,
    )

    # preprocess_bcsd_tasks(s):

    obs_ds = return_obs_task(
        obs=obs,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        latmin=latmin,
        latmax=latmax,
        lonmin=lonmin,
        lonmax=lonmax,
        variable=variable,
        obs_identifier=obs_identifier,
    )

    coarse_obs_ds = get_coarse_obs_task(
        obs_ds,
        gcm,
        scenario,
        train_period_start,
        train_period_end,
        predict_period_start,
        predict_period_end,
        variable,
        latmin,
        latmax,
        lonmin,
        lonmax,
        obs_identifier=obs_identifier,
        gcm_grid_spec=gcm_grid_spec,
        chunking_approach='matched',
    )

    spatial_anomalies_ds = get_spatial_anomalies_task(
        coarse_obs_ds,
        obs_ds,
        gcm,
        scenario,
        train_period_start,
        train_period_end,
        predict_period_start,
        predict_period_end,
        variable,
        latmin,
        latmax,
        lonmin,
        lonmax,
        obs_identifier=obs_identifier,
    )

    # the next three tasks prepare the inputs required by bcsd
    coarse_obs_full_time_ds = return_coarse_obs_full_time_task(
        coarse_obs_ds,
        gcm,
        scenario,
        train_period_start,
        train_period_end,
        predict_period_start,
        predict_period_end,
        variable,
        latmin,
        latmax,
        lonmin,
        lonmax,
        obs_identifier=obs_identifier,
        gcm_grid_spec=gcm_grid_spec,
        chunking_approach='full_time',
    )

    gcm_train_subset_full_time_ds = return_gcm_train_full_time_task(
        coarse_obs_full_time_ds,
        gcm,
        scenario,
        train_period_start,
        train_period_end,
        predict_period_start,
        predict_period_end,
        variable,
        latmin,
        latmax,
        lonmin,
        lonmax,
        gcm_identifier=gcm_identifier,
        chunking_approach='full_time',
    )

    gcm_predict_rechunked_ds = return_gcm_predict_rechunked_task(
        gcm_train_subset_full_time_ds,
        gcm,
        scenario,
        train_period_start,
        train_period_end,
        predict_period_start,
        predict_period_end,
        variable,
        latmin,
        latmax,
        lonmin,
        lonmax,
        gcm_identifier=gcm_identifier,
    )

    # fit and predict tasks(s):
    bias_corrected_ds = fit_and_predict_task(
        gcm_train_subset_full_time_ds,
        coarse_obs_full_time_ds,
        gcm_predict_rechunked_ds,
        gcm,
        scenario,
        train_period_start,
        train_period_end,
        predict_period_start,
        predict_period_end,
        variable,
        latmin,
        latmax,
        lonmin,
        lonmax,
        gcm_identifier=gcm_identifier,
    )
    # postprocess_bcsd_task(s):
    postprocess_bcsd_ds = postprocess_bcsd_task(
        bias_corrected_ds,
        spatial_anomalies_ds,
        gcm,
        scenario,
        train_period_start,
        train_period_end,
        predict_period_start,
        predict_period_end,
        variable,
        latmin,
        latmax,
        lonmin,
        lonmax,
        gcm_identifier=gcm_identifier,
    )
