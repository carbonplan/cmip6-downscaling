from prefect import Flow, Parameter, task
from xpersist import CacheStore
from xpersist.prefect.result import XpersistResult

from cmip6_downscaling import config, runtimes
from cmip6_downscaling.analysis.analysis import annual_summary, monthly_summary, run_analyses
from cmip6_downscaling.methods.bcsd import (
    fit_and_predict,
    get_coarse_obs,
    get_spatial_anomalies,
    make_flow_paths,
    postprocess_bcsd,
    return_coarse_obs_full_time,
    return_gcm_predict_rechunked,
    return_gcm_train_full_time,
    return_obs,
)
from cmip6_downscaling.tasks import pyramid

runtime = runtimes.get_runtime()

config.set(
    {
        'storage.intermediate.uri': 'az://flow-outputs/intermediates',
        'storage.results.uri': 'az://flow-outputs/results',
        'storage.temporary.uri': 'az://flow-outputs/temporary',
    }
)

target_naming_str = "{gcm}-{scenario}-{train_period_start}-{train_period_end}-{predict_period_start}-{predict_period_end}-{latmin}-{latmax}-{lonmin}-{lonmax}-{variable}.zarr"


intermediate_cache_store = CacheStore(
    config.get('storage.intermediate.uri'),
    storage_options=config.get('storage.intermediate.storage_options'),
)
results_cache_store = CacheStore(
    config.get('storage.results.uri'), storage_options=config.get('storage.results.storage_options')
)


from cmip6_downscaling.analysis.analysis import (
    monthly_summary,
    annual_summary
)

# Transform Functions into Tasks -----------------------------------------------------------


make_flow_paths_task = task(make_flow_paths, log_stdout=True, nout=4)

return_obs_task = task(
    return_obs,
    result=XpersistResult(intermediate_cache_store, serializer="xarray.zarr"),
    target="obs-ds-" + target_naming_str,
)
get_coarse_obs_task = task(
    get_coarse_obs,
    tags=['dask-resource:TASKSLOTS=1'],
    result=XpersistResult(intermediate_cache_store, serializer="xarray.zarr"),
    target="coarse-obs-ds-" + target_naming_str,
)
get_spatial_anomalies_task = task(
    get_spatial_anomalies,
    log_stdout=True,
    tags=['dask-resource:TASKSLOTS=1'],
    result=XpersistResult(intermediate_cache_store, serializer="xarray.zarr"),
    target="spatial-anomalies-ds-" + target_naming_str,
)


return_coarse_obs_full_time_task = task(
    return_coarse_obs_full_time,
    tags=['dask-resource:TASKSLOTS=1'],
    result=XpersistResult(intermediate_cache_store, serializer="xarray.zarr"),
    target="y-full-time-" + target_naming_str,
)

return_gcm_train_full_time_task = task(
    return_gcm_train_full_time,
    tags=['dask-resource:TASKSLOTS=1'],
    result=XpersistResult(intermediate_cache_store, serializer="xarray.zarr"),
    target="x-train-full-time-" + target_naming_str,
)

return_gcm_predict_rechunked_task = task(
    return_gcm_predict_rechunked,
    tags=['dask-resource:TASKSLOTS=1'],
    result=XpersistResult(intermediate_cache_store, serializer="xarray.zarr"),
    target="x-predict-rechunked-" + target_naming_str,
)

fit_and_predict_task = task(
    fit_and_predict,
    log_stdout=True,
    result=XpersistResult(intermediate_cache_store, serializer="xarray.zarr"),
    target="fit-and-predict-" + target_naming_str,
)

postprocess_bcsd_task = task(
    postprocess_bcsd,
    tags=['dask-resource:TASKSLOTS=1'],
    log_stdout=True,
    result=XpersistResult(results_cache_store, serializer="xarray.zarr"),
    target="postprocess-results-" + target_naming_str,
)

monthly_summary_task = task(
    monthly_summary,
    log_stdout=True,
    result=XpersistResult(results_cache_store, serializer=serializer),
    target="monthly-summary-" + target_naming_str,
)

annual_summary_task = task(
    annual_summary,
    log_stdout=True,
    result=XpersistResult(results_cache_store, serializer=serializer),
    target="annual-summary-" + target_naming_str,
)

run_analyses_task = task(
    run_analyses,
    log_stdout=True,
    # TODO: Force a dependency on whether postprocess_bcsd_task has
    # re-run. If it hasn't, then this task doesn't need to run again.
    # However, this step doesn't take `postprocess_bcsd_task` as an input
    # so it needs to have an explicit dependency.
)

# Main Flow -----------------------------------------------------------

with Flow(
    name='bcsd',
    storage=runtime.storage,
    run_config=runtime.run_config,
    executor=runtime.executor,
) as bcsd_flow:
    gcm = Parameter("GCM")
    scenario = Parameter("SCENARIO")
    train_period_start = Parameter("TRAIN_PERIOD_START")
    train_period_end = Parameter("TRAIN_PERIOD_END")
    predict_period_start = Parameter("PREDICT_PERIOD_START")
    predict_period_end = Parameter("PREDICT_PERIOD_END")
    variable = Parameter("VARIABLE")
    latmin = Parameter("LATMIN")
    latmax = Parameter("LATMAX")
    lonmin = Parameter("LONMIN")
    lonmax = Parameter("LONMAX")

    (
        coarse_obs_path,
        spatial_anomalies_path,
        bias_corrected_path,
        final_out_path,
    ) = make_flow_paths_task(
        GCM=gcm,
        SCENARIO=scenario,
        TRAIN_PERIOD_START=train_period_start,
        TRAIN_PERIOD_END=train_period_end,
        PREDICT_PERIOD_START=predict_period_start,
        PREDICT_PERIOD_END=predict_period_end,
        VARIABLE=variable,
        LATMIN=latmin,
        LATMAX=latmax,
        LONMIN=lonmin,
        LONMAX=lonmax,
    )
    # preprocess_bcsd_tasks(s):
    obs_ds = return_obs_task(
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
    )
    # regrid(ds: xr.Dataset, levels: int = 2, uri: str = None, other_chunks: dict = None)
    # format naming w/ prefect context
    pyramid_location = pyramid.regrid(
        postprocess_bcsd_ds, uri=config.get('storage.results.uri') + '/pyramids/' + 'test.pyr'
    )

    monthly_summary_ds = monthly_summary_task(
        postprocess_bcsd_ds,
    )

    annual_summary_ds = annual_summary_task(postprocess_bcsd_ds)

    run_analyses_task(
        {'run_id': target_naming_str, 'var': variable, 'gcm': gcm, 'scenario': scenario}
    )

    monthly_summary_ds = monthly_summary_task(
        postprocess_bcsd_ds,
        spatial_anomalies_ds,
        gcm,
        scenario,
        train_period_start,
        train_period_end,
        predict_period_start,
        predict_period_end,
        variable,
    )

    annual_summary_ds = annual_summary_task(
        postprocess_bcsd_ds,
        spatial_anomalies_ds,
        gcm,
        scenario,
        train_period_start,
        train_period_end,
        predict_period_start,
        predict_period_end,
        variable,
    )
