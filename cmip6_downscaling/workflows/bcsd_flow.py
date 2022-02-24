from datetime import timedelta

from prefect import Flow, Parameter, task
from xpersist import CacheStore
from xpersist.prefect.result import XpersistResult

from cmip6_downscaling import config, runtimes
from cmip6_downscaling.analysis.analysis import annual_summary, monthly_summary, run_analyses
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
from cmip6_downscaling.tasks import cleanup, pyramid
from cmip6_downscaling.tasks.common_tasks import (
    build_bbox,
    build_time_period_slices,
    path_builder_task,
)
from cmip6_downscaling.workflows.paths import (
    make_annual_summary_path,
    make_bcsd_output_path,
    make_bias_corrected_path,
    make_coarse_obs_path_full_space,
    make_coarse_obs_path_full_time,
    make_gcm_predict_path,
    make_monthly_summary_path,
    make_rechunked_gcm_path,
    make_return_obs_path,
    make_spatial_anomalies_path,
)

# storage_prefix = config.get("runtime.cloud.storage_prefix")

runtime = runtimes.get_runtime()

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
    max_retries=10,
    retry_delay=timedelta(seconds=10),
    result=XpersistResult(intermediate_cache_store, serializer="xarray.zarr"),
    target=make_coarse_obs_path_full_space,
)
get_spatial_anomalies_task = task(
    get_spatial_anomalies,
    tags=['dask-resource:TASKSLOTS=1'],
    max_retries=10,
    retry_delay=timedelta(seconds=5),
    result=XpersistResult(intermediate_cache_store, serializer="xarray.zarr"),
    target=make_spatial_anomalies_path,
)
return_coarse_obs_full_time_task = task(
    return_coarse_obs_full_time,
    tags=['dask-resource:TASKSLOTS=1'],
    max_retries=10,
    retry_delay=timedelta(seconds=5),
    result=XpersistResult(intermediate_cache_store, serializer="xarray.zarr"),
    target=make_coarse_obs_path_full_time,
)

return_gcm_train_full_time_task = task(
    return_gcm_train_full_time,
    tags=['dask-resource:TASKSLOTS=1'],
    max_retries=10,
    retry_delay=timedelta(seconds=5),
    result=XpersistResult(intermediate_cache_store, serializer="xarray.zarr"),
    target=make_rechunked_gcm_path,
)

return_gcm_predict_rechunked_task = task(
    return_gcm_predict_rechunked,
    tags=['dask-resource:TASKSLOTS=1'],
    max_retries=10,
    retry_delay=timedelta(seconds=5),
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
    max_retries=10,
    retry_delay=timedelta(seconds=5),
    log_stdout=True,
    result=XpersistResult(results_cache_store, serializer="xarray.zarr"),
    target=make_bcsd_output_path,
)

monthly_summary_task = task(
    monthly_summary,
    log_stdout=True,
    result=XpersistResult(results_cache_store, serializer="xarray.zarr"),
    target=make_monthly_summary_path,  # TODO: replace with the paradigm from PR #84 once it's merged (also pull that)
)

annual_summary_task = task(
    annual_summary,
    result=XpersistResult(results_cache_store, serializer="xarray.zarr"),
    target=make_annual_summary_path,
)


# Main Flow -----------------------------------------------------------

with Flow(
    name="bcsd",
    storage=runtime.storage,
    run_config=runtime.run_config,
    executor=runtime.executor,
) as bcsd_flow:
    obs = Parameter("obs")
    gcm = Parameter("gcm")
    scenario = Parameter("scenario")
    variable = Parameter("variable")

    # Note: bbox and train and predict period had to be encapsulated into tasks to prevent prefect from complaining about unused parameters.
    bbox = build_bbox(
        latmin=Parameter("latmin"),
        latmax=Parameter("latmax"),
        lonmin=Parameter("lonmin"),
        lonmax=Parameter("lonmax"),
    )
    train_period = build_time_period_slices(Parameter('train_period'))
    predict_period = build_time_period_slices(Parameter('predict_period'))
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
    if config.get('run_options.cleanup_flag') is True:
        cleanup.run_rsfip(gcm_identifier, obs_identifier)

    # preprocess_bcsd_tasks(s):
    obs_ds = return_obs_task(
        obs=obs,
        variable=variable,
        train_period=train_period,
        bbox=bbox,
        obs_identifier=obs_identifier,
    )

    coarse_obs_ds = get_coarse_obs_task(
        obs_ds=obs_ds,
        gcm=gcm,
        scenario=scenario,
        variable=variable,
        train_period=train_period,
        predict_period=predict_period,
        bbox=bbox,
        obs_identifier=obs_identifier,
        chunking_approach='full_space',
        gcm_grid_spec=gcm_grid_spec,
    )

    spatial_anomalies_ds = get_spatial_anomalies_task(
        coarse_obs=coarse_obs_ds,
        obs_ds=obs_ds,
        gcm=gcm,
        scenario=scenario,
        variable=variable,
        train_period=train_period,
        predict_period=predict_period,
        bbox=bbox,
        obs_identifier=obs_identifier,
        gcm_grid_spec=gcm_grid_spec,
    )

    # the next three tasks prepare the inputs required by bcsd
    coarse_obs_full_time_ds = return_coarse_obs_full_time_task(
        coarse_obs_ds=coarse_obs_ds,
        gcm=gcm,
        scenario=scenario,
        variable=variable,
        train_period=train_period,
        predict_period=predict_period,
        bbox=bbox,
        obs_identifier=obs_identifier,
        gcm_grid_spec=gcm_grid_spec,
    )

    gcm_train_subset_full_time_ds = return_gcm_train_full_time_task(
        coarse_obs_full_time_ds=coarse_obs_full_time_ds,
        gcm=gcm,
        scenario=scenario,
        variable=variable,
        train_period=train_period,
        predict_period=predict_period,
        bbox=bbox,
        gcm_identifier=gcm_identifier,
    )

    gcm_predict_rechunked_ds = return_gcm_predict_rechunked_task(
        gcm_train_subset_full_time_ds=gcm_train_subset_full_time_ds,
        gcm=gcm,
        scenario=scenario,
        variable=variable,
        train_period=train_period,
        predict_period=predict_period,
        bbox=bbox,
        gcm_identifier=gcm_identifier,
    )

    # fit and predict tasks(s):
    bias_corrected_ds = fit_and_predict_task(
        gcm_train_subset_full_time_ds=gcm_train_subset_full_time_ds,
        coarse_obs_full_time_ds=coarse_obs_full_time_ds,
        gcm_predict_rechunked_ds=gcm_predict_rechunked_ds,
        gcm=gcm,
        scenario=scenario,
        variable=variable,
        train_period=train_period,
        predict_period=predict_period,
        bbox=bbox,
        gcm_identifier=gcm_identifier,
    )
    # postprocess_bcsd_task(s):
    postprocess_bcsd_ds = postprocess_bcsd_task(
        bias_corrected_ds=bias_corrected_ds,
        spatial_anomalies_ds=spatial_anomalies_ds,
        gcm=gcm,
        scenario=scenario,
        variable=variable,
        train_period=train_period,
        predict_period=predict_period,
        bbox=bbox,
        gcm_identifier=gcm_identifier,
    )
    # regrid(ds: xr.Dataset, levels: int = 2, uri: str = None, other_chunks: dict = None)
    # format naming w/ prefect context

    monthly_summary_ds = monthly_summary_task(
        postprocess_bcsd_ds,
        gcm=gcm,
        scenario=scenario,
        variable=variable,
        train_period=train_period,
        predict_period=predict_period,
        bbox=bbox,
        gcm_identifier=gcm_identifier,
    )

    annual_summary_ds = annual_summary_task(
        postprocess_bcsd_ds,
        gcm=gcm,
        scenario=scenario,
        variable=variable,
        train_period=train_period,
        predict_period=predict_period,
        bbox=bbox,
        gcm_identifier=gcm_identifier,
    )

    analysis_location = run_analyses(
        {
            'gcm_identifier': gcm_identifier,
            'obs_identifier': obs_identifier,
            'gcm_grid_spec': gcm_grid_spec,
            'result_dir': config.get('storage.results.uri'),
            'intermediate_dir': config.get('storage.intermediate.uri'),
            'var': variable,
            'gcm': gcm,
            'scenario': scenario,
        },
        web_blob=config.get('storage.web_results.blob'),
        bbox=bbox,
        train_period=train_period,
        predict_period=predict_period,
        upstream_tasks=[postprocess_bcsd_ds],
    )

    pyramid_location_daily = pyramid.regrid(
        postprocess_bcsd_ds,
        uri=config.get('storage.results.uri') + pyramid_path_daily,
    )

    pyramid_location_monthly = pyramid.regrid(
        monthly_summary_ds, uri=config.get('storage.results.uri') + pyramid_path_monthly
    )

    pyramid_location_annual = pyramid.regrid(
        annual_summary_ds, uri=config.get('storage.results.uri') + pyramid_path_annual
    )
