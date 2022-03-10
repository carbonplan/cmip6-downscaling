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
    rechunked_interpolated_prediciton_task_full_time,
    rechunked_spatial_anomalies_full_time,
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
    target_grid_obs_ds_task,
)
from cmip6_downscaling.workflows.paths import (
    make_annual_summary_path,
    make_bcsd_output_path,
    make_bias_corrected_path,
    make_coarse_obs_path_full_space,
    make_coarse_obs_path_full_time,
    make_gcm_predict_path,
    make_interpolated_obs_path,
    make_interpolated_prediction_path_full_space,
    make_interpolated_prediction_path_full_time,
    make_monthly_summary_path,
    make_rechunked_gcm_path,
    make_rechunked_spatial_anomalies_path_full_time,
    make_return_obs_path,
    make_spatial_anomalies_path,
)
from cmip6_downscaling.workflows.utils import regrid_ds

runtime = runtimes.get_runtime()

intermediate_cache_store = CacheStore(
    config.get("storage.intermediate.uri"),
    storage_options=config.get("storage.intermediate.storage_options"),
)

results_cache_store = CacheStore(
    config.get("storage.results.uri"),
    storage_options=config.get("storage.results.storage_options"),
)

serializer_dump_kwargs = config.get('storage.xpersist_overwrite')
# Transform Functions into Tasks -----------------------------------------------------------


return_obs_task = task(
    return_obs,
    result=XpersistResult(
        intermediate_cache_store,
        serializer="xarray.zarr",
        serializer_dump_kwargs=serializer_dump_kwargs,
    ),
    target=make_return_obs_path,
)


get_coarse_obs_task = task(
    get_coarse_obs,
    tags=['dask-resource:TASKSLOTS=1'],
    # max_retries=10,
    # retry_delay=timedelta(seconds=10),
    result=XpersistResult(
        intermediate_cache_store,
        serializer="xarray.zarr",
        serializer_dump_kwargs=serializer_dump_kwargs,
    ),
    target=make_coarse_obs_path_full_space,
)

target_grid_obs_ds_task = task(
    target_grid_obs_ds_task,
)


interpolated_obs_task = task(
    regrid_ds,
    tags=['dask-resource:TASKSLOTS=1'],
    # max_retries=5,
    # retry_delay=timedelta(seconds=5),
    result=XpersistResult(
        intermediate_cache_store,
        serializer="xarray.zarr",
        serializer_dump_kwargs=serializer_dump_kwargs,
    ),
    target=make_interpolated_obs_path,
)
interpolated_prediction_task = task(
    regrid_ds,
    tags=['dask-resource:TASKSLOTS=1'],
    max_retries=5,
    retry_delay=timedelta(seconds=5),
    result=XpersistResult(
        intermediate_cache_store,
        serializer="xarray.zarr",
        serializer_dump_kwargs=serializer_dump_kwargs,
    ),
    target=make_interpolated_prediction_path_full_space,
)

rechunked_interpolated_prediciton_task_full_time_task = task(
    rechunked_interpolated_prediciton_task_full_time,
    # tags=['dask-resource:TASKSLOTS=1'],
    # max_retries=10,
    # retry_delay=timedelta(seconds=10),
    result=XpersistResult(
        intermediate_cache_store,
        serializer="xarrzay.zarr",
        serializer_dump_kwargs=serializer_dump_kwargs,
    ),
    target=make_interpolated_prediction_path_full_time,
)

get_spatial_anomalies_task = task(
    get_spatial_anomalies,
    log_stdout=True,
    # max_retries=5,
    # retry_delay=timedelta(seconds=5),
    result=XpersistResult(
        intermediate_cache_store,
        serializer="zarr.zarr",
        serializer_dump_kwargs=serializer_dump_kwargs,
    ),
    target=make_spatial_anomalies_path,  # "flow-outputs/prefect_intermediates/spatial_anomalies_test/ERA5/tasmax/-90.0_90.0_-180.0_180.0/1981_2010/128x256_gridsize_14_14_llcorner_-88_-180.zarr"#make_spatial_anomalies_path"
)


rechunked_spatial_anomalies_full_time_task = task(
    rechunked_spatial_anomalies_full_time,
    # tags=['dask-resource:TASKSLOTS=1'],
    # max_retries=10,
    # retry_delay=timedelta(seconds=10),
    result=XpersistResult(
        intermediate_cache_store,
        serializer="xarray.zarr",
        serializer_dump_kwargs=serializer_dump_kwargs,
    ),
    target=make_rechunked_spatial_anomalies_path_full_time,
)


return_coarse_obs_full_time_task = task(
    return_coarse_obs_full_time,
    # tags=['dask-resource:TASKSLOTS=1'],
    # max_retries=10,
    # retry_delay=timedelta(seconds=5),
    result=XpersistResult(
        intermediate_cache_store,
        serializer="xarray.zarr",
        serializer_dump_kwargs=serializer_dump_kwargs,
    ),
    target=make_coarse_obs_path_full_time,
)


return_gcm_train_full_time_task = task(
    return_gcm_train_full_time,
    # tags=['dask-resource:TASKSLOTS=1'],
    # max_retries=10,
    # retry_delay=timedelta(seconds=5),
    result=XpersistResult(
        intermediate_cache_store,
        serializer="xarray.zarr",
        serializer_dump_kwargs=serializer_dump_kwargs,
    ),
    target=make_rechunked_gcm_path,
)

return_gcm_predict_rechunked_task = task(
    return_gcm_predict_rechunked,
    # tags=['dask-resource:TASKSLOTS=1'],
    # max_retries=10,
    # retry_delay=timedelta(seconds=5),
    result=XpersistResult(
        intermediate_cache_store,
        serializer="xarray.zarr",
        serializer_dump_kwargs=serializer_dump_kwargs,
    ),
    target=make_gcm_predict_path,
)

fit_and_predict_task = task(
    fit_and_predict,
    log_stdout=True,
    # tags=['dask-resource:TASKSLOTS=1'],
    # max_retries=10,
    # retry_delay=timedelta(seconds=5),
    result=XpersistResult(
        intermediate_cache_store,
        serializer="xarray.zarr",
        serializer_dump_kwargs=serializer_dump_kwargs,
    ),
    target=make_bias_corrected_path,
)
postprocess_bcsd_task = task(
    postprocess_bcsd,
    tags=['dask-resource:TASKSLOTS=1'],
    # max_retries=10,
    # retry_delay=timedelta(seconds=5),
    log_stdout=True,
    result=XpersistResult(
        results_cache_store, serializer="xarray.zarr", serializer_dump_kwargs=serializer_dump_kwargs
    ),
    target=make_bcsd_output_path,
)

monthly_summary_task = task(
    monthly_summary,
    log_stdout=True,
    result=XpersistResult(
        results_cache_store, serializer="xarray.zarr", serializer_dump_kwargs=serializer_dump_kwargs
    ),
    target=make_monthly_summary_path,  # TODO: replace with the paradigm from PR #84 once it's merged (also pull that)
)

annual_summary_task = task(
    annual_summary,
    result=XpersistResult(
        results_cache_store, serializer="xarray.zarr", serializer_dump_kwargs=serializer_dump_kwargs
    ),
    target=make_annual_summary_path,
)


# # Main Flow -----------------------------------------------------------

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
        gcm=gcm,
        scenario=scenario,
        variable=variable,
        train_period=train_period,
        predict_period=predict_period,
        bbox=bbox,
        obs_identifier=obs_identifier,
        chunking_approach='full_space',
    )

    target_grid_obs_ds = target_grid_obs_ds_task(coarse_obs_ds)
    interpolated_obs_ds = interpolated_obs_task(
        ds_path=coarse_obs_ds,
        target_grid_ds=target_grid_obs_ds,
        gcm_grid_spec=gcm_grid_spec,
        chunking_approach='full_space',
        obs_identifier=obs_identifier,
        upstream_tasks=[coarse_obs_ds, target_grid_obs_ds],
    )
    spatial_anomalies_ds = get_spatial_anomalies_task(
        gcm=gcm,
        scenario=scenario,
        variable=variable,
        train_period=train_period,
        predict_period=predict_period,
        bbox=bbox,
        obs_identifier=obs_identifier,
        gcm_grid_spec=gcm_grid_spec,
        chunking_approach='full_space',
        upstream_tasks=[interpolated_obs_ds, obs_ds],
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

    interpolated_prediction_ds = interpolated_prediction_task(
        ds=bias_corrected_ds,
        target_grid_ds=target_grid_obs_ds,
        gcm_identifier=gcm_identifier,
        gcm_grid_spec=gcm_grid_spec,
    )

    rechunked_interpolated_prediction_ds = rechunked_interpolated_prediciton_task_full_time_task(
        interpolated_prediction_ds=interpolated_prediction_ds,
        gcm_identifier=gcm_identifier,
        gcm_grid_spec=gcm_grid_spec,
    )
    rechunked_spatial_anomalies_ds = rechunked_spatial_anomalies_full_time_task(
        spatial_anomalies_ds=spatial_anomalies_ds,
        gcm_identifier=gcm_identifier,
        gcm_grid_spec=gcm_grid_spec,
    )

    # postprocess_bcsd_task(s):
    postprocess_bcsd_ds = postprocess_bcsd_task(
        rechunked_interpolated_prediction_ds=rechunked_interpolated_prediction_ds,
        rechunked_spatial_anomalies_ds=rechunked_spatial_anomalies_ds,
        gcm=gcm,
        scenario=scenario,
        variable=variable,
        train_period=train_period,
        predict_period=predict_period,
        bbox=bbox,
        gcm_identifier=gcm_identifier,
    )

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
        postprocess_bcsd_ds, uri=config.get('storage.results.uri') + pyramid_path_daily, levels=4
    )

    pyramid_location_monthly = pyramid.regrid(
        monthly_summary_ds, uri=config.get('storage.results.uri') + pyramid_path_monthly, levels=4
    )

    pyramid_location_annual = pyramid.regrid(
        annual_summary_ds, uri=config.get('storage.results.uri') + pyramid_path_annual, levels=4
    )
