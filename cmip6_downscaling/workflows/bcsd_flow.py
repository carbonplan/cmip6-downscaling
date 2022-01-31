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
from cmip6_downscaling.tasks import pyramid
from cmip6_downscaling.tasks.common_tasks import (
    build_bbox,
    build_time_period_slices,
    path_builder_task,
)
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


# storage = Azure("prefect")
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

    # bbox and train and predict period had to be encapsulated into tasks to prevent prefect from complaining about unused parameters.
    bbox = build_bbox(
        latmin=Parameter("latmin"),
        latmax=Parameter("latmax"),
        lonmin=Parameter("lonmin"),
        lonmax=Parameter("lonmax"),
    )
    train_period = build_time_period_slices(Parameter('train_period'))
    predict_period = build_time_period_slices(Parameter('predict_period'))

    gcm_grid_spec, obs_identifier, gcm_identifier, pyramid_path = path_builder_task(
        obs=obs,
        gcm=gcm,
        scenario=scenario,
        variable=variable,
        train_period=train_period,
        predict_period=predict_period,
        bbox=bbox,
    )

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
    pyramid_location = pyramid.regrid(
        postprocess_bcsd_ds,
        uri=config.get('storage.results.uri') + pyramid_path,
    )
