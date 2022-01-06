# Imports -----------------------------------------------------------
import os

os.environ["PREFECT__FLOWS__CHECKPOINTING"] = "true"
from prefect import Flow, Parameter, task
from xpersist.prefect.result import XpersistResult

from cmip6_downscaling.config.config import (
    dask_executor,
    intermediate_cache_store,
    kubernetes_run_config,
    results_cache_store,
    serializer,
    storage,
)
from cmip6_downscaling.methods.bcsd import (
    fit_and_predict,
    get_coarse_obs,
    get_spatial_anomalies,
    make_flow_paths,
    postprocess_bcsd,
    return_obs,
    return_x_predict_rechunked,
    return_x_train_full_time,
    return_y_full_time,
)

# Transform Functions into Tasks -----------------------------------------------------------

target_naming_str = "{gcm}-{scenario}-{train_period_start}-{train_period_end}-{predict_period_start}-{predict_period_end}-{variable}.zarr"

make_flow_paths_task = task(make_flow_paths, log_stdout=True, nout=4)

# no rechunking
return_obs_task = task(
    return_obs,
    result=XpersistResult(intermediate_cache_store, serializer=serializer),
    target="obs-ds-" + target_naming_str,
)
# yes rechunking
get_coarse_obs_task = task(
    get_coarse_obs,
    tags=['dask-resource:TASKSLOTS=1'],
    result=XpersistResult(intermediate_cache_store, serializer=serializer),
    target="coarse-obs-ds-" + target_naming_str,
)
# yes rechunk
get_spatial_anomalies_task = task(
    get_spatial_anomalies,
    log_stdout=True,
    tags=['dask-resource:TASKSLOTS=1'],
    result=XpersistResult(intermediate_cache_store, serializer=serializer),
    target="spatial-anomalies-ds-" + target_naming_str,
)

# yes rechunk

return_y_full_time_task = task(
    return_y_full_time,
    tags=['dask-resource:TASKSLOTS=1'],
    result=XpersistResult(intermediate_cache_store, serializer=serializer),
    target="y-full-time-" + target_naming_str,
)
# yes rechunk

return_x_train_full_time_task = task(
    return_x_train_full_time,
    tags=['dask-resource:TASKSLOTS=1'],
    result=XpersistResult(intermediate_cache_store, serializer=serializer),
    target="x-train-full-time-" + target_naming_str,
)
# yes rechunk

return_x_predict_rechunked_task = task(
    return_x_predict_rechunked,
    tags=['dask-resource:TASKSLOTS=1'],
    result=XpersistResult(intermediate_cache_store, serializer=serializer),
    target="x-predict-rechunked-" + target_naming_str,
)
# no rechunking

fit_and_predict_task = task(
    fit_and_predict,
    log_stdout=True,
    result=XpersistResult(intermediate_cache_store, serializer=serializer),
    target="fit-and-predict-" + target_naming_str,
)
# yes rechunk

postprocess_bcsd_task = task(
    postprocess_bcsd,
    tags=['dask-resource:TASKSLOTS=1'],
    log_stdout=True,
    result=XpersistResult(results_cache_store, serializer=serializer),
    target="postprocess-results-" + target_naming_str,
)

# Main Flow -----------------------------------------------------------
# with Flow(name="bcsd-testing", storage=storage, run_config=run_config) as flow:
# with Flow(name="pr-testing") as flow:
with Flow(
    name="bcsd-pr-test",
    storage=storage,
    run_config=kubernetes_run_config,
    executor=dask_executor,
) as flow:

    gcm = Parameter("GCM")
    scenario = Parameter("SCENARIO")
    train_period_start = Parameter("TRAIN_PERIOD_START")
    train_period_end = Parameter("TRAIN_PERIOD_END")
    predict_period_start = Parameter("PREDICT_PERIOD_START")
    predict_period_end = Parameter("PREDICT_PERIOD_END")
    variable = Parameter("VARIABLE")
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
    )
    # prep_bcsd_inputs_task(s):
    y_full_time_ds = return_y_full_time_task(
        coarse_obs_ds,
        gcm,
        scenario,
        train_period_start,
        train_period_end,
        predict_period_start,
        predict_period_end,
        variable,
    )
    x_train_full_time_ds = return_x_train_full_time_task(
        y_full_time_ds,
        gcm,
        scenario,
        train_period_start,
        train_period_end,
        predict_period_start,
        predict_period_end,
        variable,
    )
    x_predict_rechunked_ds = return_x_predict_rechunked_task(
        x_train_full_time_ds,
        gcm,
        scenario,
        train_period_start,
        train_period_end,
        predict_period_start,
        predict_period_end,
        variable,
    )
    # fit and predict tasks(s):
    bias_corrected_ds = fit_and_predict_task(
        x_train_full_time_ds,
        y_full_time_ds,
        x_predict_rechunked_ds,
        gcm,
        scenario,
        train_period_start,
        train_period_end,
        predict_period_start,
        predict_period_end,
        variable,
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
    )
