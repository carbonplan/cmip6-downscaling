# Imports -----------------------------------------------------------
import os

os.environ["PREFECT__FLOWS__CHECKPOINTING"] = "true"
from funnel.prefect.result import FunnelResult
from prefect import Flow, Parameter, task

from cmip6_downscaling.config.config import (  # dask_executor,; kubernetes_run_config,; storage,
    intermediate_cache_store,
    results_cache_store,
    serializer,
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

return_obs_task = task(
    return_obs,
    result=FunnelResult(intermediate_cache_store, serializer=serializer),
    target="obs-ds",
)
get_coarse_obs_task = task(
    get_coarse_obs,
    result=FunnelResult(intermediate_cache_store, serializer=serializer),
    target="coarse-obs-ds",
)
get_spatial_anomalies_task = task(
    get_spatial_anomalies,
    result=FunnelResult(intermediate_cache_store, serializer=serializer),
    target="spatial-anomalies-ds-" + target_naming_str,
)
return_y_full_time_task = task(
    return_y_full_time,
    result=FunnelResult(intermediate_cache_store, serializer=serializer),
    target="y-full-time-" + target_naming_str,
)
return_x_train_full_time_task = task(
    return_x_train_full_time,
    result=FunnelResult(intermediate_cache_store, serializer=serializer),
    target="x-train-full-time-" + target_naming_str,
)
return_x_predict_rechunked_task = task(
    return_x_predict_rechunked,
    result=FunnelResult(intermediate_cache_store, serializer=serializer),
    target="x-predict-rechunked-" + target_naming_str,
)
fit_and_predict_task = task(
    fit_and_predict,
    result=FunnelResult(intermediate_cache_store, serializer=serializer),
    target="fit-and-predict-" + target_naming_str,
)
postprocess_bcsd_task = task(
    postprocess_bcsd,
    log_stdout=True,
    result=FunnelResult(results_cache_store, serializer=serializer),
    target="postprocess-results-" + target_naming_str,
)


# Main Flow -----------------------------------------------------------

# with Flow(name="bcsd-testing", storage=storage, run_config=run_config) as flow:
# with Flow(name="bcsd-testing", storage=storage, run_config=kubernetes_run_config, executor=dask_executor) as flow:
with Flow(name="bcsd-testing") as flow:
    gcm = Parameter("GCM")
    scenario = Parameter("SCENARIO")
    train_period_start = Parameter("TRAIN_PERIOD_START")
    train_period_end = Parameter("TRAIN_PERIOD_END")
    predict_period_start = Parameter("PREDICT_PERIOD_START")
    predict_period_end = Parameter("PREDICT_PERIOD_END")
    variable = Parameter("VARIABLE")
    # params_dict = {gcm:gcm,scenario:scenario,train_period_start:train_period_start,train_period_end:train_period_end,
    # predict_period_start:predict_period_start,predict_period_end:predict_period_end,variable:variable}

    # gcm, scenario, train_period_start,train_period_end,predict_period_start,predict_period_end,variable

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
    obs_ds = return_obs_task(train_period_start, train_period_end, variable)
    coarse_obs_ds = get_coarse_obs_task(obs_ds, variable)
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
