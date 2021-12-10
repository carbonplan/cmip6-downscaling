# Imports -----------------------------------------------------------
import os

os.environ['PREFECT__FLOWS__CHECKPOINTING'] = 'true'
from funnel.prefect.result import FunnelResult
from prefect import Flow, Parameter, task

from cmip6_downscaling.config.config import (  # dask_executor,; kubernetes_run_config,; storage,
    cache_store,
    connection_string,
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
    return_x_train_rechunked,
    return_y_rechunked,
    write_bcsd_results,
)

# Transform Functions into Tasks -----------------------------------------------------------

make_flow_paths_task = task(make_flow_paths, log_stdout=True, nout=4)

return_obs_task = task(
    return_obs, result=FunnelResult(cache_store, serializer=serializer), target='obs-ds'
)
get_coarse_obs_task = task(
    get_coarse_obs, result=FunnelResult(cache_store, serializer=serializer), target='coarse-obs-ds'
)
get_spatial_anomalies_task = task(
    get_spatial_anomalies,
    result=FunnelResult(cache_store, serializer=serializer),
    target='spatial-anomalies',
)

return_y_rechunked_task = task(
    return_y_rechunked,
    result=FunnelResult(cache_store, serializer=serializer),
    target='y-rechunked',
)
return_x_train_rechunked_task = task(
    return_x_train_rechunked,
    result=FunnelResult(cache_store, serializer=serializer),
    target='x-train-rechunked',
)
return_x_predict_rechunked_task = task(
    return_x_predict_rechunked,
    result=FunnelResult(cache_store, serializer=serializer),
    target='x-predict-rechunked',
)
fit_and_predict_task = task(
    fit_and_predict,
    result=FunnelResult(cache_store, serializer=serializer),
    target='fit-and-predict',
)
postprocess_bcsd_task = task(
    postprocess_bcsd,
    result=FunnelResult(cache_store, serializer=serializer),
    target='postprocess_results',
)
write_bcsd_results_task = task(write_bcsd_results, log_stdout=True)


# Main Flow -----------------------------------------------------------

# with Flow(name="bcsd-testing", storage=storage, run_config=run_config) as flow:
# with Flow(name="bcsd-testing", storage=storage, run_config=kubernetes_run_config, executor=dask_executor) as flow:
with Flow(name="bcsd-testing") as flow:

    flow_name = Parameter("FLOW_NAME")
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
    obs_ds = return_obs_task(train_period_start, train_period_end, variable)
    coarse_obs_ds = get_coarse_obs_task(obs_ds, variable, connection_string)
    spatial_anomalies_ds = get_spatial_anomalies_task(
        coarse_obs_ds, obs_ds, variable, connection_string
    )

    # prep_bcsd_inputs_task(s):
    y_rechunked_ds = return_y_rechunked_task(coarse_obs_ds, variable)
    x_train_rechunked_ds = return_x_train_rechunked_task(
        gcm, variable, train_period_start, train_period_end, y_rechunked_ds
    )
    x_predict_rechunked_ds = return_x_predict_rechunked_task(
        gcm, scenario, variable, predict_period_start, predict_period_end, x_train_rechunked_ds
    )

    # #fit and predict tasks(s):
    bias_corrected_ds = fit_and_predict_task(
        x_train_rechunked_ds, y_rechunked_ds, x_predict_rechunked_ds, variable, "time"
    )

    # #postprocess_bcsd_task(s):
    # postprocess_bcsd_ds = postprocess_bcsd_task(bias_corrected_ds, spatial_anomalies_ds, variable)
    # write_bcsd_results_task(postprocess_bcsd_ds, 'az://cmip6/results/{ADD_PARAMS}')
