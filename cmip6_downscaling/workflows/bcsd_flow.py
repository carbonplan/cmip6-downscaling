# Imports -----------------------------------------------------------
import os

os.environ['PREFECT__FLOWS__CHECKPOINTING'] = 'true'
from dask_kubernetes import KubeCluster, make_pod_spec
from funnel import CacheStore
from funnel.prefect.result import FunnelResult
from prefect import Flow, Parameter, task
from prefect.executors import DaskExecutor
from prefect.run_configs import KubernetesRun
from prefect.storage import Azure

from cmip6_downscaling.methods.bcsd import (  # preprocess_bcsd,; get_transformed_data,; prep_bcsd_inputs,
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

# Config -----------------------------------------------------------


storage = Azure("prefect")
image = "carbonplan/cmip6-downscaling-prefect:2021.12.06"
env_config = {
    "AZURE_STORAGE_CONNECTION_STRING": os.environ["AZURE_STORAGE_CONNECTION_STRING"],
    "EXTRA_PIP_PACKAGES": "git+https://github.com/carbonplan/cmip6-downscaling.git@param_json git+https://github.com/orianac/scikit-downscale.git@bcsd-workflow",
}
run_config = KubernetesRun(
    cpu_request=7,
    memory_request="16Gi",
    image=image,
    labels=["az-eu-west"],
    env=env_config,
)

executor = DaskExecutor(
    cluster_class=lambda: KubeCluster(
        make_pod_spec(
            image=image,
            memory_limit="16Gi",
            memory_request="16Gi",
            threads_per_worker=2,
            cpu_limit=2,
            cpu_request=2,
            env=env_config,
        )
    ),
    adapt_kwargs={"minimum": 4, "maximum": 20},
)

cache_store = CacheStore('az://flow-outputs/bcsd')
serializer = 'xarray.zarr'
connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")


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
    target='post-process',
)
write_bcsd_results_task = task(write_bcsd_results, log_stdout=True)


@task(log_stdout=True)
def show_params(
    flow_name,
    GCM,
    SCENARIO,
    TRAIN_PERIOD_START,
    TRAIN_PERIOD_END,
    PREDICT_PERIOD_START,
    PREDICT_PERIOD_END,
    VARIABLE,
):
    pass


# Main Flow -----------------------------------------------------------

# with Flow(name="bcsd-testing", storage=storage, run_config=run_config) as flow:
# with Flow(name="bcsd-testing", storage=storage, run_config=run_config, executor=executor) as flow:
with Flow(name="bcsd-testing") as flow:

    flow_name = Parameter("FLOW_NAME")
    gcm = Parameter("GCM")
    scenario = Parameter("SCENARIO")
    train_period_start = Parameter("TRAIN_PERIOD_START")
    train_period_end = Parameter("TRAIN_PERIOD_END")
    predict_period_start = Parameter("PREDICT_PERIOD_START")
    predict_period_end = Parameter("PREDICT_PERIOD_END")
    variable = Parameter("VARIABLE")

    show_params(
        flow_name,
        gcm,
        scenario,
        train_period_start,
        train_period_end,
        predict_period_start,
        predict_period_end,
        variable,
    )

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
    postprocess_bcsd_ds = postprocess_bcsd_task(bias_corrected_ds, spatial_anomalies_ds, variable)
    write_bcsd_results_task(postprocess_bcsd_ds, 'az://cmip6/results/{ADD_PARAMS}')

    # bias_corrected_path = 'az://cmip6/intermediates/bc_ssp370_MIROC6_1990_1995_tasmax.zarr'
    # spatial_anomalies_path = 'az://cmip6/intermediates/anomalies_MIROC6_1990_1995_tasmax.zarr'
    # final_out_path = 'az://cmip6/results/bcsd_ssp370_MIROC6_2090_2095_tasmax.zarr'
