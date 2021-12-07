# Imports -----------------------------------------------------------
import os

from dask_kubernetes import KubeCluster, make_pod_spec
from prefect import Flow, Parameter, task
from prefect.executors import DaskExecutor
from prefect.run_configs import KubernetesRun
from prefect.storage import Azure

from cmip6_downscaling.methods.bcsd import (
    fit_and_predict,
    make_flow_paths,
    postprocess_bcsd,
    prep_bcsd_inputs,
    preprocess_bcsd,
)

# Config -----------------------------------------------------------

connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

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

# Transform Functions into Tasks -----------------------------------------------------------

make_flow_paths_task = task(make_flow_paths, log_stdout=True, nout=4)

preprocess_bcsd_task = task(preprocess_bcsd, log_stdout=True, nout=2)

prep_bcsd_inputs_task = task(prep_bcsd_inputs, log_stdout=True, nout=3)

fit_and_predict_task = task(fit_and_predict, log_stdout=True)

postprocess_bcsd_task = task(postprocess_bcsd, log_stdout=True)


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
    print(type(VARIABLE), type(TRAIN_PERIOD_START), type(TRAIN_PERIOD_END))


# Main Flow -----------------------------------------------------------

# with Flow(name="bcsd-testing") as flow:
# with Flow(name="bcsd-testing", storage=storage, run_config=run_config) as flow:
with Flow(name="bcsd-testing", storage=storage, run_config=run_config, executor=executor) as flow:
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

    coarse_obs_path, spatial_anomalies_path = preprocess_bcsd_task(
        gcm=gcm,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        variable=variable,
        coarse_obs_path=coarse_obs_path,
        spatial_anomalies_path=spatial_anomalies_path,
        connection_string=connection_string,
        rerun=True,
    )

    (y_rechunked_path, X_train_rechunked_path, X_predict_rechunked_path,) = prep_bcsd_inputs_task(
        coarse_obs_path=coarse_obs_path,
        gcm=gcm,
        scenario=scenario,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        predict_period_start=predict_period_start,
        predict_period_end=predict_period_end,
        variable=variable,
    )

    bias_corrected_path = fit_and_predict_task(
        X_train_rechunked_path,
        y_rechunked_path,
        X_predict_rechunked_path,
        bias_corrected_path,
    )

    out_path = postprocess_bcsd_task(
        bias_corrected_path,
        spatial_anomalies_path,
        final_out_path,
        variable,
        connection_string,
    )
