import os

from dask_kubernetes import KubeCluster, make_pod_spec
from prefect import Flow, task
from prefect.executors import DaskExecutor
from prefect.run_configs import KubernetesRun
from prefect.storage import Azure

from cmip6_downscaling.methods.bcsd import (
    fit_and_predict,
    postprocess_bcsd,
    prep_bcsd_inputs,
    preprocess_bcsd,
)
from cmip6_downscaling.workflows.utils import make_flow_paths

connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

run_hyperparameters = {
    "FLOW_NAME": "BCSD_testing",
    "GCM": "MIROC6",
    "SCENARIO": "ssp370",
    "TRAIN_PERIOD_START": "1990",
    "TRAIN_PERIOD_END": "1990",
    "PREDICT_PERIOD_START": "2090",
    "PREDICT_PERIOD_END": "2090",
    "VARIABLE": "tasmax",
    "OBS": "ERA5",
}

storage = Azure("prefect")
image = "carbonplan/cmip6-downscaling-prefect:latest"
run_config = KubernetesRun(cpu_request=7, memory_request="16Gi", image=image, labels=["az-eu-west"])

executor = DaskExecutor(
    cluster_class=lambda: KubeCluster(
        make_pod_spec(
            image=image,
            memory_limit="16Gi",
            memory_request="16Gi",
            threads_per_worker=2,
            cpu_limit=2,
            cpu_request=2,
            env={
                "AZURE_STORAGE_CONNECTION_STRING": os.environ["AZURE_STORAGE_CONNECTION_STRING"],
            },
        )
    ),
    adapt_kwargs={"minimum": 4, "maximum": 20},
)


flow_name = run_hyperparameters.pop("FLOW_NAME")  # pop it out because if you leave it in the dict
# but don't call it as a parameter it'll complain

make_flow_paths_task = task(make_flow_paths, log_stdout=True, nout=4)

preprocess_bcsd_task = task(preprocess_bcsd, log_stdout=True, nout=2)

prep_bcsd_inputs_task = task(prep_bcsd_inputs, log_stdout=True, nout=3)

fit_and_predict_task = task(fit_and_predict, log_stdout=True)

postprocess_bcsd_task = task(postprocess_bcsd, log_stdout=True)

with Flow(name="bcsd-testing", storage=storage, run_config=run_config, executor=executor) as flow:

    obs = run_hyperparameters["OBS"]
    gcm = run_hyperparameters["GCM"]
    scenario = run_hyperparameters["SCENARIO"]
    train_period_start = run_hyperparameters["TRAIN_PERIOD_START"]
    train_period_end = run_hyperparameters["TRAIN_PERIOD_END"]
    predict_period_start = run_hyperparameters["PREDICT_PERIOD_START"]
    predict_period_end = run_hyperparameters["PREDICT_PERIOD_END"]
    variable = run_hyperparameters["VARIABLE"]

    coarse_obs_path, spatial_anomalies_path, bias_corrected_path, final_out_path = make_flow_paths(
        **run_hyperparameters
    )

    coarse_obs_path, spatial_anomalies_path = preprocess_bcsd(
        gcm=gcm,
        obs_id=obs,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        variable=variable,
        coarse_obs_path=coarse_obs_path,
        spatial_anomalies_path=spatial_anomalies_path,
        connection_string=connection_string,
        rerun=True,
    )

    y_rechunked_path, X_train_rechunked_path, X_predict_rechunked_path = prep_bcsd_inputs(
        coarse_obs_path,
        gcm,
        scenario,
        obs_id=obs,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        predict_period_start=predict_period_start,
        predict_period_end=predict_period_end,
        variable=variable,
    )

    bias_corrected_path = fit_and_predict(
        X_train_rechunked_path, y_rechunked_path, X_predict_rechunked_path, bias_corrected_path
    )

    out_path = postprocess_bcsd(
        bias_corrected_path, spatial_anomalies_path, final_out_path, variable, connection_string
    )
flow.run(parameters=run_hyperparameters)
