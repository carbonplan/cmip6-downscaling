# from cmip6_downscaling.workflows.utils import rechunk_dataset
import os
import random
import string

import fsspec
import intake
import numpy as np
import xarray as xr
import xesmf as xe
import zarr
from dask.distributed import Client, LocalCluster
from dask_kubernetes import KubeCluster, make_pod_spec
from prefect import Flow, Parameter, task
from prefect.executors import DaskExecutor
from prefect.run_configs import KubernetesRun
from prefect.storage import Azure
from rechunker import api
from skdownscale.pointwise_models import BcAbsolute, PointWiseDownscaler
from xarray_schema import DataArraySchema, DatasetSchema

from cmip6_downscaling.data.cmip import convert_to_360, gcm_munge, load_cmip
from cmip6_downscaling.data.observations import get_coarse_obs, get_spatial_anomolies, load_obs
from cmip6_downscaling.methods.bcsd import (
    fit_and_predict,
    postprocess_bcsd,
    prep_bcsd_inputs,
    preprocess_bcsd,
)
from cmip6_downscaling.workflows.utils import (
    calc_auspicious_chunks_dict,
    delete_chunks_encoding,
    rechunk_zarr_array,
)

connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

account_key = os.environ.get("account_key")
run_hyperparameters = {
    "FLOW_NAME": "BCSD_testing",  # want to populate with unique string name?
    # "INTERPOLATION": "bilinear",
    "GCMS": ["MIROC6"],
    # "SCENARIOS": ["ssp370"],
    "TRAIN_PERIOD_START": "1990",
    "TRAIN_PERIOD_END": "1990",
    "PREDICT_PERIOD_START": "2090",
    "PREDICT_PERIOD_END": "2090",
    "VARIABLE": "tasmax",
    "OBS": "ERA5",
}
flow_name = run_hyperparameters.pop("FLOW_NAME")  # pop it out because if you leave it in the dict
# but don't call it as a parameter it'll complain

preprocess_bcsd_task = task(preprocess_bcsd, log_stdout=True)
# TODO: make path/store templates
# TODO: think about caching with nout>1
prep_bcsd_inputs_task = task(prep_bcsd_inputs, log_stdout=True, nout=3)

fit_and_predict_task = task(fit_and_predict, log_stdout=True)

postprocess_bcsd_task = task(postprocess_bcsd, log_stdout=True)

with Flow(name="bcsd_flow", storage=storage, run_config=run_config, executor=executor) as flow:

    obs = run_hyperparameters["OBS"]
    gcm = run_hyperparameters["GCMS"]
    train_period_start = run_hyperparameters["TRAIN_PERIOD_START"]
    train_period_end = run_hyperparameters["TRAIN_PERIOD_END"]
    predict_period_start = run_hyperparameters["PREDICT_PERIOD_START"]
    predict_period_end = run_hyperparameters["PREDICT_PERIOD_END"]
    domain = test_specs["domain"]
    variable = run_hyperparameters["VARIABLE"]

    coarse_obs = preprocess_bcsd_task(
        gcm,
        obs_id=obs,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        variable=variable,
        out_bucket="cmip6",
        domain=domain,
        rerun=True,  # can remove this once we have caching working
    )

    y_rechunked_path, X_train_rechunked_path, X_predict_rechunked_path = prep_bcsd_inputs(
        coarse_obs_path,
        gcm,
        obs_id=obs,
        train_period_start=train_period_start,
        train_period_end=train_period_end,
        predict_period_start=predict_period_start,
        predict_period_end=predict_period_end,
        variable=variable,
    )

    bias_corrected_path = fit_and_predict(X_train_rechunked_path, 
                                      y_rechunked_path, 
                                      X_predict_rechunked_path, 
                                      bias_corrected_path)
                                      
    out_path = postprocess_bcsd(bias_corrected_path, spatial_anomalies_path, final_out_path, variable, connection_string)
flow.run(parameters=run_hyperparameters)
