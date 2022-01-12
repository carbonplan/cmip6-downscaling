import os

os.environ['PREFECT__FLOWS__CHECKPOINTING'] = 'true'
from dask_kubernetes import KubeCluster, make_pod_spec
from prefect.executors import DaskExecutor
from prefect.run_configs import KubernetesRun
from prefect.storage import Azure


# Azure --------------------------------------------------------------------
def return_azure_config():
    azure_config_dict = {
        'connection_string': os.environ.get("AZURE_STORAGE_CONNECTION_STRING"),
        'storage': Azure("prefect"),
        'intermediate_cache_path': 'az://flow-outputs/intermediate',
        'results_cache_path': 'az://flow-outputs/results',
        'serializer': 'xarray.zarr',
    }
    return azure_config_dict


# Prefect --------------------------------------------------------------------
def return_prefect_config():
    #  prefect agent ID
    prefect_config_dict = {'agent': ["az-eu-west"]}
    return prefect_config_dict


# Docker --------------------------------------------------------------------
def return_docker_config():
    docker_config_dict = {'image': "carbonplan/cmip6-downscaling-prefect:2021.12.06"}
    return docker_config_dict


# Kubernetes  --------------------------------------------------------------------
def return_kubernetes_config():
    kubernetes_config_dict = {'kubernetes_cpu': 7, 'kubernetes_memory': "16Gi"}
    return kubernetes_config_dict


# Dask Executor  --------------------------------------------------------------------
def return_dask_config():
    dask_config_dict = {
        "dask_executor_memory_limit": "16Gi",
        "dask_executor_memory_request": "16Gi",
        "dask_executor_threads_per_worker": 2,
        "dask_executor_cpu_limit": 2,
        "dask_executor_cpu_request": 2,
        "dask_executor_adapt_min": 4,
        "dask_executor_adapt_max": 20,
    }
    return dask_config_dict


def return_extra_pip_packages():
    extra_pip_packages = {
        "EXTRA_PIP_PACKAGES": "git+https://github.com/carbonplan/cmip6-downscaling.git git+https://github.com/pangeo-data/scikit-downscale.git git+https://github.com/NCAR/xpersist.git"
    }
    return extra_pip_packages


def return_env_config():
    env_config = {
        "AZURE_STORAGE_CONNECTION_STRING": return_azure_config()["connection_string"],
        "EXTRA_PIP_PACKAGES": return_extra_pip_packages()["EXTRA_PIP_PACKAGES"],
        'OMP_NUM_THREADS': '1',
        'MPI_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'OPENBLAS_NUM_THREADS': '1',
    }
    return env_config


def return_kubernetes_run_config():
    kubernetes_run_config = KubernetesRun(
        cpu_request=return_kubernetes_config()["kubernetes_cpu"],
        memory_request=return_kubernetes_config()["kubernetes_memory"],
        image=return_docker_config()["image"],
        labels=return_prefect_config()["agent"],
        env=return_env_config()["env_config"],
    )
    return kubernetes_run_config


def return_dask_executor():

    dask_executor = DaskExecutor(
        cluster_class=lambda: KubeCluster(
            make_pod_spec(
                image=return_docker_config()["image"],
                memory_limit=return_dask_config()["dask_executor_memory_limit"],
                memory_request=return_dask_config()["dask_executor_memory_request"],
                threads_per_worker=return_dask_config()["dask_executor_threads_per_worker"],
                cpu_limit=return_dask_config()["dask_executor_cpu_limit]"],
                cpu_request=return_dask_config()["dask_executor_cpu_request"],
                env=return_env_config()["env_config"],
            )
        ),
        adapt_kwargs={
            "minimum": return_dask_config()["dask_executor_adapt_min"],
            "maximum": return_dask_config()["dask_executor_adapt_max"],
        },
    )
    return dask_executor
