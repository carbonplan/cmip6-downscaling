import os

os.environ['PREFECT__FLOWS__CHECKPOINTING'] = 'true'
from dask_kubernetes import KubeCluster, make_pod_spec
from funnel import CacheStore
from prefect.executors import DaskExecutor
from prefect.run_configs import KubernetesRun
from prefect.storage import Azure

# Azure --------------------------------------------------------------------
CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
storage = Azure("prefect")
cache_path = 'az://flow-outputs/intermediate'
cache_store = CacheStore(cache_path)
serializer = 'xarray.zarr'

# Prefect --------------------------------------------------------------------
# prefect agent ID
agent = ["az-eu-west"]

# Docker image
image = "carbonplan/cmip6-downscaling-prefect:2021.12.06"

# Kubernetes config
kubernetes_cpu = 7
kubernetes_memory = "16Gi"

# Dask executor config
dask_executor_memory_limit = "16Gi"
dask_executor_memory_request = "16Gi"
dask_executor_threads_per_worker = 2
dask_executor_cpu_limit = 2
dask_executor_cpu_request = 2
dask_executor_adapt_min = 4
dask_executor_adapt_max = 20

extra_pip_packages = {
    "EXTRA_PIP_PACKAGES": "git+https://github.com/carbonplan/cmip6-downscaling.git@param_json git+https://github.com/orianac/scikit-downscale.git@bcsd-workflow"
}
env_config = {
    "AZURE_STORAGE_CONNECTION_STRING": CONNECTION_STRING,
    "EXTRA_PIP_PACKAGES": extra_pip_packages,
    'OMP_NUM_THREADS': '1',
    'MPI_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
}


kubernetes_run_config = KubernetesRun(
    cpu_request=kubernetes_cpu,
    memory_request=kubernetes_memory,
    image=image,
    labels=agent,
    env=env_config,
)

dask_executor = DaskExecutor(
    cluster_class=lambda: KubeCluster(
        make_pod_spec(
            image=image,
            memory_limit=dask_executor_memory_limit,
            memory_request=dask_executor_memory_request,
            threads_per_worker=dask_executor_threads_per_worker,
            cpu_limit=dask_executor_cpu_limit,
            cpu_request=dask_executor_cpu_request,
            env=env_config,
        )
    ),
    adapt_kwargs={"minimum": dask_executor_adapt_min, "maximum": dask_executor_adapt_max},
)
