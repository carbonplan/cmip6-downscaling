import os

os.environ['PREFECT__FLOWS__CHECKPOINTING'] = 'true'
from dask_kubernetes import KubeCluster, make_pod_spec
from prefect.executors import DaskExecutor
from prefect.run_configs import KubernetesRun
from prefect.storage import Azure
from xpersist import CacheStore

# Azure --------------------------------------------------------------------
connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
CONNECTION_STRING = connection_string

storage = Azure("prefect")
intermediate_cache_path = 'az://flow-outputs/intermediates'
intermediate_cache_store = CacheStore(intermediate_cache_path)
results_cache_path = 'az://flow-outputs/results'
results_cache_store = CacheStore(results_cache_path)
serializer = 'xarray.zarr'

# Prefect --------------------------------------------------------------------
# prefect agent ID
agent = ["az-eu-west"]

# Docker image
image = "carbonplan/cmip6-downscaling-prefect:2021.12.06"

# Kubernetes config
kubernetes_cpu = 4
kubernetes_memory = "8Gi"

# Dask executor config
dask_executor_memory_limit = "16Gi"
dask_executor_memory_request = "16Gi"
dask_executor_threads_per_worker = 2
dask_executor_cpu_limit = 2
dask_executor_cpu_request = 2
dask_executor_adapt_min = 4
dask_executor_adapt_max = 20

# pod config
pod_memory_limit = "7Gi"
pod_memory_request = "7Gi"
pod_threads_per_worker = 2
pod_cpu_limit = 2
pod_cpu_request = 2


OMP_NUM_THREADS = '1'
MPI_NUM_THREADS = '1'
MKL_NUM_THREADS = '1'
OPENBLAS_NUM_THREADS = '1'
DASK_DISTRIBUTED__WORKER__RESOURCES__TASKSLOTS = '1'


extra_pip_packages = "git+https://github.com/carbonplan/cmip6-downscaling.git@esmf_threading git+https://github.com/pangeo-data/scikit-downscale.git git+https://github.com/NCAR/xpersist.git"
config = {
    "AZURE_STORAGE_CONNECTION_STRING": connection_string,
    "EXTRA_PIP_PACKAGES": extra_pip_packages,
    'OMP_NUM_THREADS': OMP_NUM_THREADS,
    'MPI_NUM_THREADS': MPI_NUM_THREADS,
    'MKL_NUM_THREADS': MKL_NUM_THREADS,
    'OPENBLAS_NUM_THREADS': OPENBLAS_NUM_THREADS,
    'DASK_DISTRIBUTED__WORKER__RESOURCES__TASKSLOTS': DASK_DISTRIBUTED__WORKER__RESOURCES__TASKSLOTS,
}

kubernetes_run_config = KubernetesRun(
    cpu_request=kubernetes_cpu,
    memory_request=kubernetes_memory,
    image=image,
    labels=agent,
    env=config,
)

pod_spec = make_pod_spec(
    image=image,
    memory_limit=pod_memory_limit,
    memory_request=pod_memory_request,
    threads_per_worker=pod_threads_per_worker,
    cpu_limit=pod_cpu_limit,
    cpu_request=pod_cpu_request,
    env=config,
)
pod_spec.spec.containers[0].args.extend(['--resources', 'TASKSLOTS=1'])

dask_executor = DaskExecutor(
    cluster_class=lambda: KubeCluster(pod_spec, deploy_mode='remote'),
    adapt_kwargs={"minimum": 1, "maximum": 2},
)
