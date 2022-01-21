import os

os.environ["PREFECT__FLOWS__CHECKPOINTING"] = "true"
from abc import abstractmethod

from dask_kubernetes import KubeCluster, make_pod_spec
from prefect.executors import DaskExecutor, Executor, LocalExecutor
from prefect.run_configs import KubernetesRun, LocalRun, RunConfig
from prefect.storage import Azure, Local, Storage

# TODO: Add new config that is hybrid or local compute, but has prefect storage access (ie. what I've been using to debug. Local is now non-write permissions.)


class BaseRuntime:
    """Base configuration class that defines abstract methods (storage, run_config and executor) for subclasses."""

    serializer = 'xarray.zarr'

    @property
    @abstractmethod
    def storage(self) -> Storage:  # pragma: no cover
        pass

    @property
    @abstractmethod
    def run_config(self) -> RunConfig:  # pragma: no cover
        pass

    @property
    @abstractmethod
    def executor(self) -> Executor:  # pragma: no cover
        pass

    @property
    def intermediate_cache_path(self) -> str:  # pragma: no cover
        return f'{self._storage_prefix}flow-outputs/intermediates'

    @property
    def results_cache_path(self) -> str:  # pragma: no cover
        return f'{self._storage_prefix}flow-outputs/results'


class CloudRuntime(BaseRuntime):
    def __init__(
        self,
        connection_string=None,
        storage_prefix='az://',
        agent="az-eu-west",
        extra_pip_packages="git+https://github.com/carbonplan/cmip6-downscaling.git git+https://github.com/pangeo-data/scikit-downscale.git",
        kubernetes_cpu=7,
        kubernetes_memory="16Gi",
        image="carbonplan/cmip6-downscaling-prefect:2022.01.05",
        pod_memory_limit="4Gi",
        pod_memory_request="4Gi",
        pod_threads_per_worker=2,
        pod_cpu_limit=2,
        pod_cpu_request=2,
        deploy_mode="remote",
        adapt_min=2,
        adapt_max=2,
        dask_distributed_worker_resources_taskslots="1",
    ):

        self._connection_string = connection_string or os.environ.get(
            "AZURE_STORAGE_CONNECTION_STRING", None
        )
        self._storage_prefix = storage_prefix
        self._agent = agent
        self._extra_pip_packages = extra_pip_packages
        self._kubernetes_cpu = kubernetes_cpu
        self._kubernetes_memory = kubernetes_memory
        self._image = image
        self._pod_memory_limit = pod_memory_limit
        self._pod_memory_request = pod_memory_request
        self._pod_threads_per_worker = pod_threads_per_worker
        self._pod_cpu_limit = pod_cpu_limit
        self._pod_cpu_request = pod_cpu_request
        self._deploy_mode = deploy_mode
        self._adapt_min = adapt_min
        self._adapt_max = adapt_max
        self._dask_distributed_worker_resources_taskslots = (
            dask_distributed_worker_resources_taskslots
        )

    def __repr__(self):
        return """CloudRuntime configuration for running on prefect-cloud.  storage is `Azure("prefect")`, run_config is `KubernetesRun() and executor is ` DaskExecutor(KubeCluster())`"""

    def _generate_env(self):
        env = {
            "AZURE_STORAGE_CONNECTION_STRING": self._connection_string,
            "EXTRA_PIP_PACKAGES": self._extra_pip_packages,
            "OMP_NUM_THREADS": "1",
            "MPI_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "DASK_DISTRIBUTED__WORKER__RESOURCES__TASKSLOTS": self._dask_distributed_worker_resources_taskslots,
        }
        return env

    @property
    def storage(self) -> Storage:
        return Azure("prefect")

    @property
    def run_config(self) -> RunConfig:
        kube_run_config = KubernetesRun(
            cpu_request=self._kubernetes_cpu,
            memory_request=self._kubernetes_memory,
            image=self._image,
            labels=[self._agent],
            env=self._generate_env(),
        )
        return kube_run_config

    @property
    def executor(self) -> Executor:

        pod_spec = make_pod_spec(
            image=self._image,
            memory_limit="4Gi",
            memory_request="4Gi",
            threads_per_worker=2,
            cpu_limit=2,
            cpu_request=2,
            env=self._generate_env(),
        )
        pod_spec.spec.containers[0].args.extend(['--resources', 'TASKSLOTS=1'])

        executor = DaskExecutor(
            cluster_class=lambda: KubeCluster(pod_spec, deploy_mode='remote'),
            adapt_kwargs={"minimum": 2, "maximum": 2},
        )

        return executor


class LocalRuntime(BaseRuntime):
    def __init__(self, storage_prefix: str = './', storage_kwargs: dict = None):
        self._storage_prefix = storage_prefix
        self._storage_kwargs = storage_kwargs or {}

    def __repr__(self):
        return "LocalRuntime configuration is for running on local machines. storage is `Local()`, run_config is `LocalRun() and executor is `LocalExecutor()` "

    @property
    def storage(self) -> Storage:
        return Local(**self._storage_kwargs)

    @property
    def run_config(self) -> RunConfig:
        return LocalRun(env=self._generate_env())

    @property
    def executor(self) -> Executor:
        return LocalExecutor()

    def _generate_env(self):
        env = {
            "OMP_NUM_THREADS": "1",
            "MPI_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        return env


class TestRuntime(LocalRuntime):
    def __init__(self, storage_prefix: str = '/tmp/', storage_kwargs: dict = None):
        self._storage_prefix = storage_prefix
        self._storage_kwargs = storage_kwargs or {}

    def __repr__(self):
        return "TestRuntime configuration is for running on CI machines. storage is `Local()`, run_config is `LocalRun() and executor is `LocalExecutor()` "


class PangeoRuntime(LocalRuntime):
    def __init__(self, storage_prefix: str = 'az://', storage_kwargs: dict = None):
        self._storage_prefix = storage_prefix
        self._storage_kwargs = storage_kwargs or {}

    def __repr__(self):
        return "PangeoRuntime configuration is for running on jupyter-hubs. storage is `Local()`, run_config is `LocalRun() and executor is `LocalExecutor()` "


def get_runtime(name=None, **kwargs):
    if name == 'test':
        runtime = TestRuntime(**kwargs)
    elif name == 'local':
        runtime = LocalRuntime(**kwargs)
    elif name == 'prefect-cloud':
        runtime = CloudRuntime(**kwargs)
    elif name == 'pangeo':
        runtime = PangeoRuntime(**kwargs)
    elif os.environ.get("CI") == "true":
        runtime = TestRuntime(**kwargs)
        print('TestRuntime selected from os.environ')
    elif os.environ.get("PREFECT__BACKEND") == "cloud":
        runtime = CloudRuntime(**kwargs)
        print('PrefectCloudRuntime selected from os.environ')
    elif 'JUPYTER_IMAGE' in os.environ:
        runtime = PangeoRuntime(**kwargs)
        print('PangeoRuntime selected from os.environ')
    else:
        ValueError(
            "Name not in ['test', 'local', 'prefect-cloud', 'pangeo'] and environment variable not found for: [CI, PREFECT__BACKEND, PANGEO__BACKEND or TEST]"
        )
    return runtime
