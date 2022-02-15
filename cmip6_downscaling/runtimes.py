import os
from abc import abstractmethod

from dask_kubernetes import KubeCluster, make_pod_spec
from prefect.executors import DaskExecutor, Executor, LocalDaskExecutor, LocalExecutor
from prefect.run_configs import KubernetesRun, LocalRun, RunConfig
from prefect.storage import Azure, Local, Storage

from . import config

_threadsafe_env_vars = {
    "OMP_NUM_THREADS": "1",
    "MPI_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


class BaseRuntime:
    """Base configuration class that defines abstract methods (storage, run_config and executor) for subclasses."""

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


class CloudRuntime(BaseRuntime):
    def __init__(
        self,
        runtime_name="cloud",
        storage_options=None,
        agent=None,
        extra_pip_packages=None,
        kubernetes_cpu=None,
        kubernetes_memory=None,
        image=None,
        pod_memory_limit=None,
        pod_memory_request=None,
        pod_threads_per_worker=None,
        pod_cpu_limit=None,
        pod_cpu_request=None,
        deploy_mode=None,
        adapt_min=None,
        adapt_max=None,
        dask_distributed_worker_resources_taskslots=None,
    ):
        self._runtime_name = runtime_name
        self._storage_options = storage_options is not None or config.get(
            "runtime.cloud.storage_options.container"
        )

        self._agent = agent is not None or config.get("runtime.cloud.agent")
        self._extra_pip_packages = extra_pip_packages is not None or config.get(
            "runtime.cloud.extra_pip_packages"
        )
        self._kubernetes_cpu = kubernetes_cpu is not None or config.get(
            "runtime.cloud.kubernetes_cpu"
        )
        self._kubernetes_memory = kubernetes_memory is not None or config.get(
            "runtime.cloud.kubernetes_memory"
        )
        self._image = image is not None or config.get("runtime.cloud.image")
        self._pod_memory_limit = pod_memory_limit is not None or config.get(
            "runtime.cloud.pod_memory_limit"
        )
        self._pod_memory_request = pod_memory_request is not None or config.get(
            "runtime.cloud.pod_memory_request"
        )
        self._pod_threads_per_worker = pod_threads_per_worker is not None or config.get(
            "runtime.cloud.pod_threads_per_worker"
        )
        self._pod_cpu_limit = pod_cpu_limit is not None or config.get("runtime.cloud.pod_cpu_limit")
        self._pod_cpu_request = pod_cpu_request is not None or config.get(
            "runtime.cloud.pod_cpu_request"
        )
        self._deploy_mode = deploy_mode is not None or config.get("runtime.cloud.deploy_mode")
        self._adapt_min = adapt_min is not None or config.get("runtime.cloud.adapt_min")
        self._adapt_max = adapt_max is not None or config.get("runtime.cloud.adapt_max")
        self._dask_distributed_worker_resources_taskslots = (
            dask_distributed_worker_resources_taskslots is not None
            or config.get("runtime.cloud.dask_distributed_worker_resources_taskslots")
        )

    def __repr__(self):
        return """CloudRuntime configuration for running on prefect-cloud.  storage is `Azure("prefect")`, run_config is `KubernetesRun() and executor is ` DaskExecutor(KubeCluster())`"""

    def _generate_env(self):
        env = {
            "EXTRA_PIP_PACKAGES": self._extra_pip_packages,
            "DASK_DISTRIBUTED__WORKER__RESOURCES__TASKSLOTS": self._dask_distributed_worker_resources_taskslots,
            "PREFECT__FLOWS__CHECKPOINTING": "true",
            **_threadsafe_env_vars,
        }

        if 'AZURE_STORAGE_CONNECTION_STRING' in os.environ:
            env['AZURE_STORAGE_CONNECTION_STRING'] = os.environ['AZURE_STORAGE_CONNECTION_STRING']

        return env

    @property
    def storage(self) -> Storage:
        return Azure(self._storage_options)

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
            memory_limit=self._pod_memory_limit,
            memory_request=self._pod_memory_request,
            threads_per_worker=self._pod_threads_per_worker,
            cpu_limit=self._pod_cpu_limit,
            cpu_request=self._pod_cpu_request,
            env=self._generate_env(),
        )
        pod_spec.spec.containers[0].args.extend(["--resources", "TASKSLOTS=1"])

        executor = DaskExecutor(
            cluster_class=lambda: KubeCluster(pod_spec, deploy_mode=self._deploy_mode),
            adapt_kwargs={"minimum": self._adapt_min, "maximum": self._adapt_max},
        )

        return executor


class LocalRuntime(BaseRuntime):
    def __init__(self, storage_options: dict = None):
        self._storage_options = storage_options is not None or config.get(
            "runtime.local.storage_options"
        )

    def __repr__(self):
        return "LocalRuntime configuration is for running on local machines. storage is `Local()`, run_config is `LocalRun() and executor is `LocalExecutor()` "

    @property
    def storage(self) -> Storage:
        return Local(**self._storage_options)

    @property
    def run_config(self) -> RunConfig:
        return LocalRun(env=self._generate_env())

    @property
    def executor(self) -> Executor:
        return LocalDaskExecutor(scheduler="processes")

    def _generate_env(self):
        return _threadsafe_env_vars


class CIRuntime(LocalRuntime):
    def __init__(self, storage_options: dict = None):
        self._storage_options = storage_options is not None or config.get(
            "runtime.test.storage_options"
        )

    @property
    def executor(self) -> Executor:
        return LocalExecutor()

    def __repr__(self):
        return "CIRuntime configuration is for running on CI machines. storage is `Local()`, run_config is `LocalRun() and executor is `LocalExecutor()` "


class PangeoRuntime(LocalRuntime):
    def __init__(
        self, storage_options: dict = None, n_workers: int = None, threads_per_worker: int = None
    ):
        self._storage_options = storage_options is not None or config.get(
            "runtime.pangeo.storage_options"
        )
        self._n_workers = (
            n_workers if n_workers is not None else config.get("runtime.pangeo.n_workers")
        )
        self._threads_per_worker = (
            n_workers if n_workers is not None else config.get("runtime.pangeo.threads_per_worker")
        )

    def __repr__(self):
        return "PangeoRuntime configuration is for running on jupyter-hubs. storage is `Local()`, run_config is `LocalRun() and executor is `LocalExecutor()` "

    @property
    def storage(self) -> Storage:
        return Local(**self._storage_options)

    @property
    def run_config(self) -> RunConfig:
        return LocalRun(env=self._generate_env())

    @property
    def executor(self) -> Executor:
        return DaskExecutor(
            cluster_kwargs={
                'resources': {'TASKSLOTS': 1},
                'n_workers': self._n_workers,
                'threads_per_worker': self._threads_per_worker,
            }
        )

    def _generate_env(self):
        return _threadsafe_env_vars


def get_runtime(**kwargs):
    if config.get("run_options.runtime") == "ci":
        runtime = CIRuntime(**kwargs)
    elif config.get("run_options.runtime") == "local":
        runtime = LocalRuntime(**kwargs)
    elif config.get("run_options.runtime") == "cloud":
        runtime = CloudRuntime(**kwargs)
    elif config.get("run_options.runtime") == "pangeo":
        runtime = PangeoRuntime(**kwargs)
    elif os.environ.get("CI") == "true":
        runtime = CIRuntime(**kwargs)
        print("CIRuntime selected from os.environ")
    elif os.environ.get("PREFECT__BACKEND") == "cloud":
        runtime = CloudRuntime(**kwargs)
        print("PrefectCloudRuntime selected from os.environ")
    elif "JUPYTER_IMAGE" in os.environ:
        runtime = PangeoRuntime(**kwargs)
        print("PangeoRuntime selected from os.environ")
    else:
        raise ValueError(
            "Name not in ['ci', 'local', 'prefect-cloud', 'pangeo'] and environment variable not found for: [CI, PREFECT__BACKEND, PANGEO__BACKEND or TEST]"
        )
    return runtime
