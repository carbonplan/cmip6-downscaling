import os

os.environ["PREFECT__FLOWS__CHECKPOINTING"] = "true"
from abc import abstractmethod
from typing import Any

from dask_kubernetes import KubeCluster, make_pod_spec
from prefect.executors import DaskExecutor, LocalDaskExecutor, LocalExecutor
from prefect.run_configs import KubernetesRun, LocalRun
from prefect.storage import Azure, Local

# todo:


# prefect-cloud config has @property decs to repeat for other config subclasses.
# connection_string should only live in prefect-cloud and pangeo and hybrid. The other ones we won't write data, so we don't need write permissions.
# Add new config that is hybrid or local compute, but has prefect storage access (ie. what I've been using to debug. Local is now non-write permissions.)


class BaseConfig:
    """Base configuration class that defines abstract methods (storage, run_config and executor) for subclasses.

    Attributes
    ----------

    Methods
    -------
    storage()
        abstract definition for storage method
    run_config()
        abstract definition for run_config method
    executor()
        abstract definition for executor method
    """

    @property
    @abstractmethod
    def storage(self) -> Any:  # pragma: no cover
        pass

    @property
    @abstractmethod
    def run_config(self) -> Any:  # pragma: no cover
        pass

    @property
    @abstractmethod
    def executor(self) -> Any:  # pragma: no cover
        pass

class CloudConfig(BaseConfig):
    def __init__(self, **kwargs):
        self.connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")

        self.intermediate_cache_path = "az://flow-outputs/intermediates"
        self.results_cache_path = "az://flow-outputs/results"
        self.agent = ["az-eu-west"]
        self.extra_pip_packages = "git+https://github.com/carbonplan/cmip6-downscaling.git@esmf_threading git+https://github.com/pangeo-data/scikit-downscale.git"
        self.serializer = "xarray.zarr"
        self.kubernetes_cpu = 7
        self.kubernetes_memory = "16Gi"
        self.image = "carbonplan/cmip6-downscaling-prefect:2022.01.05"
        self.pod_memory_limit = "4Gi"
        self.pod_memory_request = "4Gi"
        self.pod_threads_per_worker = 2
        self.pod_cpu_limit = 2
        self.pod_cpu_request = 2
        self.deploy_mode = "remote"
        self.adapt_min = 2
        self.adapt_max = 2
        self.dask_distributed_worker_resources_taskslots = "1"

    def generate_env(self):
        env = {
            "AZURE_STORAGE_CONNECTION_STRING": self.connection_string,
            "EXTRA_PIP_PACKAGES": self.extra_pip_packages,
            "OMP_NUM_THREADS": "1",
            "MPI_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "DASK_DISTRIBUTED__WORKER__RESOURCES__TASKSLOTS": self.dask_distributed_worker_resources_taskslots,
        }
        return env

    @property
    def storage(self) -> Any:  # pragma: no cover
        return Azure("prefect")

    @property
    def run_config(self) -> Any:  # pragma: no cover
        kube_run_config = KubernetesRun(
            cpu_request=self.kubernetes_cpu,
            memory_request=self.kubernetes_memory,
            image=self.image,
            labels=self.agent,
            env=self.generate_env(),
        )
        return kube_run_config

    @property
    def executor(self) -> Any:  # pragma: no cover

        pod_spec = make_pod_spec(
            image=self.image,
            memory_limit="4Gi",
            memory_request="4Gi",
            threads_per_worker=2,
            cpu_limit=2,
            cpu_request=2,
            env=self.generate_env(),
        )
        pod_spec.spec.containers[0].args.extend(['--resources', 'TASKSLOTS=1'])

        executor = DaskExecutor(
            cluster_class=lambda: KubeCluster(pod_spec, deploy_mode='remote'),
            adapt_kwargs={"minimum": 2, "maximum": 2},
        )


        return executor


class LocalConfig(BaseConfig):
    def __init__(self, **kwargs):
        pass

    @property
    def storage(self) -> Any:  # pragma: no cover
        return Local()

    @property
    def run_config(self) -> Any:  # pragma: no cover
        return LocalRun()

    @property
    def executor(self) -> Any:  # pragma: no cover
        return LocalExecutor()


class TestConfig(LocalConfig):
    def __init__(self, **kwargs):
        pass

    @property
    def storage(self) -> Any:  # pragma: no cover
        return Local()

    @property
    def run_config(self) -> Any:  # pragma: no cover
        return LocalRun()

    @property
    def executor(self) -> Any:  # pragma: no cover
        return LocalExecutor()


class PangeoConfig(BaseConfig):
    def __init__(self, **kwargs):
        self.connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        self.intermediate_cache_path = "az://flow-outputs/intermediates"
        self.results_cache_path = "az://flow-outputs/results"

    @property
    def storage(self) -> Any:  # pragma: no cover
        return Local()

    @property
    def run_config(self) -> Any:  # pragma: no cover
        config = LocalRun()
        return config

    @property
    def executor(self) -> Any:  # pragma: no cover
        executor =  LocalExecutor()
        # executor = LocalDaskExecutor(scheduler="threads")
        return executor


def get_config(name=None, **kwargs):
    if name == 'test':
        config = TestConfig(**kwargs)
    elif name == 'local':
        config = LocalConfig(**kwargs)
    elif name == 'prefect-cloud':
        config = CloudConfig(**kwargs)
    elif name == 'pangeo':
        config = PangeoConfig(**kwargs)
    elif os.environ.get("CI") == "true":
        config = TestConfig(**kwargs)
        print('TestConfig selected from os.environ')
    elif os.environ.get("PREFECT__BACKEND") == "cloud":
        config = CloudConfig(**kwargs)
        print('PrefectCloudConfig selected from os.environ')
    elif 'JUPYTER_IMAGE' in os.environ:
        config = PangeoConfig(**kwargs)
        print('PangeoConfig selected from os.environ')
    else:
        ValueError(
            "Name not in ['test', 'local', 'prefect-cloud', 'pangeo'] and environment variable not found for: [CI, PREFECT__BACKEND, PANGEO__BACKEND or TEST]"
        )

    return config
