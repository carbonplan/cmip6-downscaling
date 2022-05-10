from __future__ import annotations

import os
from abc import abstractmethod
from functools import cached_property

import dask
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
    """
    Base runtime configuration class

    Subclasses must define methods (storage, run_config and executor)
    """

    def __init__(self):
        pass

    @cached_property
    @abstractmethod
    def storage(self) -> Storage:  # pragma: no cover
        pass

    @cached_property
    @abstractmethod
    def run_config(self) -> RunConfig:  # pragma: no cover
        pass

    @cached_property
    @abstractmethod
    def executor(self) -> Executor:  # pragma: no cover
        pass

    def __repr__(self):
        return (
            f"{type(self)}\n"
            f"  Storage    : {type(self.storage)}\n"
            f"  Run Config : {type(self.run_config)}\n"
            f"  Executor   : {type(self.executor)}\n"
        )


class CloudRuntime(BaseRuntime):
    def _generate_env(self):
        env = {
            "EXTRA_PIP_PACKAGES": config.get("runtime.cloud.extra_pip_packages"),
            "PREFECT__CLOUD__HEARTBEAT_MODE": "thread",
            "PREFECT__FLOWS__CHECKPOINTING": "true",
            **_threadsafe_env_vars,
        }

        if 'AZURE_STORAGE_CONNECTION_STRING' in os.environ:
            env['AZURE_STORAGE_CONNECTION_STRING'] = os.environ['AZURE_STORAGE_CONNECTION_STRING']

        return env

    @cached_property
    def storage(self) -> Storage:
        return Azure(
            container=config.get("runtime.cloud.storage_options.container"),
        )

    @cached_property
    def run_config(self) -> RunConfig:
        return KubernetesRun(
            cpu_request=config.get("runtime.cloud.kubernetes_cpu"),
            memory_request=config.get("runtime.cloud.kubernetes_memory"),
            image=config.get("runtime.cloud.image"),
            labels=[config.get("runtime.cloud.agent")],
            env=self._generate_env(),
        )

    @cached_property
    def executor(self) -> Executor:

        pod_spec = make_pod_spec(
            image=config.get("runtime.cloud.image"),
            memory_limit=config.get("runtime.cloud.pod_memory_limit"),
            memory_request=config.get("runtime.cloud.pod_memory_request"),
            threads_per_worker=config.get("runtime.cloud.pod_threads_per_worker"),
            cpu_limit=config.get("runtime.cloud.pod_cpu_limit"),
            cpu_request=config.get("runtime.cloud.pod_cpu_request"),
            env=self._generate_env(),
        )

        executor = DaskExecutor(
            cluster_class=lambda: KubeCluster(
                pod_spec, deploy_mode=config.get("runtime.cloud.deploy_mode")
            ),
            adapt_kwargs={
                "minimum": config.get("runtime.cloud.adapt_min"),
                "maximum": config.get("runtime.cloud.adapt_max"),
            },
        )

        return executor


class LocalRuntime(BaseRuntime):
    @cached_property
    def storage(self) -> Storage:
        return Local(**config.get("runtime.local.storage_options"))

    @cached_property
    def run_config(self) -> RunConfig:
        return LocalRun(env=self._generate_env())

    @cached_property
    def executor(self) -> Executor:
        return LocalDaskExecutor(scheduler="processes")

    def _generate_env(self):
        return _threadsafe_env_vars


class CIRuntime(LocalRuntime):
    @cached_property
    def storage(self) -> Storage:
        return Local(**config.get("runtime.test.storage_options"))

    @cached_property
    def executor(self) -> Executor:
        return LocalExecutor()


class PangeoRuntime(LocalRuntime):
    def __init__(self):
        dask.config.set({'temporary_directory': '/tmp/dask'})

    @cached_property
    def storage(self) -> Storage:
        return Local(**config.get("runtime.pangeo.storage_options"))

    @cached_property
    def run_config(self) -> RunConfig:
        return LocalRun(env=self._generate_env())

    @cached_property
    def executor(self) -> Executor:
        return DaskExecutor(
            cluster_kwargs={
                'resources': {'taskslots': 1},
                'n_workers': config.get("runtime.pangeo.n_workers"),
                'threads_per_worker': config.get("runtime.pangeo.threads_per_worker"),
            }
        )

    def _generate_env(self):
        return _threadsafe_env_vars


class GatewayRuntime(PangeoRuntime):
    def __init__(self):
        from dask_gateway import Gateway

        self._gateway = Gateway()

        if config.get("runtime.gateway.cluster_name"):
            # connect to an existing cluster
            self._cluster = self._gateway.get_cluster(config.get("runtime.gateway.cluster_name"))
        else:
            # create a new cluster
            options = self._gateway.cluster_options(use_local_defaults=False)
            options['worker_cores'] = config.get("runtime.gateway.worker_cores")
            options['worker_memory'] = config.get("runtime.gateway.worker_memory")
            options['image'] = config.get("runtime.gateway.image")
            options['environment'].update(self._generate_env())

            self._cluster = self._gateway.new_cluster(
                cluster_options=options, shutdown_on_close=True
            )
            self._cluster.adapt(
                minimum=config.get("runtime.gateway.adapt_min"),
                maximum=config.get("runtime.gateway.adapt_max"),
            )

    @cached_property
    def executor(self) -> Executor:

        print('cluster info:')
        print(self._cluster)
        print(self._cluster.name)
        print(self._cluster.dashboard_link)

        return DaskExecutor(
            address=self._cluster.scheduler_address,
            client_kwargs={"security": self._cluster.security},
        )

    def _generate_env(self):
        env = {**_threadsafe_env_vars}
        if config.get("runtime.gateway.extra_pip_packages"):
            env['EXTRA_PIP_PACKAGES'] = config.get("runtime.gateway.extra_pip_packages")
        if 'AZURE_STORAGE_CONNECTION_STRING' in os.environ:
            env['AZURE_STORAGE_CONNECTION_STRING'] = os.environ['AZURE_STORAGE_CONNECTION_STRING']
        return env


def get_runtime():
    if config.get("run_options.runtime") == "ci":
        runtime = CIRuntime()
    elif config.get("run_options.runtime") == "local":
        runtime = LocalRuntime()
    elif config.get("run_options.runtime") == "cloud":
        runtime = CloudRuntime()
    elif config.get("run_options.runtime") == "pangeo":
        runtime = PangeoRuntime()
    elif config.get("run_options.runtime") == "gateway":
        runtime = GatewayRuntime()
    elif os.environ.get("CI") == "true":
        runtime = CIRuntime()
        print("CIRuntime selected from os.environ")
    elif os.environ.get("PREFECT__BACKEND") == "cloud":
        runtime = CloudRuntime()
    elif "JUPYTER_IMAGE" in os.environ:
        runtime = PangeoRuntime()
    else:
        raise ValueError('unable to determine runtime')
    return runtime
