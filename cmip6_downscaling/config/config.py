import os

os.environ["PREFECT__FLOWS__CHECKPOINTING"] = "true"
from dask_kubernetes import KubeCluster, make_pod_spec
from prefect.executors import DaskExecutor
from prefect.run_configs import KubernetesRun, LocalRun
from prefect.storage import Azure


class AbstractConfig:
    # attributes
    # prefect_storage
    # prefect_run_config
    # prefect_executor
    # fsspec_fs
    def __init__(self):
        self.connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
        self.agent = "az-eu-west"
        self.storage = Azure("prefect")
        self.extra_pip_packages = "git+https://github.com/carbonplan/cmip6-downscaling.git git+https://github.com/pangeo-data/scikit-downscale.git git+https://github.com/NCAR/xpersist.git"
        self.intermediate_cache_path = "az://flow-outputs/intermediate"
        self.results_cache_path = "az://flow-outputs/results"
        self.serializer = "xarray.zarr"


class CloudConfig(AbstractConfig):
    def __init__(self):
        super().__init__()
        self.kubernetes_cpu = 7
        self.kubernetes_memory = "16Gi"
        self.image = "carbonplan/cmip6-downscaling-prefect:2021.12.06"
        self.OMP_NUM_THREADS = "1"
        self.MPI_NUM_THREADS = "1"
        self.MKL_NUM_THREADS = "1"
        self.OPENBLAS_NUM_THREADS = "1"
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
            "OMP_NUM_THREADS": self.OMP_NUM_THREADS,
            "MPI_NUM_THREADS": self.MPI_NUM_THREADS,
            "MKL_NUM_THREADS": self.MKL_NUM_THREADS,
            "OPENBLAS_NUM_THREADS": self.OPENBLAS_NUM_THREADS,
            "DASK_DISTRIBUTED__WORKER__RESOURCES__TASKSLOTS": self.dask_distributed_worker_resources_taskslots,
        }
        return env

    def dask_executor(self):

        pod_spec = make_pod_spec(
            image=self.image,
            memory_limit=self.pod_memory_limit,
            memory_request=self.pod_memory_request,
            threads_per_worker=self.pod_threads_per_worker,
            cpu_limit=self.pod_cpu_limit,
            cpu_request=self.pod_cpu_request,
            env=self.generate_env(),
        )

        pod_spec.spec.containers[0].args.extend(["--resources", "TASKSLOTS=1"])

        daskExecutor = DaskExecutor(
            cluster_class=lambda: KubeCluster(
                pod_spec,
                deploy_mode=self.deploy_mode,
                adapt_kwargs={"minimum": self.adapt_min, "maximum": self.adapt_max},
            )
        )
        return daskExecutor

    def kubernetes_run_config(self):
        kube_run_config = KubernetesRun(
            cpu_request=self.kubernetes_cpu,
            memory_request=self.kubernetes_memory,
            image=self.image,
            labels=self.agent,
            env=self.generate_env(),
        )
        return kube_run_config


class PangeoConfig(AbstractConfig):
    # with Flow(name="bcsd-testing", storage=storage, run_config=run_config) as flow:
    # Which run_config? Does this need a dask executor?
    def __init__(self):
        super().__init__()


class LocalConfig(AbstractConfig):
    # what additional args to add?
    def __init__(self):
        super().__init__()

    def local_run_config(self):
        local_run = LocalRun()
        return local_run


class TestConfig(LocalConfig):
    # for use in GitHub Actions
    # what additional args to add for testconfig? additional pip packages?
    def __init__(self):
        super().__init__()

    def local_run_config(self):
        local_run = LocalRun(env={"EXTRA_PIP_PACKAGES": self.extra_pip_packages})
        return local_run


def get_config(**kwargs):

    if os.environ.get("CI") == "true":
        config = LocalConfig(**kwargs)
    elif os.environ.get("PREFECT__BACKEND") == "cloud":
        config = CloudConfig(**kwargs)
    elif os.environ.get("PANGEO__BACKEND") == "pangeo":
        config = PangeoConfig(**kwargs)
    elif os.environ.get("TEST") == "test":
        config = TestConfig(**kwargs)
    return config


# import os

# os.environ['PREFECT__FLOWS__CHECKPOINTING'] = 'true'
# from dask_kubernetes import KubeCluster, make_pod_spec
# from prefect.executors import DaskExecutor
# from prefect.run_configs import KubernetesRun
# from prefect.storage import Azure


# # Azure --------------------------------------------------------------------
# def return_azure_config():
#     azure_config_dict = {
#         'connection_string': os.environ.get("AZURE_STORAGE_CONNECTION_STRING"),
#         'storage': Azure("prefect"),
#         'intermediate_cache_path': 'az://flow-outputs/intermediate',
#         'results_cache_path': 'az://flow-outputs/results',
#         'serializer': 'xarray.zarr',
#     }
#     return azure_config_dict


# # Prefect --------------------------------------------------------------------
# def return_prefect_config():
#     #  prefect agent ID
#     prefect_config_dict = {'agent': ["az-eu-west"]}
#     return prefect_config_dict


# # Docker --------------------------------------------------------------------
# def return_docker_config():
#     docker_config_dict = {'image': "carbonplan/cmip6-downscaling-prefect:2021.12.06"}
#     return docker_config_dict


# # Kubernetes  --------------------------------------------------------------------
# def return_kubernetes_config():
#     kubernetes_config_dict = {'kubernetes_cpu': 7, 'kubernetes_memory': "16Gi"}
#     return kubernetes_config_dict


# # Dask Executor  --------------------------------------------------------------------
# def return_dask_config():
#     dask_config_dict = {
#         "dask_executor_memory_limit": "16Gi",
#         "dask_executor_memory_request": "16Gi",
#         "dask_executor_threads_per_worker": 2,
#         "dask_executor_cpu_limit": 2,
#         "dask_executor_cpu_request": 2,
#         "dask_executor_adapt_min": 4,
#         "dask_executor_adapt_max": 20,
#     }
#     return dask_config_dict


# def return_extra_pip_packages():
#     extra_pip_packages = {
#         "EXTRA_PIP_PACKAGES": "git+https://github.com/carbonplan/cmip6-downscaling.git git+https://github.com/pangeo-data/scikit-downscale.git git+https://github.com/NCAR/xpersist.git"
#     }
#     return extra_pip_packages


# def return_env_config():
#     env_config = {
#         "AZURE_STORAGE_CONNECTION_STRING": return_azure_config()["connection_string"],
#         "EXTRA_PIP_PACKAGES": return_extra_pip_packages()["EXTRA_PIP_PACKAGES"],
#         'OMP_NUM_THREADS': '1',
#         'MPI_NUM_THREADS': '1',
#         'MKL_NUM_THREADS': '1',
#         'OPENBLAS_NUM_THREADS': '1',
#     }
#     return env_config


# def return_kubernetes_run_config():
#     kubernetes_run_config = KubernetesRun(
#         cpu_request=return_kubernetes_config()["kubernetes_cpu"],
#         memory_request=return_kubernetes_config()["kubernetes_memory"],
#         image=return_docker_config()["image"],
#         labels=return_prefect_config()["agent"],
#         env=return_env_config()["env_config"],
#     )
#     return kubernetes_run_config


# def return_dask_executor():

#     dask_executor = DaskExecutor(
#         cluster_class=lambda: KubeCluster(
#             make_pod_spec(
#                 image=return_docker_config()["image"],
#                 memory_limit=return_dask_config()["dask_executor_memory_limit"],
#                 memory_request=return_dask_config()["dask_executor_memory_request"],
#                 threads_per_worker=return_dask_config()["dask_executor_threads_per_worker"],
#                 cpu_limit=return_dask_config()["dask_executor_cpu_limit]"],
#                 cpu_request=return_dask_config()["dask_executor_cpu_request"],
#                 env=return_env_config()["env_config"],
#             )
#         ),
#         adapt_kwargs={
#             "minimum": return_dask_config()["dask_executor_adapt_min"],
#             "maximum": return_dask_config()["dask_executor_adapt_max"],
#         },
#     )
#     return dask_executor
