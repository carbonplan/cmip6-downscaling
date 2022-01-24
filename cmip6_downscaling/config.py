"""Config file used by donfig"""


_defaults = {
    "data_catalog": {
        "cmip": "https://cmip6downscaling.blob.core.windows.net/cmip6/pangeo-cmip6.json",
        "era5": "https://cmip6downscaling.blob.core.windows.net/cmip6/ERA5_daily/",
    },
    "runtime": {
        "cloud": {
            "connection_string": None,  # os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
            "storage_prefix": "az://",
            "agent": "az-eu-west",
            "extra_pip_packages": "git+https://github.com/carbonplan/cmip6-downscaling.git git+https://github.com/pangeo-data/scikit-downscale.git",
            "kubernetes_cpu": 7,
            "kubernetes_memory": "16Gi",
            "image": "carbonplan/cmip6-downscaling-prefect:2022.01.05",
            "pod_memory_limit": "4Gi",
            "pod_memory_request": "4Gi",
            "pod_threads_per_worker": 2,
            "pod_cpu_limit": 2,
            "pod_cpu_request": 2,
            "deploy_mode": "remote",
            "adapt_min": 2,
            "adapt_max": 2,
            "dask_distributed_worker_resources_taskslots": "1",
            "OMP_NUM_THREADS": "1",
            "MPI_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
        "local": {"storage_prefix": "./"},
        "CI": {
            "storage_prefix": "/tmp/",
        },
        "pangeo": {
            "storage_prefix": "./",
        },
    },
}
