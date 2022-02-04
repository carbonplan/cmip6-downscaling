"""Config file used by donfig"""

_defaults = {
    'storage': {
        'gcm_identifier_template': '{gcm}/{scenario}/{variable}/{bbox}/{train_period}/{predict_period}/',
        'obs_identifier_template': '{obs}/{variable}/{bbox}/{train_period}/',
        'intermediate': {'uri': '/tmp/flow-outputs/intermediates/', 'storage_options': {}},
        'results': {'uri': '/tmp/flow-outputs/results/', 'storage_options': {}},
        'temporary': {'uri': '/tmp/flow-outputs/temporary/', 'storage_options': {}},
        'xpersist_store_name': 'xpersist_metadata_store/',
    },
    'methods': {
        'bcsd': {
            'process_stages': {
                "intermediate": {
                    'obs_ds': {'path_template': 'obs_ds/{obs_identifier}'},
                    'coarsened_obs': {'path_template': 'coarsened_obs/{obs_identifier}'},
                    'spatial_anomalies': {'path_template': 'spatial_anomalies/{obs_identifier}'},
                    'gcm_predict': {'path_template': 'gcm_predict/{gcm_identifier}'},
                    'rechunked_gcm': {'path_template': 'rechunked_gcm/{gcm_identifier}'},
                    'bias_corrected': {'path_template': 'bias_corrected/{gcm_identifier}'},
                },
                "results": {
                    "bcsd_output": {"path_template": "bcsd_output/{gcm_identifier}"},
                    "pyramid": {"path_template": "pyramid/{gcm_identifier}"},
                },
            }
        },
        'gard': {},
        'maca': {},
    },
    "data_catalog": {
        "cmip": {
            'uri': "https://cmip6downscaling.blob.core.windows.net/cmip6/pangeo-cmip6.json",
            'storage_options': {"account_name": "cmip6downscaling"},
        },
        "era5": {
            'uri': "https://cmip6downscaling.blob.core.windows.net/cmip6/ERA5_daily/",
            'storage_options': {"account_name": "cmip6downscaling"},
        },
    },
    'run_options': {'runtime': None, 'cleanup_flag': True},
    "runtime": {
        "cloud": {
            "storage_options": {'container': 'prefect'},
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
        },
        "local": {"storage_options": {'directory': './'}},
        "test": {
            "storage_options": {'directory': './'},
        },
        "pangeo": {
            "storage_options": {'directory': './'},
        },
    },
}
