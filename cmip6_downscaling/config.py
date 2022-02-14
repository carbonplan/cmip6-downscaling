"""Config file used by donfig"""
_defaults = {
    'storage': {
        'top_level': {'storage_options': {"connection_string": "$AZURE_STORAGE_CONNECTION_STRING"}},
        'gcm_identifier_template': '{gcm}/{scenario}/{variable}/{bbox}/{train_period}/{predict_period}/',
        'obs_identifier_template': '{obs}/{variable}/{bbox}/{train_period}/',
        'intermediate': {
            'uri': 'az://flow-outputs/testing_intermediates',
            'storage_options': {"connection_string": "$AZURE_STORAGE_CONNECTION_STRING"},
        },
        'results': {
            'uri': 'az://flow-outputs/testing_results',
            'storage_options': {"connection_string": "$AZURE_STORAGE_CONNECTION_STRING"},
        },
        'temporary': {
            'uri': 'az://flow-outputs/temporary',
            'storage_options': {"connection_string": "$AZURE_STORAGE_CONNECTION_STRING"},
        },
        'xpersist_store_name': 'xpersist_metadata_store/',
        'web_results': {
            'blob': 'analysis_notebooks',
            'storage_options': {"connection_string": "$AZURE_STORAGE_CONNECTION_STRING"},
        },
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
                    "bcsd_output_monthly": {
                        "path_template": "bcsd_output_monthly/{gcm_identifier}"
                    },
                    "bcsd_output_annual": {"path_template": "bcsd_output_annual/{gcm_identifier}"},
                    "pyramid_daily": {"path_template": "pyramid_daily/{gcm_identifier}"},
                    "pyramid_monthly": {"path_template": "pyramid_monthly/{gcm_identifier}"},
                    "pyramid_annual": {"path_template": "pyramid_annual/{gcm_identifier}"},
                },
            },
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
            "storage_prefix": "az://",
            "storage_options": {
                'container': 'prefect',
                'connection_string': "$AZURE_STORAGE_CONNECTION_STRING",
            },
            "agent": "az-eu-west",
            "extra_pip_packages": "git+https://github.com/carbonplan/cmip6-downscaling.git",
            "kubernetes_cpu": 7,
            "kubernetes_memory": "16Gi",
            "image": "carbonplan/cmip6-downscaling-prefect:2022.02.08",
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
        "local": {"storage_prefix": "/tmp/", "storage_options": {'directory': './'}},
        "test": {
            "storage_prefix": "/tmp/",
            "storage_options": {
                'directory': './',
                'connection_string': "$AZURE_STORAGE_CONNECTION_STRING",
            },
        },
        "pangeo": {
            "storage_prefix": "az://",
            "storage_options": {'directory': './'},
            'n_workers': 2,
            'threads_per_worker': 2,
        },
    },
}
