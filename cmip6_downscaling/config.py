"""Config file used by donfig"""
from __future__ import annotations

_defaults = {
    'chunk_dims': {'full_space': ('time',), 'full_time': ('lat', 'lon')},
    'storage': {
        'top_level': {
            'uri': 'az://',
            'storage_options': {"connection_string": "$AZURE_STORAGE_CONNECTION_STRING"},
        },
        'intermediate': {
            'uri': 'az://flow-outputs/intermediates',
            'storage_options': {"connection_string": "$AZURE_STORAGE_CONNECTION_STRING"},
        },
        'results': {
            'uri': 'az://flow-outputs/results',
            'storage_options': {"connection_string": "$AZURE_STORAGE_CONNECTION_STRING"},
        },
        'temporary': {
            'uri': 'az://flow-outputs/temporary',
            'storage_options': {"connection_string": "$AZURE_STORAGE_CONNECTION_STRING"},
        },
        'scratch': {
            'uri': 'az://scratch',
            'storage_options': {"connection_string": "$AZURE_STORAGE_CONNECTION_STRING"},
        },
        'web_results': {
            'blob': 'analysis_notebooks',
            'storage_options': {"connection_string": "$AZURE_STORAGE_CONNECTION_STRING"},
        },
    },
    # 'methods': {
    #     'bcsd': {
    #         'process_stages': {
    #             "intermediate": {
    #                 'obs_ds': {'path_template': '/obs_ds/{obs_identifier}'},
    #                 'coarsened_obs': {'path_template': '/coarsened_obs/{obs_identifier}'},
    #                 'spatial_anomalies': {'path_template': '/spatial_anomalies/{obs_identifier}'},
    #                 'gcm_predict': {'path_template': '/gcm_predict/{gcm_identifier}'},
    #                 'rechunked_gcm': {'path_template': '/rechunked_gcm/{gcm_identifier}'},
    #                 'bias_corrected': {'path_template': '/bias_corrected/{gcm_identifier}'},
    #             },
    #             "results": {
    #                 "bcsd_output": {"path_template": "/bcsd_output/{gcm_identifier}"},
    #                 "bcsd_output_monthly": {
    #                     "path_template": "/bcsd_output_monthly/{gcm_identifier}"
    #                 },
    #                 "bcsd_output_annual": {"path_template": "/bcsd_output_annual/{gcm_identifier}"},
    #                 "pyramid_daily": {"path_template": "/pyramid_daily/{gcm_identifier}"},
    #                 "pyramid_monthly": {"path_template": "/pyramid_monthly/{gcm_identifier}"},
    #                 "pyramid_annual": {"path_template": "/pyramid_annual/{gcm_identifier}"},
    #             },
    #         },
    #     },
    #     'gard': {},
    #     'maca': {},
    # },
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
    'run_options': {'runtime': "pangeo", 'cleanup_flag': True, 'use_cache': True},
    "runtime": {
        "cloud": {
            "storage_prefix": "az://",
            "storage_options": {
                'container': 'prefect',
            },
            "agent": "az-eu-west",
            "extra_pip_packages": "git+https://github.com/carbonplan/cmip6-downscaling.git@main",
            "kubernetes_cpu": 7,
            "kubernetes_memory": "16Gi",
            "image": "carbonplan/cmip6-downscaling-prefect:2022.03.30",
            "pod_memory_limit": "8Gi",
            "pod_memory_request": "8Gi",
            "pod_threads_per_worker": 1,
            "pod_cpu_limit": 2,
            "pod_cpu_request": 2,
            "deploy_mode": "remote",
            "adapt_min": 2,
            "adapt_max": 20,
            "dask_distributed_worker_resources_taskslots": "1",
        },
        "local": {"storage_prefix": "/tmp/", "storage_options": {'directory': './'}},
        "test": {
            "storage_prefix": "/tmp/",
            "storage_options": {'directory': './'},
        },
        "pangeo": {
            "storage_prefix": "az://",
            "storage_options": {'directory': './'},
            'n_workers': 32,
            'threads_per_worker': 1,
        },
    },
}
