"""Config file used by donfig"""
from __future__ import annotations

_defaults = {
    'auth': {
        "tf_azure_storage_key": "$TF_AZURE_STORAGE_KEY",
    },
    'chunk_dims': {'full_space': ('time',), 'full_time': ('lat', 'lon')},
    'storage': {
        'top_level': {
            'uri': 'az://',
            'storage_options': {"connection_string": "$AZURE_STORAGE_CONNECTION_STRING"},
        },
        'intermediate': {
            'uri': 'az://scratch/intermediates',
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
        'static': {
            'uri': 'az://static',
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
    "data_catalog": {
        "cmip": {
            'uri': "https://cpdataeuwest.blob.core.windows.net/cp-cmip/cmip6/",
            'json': "https://cpdataeuwest.blob.core.windows.net/cp-cmip/cmip6/pangeo-cmip6.json",
        },
        "era5": {
            'uri': "https://cpdataeuwest.blob.core.windows.net/cp-cmip/training/ERA5",
            'json': "https://cpdataeuwest.blob.core.windows.net/cp-cmip/training/ERA5-azure.json",
        },
        "era5_daily": {
            'uri': "https://cpdataeuwest.blob.core.windows.net/cp-cmip/training/ERA5_daily",
            'json': "https://cpdataeuwest.blob.core.windows.net/cp-cmip/training/ERA5-daily-azure.json",
        },
        "era5_daily_winds": {
            'uri': "https://cpdataeuwest.blob.core.windows.net/cp-cmip/training/ERA5_daily_winds/",
        },
    },
    'weights': {
        'gcm_pyramid_weights': {'uri': 'az://static/xesmf_weights/cmip6_pyramids/weights.csv'},
        'downscaled_pyramid_weights': {
            'uri': 'az://static/xesmf_weights/downscaled_pyramid/weights.csv'
        },
        'gcm_obs_weights': {'uri': 'az://static/xesmf_weights/gcm_obs/weights.csv'},
    },
    'run_options': {
        'runtime': "pangeo",
        'use_cache': True,
        'generate_pyramids': False,
        'construct_analogs': True,
        'combine_regions': False,
    },
    "runtime": {
        "cloud": {
            "storage_prefix": "az://",
            "storage_options": {
                'container': 'prefect',
            },
            "agent": "az-eu-west",
            "extra_pip_packages": "git+https://github.com/carbonplan/cmip6-downscaling.git@0.1.7 --no-deps",
            "kubernetes_cpu": 15,
            "kubernetes_memory": "224Gi",
            "image": "carbonplan/cmip6-downscaling-prefect:2022.06.19",
            'n_workers': 16,
            'threads_per_worker': 1,
        },
        "local": {"storage_prefix": "/tmp/", "storage_options": {'directory': './'}},
        "test": {
            "storage_prefix": "/tmp/",
            "storage_options": {'directory': './'},
        },
        "pangeo": {
            "storage_prefix": "az://",
            "storage_options": {'directory': './'},
            'n_workers': 8,
            'threads_per_worker': 1,
        },
    },
}
