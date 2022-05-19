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
            'uri': "https://cmip6downscaling.blob.core.windows.net/cmip6/pangeo-cmip6.json",
            'storage_options': {"account_name": "cmip6downscaling"},
        },
        "era5": {
            'uri': "https://cmip6downscaling.blob.core.windows.net/cmip6/ERA5_daily/",
            'storage_options': {"account_name": "cmip6downscaling"},
        },
    },
    'weights': {
        'gcm_pyramid_weights': {'uri': 'az://static/xesmf_weights/cmip6_pyramids/weights.csv'},
        'downscaled_pyramid_weights': {
            'uri': 'az://static/xesmf_weights/downscaled_pyramid/weights.csv'
        },
        'gcm_obs_weights': {'uri': 'az://static/xesmf_weights/gcm_obs/weights.csv'},
    },
    'run_options': {'runtime': "pangeo", 'use_cache': False, 'generate_pyramids': False},
    "runtime": {
        "cloud": {
            "storage_prefix": "az://",
            "storage_options": {
                'container': 'prefect',
            },
            "agent": "az-eu-west",
            "extra_pip_packages": "git+https://github.com/carbonplan/cmip6-downscaling.git@patch/prefect-single-node-deploy scikit-downscale==0.1.3 rechunker==0.5.0 --no-deps",
            "kubernetes_cpu": 15,
            "kubernetes_memory": "224Gi",
            "image": "carbonplan/cmip6-downscaling-prefect:2022.04.12",
            'n_workers': 16,
            'threads_per_worker': 1,
        },
        "gateway": {
            "storage_prefix": "az://",
            "storage_options": {
                'container': 'prefect',
            },
            "cluster_name": '',  #
            "extra_pip_packages": "git+https://github.com/carbonplan/cmip6-downscaling.git",
            "image": "carbonplan/cmip6-downscaling-prefect:2022.04.12",
            "worker_cores": 1,
            "worker_memory": 16,  # Gi
            "adapt_min": 1,
            "adapt_max": 60,
        },
        "local": {"storage_prefix": "/tmp/", "storage_options": {'directory': './'}},
        "test": {
            "storage_prefix": "/tmp/",
            "storage_options": {'directory': './'},
        },
        "pangeo": {
            "storage_prefix": "az://",
            "storage_options": {'directory': './'},
            'n_workers': 16,
            'threads_per_worker': 1,
        },
    },
}
