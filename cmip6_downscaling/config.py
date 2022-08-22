"""Config file used by donfig"""
from __future__ import annotations

_defaults = {
    'auth': {
        "tf_azure_storage_key": "$TF_AZURE_STORAGE_KEY",  # this will fail on azure
    },
    'chunk_dims': {'full_space': ('time',), 'full_time': ('lat', 'lon')},
    'storage': {
        'top_level': {
            'uri': 's3://',
            'storage_options': {},
        },
        'intermediate': {
            'uri': 's3://carbonplan-scratch/cmip6-downscaling/intermediates',
            'storage_options': {},
        },
        'results': {
            'uri': 's3://carbonplan-cmip6/flow-outputs/results',
            'storage_options': {},
        },
        'temporary': {
            'uri': 's3://carbonplan-cmip6/flow-outputs/temporary',
            'storage_options': {},
        },
        'static': {
            'uri': 's3://carbonplan-cmip6/static',
            'storage_options': {},
        },
        'scratch': {
            'uri': 's3://carbonplan-scratch/cmip6-downscaling',
        },
        'web_results': {
            'blob': 'analysis_notebooks',  # this will fail on s3
            'storage_options': {},
        },
    },
    "data_catalog": {
        "cmip": {
            'uri': "https://cmip6-pds.s3.amazonaws.com/pangeo-cmip6.json",
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
    'run_options': {
        'runtime': "pangeo",
        'use_cache': True,
        'generate_pyramids': False,
        'construct_analogs': True,
        'combine_regions': False,
    },
    "runtime": {
        "cloud": {
            "storage_prefix": "s3://carbonplan-cmip6",
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
            "storage_prefix": "s3://carbonplan-cmip6/",
            "storage_options": {'directory': './'},
            'n_workers': 8,
            'threads_per_worker': 1,
        },
    },
}
