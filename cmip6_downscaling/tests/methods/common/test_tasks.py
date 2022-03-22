from dataclasses import asdict

import numpy as np
import pytest
import xarray as xr
from xarray_schema import DataArraySchema, DatasetSchema

from cmip6_downscaling import config

# TODO: figure out how to make this a fixture or pytest config option
config.set(
    {
        'storage.scratch.uri': '/tmp/cmip6_downscaling_tests/scratch',
        'storage.intermediate.uri': '/tmp/cmip6_downscaling_tests/intermediate',
        'storage.results.uri': '/tmp/cmip6_downscaling_tests/results',
    }
)
print(config.config)

from cmip6_downscaling.methods.common.containers import RunParameters
from cmip6_downscaling.methods.common.tasks import get_obs, make_run_parameters

params = [
    {
        'method': 'bcsd',
        'obs': 'era5',
        'model': 'canesm',
        'scenario': 'historical',
        'variable': 'tasmax',
        'latmin': '-2',
        'latmax': '2',
        'lonmin': '14.5',
        'lonmax': '18.5',
        'train_dates': ['1980', '1981'],
        'predict_dates': ['2050', '2051'],
    },
    {
        'method': 'gard',
        'obs': 'era5',
        'model': 'canesm',
        'scenario': 'historical',
        'variable': 'tasmax',
        'latmin': '-2',
        'latmax': '2',
        'lonmin': '14.5',
        'lonmax': '18.5',
        'train_dates': ['1980', '1981'],
        'predict_dates': ['2050', '2051'],
    },
]


@pytest.fixture(scope="module", params=params)
def run_parameters(request):
    return RunParameters(**request.param)


@pytest.mark.parametrize("params", params)
def test_make_run_parameters(params):
    rps = make_run_parameters.run(**params)
    assert isinstance(rps, RunParameters)
    assert asdict(rps) == params


def check_global_attrs(ds):
    for key in ['history', 'hostname', 'institution', 'source', 'tilte', 'username', 'version']:
        assert key in ds.attrs


def check_dataset_schema(ds, schema):
    schema.validate(ds)


def test_get_obs(run_parameters):
    obs_path = get_obs.run(run_parameters)

    ds = xr.open_zarr(obs_path)

    check_global_attrs(ds)
    schema = DatasetSchema(
        {
            run_parameters.variable: DataArraySchema(
                dtype=np.floating, name=run_parameters.variable, dims=['time', 'lat', 'lon']
            )
        }
    )
    check_dataset_schema(ds, schema)
