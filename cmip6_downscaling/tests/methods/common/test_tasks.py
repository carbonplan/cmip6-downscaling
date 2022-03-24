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

from cmip6_downscaling.methods.common.containers import RunParameters
from cmip6_downscaling.methods.common.tasks import get_obs, make_run_parameters, rechunk, regrid

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
    schema.validate(ds)


def test_regrid(tmp_path):

    pytest.importorskip('xesmf')

    ds = xr.tutorial.open_dataset('air_temperature').chunk({'time': 10, 'lat': -1, 'lon': -1})
    source_path = tmp_path / 'regrid_source.zarr'
    ds.to_zarr(source_path)

    target_ds = ds.isel(time=0).coarsen(lat=4, lon=4, boundary='trim').mean()
    target_grid_path = tmp_path / 'regrid_target.zarr'
    target_ds.to_zarr(target_grid_path)

    actual_path = regrid.run(source_path, target_grid_path)
    actual_ds = xr.open_zarr(actual_path)

    check_global_attrs(ds)
    expected_shape = (ds.dims['time'], target_ds.dims['latitude'], target_ds.dims['longitude'])
    schema = DatasetSchema(
        {
            'air': DataArraySchema(
                dtype=np.floating, name='air', dims=['time', 'lat', 'lon'], shape=expected_shape
            )
        }
    )
    schema.validate(actual_ds)


def test_rechunk(tmp_path):

    ds = xr.tutorial.open_dataset('air_temperature').chunk({'time': 10, 'lat': -1, 'lon': -1})
    source_path = tmp_path / 'rechunk.zarr'
    ds.to_zarr(source_path)

    actual_path = rechunk(
        source_path,
        chunking_pattern='full_time',
    )
    actual_ds = xr.open_zarr(actual_path)
