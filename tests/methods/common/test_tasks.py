from dataclasses import asdict

import numpy as np
import pytest
import xarray as xr
from upath import UPath
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
from cmip6_downscaling.methods.common.tasks import (
    get_experiment,
    get_obs,
    make_run_parameters,
    rechunk,
    regrid,
)

params = [
    {
        'method': 'bcsd',
        'obs': 'ERA5',
        'model': 'MIROC6',
        'member': 'r1i1p1f1',
        'grid_label': 'gn',
        'table_id': 'day',
        'scenario': 'ssp370',
        "features": ["tasmax", "ua"],
        'variable': 'ua',
        'latmin': '-2',
        'latmax': '2',
        'lonmin': '14.5',
        'lonmax': '18.5',
        'train_dates': ['1980', '1981'],
        'predict_dates': ['2050', '2051'],
        "bias_correction_method": "quantile_mapper",
        "bias_correction_kwargs": {"detrend": "True"},
        "model_type": "PureRegression",
        "model_params": {},
        "day_rolling_window": None,
        "year_rolling_window": None,
    },
    {
        'method': 'gard',
        'obs': 'ERA5',
        'model': 'MIROC6',
        'member': 'r1i1p1f1',
        'grid_label': 'gn',
        'table_id': 'day',
        'scenario': 'ssp370',
        "features": ["tasmax"],
        'variable': 'tasmax',
        'latmin': '-2',
        'latmax': '2',
        'lonmin': '14.5',
        'lonmax': '18.5',
        'train_dates': ['1980', '1981'],
        'predict_dates': ['2050', '2051'],
        "bias_correction_method": "quantile_mapper",
        "bias_correction_kwargs": {"detrend": "True"},
        "model_type": "PureRegression",
        "model_params": {},
        "day_rolling_window": None,
        "year_rolling_window": None,
    },
]


rechunk_params = [
    {
        'chunking_method': 'full_space',
        'chunking_schema': {'time': (2359, 561), 'lat': -1, 'lon': -1},
    },
    {'chunking_method': 'full_time', 'chunking_schema': {'time': -1, 'lat': 25, 'lon': (33, 20)}},
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
    for key in ['history', 'hostname', 'institution', 'source', 'title', 'username', 'version']:
        assert key in ds.attrs


def test_get_obs(run_parameters):
    obs_path = get_obs.run(run_parameters)
    print(obs_path)
    ds = xr.open_zarr(obs_path)
    print(ds)
    check_global_attrs(ds)
    schema = DatasetSchema(
        {
            run_parameters.variable: DataArraySchema(
                dtype=np.floating, name=run_parameters.variable, dims=['time', 'lat', 'lon']
            )
        }
    )
    schema.validate(ds)


@pytest.mark.xfail
def test_get_experiment(run_parameters):
    get_experiment_path = get_experiment.run(run_parameters, time_subset='train_period')

    ds = xr.open_zarr(get_experiment_path)
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
    source_path = UPath(tmp_path) / UPath('regrid_source.zarr')
    ds.to_zarr(source_path)

    target_ds = ds.isel(time=0).coarsen(lat=4, lon=4, boundary='trim').mean()
    target_grid_path = UPath(tmp_path) / UPath('regrid_target.zarr')
    target_ds.to_zarr(target_grid_path)

    actual_path = regrid.run(source_path, target_grid_path)
    actual_ds = xr.open_zarr(actual_path)

    check_global_attrs(actual_ds)
    expected_shape = (ds.dims['time'], target_ds.dims['lat'], target_ds.dims['lon'])
    schema = DatasetSchema(
        {
            'air': DataArraySchema(
                dtype=np.floating, name='air', dims=['time', 'lat', 'lon'], shape=expected_shape
            )
        }
    )
    schema.validate(actual_ds)


@pytest.mark.parametrize('rechunk_params', rechunk_params)
@pytest.mark.xfail
def test_rechunk(rechunk_params, tmp_path):
    # TODO Add testing parameterization to check full_space (done), full_time (done) and template match
    ds = xr.tutorial.open_dataset('air_temperature').chunk({'time': 100, 'lat': 10, 'lon': 10})
    source_path = f'{str(tmp_path)}/rechunk.zarr'
    ds.to_zarr(source_path, mode='w')

    actual_path = rechunk.run(
        source_path,
        pattern=rechunk_params['chunking_method'],
    )
    actual_ds = xr.open_zarr(actual_path)

    expected_chunks = rechunk_params['chunking_schema']
    schema = DatasetSchema(
        {
            'air': DataArraySchema(
                dtype=np.floating, name='air', dims=['time', 'lat', 'lon'], chunks=expected_chunks
            )
        }
    )

    schema.validate(actual_ds)
