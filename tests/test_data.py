import numpy as np
import pytest
from xarray_schema import DataArraySchema, DatasetSchema

from cmip6_downscaling.data.observations import open_era5

params = ['ua', 'va', 'tasmin', 'tasmax', 'pr']


@pytest.mark.parametrize('params', params)
def test_open_era5(params):
    ds = open_era5(params, slice('2020', '2020'))
    print(ds)
    print(params)
    schema = DatasetSchema(
        {params: DataArraySchema(dtype=np.floating, dims=['time', 'lat', 'lon'])},
    )
    schema.validate(ds)
