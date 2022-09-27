import numpy as np
from xarray_schema import DataArraySchema, DatasetSchema

from cmip6_downscaling.data.observations import open_era5


def test_open_era5():
    ds = open_era5(['tasmin', 'ua'], slice('2020', '2020'))
    schema = DatasetSchema(
        {'tasmin': DataArraySchema(dtype=np.floating, name='tasmin', dims=['time', 'lat', 'lon'])},
        {'ua': DataArraySchema(dtype=np.floating, name='ua', dims=['time', 'lat', 'lon'])},
    )
    schema.validate(ds)
