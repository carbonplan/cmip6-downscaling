import intake
import xarray as xr
from carbonplan.data import cat

from ..constants import KELVIN, SEC_PER_DAY
from ..data.cmip import cmip
from ..methods.bias_correction import MontlyBiasCorrection
from .utils import get_store

# time slices
train_time = slice('1960', '2009')
hist_time = slice('1950', '2014')
future_time = slice('2015', '2120')

# variable names
update_vars = ['area', 'crs', 'mask']
ds_vars = ['tasmin', 'tasmax', 'pr']
t_vars = ['tmin', 'tmax']
p_var = 'ppt'

# output chunks (for dask/zarr)
out_chunks = {'time': 198, 'x': 121, 'y': 74}


def preprocess(ds):
    ds = ds.rename({'tasmax': 'tmax', 'tasmin': 'tmin', 'pr': 'ppt'})
    ds[p_var] *= xr.Variable('time', ds.indexes['time'].days_in_month * SEC_PER_DAY)
    ds['tmin'] -= KELVIN
    ds['tmax'] -= KELVIN

    return ds


if __name__ == '__main__':
    skip_existing = True

    # dict of cmip data (raw) - just used for the simulation keys
    model_dict, data = cmip()

    # catalog of climate data
    project_cat = intake.open_catalog('../data/catalog.yaml')

    grid_ds = cat.grids.conus4k.to_dask().load()
    y_ds = cat.terraclimate.raster.to_dask().update(grid_ds[update_vars])

    t_model = MontlyBiasCorrection(correction='absolute')
    p_model = MontlyBiasCorrection(correction='relative')

    for hist_key, future_keys in model_dict.items():

        hist_ds = project_cat[hist_key].pipe(preprocess).sel(time=hist_time)

        # subset further for training
        X_ds = hist_ds.sel(time=train_time)

        # fit the historical model
        t_model.fit(X_ds[t_vars], y_ds[t_vars])
        p_model.fit(X_ds[p_var], y_ds[p_var])

        # predict for the historical data
        y_hat_ds = t_model.predict(hist_ds[t_vars])
        y_hat_ds[p_var] = p_model.predict(hist_ds[p_var])

        store = get_store(hist_key)
        if skip_existing and '.zmetadata' in store:
            print(f'{hist_key} in store, skipping...')
        else:
            # write data to the store
            y_hat_ds.chunk(out_chunks).to_zarr(store, consolidated=True, mode='w')

        for key in future_keys:

            future_ds = project_cat[key].pipe(preprocess).sel(time=future_time)

            # predict for the historical data
            y_hat_ds = t_model.predict(future_ds[t_vars])
            y_hat_ds[p_var] = p_model.predict(future_ds[p_var])

            store = get_store(key)
            if skip_existing and '.zmetadata' in store:
                print(f'{key} in store, skipping...')
            else:
                # write data to the store
                y_hat_ds.chunk(out_chunks).to_zarr(store, consolidated=True, mode='w')
