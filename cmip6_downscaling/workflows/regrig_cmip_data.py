import xesmf
from carbonplan.data import cat

from ..data.cmip import cmip
from .utils import get_store


def regrid_one_model(source_ds, target_grid, method='bilinear', reuse_weights=True):
    ''' simple wrapper around xesmf '''
    regridder = xesmf.Regridder(source_ds, target_grid, method=method, reuse_weights=reuse_weights)
    return regridder(source_ds)


if __name__ == '__main__':
    skip_existing = True
    model_dict, data = cmip()

    grid_ds = cat.grids.conus4k.to_dask().load()

    update_vars = ['area', 'crs', 'mask']
    trange = slice('1950', '2120')

    for hist_key, future_keys in model_dict.items():

        keys_to_regrid = [hist_key] + future_keys
        for key in keys_to_regrid:

            store = get_store(key)

            if skip_existing and '.zmetadata' in store:
                print(f'{key} in store, skipping...')
                continue

            print(key)
            print(data[key])

            ds = regrid_one_model(data[key].sel(time=trange), grid_ds).chunk(
                {'time': 198, 'x': 121, 'y': 74}
            )
            ds.update(grid_ds[update_vars])
            ds.to_zarr(store, mode='w', consolidated=True)
