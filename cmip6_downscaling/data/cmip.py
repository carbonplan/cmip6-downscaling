from collections import defaultdict

import intake
import pandas as pd
from intake_esm.merge_util import AggregationError

variable_ids = ['pr', 'tasmin', 'tasmax', 'rsds', 'hurs', 'ps']


def check_variable_ids_in_df(df):
    unique_vars = df['variable_id'].unique()
    return all(v in unique_vars for v in variable_ids)


def make_model_dict(hist_subset, ssp_subset):
    d = defaultdict(list)

    for key_hist in hist_subset:
        left, right = key_hist.rsplit('historical')
        left_scen = left.replace('CMIP', 'ScenarioMIP')
        if not check_variable_ids_in_df(hist_subset[key_hist].df):
            continue
        for key_ssp in ssp_subset:
            if (
                left_scen in key_ssp
                and right in key_ssp
                and check_variable_ids_in_df(ssp_subset[key_ssp].df)
            ):
                d[key_hist].append(key_ssp)
    model_dict = {k: list(set(v)) for k, v in d.items()}

    return model_dict


def fix_lons(ds):
    lon = ds.lon.copy()
    lon.values[lon.values > 180] -= 360
    ds['lon'] = lon
    return ds


def fix_times(ds):
    '''convert time coord to pandas datetime index'''
    times = ds.indexes['time']
    new_times = pd.date_range(start=times[0].strftime('%Y-%m'), periods=ds.dims['time'], freq='MS')
    ds['time'] = new_times
    return ds


def subset_conus(ds):
    ds = ds.sel(lon=slice(227, 299), lat=slice(19, 55))
    return ds


def rename(ds):
    if 'longitude' in ds:
        ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'})
    return ds


def maybe_drop_band_vars(ds):
    if ('lat_bnds' in ds) or ('lat_bnds' in ds.coords):
        ds = ds.drop('lat_bnds')
    if ('lon_bnds' in ds) or ('lon_bnds' in ds.coords):
        ds = ds.drop('lon_bnds')
    if ('time_bnds' in ds) or ('time_bnds' in ds.coords):
        ds = ds.drop('time_bnds')
    return ds


def preprocess_hist(ds):
    # consider using cmip6_preprocessing here
    return (
        ds.pipe(rename)
        .sel(time=slice('1950', '2015'))
        # .pipe(subset_conus)
        .pipe(fix_lons)
        .pipe(fix_times)
        .pipe(maybe_drop_band_vars)
    )


def preprocess_ssp(ds):
    # consider using cmip6_preprocessing here
    return (
        ds.pipe(rename)
        .sel(time=slice('2015', '2120'))
        # .pipe(subset_conus)
        .pipe(fix_lons)
        .pipe(fix_times)
        .pipe(maybe_drop_band_vars)
    )


def cmip():

    col_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
    col = intake.open_esm_datastore(col_url)

    # get all possible simulations
    full_subset = col.search(
        activity_id=['CMIP', 'ScenarioMIP'],
        experiment_id=['historical', 'ssp245', 'ssp370', 'ssp585'],
        table_id='Amon',
        grid_label='gn',
        variable_id=variable_ids,
    )

    # get historical simulations
    hist_subset = full_subset.search(
        activity_id=['CMIP'],
        experiment_id=['historical'],
        require_all_on=['variable_id'],
    )

    # get future simulations
    ssp_subset = full_subset.search(
        activity_id=['ScenarioMIP'],
        experiment_id=['ssp245', 'ssp370', 'ssp585'],
        require_all_on=['variable_id'],
    )

    model_dict = make_model_dict(hist_subset, ssp_subset)

    valid_keys = []
    for k, v in model_dict.items():
        valid_keys.extend([k] + v)

    data = {}
    zarr_kwargs = dict(consolidated=True, use_cftime=True)

    failed = {}
    for hist_key, ssp_keys in model_dict.items():
        print(hist_key)
        try:
            data[hist_key] = hist_subset[hist_key](
                zarr_kwargs=zarr_kwargs, preprocess=preprocess_hist
            ).to_dask()
        except (OSError, AggregationError, IndexError, RuntimeError) as e:
            print(f'key failed: {hist_key}')
            failed[hist_key] = e
            continue

        for ssp_key in ssp_keys:
            print(ssp_key)
            try:
                data[ssp_key] = ssp_subset[ssp_key](
                    zarr_kwargs=zarr_kwargs, preprocess=preprocess_ssp
                ).to_dask()
            except (OSError, AggregationError, IndexError, RuntimeError) as e:
                print(f'key failed: {ssp_key}')
                failed[ssp_key] = e

    for k in list(data):
        if k not in valid_keys:
            del data[k]

    print(f'done with cmip but these keys failed: {failed}')

    return model_dict, data
