import os
from collections import defaultdict

import intake
import pandas as pd
import xarray as xr
import zarr
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


def load_cmip_dictionary(
<<<<<<< HEAD
    activity_ids=["CMIP", "ScenarioMIP"],
    experiment_ids=["historical", "ssp370"],  # , "ssp126", "ssp245",  "ssp585"
=======
    activity_ids=["CMIP"],
    experiment_ids=["historical"],  # , "ssp126", "ssp245",  "ssp585"
>>>>>>> origin
    member_ids=["r1i1p1f1"],
    source_ids=["MIROC6"],  # BCC-CSM2-MR"]
    table_ids=["day"],
    grid_labels=["gn"],
    variable_ids=["tasmax"],
<<<<<<< HEAD
=======
    return_type='zarr',
>>>>>>> origin
):
    """Loads CMIP6 GCM dataset dictionary based on input criteria.

    Parameters
    ----------
    activity_ids : list, optional
        [activity_ids in CMIP6 catalog], by default ["CMIP", "ScenarioMIP"],
    experiment_ids : list, optional
        [experiment_ids in CMIP6 catalog], by default ["historical", "ssp370"],  ex:#  "ssp126", "ssp245",  "ssp585"
    member_ids : list, optional
        [member_ids in CMIP6 catalog], by default ["r1i1p1f1"]
    source_ids : list, optional
        [source_ids in CMIP6 catalog], by default ["MIROC6"]
    table_ids : list, optional
        [table_ids in CMIP6 catalog], by default ["day"]
    grid_labels : list, optional
        [grid_labels in CMIP6 catalog], by default ["gn"]
    variable_ids : list, optional
        [variable_ids in CMIP6 catalog], by default ['tasmax']

    Returns
    -------
    [dictionary]
        [dictionary containing available xarray datasets]
    """
    col_url = "https://cmip6downscaling.blob.core.windows.net/cmip6/pangeo-cmip6.json"
<<<<<<< HEAD
    print("intake")
    full_subset = intake.open_esm_datastore(col_url).search(
        activity_id=activity_ids,
        experiment_id=experiment_ids,
        member_id=member_ids,
        source_id=source_ids,
        table_id=table_ids,
        grid_label=grid_labels,
        variable_id=variable_ids,
    )
    print("to dictionary")
    ds_dict = full_subset.to_dataset_dict(
        zarr_kwargs={"consolidated": True, "decode_times": True, "use_cftime": True},
        storage_options={
            "account_name": "cmip6downscaling",
            "account_key": os.environ.get("AccountKey", None),
        },
        progressbar=False,
    )
    print("return dict")
    return ds_dict
=======

    stores = (
        intake.open_esm_datastore(col_url)
        .search(
            activity_id=activity_ids,
            experiment_id=experiment_ids,
            member_id=member_ids,
            source_id=source_ids,
            table_id=table_ids,
            grid_label=grid_labels,
            variable_id=variable_ids,
        )
        .df['zstore']
        .to_list()
    )
    if len(stores) > 1:
        raise ValueError('can only get 1 store at a time')
    if return_type == 'zarr':
        ds = zarr.open_consolidated(stores[0], mode='r')
    elif return_type == 'xr':
        ds = xr.open_zarr(stores[0], consolidated=True)

    ds = gcm_munge(ds)

    return ds

>>>>>>> origin

def convert_to_360(lon):
    if lon > 0:
        return lon
    elif lon < 0:
        return 360 + lon

<<<<<<< HEAD
def gcm_munge(ds):
    if ds.lat[0] < ds.lat[-1]:
        ds = ds.reindex({"lat": ds.lat[::-1]})
    ds = ds.drop(["lat_bnds", "lon_bnds", "time_bnds", "height", "member_id"]).squeeze(
        drop=True
    )
    return ds
=======

def gcm_munge(ds):
    if ds.lat[0] < ds.lat[-1]:
        ds = ds.reindex({"lat": ds.lat[::-1]})
    ds = maybe_drop_band_vars(ds)
    if 'height' in ds:
        ds = ds.drop('height')
    ds = ds.squeeze(drop=True)
    return ds
>>>>>>> origin
