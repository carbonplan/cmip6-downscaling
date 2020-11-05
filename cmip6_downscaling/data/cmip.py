import os
from collections import defaultdict

import intake
import xarray as xr
import zarr


def make_model_dict(hist_subset, ssp_subset):
    d = defaultdict(list)

    for key_hist in hist_subset:
        left, right = key_hist.rsplit('historical')
        left_scen = left.replace('CMIP', 'ScenarioMIP')
        if not hist_subset[key_hist].df.nunique()['variable_id'] >= 3:
            continue
        for key_ssp in ssp_subset:
            if (
                left_scen in key_ssp
                and right in key_ssp
                and ssp_subset[key_ssp].df.nunique()['variable_id'] >= 3
            ):
                d[key_hist].append(key_ssp)
    model_dict = {k: list(set(v)) for k, v in d.items()}

    return model_dict


def fix_lons(ds):
    lon = ds.lon.copy()
    lon.values[lon.values > 180] -= 360
    ds['lon'] = lon
    return ds


def subset_conus(ds):
    ds = ds.sel(lon=slice(227, 299), lat=slice(19, 55))
    return ds


def rename(ds):
    if 'longitude' in ds:
        ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'})
    return ds


def preprocess_hist(ds):
    # consider using cmip6_preprocessing here
    return ds.pipe(rename).pipe(subset_conus).pipe(fix_lons).sel(time=slice('1960', '2014'))


def preprocess_ssp(ds):
    # consider using cmip6_preprocessing here
    return ds.pipe(rename).pipe(subset_conus).pipe(fix_lons).sel(time=slice('2015', '2100'))


def cmip():

    col_url = 'https://raw.githubusercontent.com/NCAR/intake-esm-datastore/master/catalogs/pangeo-cmip6.json'
    col = intake.open_esm_datastore(col_url)

    # get all possible simulations
    full_subset = col.search(
        activity_id=['CMIP', 'ScenarioMIP'],
        experiment_id=['historical', 'ssp245', 'ssp370', 'ssp585'],
        member_id='r1i1p1f1',  # remove this at some point
        table_id='Amon',
        grid_label='gn',
        variable_id=['pr', 'tasmin', 'tasmax'],
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
    zarr_kwargs = dict(consolidated=True)
    data.update(hist_subset.to_dataset_dict(zarr_kwargs=zarr_kwargs, preprocess=preprocess_hist))
    data.update(ssp_subset.to_dataset_dict(zarr_kwargs=zarr_kwargs, preprocess=preprocess_ssp))

    for k in list(data):
        if k not in valid_keys:
            del data[k]

    return model_dict, data


def sample_data():

    samples = [
        'CMIP.AWI.AWI-CM-1-1-MR.historical.Amon.gn',
        'CMIP.BCC.BCC-CSM2-MR.historical.Amon.gn',
        'CMIP.CSIRO.ACCESS-ESM1-5.historical.Amon.gn',
        'CMIP.MIROC.MIROC6.historical.Amon.gn',
        'CMIP.HAMMOZ-Consortium.MPI-ESM-1-2-HAM.historical.Amon.gn',
        'CMIP.NUIST.NESM3.historical.Amon.gn',
        'CMIP.CSIRO-ARCCSS.ACCESS-CM2.historical.Amon.gn',
        'CMIP.CCCma.CanESM5.historical.Amon.gn',
        'CMIP.CAS.FGOALS-g3.historical.Amon.gn',
        'CMIP.MPI-M.MPI-ESM1-2-LR.historical.Amon.gn',
        'CMIP.MRI.MRI-ESM2-0.historical.Amon.gn',
        'CMIP.CMCC.CMCC-CM2-SR5.historical.Amon.gn',
        'ScenarioMIP.CCCma.CanESM5.ssp370.Amon.gn',
        'ScenarioMIP.MRI.MRI-ESM2-0.ssp245.Amon.gn',
        'ScenarioMIP.MPI-M.MPI-ESM1-2-LR.ssp245.Amon.gn',
        'ScenarioMIP.BCC.BCC-CSM2-MR.ssp245.Amon.gn',
        'ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp245.Amon.gn',
        'ScenarioMIP.CSIRO-ARCCSS.ACCESS-CM2.ssp245.Amon.gn',
        'ScenarioMIP.CAS.FGOALS-g3.ssp370.Amon.gn',
        'ScenarioMIP.AWI.AWI-CM-1-1-MR.ssp370.Amon.gn',
        'ScenarioMIP.BCC.BCC-CSM2-MR.ssp370.Amon.gn',
        'ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp370.Amon.gn',
        'ScenarioMIP.MIROC.MIROC6.ssp370.Amon.gn',
        'ScenarioMIP.AWI.AWI-CM-1-1-MR.ssp585.Amon.gn',
        'ScenarioMIP.MRI.MRI-ESM2-0.ssp585.Amon.gn',
        'ScenarioMIP.CSIRO-ARCCSS.ACCESS-CM2.ssp370.Amon.gn',
        'ScenarioMIP.CAS.FGOALS-g3.ssp245.Amon.gn',
        'ScenarioMIP.MIROC.MIROC6.ssp245.Amon.gn',
        'ScenarioMIP.CMCC.CMCC-CM2-SR5.ssp585.Amon.gn',
        'ScenarioMIP.CSIRO-ARCCSS.ACCESS-CM2.ssp585.Amon.gn',
        'ScenarioMIP.MPI-M.MPI-ESM1-2-LR.ssp585.Amon.gn',
        'ScenarioMIP.MIROC.MIROC6.ssp585.Amon.gn',
        'ScenarioMIP.MPI-M.MPI-ESM1-2-LR.ssp370.Amon.gn',
        'ScenarioMIP.NUIST.NESM3.ssp585.Amon.gn',
        'ScenarioMIP.MRI.MRI-ESM2-0.ssp370.Amon.gn',
        'ScenarioMIP.CAS.FGOALS-g3.ssp585.Amon.gn',
        'ScenarioMIP.CCCma.CanESM5.ssp245.Amon.gn',
        'ScenarioMIP.CMCC.CMCC-CM2-SR5.ssp245.Amon.gn',
        'ScenarioMIP.HAMMOZ-Consortium.MPI-ESM-1-2-HAM.ssp370.Amon.gn',
        'ScenarioMIP.CCCma.CanESM5.ssp585.Amon.gn',
        'ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp585.Amon.gn',
        'ScenarioMIP.CMCC.CMCC-CM2-SR5.ssp370.Amon.gn',
        'ScenarioMIP.NUIST.NESM3.ssp245.Amon.gn',
        'ScenarioMIP.AWI.AWI-CM-1-1-MR.ssp245.Amon.gn',
        'ScenarioMIP.BCC.BCC-CSM2-MR.ssp585.Amon.gn',
    ]

    cmip_point_data = {}
    for key in samples:
        store = zarr.storage.ABSStore(
            'carbonplan-data',
            prefix=f'carbonplan-scratch/downscaling-point-data/{key}',
            account_name='carbonplan',
            account_key=os.environ['BLOB_ACCOUNT_KEY'],
        )

        cmip_point_data[key] = xr.open_zarr(store)

    store = zarr.storage.ABSStore(
        'carbonplan-data',
        prefix='carbonplan-scratch/downscaling-point-data/terraclimate',
        account_name='carbonplan',
        account_key=os.environ['BLOB_ACCOUNT_KEY'],
    )
    obs_points = xr.open_zarr(store)

    return cmip_point_data, obs_points
