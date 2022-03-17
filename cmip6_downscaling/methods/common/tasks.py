import os
from dataclasses import asdict
from pathlib import PosixPath

import papermill as pm
import xarray as xr
from azure.storage.blob import BlobServiceClient, ContentSettings
from prefect import task
from upath import UPath

from cmip6_downscaling import config
from cmip6_downscaling.data.cmip import load_cmip
from cmip6_downscaling.data.observations import open_era5
from cmip6_downscaling.methods.common.utils import lon_to_180, subset_dataset

from .containers import RunParameters

intermediate_dir = UPath(config.get("storage.intermediate.uri"))

use_cache = True  # TODO: this should be a config option


@task
def make_run_parameters(**kwargs) -> RunParameters:
    """Prefect task wrapper for RunParameters"""
    return RunParameters(**kwargs)


@task
def get_obs(run_parameters: RunParameters) -> UPath:

    target = (
        intermediate_dir
        / "get_obs"
        / "{obs}_{variable}_{latmin}_{latmax}_{lonmin}_{lonmax}_{train_dates[0]}_{train_dates[1]}".format(
            **asdict(run_parameters)
        )
    )
    if use_cache and (target / '.zmetadata').exists():
        print(f'found existing target: {target}')
        return target

    ds = open_era5(run_parameters.variable, run_parameters.train_period).pipe(lon_to_180)
    subset = subset_dataset(
        ds,
        run_parameters.variable,
        run_parameters.train_period.time_slice,
        run_parameters.bbox,
        chunking_schema={'time': 365, 'lat': 150, 'lon': 150},
    )
    del subset[run_parameters.variable].encoding['chunks']

    subset.to_zarr(target, mode='w')
    return target


@task
def get_experiment(run_parameters: RunParameters, time_subset: str) -> UPath:

    target = (
        intermediate_dir
        / "get_experiment"
        / time_subset
        / "{model}_{scenario}_{variable}_{latmin}_{latmax}_{lonmin}_{lonmax}_{train_dates[0]}_{train_dates[1]}_{predict_dates[0]}_{predict_dates[1]}".format(
            **asdict(run_parameters)
        )
    )
    if use_cache and (target / '.zmetadata').exists():
        print(f'found existing target: {target}')
        return target
    print(target)
    ds = load_cmip(
        source_ids=run_parameters.model, return_type='xr', variable_ids=run_parameters.variable
    ).pipe(lon_to_180)

    if time_subset == 'train_period':
        time_period = run_parameters.train_period.time_slice
    elif time_subset == 'predict_period':
        time_period = run_parameters.predict_period.time_slice
    else:
        raise ValueError('time_subset arg is not train_period or predict_period.')
    subset = subset_dataset(ds, run_parameters.variable, time_period, run_parameters.bbox)
    # Note: dataset is chunked into time:365 chunks to standardize leap-year chunking.
    subset = subset.chunk({'time': 365})
    del subset[run_parameters.variable].encoding['chunks']

    subset.to_zarr(target, mode='w')
    return target


@task
def rechunk(path: UPath, pattern: str = None) -> UPath:

    # TODO: think about how to cache this result
    target = intermediate_dir / "rechunk" / "todo"
    if use_cache and (target / '.zmetadata').exists():
        print(f'found existing target: {target}')
        return target

    # TODO

    return target


# @task
def monthly_summary(ds_path: UPath, run_parameters: RunParameters) -> UPath:

    target = intermediate_dir / "monthly_summary" / run_parameters.run_id
    if use_cache and (target / '.zmetadata').exists():
        print(f'found existing target: {target}')
        return target

    ds = xr.open_zarr(ds_path)

    out_ds = xr.Dataset()
    for var in ds:
        if var in ['tasmax', 'tasmin']:
            out_ds[var] = ds[var].resample(time='1MS').mean(dim='time')
        elif var in ['pr']:
            out_ds[var] = ds[var].resample(time='1MS').sum(dim='time')
        else:
            print(f'{var} not implemented')

    out_ds.to_zarr(target, mode='w')

    return target


@task
def annual_summary(ds_path: UPath, run_parameters: RunParameters) -> UPath:

    target = intermediate_dir / "annual_summary" / run_parameters.run_id
    if use_cache and (target / '.zmetadata').exists():
        print(f'found existing target: {target}')
        return target

    ds = xr.open_zarr(ds_path)

    out_ds = xr.Dataset()
    for var in ds:
        if var in ['tasmax', 'tasmin']:
            out_ds[var] = ds[var].resample(time='YS').mean()
        elif var in ['pr']:
            out_ds[var] = ds[var].resample(time='YS').sum()
        else:
            print(f'{var} not implemented')

    out_ds.to_zarr(target, mode='w')

    return target


@task
def pyramid(
    ds_path: UPath, run_parameters: RunParameters, key: str = 'daily', levels: int = 4
) -> UPath:

    target = intermediate_dir / "pyramid" / run_parameters.run_id
    if use_cache and (target / '.zmetadata').exists():
        print(f'found existing target: {target}')
        return target

    # TODO

    return target


@task
def run_analyses(ds_path: UPath, run_parameters: RunParameters) -> UPath:
    """Prefect task to run the analyses on results from a downscaling run.

    Parameters
    ----------
    ds_path : UPath
    run_parameters : RunParameters
        Downscaling run parameter container

    Returns
    -------
    PosixPath
        The local location of an executed notebook path.
    """

    from cmip6_downscaling.analysis import metrics

    root = PosixPath(metrics.__file__)
    template_path = root.parent / 'analyses_template.ipynb'
    executed_notebook_path = root.parent / f'analyses_{run_parameters.run_id}.ipynb'
    executed_html_path = root.parent / f'analyses_{run_parameters.run_id}.html'

    parameters = asdict(run_parameters)

    # TODO: figure out how to unpack these fields in the notebook
    # asdict will return lists for train_dates and predict_dates
    # parameters['train_period_start'] = train_period.start
    # parameters['train_period_end'] = train_period.stop
    # parameters['predict_period_start'] = predict_period.start
    # parameters['predict_period_end'] = predict_period.stop

    # execute notebook with papermill
    pm.execute_notebook(template_path, executed_notebook_path, parameters=parameters)

    # convert from ipynb to html
    # TODO: move this to stand alone function
    # Q: can we control the output path name?
    os.system(f"jupyter nbconvert {executed_notebook_path} --to html")

    # TODO: move to stand alone function
    connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING', None)
    if connection_string is not None:
        # if you have a connection_string, copy the html to azure, if not just return
        # because it is already in your local machine
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        # TODO: fix b/c the run_id has slashes now!!!
        blob_name = config.get('storage.web_results.blob') / parameters.run_id / 'analyses.html'
        blob_client = blob_service_client.get_blob_client(container='$web', blob=blob_name)
        # clean up before writing
        try:
            blob_client.delete_blob()
        except:  # TODO: raise specific error
            pass

        #  need to specify html content type so that it will render and not download
        with open(executed_html_path, "rb") as data:
            blob_client.upload_blob(
                data, content_settings=ContentSettings(content_type='text/html')
            )

    return executed_notebook_path
