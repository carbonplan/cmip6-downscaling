import os
from typing import Tuple

import cartopy.crs as ccrs
import fsspec
import matplotlib.pyplot as plt
import pandas as pd
import papermill as pm
import xarray as xr
from azure.storage.blob import BlobServiceClient, ContentSettings
from prefect import task

from cmip6_downscaling.data.cmip import convert_to_360
from cmip6_downscaling.workflows.paths import get_notebook_paths
from cmip6_downscaling.workflows.utils import BBox

from .qaqc import make_qaqc_ds

connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
fs = fsspec.filesystem('az', connection_string=connection_string)


def qaqc_checks(ds: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:
    '''
    Create the temporal and spatial summaries of a handful of QAQC
    analyses - nans, aphysical quantities

    Parameters
    ----------
    ds : xr.Dataset
        Climate dataset with dimensions ['time', 'lat', 'lon']


    Returns
    -------
    Xarray Dataset, Xarray Dataset
        Temporal and spatial summaries of QAQC analyses
    '''
    qaqc_ds = make_qaqc_ds(ds)
    annual_qaqc_ts = qaqc_ds.groupby('time.year').sum().sum(dim=['lat', 'lon']).to_dataframe()
    qaqc_maps = qaqc_ds.sum(dim='time')
    return annual_qaqc_ts, qaqc_maps


def monthly_summary(ds: xr.Dataset) -> xr.Dataset:
    """Creates an monthly summary dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Downscaled dataset at daily timestep

    Returns
    ----------
    xr.Dataset
        Downscaled dataset at monthly timestep
    """
    out_ds = xr.Dataset()
    for var in ds:
        if var in ['tasmax', 'tasmin']:
            out_ds[var] = ds[var].resample(time='1MS').mean(dim='time')
        elif var in ['pr']:
            out_ds[var] = ds[var].resample(time='1MS').sum(dim='time')
        else:
            print(f'{var} not implemented')

    return out_ds


def annual_summary(ds: xr.Dataset) -> xr.Dataset:
    """Creates an annual summary dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Downscaled dataset at daily timestep

    Returns
    ----------
    xr.Dataset
        Downscaled dataset at annual timestep
    """
    out_ds = xr.Dataset()
    for var in ds:
        if var in ['tasmax', 'tasmin']:
            out_ds[var] = ds[var].resample(time='YS').mean()
        elif var in ['pr']:
            out_ds[var] = ds[var].resample(time='YS').sum()
        else:
            print(f'{var} not implemented')

    return out_ds


@task(log_stdout=True, tags=['dask-resource:TASKSLOTS=1'])
def run_analyses(
    parameters: dict, bbox: BBox, train_period: slice, predict_period: slice, web_blob: str
) -> str:
    """Prefect task to run the analyses on results from a downscaling run.

    Parameters
    ----------
    parameters : dict
        Parameters provided from the downscaling run to uniquely identify
        the output dataset (e.g. gcm, training period, obs dataset, etc.)
    bbox : BBox
        The bounding box over which you are running.
    train_period : slice
        Training period (e.g. slice('1981', '2010'))
    predict_period : slice
        Prediction period (e.g. slice('1981', '2099')). Currently this assumes
        that the downscaled dataset has the historical and future timeseries all
        in one file.
    web_blob : str
        The location in the web bucket where you want these files to go. Likely
        specified in a yaml file in your local setup.

    Returns
    -------
    str
        The local location of an executed notebook path.
    """
    gcm_identifier = parameters['gcm_identifier']
    template_path, executed_notebook_path, executed_html_path = get_notebook_paths(
        gcm_identifier.replace('/', '_')
    )

    parameters['train_period_start'] = train_period.start
    parameters['train_period_end'] = train_period.stop
    parameters['predict_period_start'] = predict_period.start
    parameters['predict_period_end'] = predict_period.stop
    parameters['latmax'] = bbox.lat_slice.start
    parameters['latmin'] = bbox.lat_slice.stop
    parameters['lonmin'] = bbox.lon_slice.start
    parameters['lonmax'] = bbox.lon_slice.stop
    pm.execute_notebook(template_path, executed_notebook_path, parameters=parameters)
    # convert from ipynb to html
    os.system(f"jupyter nbconvert {executed_notebook_path} --to html")
    connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING', None)
    if connection_string is not None:
        # if you have a connection_string, copy the html to azure, if not just return
        # because it is already in your local machine
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        # TODO: fix b/c the run_id has slashes now!!!
        blob_name = web_blob + gcm_identifier + 'analyses.html'
        blob_client = blob_service_client.get_blob_client(container='$web', blob=blob_name)
        # clean up before writing
        try:
            blob_client.delete_blob()
        except:
            pass

        #  need to specify html content type so that it will render and not download
        with open(executed_html_path, "rb") as data:
            blob_client.upload_blob(
                data, content_settings=ContentSettings(content_type='text/html')
            )
        print(
            f'**** Your notebook is hosted here! *****\nhttps://cmip6downscaling.z6.web.core.windows.net/{blob_name}'
        )
    return executed_notebook_path


def load_top_cities(
    num_cities: int = 100, plot: bool = False, add_additional_cities: bool = True
) -> pd.DataFrame:
    """Load in the biggest (by population) `num_cities` (default 100) cities in the world,
    when limiting it to one in each country. This is a way to get
    a sense of performance in individual pixels where many people live but
    also have geographic diversity.


    Parameters
    ----------
    num_cities : int, optional
        How many cities to evaluate, by default 100
    plot : bool, optional
        Whether or not you want to return a plot of the
         cities, by default False
    add_additional_cities : bool, optional
        Whether you want to add 8 additional cities that
        help get you better spatial coverage,
        particularly in US, if num_cities=100

    Returns
    -------
    pd.DataFrame
        Dataframe with columns ['city', 'lat', 'lng']
    """

    cities = pd.read_csv('https://cmip6downscaling.blob.core.windows.net/cmip6/worldcities.csv')
    top_cities = (
        cities.sort_values('population', ascending=False)
        .groupby('country')
        .first()
        .sort_values('population', ascending=False)[0:num_cities][['city', 'lat', 'lng']]
    )
    if add_additional_cities:
        additional_cities = [
            'Seattle',
            'Los Angeles',
            'Denver',
            'Chicago',
            'Anchorage',
            'Perth',
            'Paramaribo',
            'Fortaleza',
        ]
        for additional_city in additional_cities:
            top_cities = top_cities.append(
                cities[cities['city'] == additional_city][['city', 'lat', 'lng']]
            )
    if plot:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.stock_img()
        for (lat, lon) in top_cities[['lat', 'lng']].values:
            plt.plot(
                lon,
                lat,
                color='blue',
                marker='o',
            )
    return top_cities


def select_points(ds: xr.Dataset, top_cities: pd.DataFrame) -> xr.Dataset:
    """Select the nearest pixels to each of the cities listed in `top_cities`


    Parameters
    ----------
    ds : xr.Dataset
        Dataset including dimensions ['lat', 'lon']
    top_cities: pd.DataFrame
        Dataframe with columns ['city', 'lat', 'lng']

    Returns
    -------
    xr.Dataset
        Dataset with dimension 'city' (and maybe others like 'time')
        with coordinate labels of the cities from `top_cities`
    """
    cities = ds.sel(
        lat=xr.DataArray(top_cities.lat.values, dims='cities'),
        lon=xr.DataArray(top_cities.lng.values, dims='cities'),
        method='nearest',  # you can't apply tolerance in 2d selections like this.
        # as a result, if you're running a subset the data won't be valid for cities
        # that aren't in the subset (but it will still return non-nan values)
    )
    cities = cities.assign_coords({'cities': top_cities.city.values})
    return cities


def grab_top_city_data(ds_list: list, top_cities: pd.DataFrame) -> list:
    """Given a list of datasets, perform the `select_points` operation on each of them.

    Parameters
    ----------
    ds_list : list
        List of xr.Datasets
    top_cities: pd.DataFrame
        Dataframe with columns ['city', 'lat', 'lng']

    Returns
    -------
    list
        List of datasets with dimension ['city']
    """
    city_ds_list = []
    for ds in ds_list:
        city_ds_list.append(select_points(ds, top_cities).compute())
    return city_ds_list


def get_seasonal(ds: xr.Dataset, aggregator: str = 'mean') -> xr.Dataset:
    """Aggregate to seasonal
    Parameters
    ----------
    ds : xr.Dataset
        dataset with climate on at least seasonal basis (but likely daily)
    aggregator: str
        kind of aggregation you want to do (e.g. mean, max, min, stdev)

    Returns
    -------
    xr.Dataset
        Dataset collapsed along the time dimension into a seasonally
        aggregated dataset
    """
    return getattr(ds.groupby('time.season'), aggregator)()


def change_ds(
    ds_historic: xr.Dataset,
    ds_future: xr.Dataset,
    metrics: list = ['mean', 'std', 'percentile1', 'percentile5', 'percentile95', 'percentile99'],
) -> xr.Dataset:
    """Calculate change in a variety of metrics between a historic and future period

    Parameters
    ----------
    ds_historic : xr.Dataset
        Historical climate with dimension 'time'
    ds_future : xr.Dataset
        Future climate with dimension 'time'
    metrics : list, optional
        List of metrics you want to calculate, by default ['mean', 'std', 'percentile1', 'percentile5', 'percentile95', 'percentile99']

    Returns
    -------
    xr.Dataset
        Dataset with changes for the metrics listed in `metrics`
    """
    ds = xr.Dataset()
    for metric in metrics:
        if metric == 'mean':
            change = ds_future.mean(dim='time') - ds_historic.mean(dim='time')
        elif metric == 'median':
            change = ds_future.median(dim='time') - ds_historic.median(dim='time')
        elif metric == 'std':
            change = ds_future.std(dim='time') - ds_historic.std(dim='time')
        elif 'percentile' in metric:
            # parse the percentile
            percentile = float(metric.split('percentile')[1]) / 100
            # wet_day = wet_day.chunk({'time': -1})
            change = ds_future.quantile(percentile, dim='time') - ds_historic.quantile(
                percentile, dim='time'
            )
        ds[metric] = change
    return ds
