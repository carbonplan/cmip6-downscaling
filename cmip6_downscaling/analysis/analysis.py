from __future__ import annotations

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from .qaqc import make_qaqc_ds


def qaqc_checks(ds: xr.Dataset, checks: list) -> tuple[xr.Dataset, xr.Dataset]:
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
    qaqc_ds = make_qaqc_ds(ds, checks)
    annual_qaqc_ts = qaqc_ds.groupby('time.year').sum().sum(dim=['lat', 'lon']).to_dataframe()
    qaqc_maps = qaqc_ds.sum(dim='time')
    return annual_qaqc_ts, qaqc_maps


def load_big_cities(
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

    cities = pd.read_csv('https://cmip6downscaling.blob.core.windows.net/static/worldcities.csv')
    big_cities = (
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
            'Paramaribo',
            'Fortaleza',
        ]
        for additional_city in additional_cities:
            big_cities = pd.concat(
                [big_cities, cities[cities['city'] == additional_city][['city', 'lat', 'lng']]]
            )

    if plot:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.stock_img()
        for (lat, lon) in big_cities[['lat', 'lng']].values:
            plt.plot(
                lon,
                lat,
                color='blue',
                marker='o',
            )
    return big_cities


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


def grab_big_city_data(ds_list: list, big_cities: pd.DataFrame) -> list:
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
        city_ds_list.append(select_points(ds, big_cities).compute())
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


def change_ds(ds_historic: xr.Dataset, ds_future: xr.Dataset, metrics: list = None) -> xr.Dataset:
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
    if metrics is None:
        metrics = [
            'mean',
            'std',
            'percentile1',
            'percentile5',
            'percentile95',
            'percentile99',
        ]

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
