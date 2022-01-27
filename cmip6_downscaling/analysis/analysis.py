import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import papermill as pm
import xarray as xr
from prefect import task
from cmip6_downscaling.data.cmip import convert_to_360

from .qaqc import make_qaqc_ds


def qaqc_checks(ds):
    # run_qaqc_dir =  f"{analysis_dir}/qaqc/{run_id}"
    qaqc_ds = make_qaqc_ds(ds)
    annual_qaqc_ts = qaqc_ds.groupby('time.year').sum().sum(dim=['lat', 'lon']).to_dataframe()
    qaqc_maps = qaqc_ds.sum(dim='time')
    return annual_qaqc_ts, qaqc_maps


def monthly_summary(ds):
    """Creates an monthly summary dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Downscaled dataset
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


def annual_summary(ds):
    """Creates an annual summary dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Downscaled dataset
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
def run_analyses(parameters):
    print('wooohooo')    
    # pm.execute_notebook(
    #     'analyses.ipynb', 'analyses_{}.ipynb'.format(parameters['run_id']), parameters=parameters
    # )
    return 'i love donuts' # TODO: path to analysis notebook

def load_top_cities(plot=False):
    cities = pd.read_csv('worldcities.csv')
    top_cities = (
        cities.sort_values('population', ascending=False)
        .groupby('country')
        .first()
        .sort_values('population', ascending=False)[0:100][['city', 'lat', 'lng']]
    )
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


def select_points(ds, top_cities):
    cities = ds.sel(
        lat=xr.DataArray(top_cities.lat.values, dims='cities'),
        lon=xr.DataArray(top_cities.lng.apply(convert_to_360).values, dims='cities'),
        method='nearest',
    )
    cities = cities.assign_coords({'cities': top_cities.city.values})
    return cities


def grab_top_city_data(ds_list, top_cities):
    city_ds_list = []
    for ds in ds_list:
        city_ds_list.append(select_points(ds, top_cities).compute())
    return city_ds_list


def get_seasonal(ds, aggregator='mean'):
    if aggregator == 'mean':
        return ds.groupby('time.season').mean()
    elif aggregator == 'stdev':
        return ds.groupby('time.season').std()
    elif aggregator == 'min':
        return ds.groupby('time.season').min()
    elif aggregator == 'max':
        return ds.groupby('time.season').max()
    else:
        raise NotImplementedError


def change_ds(
    ds_historic,
    ds_future,
    metrics=['mean', 'std', 'percentile1', 'percentile5', 'percentile95', 'percentile99'],
):
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
