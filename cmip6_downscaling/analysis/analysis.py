from .qaqc import make_qaqc_ds
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

analysis_dir = f"az://cmip6/results/analysis"

def qaqc_checks(ds):
    # run_qaqc_dir =  f"{analysis_dir}/qaqc/{run_id}"
    qaqc_ds = make_qaqc_ds(ds)
    annual_qaqc_ts = qaqc_ds.groupby('time.year').sum().sum(dim=['lat', 'lon']
                                            ).to_dataframe()
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
            out_ds[var] = ds[var].groupby('time.month').mean()
        elif var in ['pr']:
            out_ds[var] = ds[var].groupby('time.month').sum()
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
            out_ds[var] = ds[var].groupby('time.year').mean()
        elif var in ['pr']:
            out_ds[var] = ds[var].groupby('time.year').sum()
        else:
            print(f'{var} not implemented')

    return out_ds

def run_analysis_notebook(gcm,
    scenario,
    train_period_start,
    train_period_end,
    predict_period_start,
    predict_period_end,
    variable,
    latmin, 
    latmax,
    lonmin, 
    lonmax):
    """Create a jupyter notebook with analyses about the run with 
    parameters specified here.

    Parameters
    ----------
    gcm : str
        Input GCM
    scenario: str
        Input GCM scenario
    train_period_start : str
        Date for training period start (e.g. '1985')
    train_period_end : str
        Date for training period end (e.g. '2015')
    predict_period_start : str
        Date for prediction period start (e.g. '2090')
    predict_period_end : str
        Date for prediction period end (e.g. '2090')
    variable: str
        The variable included in the dataset.
    latmin : str
        Minimum latitude in domain
    latmax : str
        Maximum latitude in domain
    lonmin : str
        Minimum longitude in domain
    lonmax : str
        Maximum longitude in domain
    """
    run_id = f'{gcm}-{scenario}-{train_period_start}-{train_period_end}-{predict_period_start}-{predict_period_end}-{latmin}-{latmax}-{lonmin}-{lonmax}-{variable}.zarr'
    pm.execute_notebook(
        '/home/jovyan/cmip6-downscaling/notebooks/analyses.ipynb',
        f'/home/jovyan/cmip6-downscaling/notebooks/output/analyses_{run_id}.ipynb',
        parameters={'run_id': run_id}
        )


def load_top_cities(plot=False):
    cities = pd.read_csv('worldcities.csv')
    top_cities = cities.sort_values('population', 
                                ascending=False).groupby('country').first(
                        ).sort_values('population', ascending=False)[0:100][['city', 'lat', 'lng']]
    additional_cities = ['Seattle', 'Los Angeles', 'Denver', 'Chicago', 'Anchorage', 'Perth', 'Paramaribo', 'Fortaleza']
    for additional_city in additional_cities:
        top_cities = top_cities.append(cities[cities['city']==additional_city][['city', 'lat', 'lng']])
    if plot:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.stock_img()
        for (lat, lon) in top_cities[['lat', 'lng']].values:
            plt.plot(lon, lat, color='blue', marker='o',)
    return top_cities

def select_points(ds, top_cities):
    return ds.sel(lat=xr.DataArray(top_cities.lat.values, dims='cities'), 
                lon=xr.DataArray(top_cities.lng.apply(convert_to_360).values, dims='cities'), 
                method='nearest')

def grab_top_city_data(obs_ds, downscaled_ds, top_cities):
    obs = select_points(obs_ds, top_cities).compute()
    downscaled_ds = select_points(downscaled_ds, top_cities).compute()
    return obs, downscaled_ds

def get_seasonal(ds, aggregator='mean'):
    if aggregator=='mean':
        return ds.groupby('time.season').mean()
    elif aggregator=='stdev':
        return ds.groupby('time.season').std()
    elif aggregator=='min':
        return ds.groupby('time.season').min()
    elif aggregator=='max':
        return ds.groupby('time.season').max()
    else:
        raise NotImplementedError

