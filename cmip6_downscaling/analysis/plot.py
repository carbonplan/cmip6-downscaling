import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from carbonplan import styles
import xarray as xr
import seaborn as sns


def plot_time_mean(ds1, ds2, diff=True, limits_dict=None):

    vars_to_plot = list(ds1.data_vars.keys())

    fig, axarr = plt.subplots(ncols=3, nrows=len(vars_to_plot))

    for i, var in enumerate(vars_to_plot):
        if var in list(limits_dict.keys()):
            vmin = limits_dict[var]['vmin']
            vmax = limits_dict[var]['vmax']
        else:
            vmin, vmax = None, None
        ds2[var].plot(ax=axarr[i, 0], vmin=vmin, vmax=vmax)
        ds2[var].plot(ax=axarr[i, 1], vmin=vmin, vmax=vmax)
        (ds2 - ds1)[var].plot(ax=axarr[i, 2])

    return fig, axarr


def plot_seasonal_mean(ds1, ds2, limits_dict=None):

    vars_to_plot = list(ds1.data_vars.keys())

    fig, axarr = plt.subplots(nrows=len(vars_to_plot))

    for i, var in enumerate(vars_to_plot):
        if var in list(limits_dict.keys()):
            vmin = limits_dict[var]['vmin']
            vmax = limits_dict[var]['vmax']
        else:
            vmin, vmax = None, None
        ds2[var].plot(ax=axarr[i], vmin=vmin, vmax=vmax)
        ds2[var].plot(ax=axarr[i], vmin=vmin, vmax=vmax)

    return fig, axarr


def plot_cdfs(obs_ds, 
                historical_downscaled, 
                future_downscaled,
                top_cities, 
                training_period, 
                future_period,
                historical_gcm: xr.Dataset = None,
                future_gcm: xr.Dataset = None,
                ncols: int = 4,
                sharex: bool = True):
    fig, axarr = plt.subplots(ncols=ncols, nrows=(int(len(obs_ds.cities)/ncols))+1, figsize=(15,60), sharey=True, sharex=sharex)
    for i, city in enumerate(top_cities.city.values):
        ax = axarr.reshape(-1)[i]
        sns.ecdfplot(data=obs_ds.isel(cities=i), label=f'ERA5 ({training_period.start}-{training_period.stop})', ax=ax, color='#1b1e23')
        sns.ecdfplot(data=historical_downscaled.isel(cities=i), 
                    label=f'Downscaled GCM ({training_period.start}-{training_period.stop})', 
                    color='#8b9fd1',
                    ax=ax)
        sns.ecdfplot(data=future_downscaled.isel(cities=i), label=f'Downscaled GCM ({future_period.start}-{future_period.stop})', ax=ax,
        color='#f16f71')
        if historical_gcm is not None:
            sns.ecdfplot(data=historical_gcm.isel(cities=i), label=f'Raw GCM ({training_period.start}-{training_period.stop})', ax=ax, color='#8b9fd1', linestyle='--')
        if future_gcm is not None:
            sns.ecdfplot(data=future_gcm.isel(cities=i), label=f'Raw GCM ({future_period.start}-{future_period.stop})', ax=ax, color='#f16f71', linestyle='--')

        ax.set_title(city)
    plt.legend()


def plot_values_and_difference(ds1, ds2, plot_diff=True, diff_limit=10, 
                            cbar_kwargs={}, title1='', title2='', title3='', city_coords=None, variable='', metric='mean'):
    fig, axarr = plt.subplots(ncols=3, figsize=(24,3), subplot_kw={'projection': ccrs.PlateCarree()})
    var_limit = {'tasmax': {'mean': 0.25,
                'std': 1,
                'percentile1': 1,
                'percentile5': 1,
                'percentile95': 2,
                'percentile99': 2}}

    if city_coords is not None:
        plot1 = axarr[0].scatter(x=city_coords.lon, y=city_coords.lat,
            c=ds1,
            cmap="fire_light", 
            transform=ccrs.PlateCarree())
        fig.colorbar(plot1, ax=axarr[0]).set_label(f'{variable}')

        plot2 = axarr[1].scatter(x=city_coords.lon, y=city_coords.lat,
            c=ds2,
            cmap="fire_light", 
            transform=ccrs.PlateCarree())
        fig.colorbar(plot2, ax=axarr[1]).set_label(f'{variable}')

        diff = axarr[2].scatter(x=city_coords.lon, y=city_coords.lat,
            c=(ds2-ds1),
            cmap="orangeblue_light_r",
            transform=ccrs.PlateCarree(), vmin=-var_limit[variable][metric],
                                        vmax=var_limit[variable][metric])
        fig.colorbar(diff, ax=axarr[2]).set_label(f'{variable}')
    else:
        ds1.plot(ax=axarr[0], cmap='fire_light', cbar_kwargs=cbar_kwargs)
        ds2.plot(ax=axarr[1], cmap='fire_light', cbar_kwargs=cbar_kwargs)
        (ds2 - ds1).plot(ax=axarr[2], cmap='orangeblue_light_r', vmin=-diff_limit, vmax=diff_limit, cbar_kwargs={'label': 'Difference (middle - left)'})


    axarr[0].set_title(title1)
    axarr[1].set_title(title2)

    for ax in axarr:
        ax.coastlines()

def plot_seasonal(ds1, ds2):
    fig, axarr = plt.subplots(ncols=4, nrows=3, 
                subplot_kw={'projection': ccrs.PlateCarree()},
                figsize=(10,6))
    for i, season in enumerate(['DJF', 'JJA', 'MAM', 'SON']):
        ds1.sel(season=season).plot(ax=axarr[0,i], cmap='fire_light')
    for i, season in enumerate(['DJF', 'JJA', 'MAM', 'SON']):
        ds2.sel(season=season).plot(ax=axarr[1,i], cmap='fire_light')        
    for i, season in enumerate(['DJF', 'JJA', 'MAM', 'SON']):
        (ds2-ds1).sel(season=season).plot(ax=axarr[2,i], cmap='orangeblue_light_r')
    for ax in axarr.reshape(-1):
        ax.coastlines()
    plt.tight_layout()