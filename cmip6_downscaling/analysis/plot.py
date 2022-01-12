import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from carbonplan import styles


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


def plot_cdfs(obs_ds, historical_downscaled, future_downscaled, top_cities, training_period, future_period):
    fig, axarr = plt.subplots(nrows=22, ncols=5, figsize=(15,60), sharey=True, sharex=True)
    for i, city in enumerate(top_cities.city.values):
        ax = axarr.reshape(-1)[i]
        sns.ecdfplot(data=obs.isel(cities=i), label=f'ERA5 ({training_period.start}-{training_period.stop})', ax=ax)
        sns.ecdfplot(data=historical_downscaled.isel(cities=i), label=f'Downscaled GCM ({training_period.start}-{training_period.stop})', ax=ax)
        sns.ecdfplot(data=future_downscaled.isel(cities=i), label=f'Downscaled GCM ({future_period.start}-{future_period.stop})', ax=ax)
        ax.set_title(city)
    plt.legend()


def plot_values_and_difference(ds1, ds2, plot_diff=True, diff_min=-10, diff_max=10, cbar_kwargs={}, title1='', title2=''):
    fig, axarr = plt.subplots(ncols=3, figsize=(24,3), subplot_kw={'projection': ccrs.PlateCarree()})
    ds1.plot(ax=axarr[0], cmap='fire_light', cbar_kwargs=cbar_kwargs)
    axarr[0].set_title(title1)
    ds2.plot(ax=axarr[1], cmap='fire_light', cbar_kwargs=cbar_kwargs)
    axarr[1].set_title(title2)

    (ds2 - ds1).plot(ax=axarr[2], cmap='orangeblue_light_r', vmin=diff_min, vmax=diff_max, cbar_kwargs={'label': 'Difference (middle - left)'})
    for ax in axarr:
        ax.coastlines()

def plot_seasonal(ds1, ds2):
    fig, axarr = plt.subplots(ncols=4, nrows=3, subplot_kw={'projection': ccrs.PlateCarree()})
    for i, season in enumerate(['DJF', 'JJA', 'MAM', 'SON']):
        ds1.sel(season=season).plot(ax=axarr[0,i], cmap='fire_light')
    for i, season in enumerate(['DJF', 'JJA', 'MAM', 'SON']):
        ds2.sel(season=season).plot(ax=axarr[1,i], cmap='fire_light')        
    for i, season in enumerate(['DJF', 'JJA', 'MAM', 'SON']):
        (ds2-ds1).sel(season=season).plot(ax=axarr[2,i], cmap='orangeblue_light_r')
    for ax in axarr.reshape(-1):
        ax.coastlines()