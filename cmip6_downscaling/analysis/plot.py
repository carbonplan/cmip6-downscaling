from __future__ import annotations

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xarray as xr
from carbonplan import styles

styles.mpl.set_theme(style='carbonplan_light')


def plot_cdfs(
    obs: xr.Dataset,
    top_cities: pd.DataFrame,
    train_period: slice,
    predict_period: slice,
    historical_downscaled: xr.Dataset = None,
    future_downscaled: xr.Dataset = None,
    historical_gcm: xr.Dataset = None,
    future_gcm: xr.Dataset = None,
    ncols: int = 4,
    sharex: bool = True,
) -> mpl.figure.Figure:
    """Plot cdfs of individual pixels

    Parameters
    ----------
    obs : xr.Dataset
        observed dataset with dimensions ('time', 'city')
    historical_downscaled : xr.Dataset
        historical dataset with dimensions ('time', 'city'), by default None
    future_downscaled : xr.Dataset
        future downscaled dataset with dimensions ('time', 'city'), by default None
    top_cities : pd.DataFrame
        dataframe with cities and their locations ('lat', 'lng')
    train_period : slice
        training period, likely something like slice('1980', '2010')
    predict_period : slice
        future period, likely something like slice('1980', '2010')
    historical_gcm : xr.Dataset, optional
        raw historical gcm with dimensions ('time', 'city'), by default None
    future_gcm : xr.Dataset, optional
        raw future gcm dataset with dimensions ('time', 'city'), by default None
    ncols : int, optional
        how many cols you want in your figure, by default 4
    sharex : bool, optional
        whether to force plots to share x-axis, when True it helps with cross-domain
        comparisons, when false it supports QAQC checking for individual locations,
        by default True

    Returns
    -------
    mpl.figure.Figure
        Figure
    """
    fig, axarr = plt.subplots(
        ncols=ncols,
        nrows=(int(len(obs.cities) / ncols)) + 1,
        figsize=(15, 60),
        sharey=True,
        sharex=sharex,
    )
    for i, city in enumerate(top_cities.city.values):
        ax = axarr.reshape(-1)[i]
        sns.ecdfplot(
            data=obs.isel(cities=i),
            label=f'ERA5 ({train_period.start}-{train_period.stop})',
            ax=ax,
            color='#1b1e23',
        )
        if historical_downscaled is not None:
            sns.ecdfplot(
                data=historical_downscaled.isel(cities=i),
                label=f'Downscaled GCM ({train_period.start}-{train_period.stop})',
                color='#8b9fd1',
                ax=ax,
            )
        if future_downscaled is not None:
            sns.ecdfplot(
                data=future_downscaled.isel(cities=i),
                label=f'Downscaled GCM ({predict_period.start}-{predict_period.stop})',
                ax=ax,
                color='#f16f71',
            )
        if historical_gcm is not None:
            sns.ecdfplot(
                data=historical_gcm.isel(cities=i),
                label=f'Raw GCM ({train_period.start}-{train_period.stop})',
                ax=ax,
                color='#8b9fd1',
                linestyle='--',
            )
        if future_gcm is not None:
            sns.ecdfplot(
                data=future_gcm.isel(cities=i),
                label=f'Raw GCM ({predict_period.start}-{predict_period.stop})',
                ax=ax,
                color='#f16f71',
                linestyle='--',
            )

        ax.set_title(city)
    plt.legend()
    plt.close()
    return fig


def plot_values_and_difference(
    ds1: xr.Dataset,
    ds2: xr.Dataset,
    diff_limit: float = 10.0,
    cbar_kwargs: dict = {},
    title1: str = '',
    title2: str = '',
    title3: str = '',
    city_coords: pd.DataFrame = None,
    variable: str = '',
    metric: str = 'mean',
) -> mpl.figure.Figure:
    """Plot two datasets and their difference

    Parameters
    ----------
    ds1 : xr.Dataset
        Dataset with dimensions ('lat', 'lon)
    ds2 : xr.Dataset
        Dataset with dimensions ('lat', 'lon)
    diff_limit : float, optional
        The upper/lower bound for the difference plot cmap, by default 10.0
    cbar_kwargs : dict, optional
        passing kwargs to the cbar, by default {}
    title1 : str, optional
        title for panel 1, by default ''
    title2 : str, optional
        title for panel 2, by default ''
    title3 : str, optional
        title for panel 3, by default ''
    city_coords : pd.DataFrame, optional
        if defined, plot will be a scatter plot of locations in `city_coords`, by default None
    variable : str, optional
        name of the variable if you want to add it to plot, by default ''
    metric : str, optional
        metric of interest, helps to tailor the variable limits, by default 'mean'

    Returns
    -------
    Tuple[mpl.figure.Figure, mpl.axes.Axes]
        Figure and axes
    """
    fig, axarr = plt.subplots(
        ncols=3, figsize=(24, 3), subplot_kw={'projection': ccrs.PlateCarree()}
    )
    var_limit = {
        'tasmax': {
            'mean': 0.1,
            'median': 2,
            'std': 3,
            'percentile1': 5,
            'percentile5': 2,
            'percentile95': 2,
            'percentile99': 5,

        }
    }

    if city_coords is not None:
        plot1 = axarr[0].scatter(
            x=city_coords.lon,
            y=city_coords.lat,
            c=ds1,
            cmap="fire_light",
            transform=ccrs.PlateCarree(),
        )
        fig.colorbar(plot1, ax=axarr[0]).set_label(f'{variable}')

        plot2 = axarr[1].scatter(
            x=city_coords.lon,
            y=city_coords.lat,
            c=ds2,
            cmap="fire_light",
            transform=ccrs.PlateCarree(),
        )
        fig.colorbar(plot2, ax=axarr[1]).set_label(f'{variable}')

        diff = axarr[2].scatter(
            x=city_coords.lon,
            y=city_coords.lat,
            c=(ds2 - ds1),
            cmap="orangeblue_light_r",
            transform=ccrs.PlateCarree(),
            vmin=-var_limit[variable][metric],
            vmax=var_limit[variable][metric],
        )
        fig.colorbar(diff, ax=axarr[2]).set_label(f'{variable}')
    else:
        ds1.plot(ax=axarr[0], cmap='fire_light', cbar_kwargs=cbar_kwargs)
        ds2.plot(ax=axarr[1], cmap='fire_light', cbar_kwargs=cbar_kwargs)
        (ds2 - ds1).plot(
            ax=axarr[2],
            cmap='orangeblue_light_r',
            vmin=-var_limit[variable][metric],
            vmax=var_limit[variable][metric],
            cbar_kwargs={'label': 'Difference (middle - left)'},
        )

    axarr[0].set_title(title1)
    axarr[1].set_title(title2)

    for ax in axarr:
        ax.coastlines()
    return fig


def plot_seasonal(ds1: xr.Dataset, ds2: xr.Dataset) -> mpl.figure.Figure:
    """Plot the seasonality of two datasets and the difference between them

    Parameters
    ----------
    ds1 : xr.Dataset
        Dataset of dimensions ('lat', 'lon', 'season')
    ds2 : xr.Dataset
        Dataset of dimensions ('lat', 'lon', 'season')

    Returns
    -------
    mpl.figure.Figure
        Figure
    """
    fig, axarr = plt.subplots(
        ncols=4, nrows=3, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(10, 6)
    )
    seasons = ['DJF', 'JJA', 'MAM', 'SON']
    col_ds_list = [ds1, ds2, ds2 - ds1]
    cmaps = ['fire_light', 'fire_light', 'orangeblue_light_r']

    for j, ds in enumerate(col_ds_list):
        for i, season in enumerate(seasons):
            ds.sel(season=season).plot(ax=axarr[j, i], cmap=cmaps[j])
            axarr[j, i].coastlines()
    plt.tight_layout()
    plt.close()
    return fig
