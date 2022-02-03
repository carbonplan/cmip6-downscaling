import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xarray as xr
from carbonplan import styles

from cmip6_downscaling.workflows.paths import (
    make_bcsd_output_path,
    make_bias_corrected_path,
    make_coarse_obs_path,
    make_gcm_predict_path,
    make_rechunked_gcm_path,
    make_return_obs_path,
    make_spatial_anomalies_path,
)

styles.mpl.set_theme(style='carbonplan_light')


def plot_cdfs(
    obs_ds: xr.Dataset,
    historical_downscaled: xr.Dataset,
    future_downscaled: xr.Dataset,
    top_cities: pd.DataFrame,
    training_period: slice,
    future_period: slice,
    historical_gcm: xr.Dataset = None,
    future_gcm: xr.Dataset = None,
    ncols: int = 4,
    sharex: bool = True,
) -> mpl.figure.Figure:
    """Plot cdfs of individual pixels

    Parameters
    ----------
    obs_ds : xr.Dataset
        observed dataset with dimensions ('time', 'city')
    historical_downscaled : xr.Dataset
        historical dataset with dimensions ('time', 'city')
    future_downscaled : xr.Dataset
        future downscaled dataset with dimensions ('time', 'city')
    top_cities : pd.DataFrame
        dataframe with cities and their locations ('lat', 'lng')
    training_period : slice
        training period, likely something like slice('1980', '2010')
    future_period : slice
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
        nrows=(int(len(obs_ds.cities) / ncols)) + 1,
        figsize=(15, 60),
        sharey=True,
        sharex=sharex,
    )
    for i, city in enumerate(top_cities.city.values):
        ax = axarr.reshape(-1)[i]
        sns.ecdfplot(
            data=obs_ds.isel(cities=i),
            label=f'ERA5 ({training_period.start}-{training_period.stop})',
            ax=ax,
            color='#1b1e23',
        )
        sns.ecdfplot(
            data=historical_downscaled.isel(cities=i),
            label=f'Downscaled GCM ({training_period.start}-{training_period.stop})',
            color='#8b9fd1',
            ax=ax,
        )
        sns.ecdfplot(
            data=future_downscaled.isel(cities=i),
            label=f'Downscaled GCM ({future_period.start}-{future_period.stop})',
            ax=ax,
            color='#f16f71',
        )
        if historical_gcm is not None:
            sns.ecdfplot(
                data=historical_gcm.isel(cities=i),
                label=f'Raw GCM ({training_period.start}-{training_period.stop})',
                ax=ax,
                color='#8b9fd1',
                linestyle='--',
            )
        if future_gcm is not None:
            sns.ecdfplot(
                data=future_gcm.isel(cities=i),
                label=f'Raw GCM ({future_period.start}-{future_period.stop})',
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
            'mean': 0.25,
            'std': 1,
            'percentile1': 1,
            'percentile5': 1,
            'percentile95': 2,
            'percentile99': 2,
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
            vmin=-diff_limit,
            vmax=diff_limit,
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


def plot_each_step_bcsd(
    gcm_identifier: str,
    obs_identifier: str,
    result_dir: str,
    intermediate_dir: str,
    train_period: slice,
    var: str,
) -> mpl.figure.Figure:
    """Plot the training period mean of each intermediary and output file in the
     bcsd process. For the spatial anomalies it just plots the first month.

    Parameters
    ----------
    gcm_identifier : str
        unique identifier for a run
    obs_identifier : str
        unique identifier for the obs used in training
    result_dir : str
        Location of the final results
    intermediate_dir : str
        Location of intermediate files
    train_period : slice
        Period used for training
    var : str
        Variable of interest

    Returns
    -------
    mpl.figure.Figure
        Figure
    """
    steps = [
        make_return_obs_path(obs_identifier),
        make_coarse_obs_path(obs_identifier),
        make_spatial_anomalies_path(obs_identifier),
        make_rechunked_gcm_path(gcm_identifier),
        make_gcm_predict_path(gcm_identifier),
        make_bias_corrected_path(gcm_identifier),
        make_bcsd_output_path(gcm_identifier),
    ]
    fig, axarr = plt.subplots(ncols=len(steps), figsize=(20, 3))
    for i, path in enumerate(steps):
        prefix = path.split('/')[0]
        print(prefix)

        if prefix == 'bcsd_output':
            data_location = result_dir
        else:
            data_location = intermediate_dir
        ds = xr.open_zarr('/'.join([data_location, path]))
        if prefix == 'spatial_anomalies':
            ds[var].isel(month=0).plot(ax=axarr[i])
        else:
            ds[var].sel(time=train_period).mean(dim='time').plot(ax=axarr[i])
        axarr[i].set_title(prefix)
    plt.tight_layout()
    plt.close()
    return fig
