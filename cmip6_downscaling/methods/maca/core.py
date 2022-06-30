from __future__ import annotations

import numpy as np
import xarray as xr
import xesmf as xe
from skdownscale.pointwise_models import EquidistantCdfMatcher, PointWiseDownscaler
from sklearn.linear_model import LinearRegression

from .utils import add_circular_temporal_pad, generate_batches, pad_with_edge_year


def epoch_trend(
    data: xr.Dataset,
    historical_period: slice,
    day_rolling_window: int = 21,
    year_rolling_window: int = 31,
) -> xr.Dataset:
    """
    Calculate the epoch trend as a multi-day, multi-year rolling average. The trend is calculated as the anomaly
    against the historical mean (also a multi-day rolling average).

    Parameters
    ----------
    data : xr.Dataset
        Data to calculate epoch trend on
    historical_period : slice
        Slice indicating the historical period to be used when calculating historical mean
    day_rolling_window : int
        Number of days to include when calculating the rolling average
    year_rolling_window : int
        Number of years to include when calculating the rolling average

    Returns
    -------
    trend : xr.Dataset
        The long term average trend
    """
    # the rolling windows need to be odd numbers since the rolling average is centered
    assert day_rolling_window % 2 == 1
    d_offset = int((day_rolling_window - 1) / 2)

    assert year_rolling_window % 2 == 1
    y_offset = int((year_rolling_window - 1) / 2)

    # get historical mean as a rolling average -- the result has one number for each day of year
    # which is the average of the neighboring day of years over multiple years
    padded = add_circular_temporal_pad(data=data.sel(time=historical_period), offset=d_offset)
    hist_mean = (
        padded.rolling(time=day_rolling_window, center=True)
        .mean()
        .dropna('time')
        .groupby("time.dayofyear")
        .mean()
    )

    # get rolling average for the entire data -- the result has one number for each day in the entire time series
    # which is the average of the neighboring day of years in neighboring years
    padded = add_circular_temporal_pad(data=data, offset=d_offset)

    def func(x):
        window = min(len(x.time), year_rolling_window)
        return x.rolling(time=window, center=True).mean()

    rolling_doy_mean = (
        padded.rolling(time=day_rolling_window, center=True)
        .mean()
        .dropna('time')
        .groupby('time.dayofyear')
        .apply(func)
        .dropna('time')
    )

    # repeat the first/last year to cover the time periods without enough data for the rolling average
    for i in range(y_offset):
        rolling_doy_mean = pad_with_edge_year(rolling_doy_mean)

    # remove historical mean from rolling average, leaving trend as the anamoly
    trend = rolling_doy_mean.groupby('time.dayofyear') - hist_mean

    return trend


def get_doy_mask(
    source_doy: xr.DataArray,
    target_doy: xr.DataArray,
    doy_range: int = 45,
) -> xr.DataArray:
    """
    Given two 1D dataarrays containing day of year informations: source_doy and target_doy , return a matrix of shape
    len(target_doy) x len(source_doy). Cell (i, j) is True if the source doy j is within doy_range days of the target
    doy i, and False otherwise

    Parameters
    ----------
    source_doy : xr.DataArray
        1D xr data array with day of year information
    target_doy : xr.DataArray
        1D xr data array with day of year information

    Returns
    -------
    mask : xr.DataArray
        2D xr data array of boolean type in the shape of len(target_doy) x len(source_doy)
    """
    # get the min and max doy within doy_range days to target doy
    target_doy_min = target_doy - doy_range
    target_doy_max = target_doy + doy_range
    # make sure the range is within 0-365
    # TODO: what to do with leap years???
    target_doy_min[target_doy_min <= 0] += 365
    target_doy_max[target_doy_max > 365] -= 365

    # if the min is larger than max, the target doy is at the edge of a year, and we can accept the
    # source if any one of the condition is True
    one_sided = target_doy_min > target_doy_max
    edges = ((source_doy >= target_doy_min) | (source_doy <= target_doy_max)) & (one_sided)
    # otherwise the source doy needs to be within min and max
    within = (source_doy >= target_doy_min) & (source_doy <= target_doy_max) & (~one_sided)

    # mask is true if either one of the condition is satisfied
    mask = edges | within

    return mask


def bias_correction(
    ds_gcm: xr.Dataset,
    ds_obs: xr.Dataset,
    variables: list[str],
    batch_size: int = 15,
    buffer_size: int = 15,
) -> xr.Dataset:
    """
    Run bias correction as it is done in the MACA method.

    The bias correction is performed using the Equidistant CDF matching method in batches.
    Neighboring day of years are bias corrected together with a buffer. That is, with a batch
    size of 15 and a buffer size of 15, the 45 neighboring days of year are bias corrected
    together, but only the result of the center 15 days are used. The historical GCM is mapped
    to historical coarsened observation in the bias correction.

    Parameters
    ----------
    ds_gcm : xr.Dataset
        GCM dataset, must have a dimension called time on which we can call .dt.dayofyear on
    ds_obs : xr.Dataset
        Observation dataset, must have a dimension called time on which we can call .dt.dayofyear on
    variables : List[str]
        Names of the variables used in obs and gcm dataset (including features and label)
    batch_size : Optional[int]
        The batch size in terms of day of year to bias correct together
    buffer_size : Optional[int]
        The buffer size in terms of day of year to include in the bias correction

    Returns
    -------
    ds_out : xr.Dataset
        The bias corrected dataset

    See Also
    --------
    https://climate.northwestknowledge.net/MACA/MACAmethod.php
    """

    # map_blocks work around
    if 't2' in ds_obs.coords:
        ds_obs = ds_obs.rename({'t2': 'time'})

    if isinstance(variables, str):
        variables = [variables]

    doy_gcm = ds_gcm.time.dt.dayofyear
    doy_obs = ds_obs.time.dt.dayofyear

    ds_out = xr.Dataset()
    for var in variables:
        if var in ['pr', 'huss', 'vas', 'uas']:
            kind = 'ratio'
        else:
            kind = 'difference'

        bias_correction_model = PointWiseDownscaler(
            EquidistantCdfMatcher(
                kind=kind, extrapolate=None  # cdf in maca implementation spans [0, 1]
            )
        )

        batches, cores = generate_batches(
            n=doy_gcm.max().values, batch_size=batch_size, buffer_size=buffer_size, one_indexed=True
        )

        bc_result = []
        # TODO: currently running in sequence but can be mapped out into separate workers/runners
        for i, (b, c) in enumerate(zip(batches, cores)):
            gcm_batch = ds_gcm.sel(time=doy_gcm.isin(b))
            obs_batch = ds_obs.sel(time=doy_obs.isin(b))

            train_x, train_y = xr.align(gcm_batch[[var]], obs_batch[var], join='inner', copy=True)

            bias_correction_model.fit(train_x, train_y)

            bc_data = bias_correction_model.predict(X=gcm_batch)
            bc_result.append(bc_data.sel(time=bc_data.time.dt.dayofyear.isin(c)))

        ds_out[var] = xr.concat(bc_result, dim='time').sortby('time')

    return ds_out


def construct_analogs(
    ds_gcm: xr.Dataset,
    ds_obs_coarse: xr.Dataset,
    ds_obs_fine: xr.Dataset,
    label: str,
    n_analogs: int = 10,
    doy_range: int = 45,
) -> xr.Dataset:
    """
    Find analog days for each coarse scale GCM day from coarsened observations, then use the fine scale versions of
    these analogs to construct the downscaled GCM data. The fine scale analogs are combined together using a linear
    combination where the coefficients come from a linear regression of coarsened observation to the GCM day to be
    downscaled. Analogs are selected based on the lowest RMSE between coarsened obs and target GCM pattern. See
    https://climate.northwestknowledge.net/MACA/MACAmethod.php for more details.

    Parameters
    ----------
    ds_gcm : xr.Dataset
        GCM dataset, original/coarse resolution
    ds_obs_coarse : xr.Dataset
        Observation dataset coarsened to the GCM resolution
    ds_obs_fine : xr.Dataset
        Observation dataset, original/fine resolution
    label : str
        Name of variable to be downscaled
    n_analogs : int
        Number of analog days to look for
    doy_range : int
        The range of day of year to look for analogs within

    Returns
    -------
    downscaled : xr.Dataset
        The downscaled dataset
    """
    for dim in ['time', 'lat', 'lon']:
        for ds in [ds_gcm, ds_obs_coarse, ds_obs_fine]:
            assert dim in ds.dims
    if len(ds_obs_coarse.time) != len(ds_obs_fine.time):
        raise ValueError('ds_obs_coarse is not the same length as ds_obs_fine')

    # work with dataarrays instead of datasets
    da_gcm = ds_gcm[label]
    da_obs_coarse = ds_obs_coarse[label]
    da_obs_fine = ds_obs_fine[label]

    # initialize a regridder to interpolate the residuals from coarse to fine scale later
    def _make_template(da):
        temp = da.isel(time=0)
        template = temp.to_dataset(name=da.name)
        template['mask'] = temp.notnull()
        return template

    coarse_template = _make_template(da_obs_coarse)
    fine_template = _make_template(da_obs_fine)

    regridder = xe.Regridder(
        coarse_template,
        fine_template,
        "bilinear",
        extrap_method="nearest_s2d",
    )

    # get dimension sizes from input data
    domain_shape_coarse = (len(da_obs_coarse.lat), len(da_obs_coarse.lon))
    n_pixel_coarse = domain_shape_coarse[0] * domain_shape_coarse[1]

    # rename the time dimension to keep track of them
    X = da_obs_coarse.rename({'time': 'ndays_in_obs'})  # coarse obs
    y = da_gcm.rename({'time': 'ndays_in_gcm'})  # coarse gcm

    # get rmse between each GCM slices to be downscaled and each observation slices
    # will have the shape ndays_in_gcm x ndays_in_obs
    rmse = np.sqrt(((X - y) ** 2).sum(dim=['lat', 'lon'])) / n_pixel_coarse
    rmse = rmse.load()

    # get a day of year mask in the same shape of rmse according to the day range input
    mask = get_doy_mask(
        source_doy=X.ndays_in_obs.dt.dayofyear,
        target_doy=y.ndays_in_gcm.dt.dayofyear,
        doy_range=doy_range,
    )

    # find the indices with the lowest rmse within the day of year constraint
    dim_order = ['ndays_in_gcm', 'ndays_in_obs']
    inds = (
        rmse.where(mask)
        .argsort(axis=rmse.get_axis_num('ndays_in_obs'))
        .isel(ndays_in_obs=slice(0, n_analogs))
        .transpose(*dim_order)
    )
    # rearrage the data into tabular format in order to train linear regression models to get coefficients
    X = X.stack(pixel_coarse=['lat', 'lon'])
    X = X.where(X.notnull(), drop=True)
    y = y.stack(pixel_coarse=['lat', 'lon'])
    y = y.where(y.notnull(), drop=True)

    # initialize models to be used
    lr_model = LinearRegression()

    # initialize temporary variables
    downscaled = []
    residuals = []
    coefs = []
    intercepts = []
    obs_analogs = []

    # pre-load to speed up for loop (lots of indexing into x & y)
    X = X.load()
    y = y.load()

    # train a linear regression model for each day in coarsen GCM dataset, where the features are each coarsened observation
    # analogs, and examples are each pixels within the coarsened domain
    for i in range(len(y)):

        # get data from the GCM day being downscaled
        yi = y.isel(ndays_in_gcm=i)
        # get data from the coarsened obs analogs
        ind = inds.isel(ndays_in_gcm=i).values

        # save obs_analogs for later
        obs_analogs.append(da_obs_fine.isel(time=ind).rename({'time': 'analog'}).drop('analog'))

        xi = X.isel(ndays_in_obs=ind).transpose('pixel_coarse', 'ndays_in_obs')

        # fit model
        lr_model.fit(xi.data, yi.data)
        coefs.append(lr_model.coef_)
        intercepts.append(lr_model.intercept_)

        residual = yi - lr_model.predict(xi.data)
        residual = residual.unstack('pixel_coarse')

        residuals.append(residual)

    # reconstruct xarray objects from temporary lists
    residuals = xr.concat(residuals, dim='ndays_in_gcm').rename({'ndays_in_gcm': 'time'})
    coefs = xr.DataArray(coefs, dims=('time', 'analog'), coords={'time': residuals.time})
    intercepts = xr.DataArray(intercepts, coords={'time': residuals.time})
    obs_analogs = xr.concat(obs_analogs, dim='time')

    # interpolate residuals to fine grid
    interpolated_residual = regridder(residuals)

    # combine obs analogs with residuals
    fine_pred = (obs_analogs * coefs).sum(dim='analog') + intercepts + interpolated_residual

    downscaled = fine_pred.to_dataset(name=label)

    return downscaled
