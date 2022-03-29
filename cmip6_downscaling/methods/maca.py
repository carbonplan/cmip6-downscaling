from __future__ import annotations

import numpy as np
import xarray as xr
import xesmf as xe
from skdownscale.pointwise_models import EquidistantCdfMatcher, PointWiseDownscaler
from sklearn.linear_model import LinearRegression

from cmip6_downscaling.workflows.utils import generate_batches


def maca_bias_correction(
    ds_gcm: xr.Dataset,
    ds_obs: xr.Dataset,
    historical_period: slice,
    variables: str | list[str],
    batch_size: int | None = 15,
    buffer_size: int | None = 15,
) -> xr.Dataset:
    """
    Run bias correction as it is done in the MACA method. See https://climate.northwestknowledge.net/MACA/MACAmethod.php
    for more details. Briefly, the bias correction is performed using the Equidistant CDF matching method in batches.
    Neighboring day of years are bias corrected together with a buffer. That is, with a batch size of 15 and a buffer size
    of 15, the 45 neighboring days of year are bias corrected together, but only the result of the center 15 days are used.
    The historical GCM is mapped to historical coarsened observation in the bias correction.

    Parameters
    ----------
    ds_gcm: xr.Dataset
        GCM dataset, must have a dimension called time on which we can call .dt.dayofyear on
    ds_obs: xr.Dataset
        Observation dataset, must have a dimension called time on which we can call .dt.dayofyear on
    historical_period: slice
        The historical period
    variables: List[str]
        Names of the variables used in obs and gcm dataset (including features and label)
    batch_size: Optional[int]
        The batch size in terms of day of year to bias correct together
    buffer_size: Optional[int]
        The buffer size in terms of day of year to include in the bias correction

    Returns
    -------
    ds_out: xr.Dataset
        The bias corrected dataset
    """
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

            train_x = gcm_batch.sel(time=historical_period)[[var]]
            train_y = obs_batch.sel(time=historical_period)[var]

            # TODO: this is a total hack to get around the different calendars of observation dataset and GCM
            # should be able to remove once we unify all calendars
            if len(train_x.time) > len(train_y.time):
                train_x = train_x.isel(time=slice(0, len(train_y.time)))
            elif len(train_x.time) < len(train_y.time):
                train_y = train_y.isel(time=slice(0, len(train_x.time)))
            train_x = train_x.assign_coords({'time': train_y.time})

            bias_correction_model.fit(
                train_x.unify_chunks(),  # dataset
                train_y.unify_chunks(),  # dataarray
            )

            bc_data = bias_correction_model.predict(X=gcm_batch.unify_chunks())
            # needs the .compute here otherwise the code fails with errors
            # TODO: not sure if this is scalable
            bc_result.append(bc_data.sel(time=bc_data.time.dt.dayofyear.isin(c)).compute())

        ds_out[var] = xr.concat(bc_result, dim='time').sortby('time')

    return ds_out


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
    source_doy: xr.DataArray
        1D xr data array with day of year information
    target_doy: xr.DataArray
        1D xr data array with day of year information

    Returns
    -------
    mask: xr.DataArray
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


def maca_construct_analogs(
    ds_gcm: xr.Dataset,
    ds_obs_coarse: xr.Dataset,
    ds_obs_fine: xr.Dataset,
    label: str,
    n_analogs: int = 10,
    doy_range: int = 45,
    **kwargs,
) -> xr.DataArray:
    """
    Find analog days for each coarse scale GCM day from coarsened observations, then use the fine scale versions of
    these analogs to construct the downscaled GCM data. The fine scale analogs are combined together using a linear
    combination where the coefficients come from a linear regression of coarsened observation to the GCM day to be
    downscaled. Analogs are selected based on the lowest RMSE between coarsened obs and target GCM pattern. See
    https://climate.northwestknowledge.net/MACA/MACAmethod.php for more details.

    Parameters
    ----------
    ds_gcm: xr.Dataset
        GCM dataset, original/coarse resolution
    ds_obs_coarse: xr.Dataset
        Observation dataset coarsened to the GCM resolution
    ds_obs_fine: xr.Dataset
        Observation dataset, original/fine resolution
    label: str
        Name of variable to be downscaled
    n_analogs: int
        Number of analog days to look for
    doy_range: int
        The range of day of year to look for analogs within

    Returns
    -------
    downscaled: xr.Dataset
        The downscaled dataset
    """
    # make sure the input data is valid with the correct shape and dims
    for dim in ['time', 'lat', 'lon']:
        for ds in [ds_gcm, ds_obs_coarse, ds_obs_fine]:
            assert dim in ds.dims
    assert len(ds_obs_coarse.time) == len(ds_obs_fine.time)

    # work with dataarrays instead of datasets
    ds_gcm = ds_gcm[label]
    ds_obs_coarse = ds_obs_coarse[label]
    ds_obs_fine = ds_obs_fine[label]

    # get dimension sizes from input data
    domain_shape_coarse = (len(ds_obs_coarse.lat), len(ds_obs_coarse.lon))
    n_pixel_coarse = domain_shape_coarse[0] * domain_shape_coarse[1]

    # rename the time dimension to keep track of them
    X = ds_obs_coarse.rename({'time': 'ndays_in_obs'})  # coarse obs
    y = ds_gcm.rename({'time': 'ndays_in_gcm'})  # coarse gcm

    # get rmse between each GCM slices to be downscaled and each observation slices
    # will have the shape ndays_in_gcm x ndays_in_obs
    rmse = np.sqrt(((X - y) ** 2).sum(dim=['lat', 'lon'])) / n_pixel_coarse

    # get a day of year mask in the same shape of rmse according to the day range input
    mask = get_doy_mask(
        source_doy=X.ndays_in_obs.dt.dayofyear,
        target_doy=y.ndays_in_gcm.dt.dayofyear,
        doy_range=doy_range,
    )

    # find the indices with the lowest rmse within the day of year constraint
    dim_order = ['ndays_in_gcm', 'ndays_in_obs']
    inds = (
        xr.apply_ufunc(
            np.argsort,
            rmse.where(mask),
            vectorize=True,
            input_core_dims=[['ndays_in_obs']],
            output_core_dims=[['ndays_in_obs']],
            dask='parallelized',
            output_dtypes=['int'],
            dask_gufunc_kwargs={'allow_rechunk': True},
        )
        .isel(ndays_in_obs=slice(0, n_analogs))
        .transpose(*dim_order)
        .compute()
    )

    # rearrage the data into tabular format in order to train linear regression models to get coefficients
    X = X.stack(pixel_coarse=['lat', 'lon'])
    y = y.stack(pixel_coarse=['lat', 'lon'])

    # initialize models to be used
    lr_model = LinearRegression()

    # TODO: check if the rechunk can be removed
    # initialize a regridder to interpolate the residuals from coarse to fine scale later
    coarse_template = ds_obs_coarse.isel(time=0).chunk({'lat': -1, 'lon': -1})
    fine_template = ds_obs_fine.isel(time=0).chunk({'lat': -1, 'lon': -1})
    regridder = xe.Regridder(
        coarse_template,
        fine_template,
        "bilinear",
        extrap_method="nearest_s2d",
    )

    # initialize output
    downscaled = []
    # train a linear regression model for each day in coarsen GCM dataset, where the features are each coarsened observation
    # analogs, and examples are each pixels within the coarsened domain
    for i in range(len(y)):
        # get data from the GCM day being downscaled
        yi = y.isel(ndays_in_gcm=i)
        # get data from the coarsened obs analogs
        ind = inds.isel(ndays_in_gcm=i).values
        xi = X.isel(ndays_in_obs=ind).transpose('pixel_coarse', 'ndays_in_obs')

        # fit model
        lr_model.fit(xi, yi)
        # save the residuals to be interpolated and included in the final prediction
        residual = yi - lr_model.predict(xi)

        # chunk so that residual is spatially contiguous before interpolation
        # TODO: check if the rechunk can be removed
        residual = residual.unstack('pixel_coarse').chunk({'lat': -1, 'lon': -1})
        interpolated_residual = regridder(residual)

        # construct fine scale prediction by combining the fine scale analogs with the coefficients found in the linear model
        # then add the intercept and interpolated residuals
        fine_pred = (
            (ds_obs_fine.isel(time=ind).transpose('lat', 'lon', 'time') * lr_model.coef_).sum(
                dim='time'
            )
            + lr_model.intercept_
            + interpolated_residual
        )
        fine_pred = fine_pred.rename({'ndays_in_gcm': 'time'}).drop('dayofyear')
        downscaled.append(fine_pred)

    downscaled = xr.concat(downscaled, dim='time').sortby('time')
    downscaled = downscaled.to_dataset(name=label)
    return downscaled
