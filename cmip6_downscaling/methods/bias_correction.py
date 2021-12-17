from sklearn.preprocessing import QuantileTransformer, StandardScaler
from skdownscale.pointwise_models.utils import default_none_kwargs
from skdownscale.pointwise_models import (
    PointWiseDownscaler,
    QuantileMappingReressor,
    TrendAwareQuantileMappingRegressor,
)

VALID_CORRECTIONS = ['absolute', 'relative']


def maybe_ensemble_mean(ds):
    if 'member' in ds.dims:
        return ds.mean('member')
    return ds


class MontlyBiasCorrection:
    """simple estimator class to handling monthly bias correction

    Parameters
    ----------
    correction : str
        Choice of correction method, either `absolute` or `relative`.
    """

    def __init__(self, correction='absolute'):

        if correction in VALID_CORRECTIONS:
            self.correction = correction
        else:
            raise ValueError(
                f'Invalid correction ({correction}), valid corrections include: {VALID_CORRECTIONS}'
            )

    def fit(self, X, y):
        """Fit the model

        Calculates the climatology of X and y and stores the correction factor

        Parameters
        ----------
        X : xarray.DataArray or xarray.Dataset
            Training data.
        y : xarray.DataArray or xarray.Dataset (same as X)
            Training targets.
        """

        self.x_climatology_ = X.pipe(maybe_ensemble_mean).groupby('time.month').mean()
        self.y_climatology_ = y.groupby('time.month').mean()

        if self.correction == 'absolute':
            self.correction_ = self.x_climatology_ - self.y_climatology_
        elif self.correction == 'relative':
            self.correction_ = self.x_climatology_ / self.y_climatology_
        else:
            raise ValueError(f'Invalid correction: {self.correction}')

        return self

    def predict(self, X):
        """Apply transforms to the data, and predict with the final estimator

        Parameters
        ----------
        X : xarray.DataArray or xarray.Dataset
            Data to predict with. Data structure must be the same as was applied to the
            `fit()` method.
        """

        if self.correction == 'absolute':
            corrected = X.groupby('time.month') - self.correction_
        elif self.correction == 'relative':
            corrected = X.groupby('time.month') / self.correction_
        else:
            raise ValueError(f'Invalid correction: {self.correction}')

        return corrected.drop('month')

    def persist(self, **kwargs):
        """Persist correction to dask arrays"""
        self.correction_ = self.correction_.persist(**kwargs)
        return self

    def compute(self, **kwargs):
        """Load correction as numpy arrays"""
        self.correction_ = self.correction_.compute(**kwargs)
        return self


def bias_correct_obs_by_method(
    da_obs: Union[xr.DataArray, xr.Dataset],
    method: str,
    bc_kwargs: Dict[str, Any],
) -> xr.DataArray:
    if method == 'quantile_transform':
        if 'n_quantiles' not in bc_kwargs:
            bc_kwargs['n_quantiles'] = len(da_obs)
        qt = PointWiseDownscaler(model=QuantileTransformer(**bc_kwargs))
        qt.fit(da_obs)
        return qt.transform(da_obs)

    elif method == 'z_score':
        # transform obs
        sc = PointWiseDownscaler(model=StandardScaler(**bc_kwargs))
        sc.fit(da_obs)
        return sc.transform(da_obs)

    elif method in ['quantile_map', 'detrended_quantile_map', 'none']:
        return da_obs

    else:
        availalbe_methods = [
            'quantile_transform',
            'z_score',
            'quantile_map',
            'detrended_quantile_map',
            'none',
        ]
        raise NotImplementedError(f'bias correction method must be one of {availalbe_methods}')


def bias_correct_gcm_by_method(
    da_gcm: Union[xr.DataArray, xr.Dataset],
    da_obs: Union[xr.DataArray, xr.Dataset],
    historical_period: slice,
    method: str,
    bc_kwargs: Dict[str, Any],
):
    if method == 'quantile_transform':
        # transform gcm
        if 'n_quantiles' not in bc_kwargs:
            bc_kwargs['n_quantiles'] = len(da_gcm.sel(time=historical_period))
        qt = PointWiseDownscaler(model=QuantileTransformer(**bc_kwargs))
        qt.fit(da_gcm.sel(time=historical_period))
        return qt.transform(da_gcm)

    elif method == 'z_score':
        # transform gcm
        sc = PointWiseDownscaler(model=StandardScaler(**bc_kwargs))
        sc.fit(da_gcm.sel(time=historical_period))
        return sc.transform(da_gcm)

    # TODO: test to see QuantileMappingReressor and TrendAwareQuantileMappingRegressor
    # can handle multiple variables at once 
    elif method == 'quantile_map':
        qm = PointWiseDownscaler(model=QuantileMappingReressor(**bc_kwargs), dim='time')
        qm.fit(da_gcm.sel(time=historical_period), da_obs)
        return qm.predict(da_gcm)

    elif self.bias_correction_method == 'detrended_quantile_map':
        qm = PointWiseDownscaler(
            TrendAwareQuantileMappingRegressor(QuantileMappingReressor(**bc_kwargs))
        )
        qm.fit(da_gcm.sel(time=historical_period), da_obs)
        return qm.predict(da_gcm)

    elif method == 'none':
        return da_gcm

    else:
        availalbe_methods = [
            'quantile_transform',
            'z_score',
            'quantile_map',
            'detrended_quantile_map',
            'none',
        ]
        raise NotImplementedError(f'bias correction method must be one of {availalbe_methods}')
