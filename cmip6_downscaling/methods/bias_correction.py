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


def bias_correct_obs_by_var(
    da_obs: xr.DataArray,
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


def bias_correct_obs(
    ds_obs: xr.Dataset,
    methods: Union[str, Dict[str, str]],
    bc_kwargs: Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]] = None,
) -> xr.DataArray:
    """
    Bias correct observation data according to methods and kwargs. 

    Parameters
    ----------
    ds_obs : xr.Dataset
        Observation dataset 
    methods : string or dict of string
        Bias correction methods to be used. If string, the same method would be applied to all variables. 
        If dict, the variables that are not in the keys of the dictionary would not be transformed. 
    bc_kwargs: dict, dict of dicts, or None 
        Keyword arguments to be used with the bias correction method 

    Returns
    -------
    ds_obs_bias_corrected : xr.Dataset
        Bias corrected observation dataset 
    """
    bias_corrected = xr.Dataset()

    for v in ds_obs.data_vars:
        method = methods.get(v, 'none') if isinstance(methods, dict) else methods 
        if bc_kwargs is not None and v in bc_kwargs:
            bc_kws = default_none_kwargs(bc_kwargs[v], copy=True)
        elif bc_kwargs is not None:
            kws = default_none_kwargs(bc_kwargs, copy=True)
        else:
            kws = default_none_kwargs({}, copy=True)

        bias_corrected[v] = bias_correct_obs_by_var(
            da_obs=ds_obs[v],
            method=method,
            bc_kwargs=kws
        )

    return bias_corrected


def bias_correct_gcm_by_var(
    da_gcm: xr.DataArray,
    da_obs: xr.DataArray,
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


def bias_correct_gcm(
    ds_gcm: xr.Dataset,
    ds_obs: xr.Dataset,
    historical_period_start: str,
    historical_period_end: str,
    methods: Union[str, Dict[str, str]],
    bc_kwargs: Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]] = None,
) -> xr.DataArray:
    """
    Bias correct gcm data to the provided observation data according to methods and kwargs. 

    Parameters
    ----------
    ds_gcm : xr.Dataset
        GCM dataset to be bias corrected 
    ds_obs : xr.Dataset
        Observation dataset to bias correct to 
    historical_period_start : str
        Start year of the historical/training period 
    historical_period_end : str
        End year of the historical/training period 
    methods : string or dict of string
        Bias correction methods to be used. If string, the same method would be applied to all variables. 
        If dict, the variables that are not in the keys of the dictionary would not be transformed. 
    bc_kwargs: dict, dict of dicts, or None 
        Keyword arguments to be used with the bias correction method 

    Returns
    -------
    ds_gcm_bias_corrected : xr.Dataset
        Bias corrected GCM dataset 
    """
    historical_period = slice(historical_period_start, historical_period_end)

    bias_corrected = xr.Dataset()

    for v in ds_gcm.data_vars:
        assert v in ds_obs.data_vars
        method = methods.get(v, 'none') if isinstance(methods, dict) else methods 
        if bc_kwargs is not None and v in bc_kwargs:
            bc_kws = default_none_kwargs(bc_kwargs[v], copy=True)
        elif bc_kwargs is not None:
            kws = default_none_kwargs(bc_kwargs, copy=True)
        else:
            kws = default_none_kwargs({}, copy=True)

        bias_corrected[v] = bias_correct_gcm_by_var(
            da_gcm=ds_gcm[v],
            da_obs=ds_obs[v],
            historical_period=historical_period,
            method=method,
            bc_kwargs=kws
        )

    return bias_corrected
