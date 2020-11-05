VALID_CORRECTIONS = ['absolute', 'relative']


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

        self.x_climatology_ = X.groupby('time.month').mean()
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

        return corrected
