import xarray as xr


def reconstruct_finescale(ds: xr.Dataset, spatial_anomaly: xr.Dataset = None):
    """Add the spatial anomalies back into the interpolated fine scale dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset or data array you're wanting to chunk. With dimensions ('month', 'lat', 'lon')
    spatial_anomaly : xr.Dataset, optional
        The dataset of monthly spatial anomalies resulting from taking the difference between
        the fine scale obs and the interpolated obs. With dimensions ('month', 'lat', 'lon')

    Returns
    -------
    reconstructed : xr.Dataset
        Finescale dataset with spatial heterogeneity added back in
    """
    reconstructed = ds.groupby('time.month') + spatial_anomaly
    return reconstructed
