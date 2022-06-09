import xarray as xr
import xesmf as xe


def bilinear_interpolate(ds: xr.Dataset, output_degree: float) -> xr.Dataset:
    """
    Bilinear inperpolate dataset to a global grid with specified step size

    Parameters
    ----------
    ds : str
        Input dataset
    output_degree : float
        Step size for output dataset

    Returns
    -------
    xr.Dataset
        regridded dataset
    """

    target_grid_ds = xe.util.grid_global(output_degree, output_degree, cf=True)
    regridder = xe.Regridder(ds, target_grid_ds, "bilinear", extrap_method="nearest_s2d")
    return regridder(ds)


def conservative_interpolate(ds: xr.Dataset, output_degree: float) -> xr.Dataset:
    """
    Conservative inperpolate dataset to a global grid with specified step size

    Parameters
    ----------
    ds : str
        Input dataset
    output_degree : float
        Step size for output dataset

    Returns
    -------
    xr.Dataset
        regridded dataset
    """
    target_grid_ds = xe.util.grid_global(output_degree, output_degree, cf=True)
    # conservative area regridding needs lat_bands and lon_bands
    regridder = xe.Regridder(ds, target_grid_ds, "conservative")
    return regridder(ds)
