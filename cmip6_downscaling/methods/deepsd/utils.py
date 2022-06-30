import fsspec
import numpy as np
import xarray as xr
import xesmf as xe

EPSILON = 1e-6  # small value to add to the denominator when normalizing to avoid division by 0
INPUT_SIZE = 51  # number of pixels in a patch example used for training deepsd model (in both lat/lon (or x/y) directions)
PATCH_STRIDE = 20  # number of pixels to skip when generating patches for deepsd training
INFERENCE_BATCH_SIZE = 500  # number of timesteps in each inference iteration
starting_resolutions = {
    'ERA5': 2.0,
    'GISS-E2-1-G': 2.0,
    'BCC-CSM2-MR': 1.0,
    'AWI-CM-1-1-MR': 1.0,
    'BCC-ESM1': 2.0,
    'SAM0-UNICON': 1.0,
    'CanESM5': 2.0,
    'MRI-ESM2-0': 1.0,
    'MPI-ESM-1-2-HAM': 2.0,
    'MPI-ESM1-2-HR': 1.0,
    'MPI-ESM1-2-LR': 2.0,
    'NESM3': 2.0,
    'NorESM2-LM': 2.0,
    'FGOALS-g3': 2.0,
    'MIROC6': 1.0,
    'ACCESS-CM2': 1.0,
    'NorESM2-MM': 1.0,
    'ACCESS-ESM1-5': 1.0,
    'AWI-ESM-1-1-LR': 2.0,
    'TaiESM1': 1.0,
    'NorCPM1': 2.0,
    'CMCC-ESM2': 1.0,
}
stacked_model_path = 'az://cmip6downscaling/training/deepsd/deepsd_models/{var}_{starting_resolution}d_to_0_25d/frozen_graph.pb'
output_node_name = '{var}_0_25/prediction:0'


def res_to_str(r):
    return str(np.round(r, 2)).replace('.', '_')


def bilinear_interpolate(ds: xr.Dataset, output_degree: float) -> xr.Dataset:
    """
    Bilinear inperpolate dataset to a global grid with specified step size

    Parameters
    ----------
    ds : xr.Dataset
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
    return regridder(ds, keep_attrs=True)


def conservative_interpolate(ds: xr.Dataset, output_degree: float) -> xr.Dataset:
    """
    Conservative inperpolate dataset to a global grid with specified spacing

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    output_degree : float
        Spacing for output dataset

    Returns
    -------
    xr.Dataset
        Regridded dataset
    """
    target_grid_ds = xe.util.grid_global(output_degree, output_degree, cf=True)
    # conservative area regridding needs lat_bands and lon_bands
    regridder = xe.Regridder(ds, target_grid_ds, "conservative")
    return regridder(ds, keep_attrs=True)


def normalize(
    ds: xr.Dataset, dims: list[str] = ['lat', 'lon'], epsilon: float = 1e-6
) -> xr.Dataset:
    """
    Normalize dataset

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset
    dim : list
        Dimensions over which to apply mean and standard deviation
    epsilon : float
        Value to add to standard deviation during normalization

    Returns
    -------
    xr.Dataset
        Normalized dataset
    """
    mean = ds.mean(dim=dims).compute()
    std = ds.std(dim=dims).compute()
    norm = (ds - mean) / (std + epsilon)

    return norm


def build_grid_spec(
    output_degree,
):
    output_degree = np.round(output_degree, 2)
    gcm_grid = xe.util.grid_global(output_degree, output_degree, cf=True)

    nlat = len(gcm_grid.lat)
    nlon = len(gcm_grid.lon)
    lat_spacing = int(np.round(abs(gcm_grid.lat[0] - gcm_grid.lat[1]), 1) * 10)
    lon_spacing = int(np.round(abs(gcm_grid.lon[0] - gcm_grid.lon[1]), 1) * 10)
    min_lat = int(np.round(gcm_grid.lat.min(), 1))
    min_lon = int(np.round(gcm_grid.lon.min(), 1))

    grid_spec = f'{nlat:d}x{nlon:d}_gridsize_{lat_spacing:d}_{lon_spacing:d}_llcorner_{min_lat:d}_{min_lon:d}'
    return grid_spec


def make_coarse_elev_path(
    output_degree,
):
    grid_spec = build_grid_spec(output_degree)
    return f'az://scratch/deepsd/intermediate/elev/ERA5_full_space_{grid_spec}.zarr'


def get_elevation_data(output_degree):
    elev_path = make_coarse_elev_path(output_degree)
    elev_store = fsspec.get_mapper(elev_path)
    return xr.open_zarr(elev_store)


def initialize_empty_dataset(lats, lons, times, output_path, var, chunks, attrs={}):
    """
    Create an empty zarr store for output from inference

    Parameters
    ----------
    lats : coords
        Coordinates for the new dataset
    lons : coords
        Coordinates for the new dataset
    times : coords
        Coordinates for the new dataset
    output_path : UPath
        Path to the zarr store
    var : std
        Name to give the variable in the empty dataset
    chunks : dict
        Chunking scheme for the empty dataset
    attrs : dict
        Attrs for the empty dataset

    Returns
    -------
    xr.Dataset
        Normalized dataset
    """
    ds = xr.DataArray(
        np.empty(shape=(len(times), len(lats), len(lons)), dtype=np.float32),
        dims=["time", "lat", "lon"],
        coords=[times, lats, lons],
        attrs=attrs,
    )
    ds = ds.to_dataset(name=var).chunk(chunks)

    print(output_path)
    ds.to_zarr(output_path, mode="w", compute=False)
