from .qaqc import make_qaqc_ds


analysis_dir = f"az://cmip6/results/analysis"

def qaqc_checks(ds):
    # run_qaqc_dir =  f"{analysis_dir}/qaqc/{run_id}"
    qaqc_ds = make_qaqc_ds(ds)
    annual_qaqc_ts = qaqc_ds.groupby('time.day').sum().sum(dim=['lat', 'lon']
                                            ).to_dataframe()
    qaqc_maps = qaqc_ds.sum(dim='time')
    return annual_qaqc_ts, qaqc_maps
        
def monthly_summary(ds):
    """Creates an monthly summary dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Downscaled dataset
    """
    out_ds = xr.Dataset()
    for var in ds:
        if var in ['tasmax', 'tasmin']:
            out_ds[var] = ds[var].groupby('time.month').mean()
        elif var in ['pr']:
            out_ds[var] = ds[var].groupby('time.month').sum()
        else:
            print(f'{var} not implemented')

    return out_ds

def annual_summary(ds):
    """Creates an annual summary dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Downscaled dataset
    """
    out_ds = xr.Dataset()
    for var in ds:
        if var in ['tasmax', 'tasmin']:
            out_ds[var] = ds[var].groupby('time.year').mean()
        elif var in ['pr']:
            out_ds[var] = ds[var].groupby('time.year').sum()
        else:
            print(f'{var} not implemented')

    return out_ds