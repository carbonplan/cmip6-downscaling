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

def run_analysis_notebook(gcm,
    scenario,
    train_period_start,
    train_period_end,
    predict_period_start,
    predict_period_end,
    variable,
    latmin, 
    latmax,
    lonmin, 
    lonmax):
    """Create a jupyter notebook with analyses about the run with 
    parameters specified here.

    Parameters
    ----------
    gcm : str
        Input GCM
    scenario: str
        Input GCM scenario
    train_period_start : str
        Date for training period start (e.g. '1985')
    train_period_end : str
        Date for training period end (e.g. '2015')
    predict_period_start : str
        Date for prediction period start (e.g. '2090')
    predict_period_end : str
        Date for prediction period end (e.g. '2090')
    variable: str
        The variable included in the dataset.
    latmin : str
        Minimum latitude in domain
    latmax : str
        Maximum latitude in domain
    lonmin : str
        Minimum longitude in domain
    lonmax : str
        Maximum longitude in domain
    """
    run_id = f'{gcm}-{scenario}-{train_period_start}-{train_period_end}-{predict_period_start}-{predict_period_end}-{latmin}-{latmax}-{lonmin}-{lonmax}-{variable}.zarr'
    pm.execute_notebook(
        '/home/jovyan/cmip6-downscaling/notebooks/analyses.ipynb',
        f'/home/jovyan/cmip6-downscaling/notebooks/output/analyses_{run_id}.ipynb',
        parameters={'run_id': run_id}
        )
