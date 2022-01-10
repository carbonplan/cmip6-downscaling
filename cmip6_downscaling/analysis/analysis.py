from .qaqc import make_qaqc_ds


analysis_dir = f"az://cmip6/results/analysis"


def write_qaqc_annual_timeseries(qaqc_ds, out_dir):
    qaqc_ds.groupby('time.year').sum().sum(dim=['lat', 'lon']
                                                ).to_dataframe().to_csv(out_dir+'/qaqc.csv')

def write_qaqc_maps(qaqc_ds, out_dir):
    maps = qaqc_ds.sum(dim='time')
    fig, axarr = plt.subplots(ncols=len(maps), nrows=len(maps['qaqc_check']))
    for col, variable in enumerate(maps.data_vars):
        for row, qaqc_check in enumerate(maps['qaqc_check']):
            maps[variable].sel(qaqc_check=qaqc_check).plot(ax=axarr[row, col])
    fig.savefig(out_dir+'/maps.png')


def qaqc_checks(ds):
    # run_qaqc_dir =  f"{analysis_dir}/qaqc/{run_id}"
    qaqc_ds = make_qaqc_ds(ds)
    try:
        assert qaqc_ds.sum().to_array().values[0] == 0, 'Warning: QAQC issues detected. Creating QAQC summaries'
    except AssertionError:
    # only want to write out these files if we detect qaqc issues
        annual_qaqc_ts = qaqc_ds.groupby('time.year').sum().sum(dim=['lat', 'lon']
                                                ).to_dataframe()
        qaqc_maps = qaqc_ds.sum(dim='time')
        return annual_qaqc_ts, qaqc_maps
        
def write_human_readable_analyses(run_id):
    out_dir = '/home/jovyan/cmip6_analyses/{run_id}/qaqc'
    analysis_run_dir = f"{analysis_dir}/qaqc/{run_id}"
    qaqc_ds = xr.open_zarr(analysis_run_dir+'/qaqc.zarr')
    write_qaqc_maps(qaqc_ds)




