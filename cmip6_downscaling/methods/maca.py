import xarray as xr
from typing import Union, List, Optional 
from skdownscale.pointwise_models import EquidistantCdfMatcher, PointWiseDownscaler
from cmip6_downscaling.workflows.utils import generate_batches


def maca_bias_correction(
    ds_gcm: xr.Dataset,
    ds_obs: xr.Dataset,
    historical_period: slice,
    variables: Union[str, List[str]],
    batch_size: Optional[int] = 15,
    buffer_size: Optional[int] = 15,
) -> xr.Dataset:
    """
    ds_gcm and ds_obs must have a dimension called time on which we can call .dt.dayofyear on
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
        for b, c in zip(batches, cores):
            # TODO: do we need these loads? 
            gcm_batch = ds_gcm.sel(time=doy_gcm.isin(b)) #.load()
            obs_batch = ds_obs.sel(time=doy_obs.isin(b)) #.load()

            train_x = gcm_batch.sel(time=historical_period)[[var]]
            train_y = obs_batch.sel(time=historical_period)[var]

            # TODO: this is a total hack to get around the different calendars of observation dataset and GCM 
            if len(train_x.time) > len(train_y.time):
                train_x = train_x.isel(time=slice(0, len(train_y.time)))
            elif len(train_x.time) < len(train_y.time):
                train_y = train_y.isel(time=slice(0, len(train_x.time)))
            
            train_x = train_x.assign_coords({'time': train_y.time})

            bias_correction_model.fit(
                train_x.unify_chunks(),  # dataset
                train_y.unify_chunks(),  # dataarray
            )

            bc_data = bias_correction_model.predict(X=gcm_batch)
            # TODO: do we need this load 
            # bc_data.load()
            # del gcm_batch, obs_batch
            bc_result.append(bc_data.sel(time=bc_data.time.dt.dayofyear.isin(c)))

        # TODO: are these dataarrays or datasets? might need to cast into dataset 
        ds_out[var] = xr.concat(bc_result, dim='time').sortby('time')

    return ds_out