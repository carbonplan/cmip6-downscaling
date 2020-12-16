def weighted_mean(ds, *args, **kwargs):
    weights = ds.time.dt.days_in_month
    return ds.weighted(weights).mean(dim='time')


def time_mean(obj):
    obj = obj.resample(time='AS').map(weighted_mean, dim='time')
    return obj.mean('time')


def seasonal_cycle_mean(obj):
    return obj.mean(('x', 'y'))


def seasonal_cycle_std(obj):
    return obj.mean(('x', 'y'))


# def extreme_scaling(obj, factors=[1, 2, 4, 8, 16], q=0.02):

#     for factor in factors:

#         obj_coarse = obj.coarsen(x=factor, y=factor).mean()
#         obj_quantile = obj_coarse.quantile(q=q).mean(('x', 'y'))


def calc(obj, compute=False):

    metrics = {}
    if compute:
        metrics['time_mean'] = time_mean(obj).compute()
        metrics['seasonal_cycle_mean'] = seasonal_cycle_mean(obj).compute()
    else:
        metrics['time_mean'] = time_mean(obj)
        metrics['seasonal_cycle_mean'] = seasonal_cycle_mean(obj)

    return metrics
