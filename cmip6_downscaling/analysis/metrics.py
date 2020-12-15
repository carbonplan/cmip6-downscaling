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


def calc(x_obj, y_obj):

    metrics = {}

    metrics['x_time_mean'] = time_mean(x_obj)
    metrics['y_time_mean'] = time_mean(y_obj)
    metrics['x_seasonal_cycle_mean'] = seasonal_cycle_mean(x_obj)
    metrics['y_seasonal_cycle_mean'] = seasonal_cycle_mean(y_obj)

    return metrics
