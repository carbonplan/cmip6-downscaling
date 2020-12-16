import matplotlib.pyplot as plt


def plot_time_mean(ds1, ds2, diff=True, limits_dict=None):

    vars_to_plot = list(ds1.data_vars.keys())

    fig, axarr = plt.subplots(ncols=3, nrows=len(vars_to_plot))

    for i, var in enumerate(vars_to_plot):
        if var in list(limits_dict.keys()):
            vmin = limits_dict[var]['vmin']
            vmax = limits_dict[var]['vmax']
        else:
            vmin, vmax = None, None
        ds2[var].plot(ax=axarr[i, 0], vmin=vmin, vmax=vmax)
        ds2[var].plot(ax=axarr[i, 1], vmin=vmin, vmax=vmax)
        (ds2 - ds1)[var].plot(ax=axarr[i, 2])

    return fig, axarr


def plot_seasonal_mean(ds1, ds2, limits_dict=None):

    vars_to_plot = list(ds1.data_vars.keys())

    fig, axarr = plt.subplots(nrows=len(vars_to_plot))

    for i, var in enumerate(vars_to_plot):
        if var in list(limits_dict.keys()):
            vmin = limits_dict[var]['vmin']
            vmax = limits_dict[var]['vmax']
        else:
            vmin, vmax = None, None
        ds2[var].plot(ax=axarr[i], vmin=vmin, vmax=vmax)
        ds2[var].plot(ax=axarr[i], vmin=vmin, vmax=vmax)

    return fig, axarr
