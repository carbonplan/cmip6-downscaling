import pandas as pd


def add_circular_temporal_pad(data, offset, timeunit='D'):
    """
    pad the beginning of data with the last values of data, and pad the end of data with the first values of data 
    
    data must have a dimension called time 
    """
    
    padded = data.pad(time=offset, mode='wrap')

    time_coord = padded.time.values
    time_coord[:offset] = (data.time[:offset] - pd.Timedelta(offset, timeunit)).values
    time_coord[-offset:] = (data.time[-offset:] + pd.Timedelta(offset, timeunit)).values

    padded = padded.assign_coords({'time': time_coord})
    
    return padded


def days_in_year(year, use_leap_year=False):
    if (year % 4 == 0) and use_leap_year:
        return 366
    return 365
    # there might be calendars with 360 days 

    
def pad_with_edge_year(data):
    """
    pad data with repeating values at the edge, similar to the behavior of np.pad(mode='edge') but uses the 365 edge values 
    instead of repeating only 1 edge value (366 if leap year)
    
    offset is the number of years to pad on each side
    data must have a dimension called time 
    """
    def pad_with(vector, pad_width, iaxis, kwargs):
        pstart = pad_width[0]
        pend = pad_width[1]
        if pstart > 0:
            start = vector[pstart:pstart*2]
            vector[:pstart] = start
        if pend > 0:
            end = vector[-pend*2:-pend]
            vector[-pend:] = end
    
    # TODO: figure out whether use_leap_year should be true or false based on the 
    prev_year = days_in_year(data.time[0].dt.year.values - 1, use_leap_year=False)
    next_year = days_in_year(data.time[-1].dt.year.values + 1, use_leap_year=False)
    padded = data.pad({'time':(prev_year, next_year)}, mode=pad_with)
    time_coord = padded.time.values
    time_coord[:prev_year] = (data.time[:prev_year] - pd.Timedelta(prev_year, 'D')).values
    time_coord[-next_year:] = (data.time[-next_year:] + pd.Timedelta(next_year, 'D')).values
    
    padded = padded.assign_coords({'time': time_coord})
    
    return padded


def epoch_adjustment(data, historical_period, day_rolling_window=21, year_rolling_window=31):
    """
    data must have a dimension called time 
    historical_period should be a slice() object that can be directly used in xr.DataArray.sel()
    """

    assert day_rolling_window % 2 == 1
    d_offset = int((day_rolling_window - 1) / 2)

    assert year_rolling_window % 2 == 1
    y_offset = int((year_rolling_window - 1) / 2)
    
    # get historical average rolling average 
    padded = add_circular_temporal_pad(
        data=data.sel(time=historical_period),
        offset=d_offset
    )
    hist_mean = padded.rolling(time=day_rolling_window, center=True).mean().dropna('time').groupby("time.dayofyear").mean()
    
    # get rolling average for the entire data  
    padded = add_circular_temporal_pad(
        data=data,
        offset=d_offset
    )
    func = lambda x: x.rolling(time=year_rolling_window, center=True).mean()
    rolling_doy_mean = padded.rolling(time=day_rolling_window, center=True).mean().dropna('time').groupby('time.dayofyear').apply(func).dropna('time')
    
    rolling_doy_mean = rolling_doy_mean.load()
    
    # repeat the first/last year   
    for i in range(y_offset): 
        rolling_doy_mean = pad_with_edge_year(rolling_doy_mean)
                
    trend = (rolling_doy_mean.groupby('time.dayofyear') - hist_mean)
    ea_data = data - trend
        
    return ea_data, trend
