from __future__ import annotations

import numpy as np
import xarray as xr

from ..constants import KELVIN, MB_PER_KPA

sat_pressure_0c = 6.112  # [milibar]
min_vap = 0.005  # lower limit for vapor pressure


def dewpoint(e):
    """Calculate the ambient dewpoint given the vapor pressure.

    Parameters
    ----------
    e : scalar or array-like
        Water vapor partial pressure [milibar]

    Returns
    -------
    dewpoint : scalar or array-like
        dewpoint temperature [C]

    See Also
    --------
    metpy.calc.dewpoint
    """
    e_milibar = e  # [milibar]
    val = np.log(e_milibar / sat_pressure_0c)
    return 243.5 * val / (17.67 - val)  # dewpoint temperature [C]


def saturation_vapor_pressure(temperature):
    """Calculate the saturation water vapor (partial) pressure.

    Parameters
    ----------
    temperature : scalar or array-like
        air temperature [K]

    Returns
    -------
    svp : scalar or array-like
        The saturation water vapor (partial) pressure [milibar]

    See Also
    --------
    metpy.calc.saturation_vapor_pressure
    """
    # temperature [k]
    return sat_pressure_0c * np.exp(
        17.67 * (temperature - KELVIN) / (temperature - 29.65)
    )  # [milibar]


def dewpoint_from_relative_humidity(temperature, rh):
    """Calculate the ambient dewpoint given air temperature and relative humidity.

    Parameters
    ----------
    temperature : scalar or array-like
        air temperature [K]
    rh : scalar or array-like
        relative humidity expressed as a ratio in the range 0 < rh <= 1

    Returns
    -------
    dewpoint : scalar or array-like
        The dewpoint temperature [C]

    See Also
    --------
    metpy.calc.dewpoint_from_relative_humidity
    """
    return dewpoint(rh * saturation_vapor_pressure(temperature))


def relative_humidity_from_dewpoint(temperature, dewpt):
    """Calculate the relative humidity.

    Uses temperature and dewpoint in celsius to calculate relative
    humidity using the ratio of vapor pressure to saturation vapor pressures.

    Parameters
    ----------
    temperature : scalar or array-like
        air temperature [K]
    dewpt : scalar or array-like
        dewpoint temperature [K]

    Returns
    -------
    scalar or array-like
        relative humidity

    See Also
    --------
    metpyt.calc.relative_humidity_from_dewpoint
    """
    e = saturation_vapor_pressure(dewpt)
    e_s = saturation_vapor_pressure(temperature)
    return e / e_s


def process(ds: xr.Dataset) -> xr.Dataset:
    """Calculate missing derived variables

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset

    Returns
    -------
    ds : xr.Dataset
        Output dataset, includes the follwoing variables: {'tmean', 'vap', 'rh', 'tdew', 'vpd'}
    """

    if 'tmean' not in ds:
        ds['tmean'] = (ds['tmax'] + ds['tmin']) / 2  # [C]

    sat_vp = saturation_vapor_pressure(ds['tmean'] + KELVIN) / MB_PER_KPA

    if 'vap' not in ds and 'rh' in ds:
        ds['vap'] = ds['rh'] * sat_vp
        ds['vap'] = ds['vap'].clip(min=min_vap)
        ds['rh'] = ds['vap'] / sat_vp
        ds['tdew'] = dewpoint_from_relative_humidity(ds['tmean'] + KELVIN, ds['rh'])
    elif 'rh' not in ds and 'vap' in ds:
        ds['vap'] = ds['vap'].clip(min=min_vap)
        ds['rh'] = ds['vap'] / sat_vp
        ds['tdew'] = dewpoint(ds['vap'] * MB_PER_KPA)
    elif 'rh' not in ds and 'tdew' in ds:
        ds['rh'] = relative_humidity_from_dewpoint(ds['tmean'] + KELVIN, ds['tdew'] + KELVIN)
        ds['vap'] = ds['rh'] * sat_vp
        ds['vap'] = ds['vap'].clip(min=min_vap)
        ds['rh'] = ds['vap'] / sat_vp
        ds['tdew'] = dewpoint_from_relative_humidity(ds['tmean'] + KELVIN, ds['rh'])
    else:
        raise ValueError('not able to calculate vap/rh/tdew with given input variables')

    if 'vpd' not in ds:
        ds['vpd'] = sat_vp - ds['vap']

    if any(v not in ds for v in ['tmean', 'vap', 'rh', 'tdew', 'vpd']):
        raise ValueError(f'some derived variables were not calculated: {ds}')

    return ds
