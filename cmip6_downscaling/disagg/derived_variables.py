import numpy as np

from ..constants import KELVIN

sat_pressure_0c = 6.112  # [milibar]


def dewpoint(e):
    e_milibar = e  # [milibar]
    val = np.log(e_milibar / sat_pressure_0c)
    return 243.5 * val / (17.67 - val)  # dewpoint temperature [C]


def saturation_vapor_pressure(temperature):
    # temperature [k]
    return sat_pressure_0c * np.exp(17.67 * (temperature - KELVIN) / (temperature - 29.65))


def dewpoint_from_relative_humidity(temperature, rh):
    return dewpoint(rh * saturation_vapor_pressure(temperature + KELVIN))


def relative_humidity_from_dewpoint(temperature, dewpt):
    e = saturation_vapor_pressure(dewpt)
    e_s = saturation_vapor_pressure(temperature)
    return e / e_s


def process(ds):

    if 'tmean' not in ds:
        ds['tmean'] = (ds['tmax'] + ds['tmin']) / 2

    sat_vp = saturation_vapor_pressure(ds['tmean'] + KELVIN)

    if 'vap' not in ds and 'rh' in ds:
        ds['vap'] = ds['rh'] * sat_vp

    if 'rh' not in ds and 'vap' in ds:
        ds['rh'] = ds['vap'] / sat_vp

    if 'tdew' not in ds and 'vap' in ds:
        # calc tdew
        ds['tdew'] = dewpoint(ds['vap'] * 10.0)

    if 'tdew' not in ds and 'rh' in ds:
        ds['tdew'] = dewpoint_from_relative_humidity(ds['tmean'] + KELVIN, ds['rh'])

    if 'rh' not in ds and 'tdew' in ds:
        ds['rh'] = relative_humidity_from_dewpoint(ds['tmean'] + KELVIN, ds['tdew'] + KELVIN)

    if 'vap' not in ds and 'rh' in ds:
        # repeated from above (now that we've calculated rh)
        ds['vap'] = ds['rh'] * sat_vp

    if 'vpd' not in ds:
        ds['vpd'] = sat_vp - ds['vap']

    if not all(v in ds for v in ['tmean', 'vap', 'rh', 'tdew', 'vpd']):
        raise ValueError('some derived variables were not calculated: %s' % ds)

    return ds
