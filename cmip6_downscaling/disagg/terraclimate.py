from __future__ import annotations

import math

import numba
import numpy as np
import pandas as pd
from climate_indices import palmer

from .. import CLIMATE_NORMAL_PERIOD
from ..constants import KELVIN, MGM2D_PER_WM2, MIN_PER_DAY, MM_PER_IN, MONTHS_PER_YEAR, SEC_PER_DAY

days_in_month = np.array([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
d2 = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])
d1 = np.array([0, 1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])

DAYS_PER_YEAR = 365
GSC = 0.082  # MJ m -2 min-1 (solar constant)
SMALL_PPT = 1e-10  # fill value for months with zero precipitation
# CLIMATE_NORMAL_PERIOD = (1960, 1990)
# CLIMATE_NORMAL_PERIOD = (1970, 2000)


def by_month(x):
    return x.month


@numba.njit
def mf(t: float, t0: float = -2.0, t1: float = 6.5) -> float:
    if t < t0:
        return 0
    elif t > t1:
        return 1
    else:
        # see also https://github.com/abatz/WATERBALANCE/blob/master/runsnow.m
        # ann params from Dai 2008, Table 1a (row 1)
        a = -48.2292
        b = 0.7205
        c = 1.1662
        d = 1.0223
        # parametric form from Dai 2008
        return 1 - (a * (np.tanh(b * (t - c)) - d) / 100.0)


@numba.njit
def monthly_pet(
    radiation: float,
    tmax: float,
    tmin: float,
    wind: float,
    dpt: float,
    tmean_prev: float,
    lat: float,
    elev: float,
    month: float,
    albedo: float = 0.23,
) -> float:
    """Calculate monthly Reference ET estimates using the Penman-Montieth equation

    This function runs Reference ET estimates for monthly timesteps using methods based on
    the Penman-Montieth equation as presented in Allen et al (1998).  It incorporates a
    modification which adjusts stomatal conductance downwards at temperatures below 5 C.

    This version has been ported over from John Abatzoglou's implementation used
    for Terraclimate (https://github.com/abatz/WATERBALANCE)

    Parameters
    ----------
    radiation : float
        Monthly average shortwave radiation in MJ/m^2/day
    tmax : float
        Monthly average maximum temperature in C
    tmin : float
        Monthly average minimum temperature in C
    wind : float
        Monthly average wind speed in m/s at 10m above ground
    dpt : float
        Dewpoint temperature in C
    tmean_prev : float
        Mean temp of previous month in C
    lat : float
        Latitude (degrees)
    elev : float
        Elevation (meters)
    month : int
        Month of year
    albedo : float
        Surface albedo, default=0.23

    Returns
    -------
    eto : float
    """

    doy = (d1[month] + d2[month]) / 2  # use middle day of month to represent monthly average.
    n_days = days_in_month[month]

    # calculate soil heat flux (total for the month) using change in temperature from previous month
    tmean = (tmax + tmin) / 2
    G = 0.14 * (tmean - tmean_prev) * n_days  # KEY CHANGE
    # convert to wind height at 2m
    hw = 10  # height of wind measurements
    wind = wind * (4.87 / math.log(67 * hw - 5.42))  # convert to wind height at 2m

    # stomatal conductance adjustment for low temperatures
    sr = 100  # stomatal resistance sec/m
    ks_min = 0.01  # minimum value for temps below T1
    Tl = -10  # minimum temp (sc goes to ks_min below this temp)
    T0 = 5  # optimal temp
    Th = 100  # maximum temp (sc goes to zero above this)
    thresh = 5  # temperature threshold below which to apply Jarvis equation (ks=1 above this temp)
    b4 = (Th - T0) / (Th - Tl)
    b3 = 1 / ((T0 - Tl) * (Th - T0) ** b4)
    ks = np.maximum(np.minimum(b3 * (tmean - Tl) * (Th - tmean) ** b4, 1), ks_min)

    # ks[np.isnan(ks)] = ks_min
    # ks[tmean >= thresh] = 1
    if np.isnan(ks):
        ks = ks_min
    if tmean >= thresh:
        ks = 1

    # convert to stomatal resistance.
    sr = sr / ks

    # ra is aerodynamic resistance, rs is bulk surface resistance
    ra = (
        208 / wind
    )  # (log((2-2/3*0.12)/(0.123*0.12))*log((2-2/3*0.12)/(0.1*0.123*0.12)))/(0.41**2*wind) # equal to 208/wind for hh=hw=2.
    rs = sr / (0.5 * 24 * 0.12)  # value of 70 when sr=100

    # Saturation vapor pressure
    P = 101.3 * ((293 - 0.0065 * elev) / 293) ** 5.26  # Barometric pressure in kPa

    es = (
        0.6108 * np.exp(tmin * 17.27 / (tmin + 237.3)) / 2
        + 0.6108 * np.exp(tmax * 17.27 / (tmax + 237.3)) / 2
    )
    ea = 0.6108 * np.exp(dpt * 17.27 / (dpt + 237.3))
    vpd = es - ea

    vpd = np.maximum(
        0, vpd
    )  # added because this can be negative if dewpoint temperature is greater than mean temp (implying vapor pressure greater than saturation).

    # delta - Slope of the saturation vapor pressure vs. air temperature curve at the average hourly air temperature
    delta = (4098 * es) / (tmean + 237.3) ** 2

    lhv = 2.501 - 2.361e-3 * tmean  # latent heat of vaporization
    cp = 1.013 * 10**-3  # specific heat of air

    gamma = cp * P / (0.622 * lhv)  # Psychrometer constant (kPa C-1)
    pa = P / (1.01 * (tmean + KELVIN) * 0.287)  # mean air density at constant pressure

    # Calculate potential max solar radiation or clear sky radiation
    phi = np.pi * lat / 180
    dr = 1 + 0.033 * np.cos(2 * np.pi / DAYS_PER_YEAR * doy)
    delt = 0.409 * np.sin(2 * np.pi / DAYS_PER_YEAR * doy - 1.39)
    omegas = np.arccos(-np.tan(phi) * np.tan(delt))
    Ra = (
        MIN_PER_DAY
        / np.pi
        * GSC
        * dr
        * (omegas * np.sin(phi) * np.sin(delt) + np.cos(phi) * np.cos(delt) * np.sin(omegas))
    )  # Daily extraterrestrial radiation
    Rso = Ra * (
        0.75 + 2e-5 * elev
    )  # For a cloudless day, Rs is roughly 75% of extraterrestrial radiation (Ra)
    Rso = np.maximum(0, Rso)
    # radfraction is a measure of relative shortwave radiation, or of
    # possible radiation (cloudy vs. clear-sky)
    radfraction = radiation / Rso
    radfraction = np.minimum(1, radfraction)

    # longwave  and net radiation
    longw = (
        4.903e-9
        * ((tmax + KELVIN) ** 4 + (tmin + KELVIN) ** 4)
        / 2
        * (0.34 - 0.14 * np.sqrt(ea))
        * (1.35 * radfraction - 0.35)
    )
    netrad = (radiation * (1 - albedo) - longw) * n_days

    # PET
    pet = (
        0.408
        * ((delta * (netrad - G)) + (pa * cp * vpd / ra * SEC_PER_DAY * n_days))
        / (delta + gamma * (1 + rs / ra))
    )

    return pet


@numba.jit
def hydromod(
    t_mean: float,
    ppt: float,
    pet: float,
    awc: float,
    soil: float,
    swe: float,
    mfsnow: float,
) -> dict[str, float]:
    """
    Run simple hydro model to calculate water balance terms. This is
    a translation from matlab into python of a simple hydro model
    written by John Abatzoglou. The original model is located here:
    https://github.com/abatz/WATERBALANCE/blob/master/hydro_tax_ro.m

    This function computes monthly estimated water balance terms given
    specified parameters and states.

    Parameters
    ----------
    t_mean : float
        Mean monthly temperatures in C
    ppt : float
        Monthly accumulated precipitation in mm
    pet : float
        Potential evapotranspiration
    awc : float
        Available water content (constant)
    soil : float
        Soil water storage at beginning of month.
    swe : float
        Snow storage at beginning of month.
    mfsnow : float
        Melt fraction of snow and/or rain/snow fraction.
    Returns
    -------
    data : dict
        Dictionary with data values for aet, def, q, swe, and soil
    """

    snowfall = (1 - mfsnow) * ppt
    rain = ppt - snowfall
    melt = mfsnow * (snowfall + swe)
    input_h2o = rain + melt
    extra_runoff = input_h2o * 0.05
    input_h2o *= 0.95

    swe = (1 - mfsnow) * (snowfall + swe)
    # change in soil is determined by difference from available increase
    # in h2o and pet
    delta_soil = input_h2o - pet

    if (delta_soil < 0) and (swe > 0) and (swe > -delta_soil):
        # if swe has enough to fulfill delta_soil,
        # remove the delta_soil from swe, assign it to
        # snow_drink and then zero-out delta_soil
        swe += delta_soil
        snow_drink = -delta_soil
        delta_soil = 0
    elif (delta_soil < 0) and (swe > 0) and (swe < -delta_soil):
        # if swe doesn't have enough capacity to fulfill
        # the delta_soil need, bring delta_soil closer to zero by drinking
        # swe and then zero-out swe
        delta_soil += swe
        snow_drink = swe
        swe = 0
    else:
        snow_drink = 0

    if -delta_soil > soil:
        # if the need is greater than availability in soil then
        # constrain it to soil (this holds the water balance)
        delta_soil = -soil

    if delta_soil < 0:
        # if delta_soil is negative a.k.a. soil will drain
        drain_soil = delta_soil * (1 - np.exp(-soil / awc))
    else:
        drain_soil = 0

    # In this model aet is calculated based upon a demand/supply
    # relationship where demand is PET and supply is the sum of
    # input_h2o and snow_drink (with input_h2o being 95% of the
    # runoff from melt+rain and snow_drink is like a sublimation term)
    supply = input_h2o + snow_drink

    if pet >= supply:
        # if there is more evaporative demand than supply,
        # aet will be constrained by the soil drainage
        # and your deficit will be the difference between
        # pet and aet
        aet = supply - drain_soil
        deficit = pet - aet
        runoff = 0
        soil = soil + drain_soil  # remove from soil
    else:
        # if there is enough water to satisfy pet then
        # aet will match pet and there is no deficit
        aet = pet
        deficit = 0

        if (soil + delta_soil) > awc:
            # if supply exceeds demand and the updated soil water is greater than
            # the available capacity
            runoff = max(0, soil + delta_soil - awc)
            runoff_snow = min(extra_runoff, melt)

            rain_input = rain - extra_runoff - runoff_snow
            excess_after_liquid = max(0, rain_input - pet)
            excess_rain_only = max(0, soil + excess_after_liquid - awc)
            runoff_snow += runoff - excess_rain_only
            soil = awc
        else:
            # if supply exceeds demand and updated soil water
            # is less than available water capacity you update
            # the soil by the change in soil
            soil += delta_soil
            runoff = 0

    # add the extra runoff component
    runoff += extra_runoff

    return {
        'aet': aet,
        'def': deficit,
        'q': runoff,
        'swe': swe,
        'soil': soil,
    }


def pdsi(
    ppt: pd.Series,
    pet: pd.Series,
    awc: float,
    pad_years: int = 10,
    y1: int = CLIMATE_NORMAL_PERIOD[0],
    y2: int = CLIMATE_NORMAL_PERIOD[1],
) -> np.ndarray:
    """Calculate the Palmer Drought Severity Index (PDSI)

    This is a simple wrapper of the climate_idicies package implementation of pdsi. The wrapper
    includes a spin up period (`pad_years`) using a repeated climatology calculated between `y1`
    and `y2`.

    Note that this is not a perfect reproduction of the Terraclimate PDSI implementation. See
    https://github.com/carbonplan/cmip6-downscaling/issues/4 for more details.

    Parameters
    ----------
    ppt : pd.Series
        Monthly precipitation timeseries (mm)
    pet : pd.Series
        Monthly PET timeseries (mm)
    awc : float
        Soil water capacity (mm)
    pad_years : int
        Number of years of the climatology to prepend to the timeseries of ppt and pet
    y1 : int
        Start year for climate normal period
    y2 : int
        End year for climate normal period

    Returns
    -------
    pdsi : pd.Series
        Timeseries of PDSI (unitless)
    """

    pad_months = pad_years * MONTHS_PER_YEAR
    assert len(ppt) > pad_months
    y0 = ppt.index.year[0] - pad_years  # start year (with pad)

    awc_in = awc / MM_PER_IN

    # calculate the climatology for ppt and pet (for only the climate normal period)
    df = pd.concat([pet, ppt], axis=1)
    climatology = df.loc[str(y1) : str(y2)].groupby(by=by_month).mean()

    # repeat climatology for pad_years, then begine the time series
    ppt_extended = np.concatenate([np.tile(climatology['ppt'].values, pad_years), ppt.values])
    ppt_extended_in = ppt_extended / MM_PER_IN
    pet_extended = np.concatenate([np.tile(climatology['pet'].values, pad_years), pet.values])
    pet_extended_in = pet_extended / MM_PER_IN

    # set all zero ppt months to SMALL_PPT: this gets around a divide by zero
    # in the pdsi function below.
    ppt_zeros = ppt_extended_in <= 0
    if ppt_zeros.any():
        ppt_extended_in[ppt_zeros] = SMALL_PPT

    pdsi_vals = palmer.pdsi(ppt_extended_in, pet_extended_in, awc_in, y0, y1, y2)[0]
    pdsi_vals = pdsi_vals.clip(-16, 16)
    out = pd.Series(pdsi_vals[pad_months:], index=ppt.index)

    return out


@numba.jit
def model(
    df: pd.DataFrame,
    awc: float,
    lat: float,
    elev: float,
    snowpack_prev: float | None = None,
    soil_prev: float | None = None,
    tmean_prev: float | None = None,
) -> pd.DataFrame:
    """Terraclimate hydrology model

    Given a dataframe of monthly hydrometeorologic data, return a new dataframe of derived
    variables including {aet (mm), def (mm), pet (mm), q (mm), soil (mm), swe (mm)}.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing columns {tmean (C), ppt (mm), srad (w m-2), tmax (C), tmin (C),
        ws (m s-1), tdew (C)}. Must include a valid DatetimeIndex.
    awc : float
        Soil water capacity (mm)
    lat : float
        Latitude (degrees)
    elev : float
        Elevation (meters)
    snowpack_prev : float, optional
        Snowpack at the beginning of the month. If None this is taken to be zero.
    soil_prev : float, optional
        Soil water content for the previous month (mm)

    Returns
    -------
    out_df : pd.DataFrame
        aet (mm), def (mm), pet (mm), q (mm), soil (mm), swe (mm)
    """

    if snowpack_prev is None:
        snowpack_prev = 0.0
    if tmean_prev is None:
        tmean_prev = df['tmean'][0]
    if soil_prev is None:
        soil_prev = awc

    for i, row in df.iterrows():

        radiation = row['srad'] * MGM2D_PER_WM2

        # run snow routine
        mfsnow = mf(row['tmean'])

        # run pet routine
        pet = monthly_pet(
            radiation,
            row['tmax'],
            row['tmin'],
            row['ws'],
            row['tdew'],
            tmean_prev,
            lat,
            elev,
            i.month,
        )

        # Reduce PET when there is snow
        pet *= mfsnow

        # run simple hydrology model
        hydro_out = hydromod(
            row['tmean'],
            row['ppt'],
            pet,
            awc,
            soil_prev,
            snowpack_prev,
            mfsnow,
        )

        # populate output dataframe
        df.at[i, 'aet'] = hydro_out['aet']
        df.at[i, 'def'] = hydro_out['def']
        df.at[i, 'pet'] = pet
        df.at[i, 'q'] = hydro_out['q']
        df.at[i, 'soil'] = hydro_out['soil']
        df.at[i, 'swe'] = hydro_out['swe']

        # save state variables
        tmean_prev = row['tmean']
        snowpack_prev = hydro_out['swe']
        soil_prev = hydro_out['soil']

    df['pdsi'] = pdsi(df['ppt'], df['pet'], awc)

    return df
