"""
    ---------------------------------------------------------------------------
    Copyright (C) 2020 Francesco Darugna <fd@geopp.de>  Geo++ GmbH,
                       Jannes B. Wübbena <jw@geopp.de>  Geo++ GmbH.
    A list of all the historical SSRZ Python Demonstrator contributors in
    CREDITS.info
    The first author has received funding from the European Union's
    Horizon 2020 research and innovation programme under the
    Marie Sklodowska-Curie Grant Agreement No 722023.
    ---------------------------------------------------------------------------

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    ---------------------------------------------------------------------------
    ****************************************************************************
    Description:
    Collection of methods to compute tropospheric models and mapping functions.
    ****************************************************************************
"""
import numpy as np
import doctest
import interp_module as interp


# Data table of known geoid heights
# autopep8: off
_data_table = [
[ 13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13,  13 ],
[ 33,  34,  28,  23,  17,  13,   9,   4,   4,   1,  -2,  -2,   0,   2,   3,   2,   1,   1,   3,   1,  -2,  -3,  -3,  -3,  -1,   3,   1,   5,   9,  11,  19,  27,  31,  34,  33,  34 ],
[ 51,  43,  29,  20,  12,   5,  -2, -10, -14, -12, -10, -14, -12,  -6,  -2,   3,   6,   4,   2,   2,   1,  -1,  -3,  -7, -14, -24, -27, -25, -19,   3,  24,  37,  47,  60,  61,  58 ],
[ 47,  41,  21,  18,  14,   7,  -3, -22, -29, -32, -32, -26, -15,  -2,  13,  17,  19,   6,   2,   9,  17,  10,  13,   1, -14, -30, -39,  46, -42, -21,   6,  29,  49,  65,  60,  57 ],
[ 47,  48,  42,  28,  12, -10, -19, -33, -43, -42, -43, -29,  -2,  17,  23,  22,   6,   2,  -8,   8,   8,   1, -11, -19, -16, -18, -22, -35, -40, -26, -12,  24,  45,  63,  62,  59 ],
[ 52,  48,  35,  40,  33,  -9, -28, -39, -48, -59, -50, -28,   3,  23,  37,  18,  -1, -11, -12, -10, -13, -20, -31, -34, -21, -16, -26, -34, -33, -35, -26,   2,  33,  59,  52,  51 ],
[ 36,  28,  29,  17,  12, -20, -15, -40, -33, -34, -34, -28,   7,  29,  43,  20,   4,  -6,  -7,  -5,  -8, -15, -28, -40, -42, -29, -22, -26, -32, -51, -40, -17,  17,  31,  34,  44 ],
[ 31,  26,  15,   6,   1, -29, -44, -61, -67, -59, -36, -11,  21,  39,  49,  39,  22,  10,   5,  10,   7,  -7, -23, -39, -47, -34,  -9, -10, -20, -45, -48, -32,  -9,  17,  25,  31 ],
[ 22,  23,   2,  -3,  -7, -36, -59, -90, -95, -63, -24,  12,  53,  60,  58,  46,  36,  26,  13,  12,  11,   2, -11, -28, -38, -29, -10,   3,   1, -11, -41, -42, -16,   3,  17,  33 ],
[ 18,  12, -13,  -9, -28, -49, -62, -89 -102, -63,  -9,  33,  58,  73,  74,  63,  50,  32,  22,  16,  17,  13,   1, -12, -23, -20, -14,  -3,  14,  10, -15, -27, -18,   3,  12,  20 ],
[ 12,  13,  -2, -14, -25, -32, -38, -60, -75,  63, -26,   0,  35,  52,  68,  76,  64,  52,  36,  22,  11,   6,  -1,  -8, -10,  -8, -11,  -9,   1,  32,   4, -18, -13,  -9,   4,  14 ],
[ 17,  23,  21,   8,  -9, -10, -11, -20, -40, -47, -45, -25,   5,  23,  45,  58,  57,  63,  51,  27,  10,   0,  -9, -11,  -5,  -2,  -3,  -1,   9,  35,  20,  -5,  -6,  -5,   0,  13 ],
[ 22,  27,  34,  29,  14,  15,  15,   7,  -9, -25, -37, -39, -23, -14,  15,  33,  34,  45,  46,  22,   5,  -2,  -8, -13, -10,  -7,  -4,   1,   9,  32,  16,   4,  -8,   4,  12,  15 ],
[ 18,  26,  31,  33,  39,  41,  30,  24,  13,  -2, -20, -32, -33, -27, -14,  -2,   5,  20,  21,   6,   1,  -7, -12, -12, -12, -10,  -7,   1,   8,  23,  15,  -2,  -6,   6,  21,  24 ],
[ 25,  26,  34,  39,  45,  45,  38,  39,  28,  13,  -1, -15, -22, -22, -18, -15, -14, -10, -15, -18, -18, -16, -17, -15, -10, -10,  -8,  -2,   6,  14,  13,   3,   3,  10,  20,  27 ],
[ 16,  19,  25,  30,  35,  35,  33,  30,  27,  10,  -2, -14, -23, -30, -33, -29, -35, -43, -45, -43, -37, -32, -30, -26, -23, -22, -16, -10,  -2,  10,  20,  20,  21,  24,  22,  17 ],
[ 16,  16,  17,  21,  20,  26,  26,  22,  16,  10,  -1, -16, -29, -36, -46, -55, -54, -59, -61, -60, -61, -55, -49, -44, -38, -31, -25, -16,  -6,   1,   4,   5,   4,   2,   6,  12 ],
[ -4,  -1,   1,   4,   4,   6,   5,   4,   2,  -6, -15, -24, -33, -40, -48, -50, -53, -52, -53, -54, -55, -52, -48, -42, -38, -38, -29, -26, -26, -24, -23, -21, -19, -16, -12,  -8 ],
[-30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30, -30 ]
]
# autopep8: on


def compute_geoid_hgt(lat, long):
    """ Interpolation of WGS 84 Geoid Heights
    Input:
        - lat  : latitude of the desidered point
        - long : longitude of the desidered point
    Output:
        - N    : geoid height at the desidered point using a bilinear
                 interpolation

    Bilinear interpolation uses a grid table of known values of the
    geoid height. The grid considers a stepsize of 10° both in latitude and
    longitude.  Latitude is considered in [-90, 90] (be aware that the order
    is reverse: from -90 to 90!) and the longitude in [0, 350].

    Reference:
    DEPARTMENT OF DEFENSE WORLD GEODETIC SYSTEM,1984
    ITS DEFINITION AND RELATIONSHIPS WITHLOCAL GEODETIC SYSTEMS

    Example:
    >>> '%.8f' % compute_geoid_hgt(17.98, 34.1234)
    '2.20359268'
    >>> '%.8f' % compute_geoid_hgt(87.38, 340.766)
    '18.26006920'
    """

    # Import the known geoid heights
    N_ref = _data_table
    # Define the vectors of latitude and longitude of the grid of known
    # geoid heights
    phi = np.linspace(90, -90, 19)  # latitude
    lam = np.linspace(0, 350, 36)   # longitude
    # Find the closest interval for lat
    index = (np.abs(phi-lat)).argmin()
    if lat > 0:
        if phi[index] > lat:
            i_lat_1 = index + 1
            i_lat_2 = index
            lat_1 = phi[index + 1]
            lat_2 = phi[index]
        else:
            i_lat_1 = index
            i_lat_2 = index - 1
            lat_1 = phi[index]
            lat_2 = phi[index - 1]
    else:
        if phi[index] > lat:
            i_lat_1 = index
            i_lat_2 = index + 1
            lat_1 = phi[index]
            lat_2 = phi[index + 1]
        else:
            i_lat_1 = index - 1
            i_lat_2 = index
            lat_1 = phi[index - 1]
            lat_2 = phi[index]
    # Find the closest interval for long
    # shift long to range 0:360 if necessary
    while long < 0:
        long += 360
    while long >= 360:
        long -= 360
    jndex = (np.abs(lam - long)).argmin()
    if lam[jndex] > long:
        j_lon_1 = jndex - 1
        j_lon_2 = jndex
        lon_1 = lam[jndex - 1]
        lon_2 = lam[jndex]
    else:
        j_lon_1 = jndex
        if j_lon_1 == 35:
            j_lon_2 = 0
        else:
            j_lon_2 = jndex + 1
        lon_1 = lam[j_lon_1]
        lon_2 = lam[j_lon_2]

    # Rectangle which contains the desidered value
    N1 = N_ref[i_lat_1][j_lon_1]
    N2 = N_ref[i_lat_1][j_lon_2]
    N3 = N_ref[i_lat_2][j_lon_2]
    N4 = N_ref[i_lat_2][j_lon_1]

    N_known = [N1, N2, N3, N4]
    x1 = [lat_1, lat_2]
    x2 = [lon_1, lon_2]

    # Interpolation
    N = interp.do_bil_int(N_known, x1, x2, lat, long)

    return N


def get_unb3_weather(latitude, doy):
    """Get the weather parameters according to the UNB3 weather model

    A reference of the model can be found for example in
    Leandro et al., UNB Neutral Atmosphere Models: Development and Performance
    Proceedings of ION NTM 2006

    The air pressure, temperature, water vapor pressure, temperature
    lapse rate and the height correction factor are determined by
    a model consisting of a latitude dependent constant term
    and an amplitude term that varies with a period of 1 year.

    The latitude dependency is realised by giving values for
    15,30,45,60,75 degrees and linear interpolation between them.
    For latitudes <15 or >75 degrees, no interpolation is done.

    Examples and unit tests for the different latitude regions.
    Reference values come from the 'mobswthr4parm' routine from
    gw/tropolib/src/mobswthr.for:
    >>> '%.2f,%.2f,%.2f,%.5f,%.2f' % get_unb3_weather(8,100)
    '1013.25,299.65,26.31,0.00630,2.77'
    >>> '%.2f,%.2f,%.2f,%.5f,%.2f' % get_unb3_weather(18,18)
    '1014.79,297.17,23.66,0.00620,2.78'
    >>> '%.2f,%.2f,%.2f,%.5f,%.2f' % get_unb3_weather(-32,300)
    '1017.19,292.38,20.10,0.00598,3.06'
    >>> '%.2f,%.2f,%.2f,%.5f,%.2f' % get_unb3_weather(-52,200)
    '1015.86,265.38,3.13,0.00495,1.64'
    >>> '%.2f,%.2f,%.2f,%.5f,%.2f' % get_unb3_weather(74,0)
    '1013.43,251.34,1.17,0.00403,1.28'
    >>> '%.2f,%.2f,%.2f,%.5f,%.2f' % get_unb3_weather(89,365)
    '1013.44,250.83,1.11,0.00398,1.28'

    Input arguments:
    latitude          latitude of the relevant position in degrees
    doy               day of year between 0 and 366

    Return values:
    res_pressure      barometric pressure in mbar at geoid
    res_temp          temperature in K at geoid
    res_wvp           water vapor pressure in mbar at geoid
    res_lr            temperature lapse rate in K/m
    res_hf            water vapor pressure height factor (unitless)
    """
    # model parameters from Langley publications
    pressures = np.array([1013.25, 1017.25, 1015.75, 1011.75, 1013.00])
    temperatures = np.array([299.65, 294.15, 283.15, 272.15, 263.65])
    water_vapor_press = np.array([26.31, 21.79, 11.66, 6.78, 4.11])
    lapse_rate = np.array([0.00630, 0.00605, 0.00558, 0.00539, 0.00453])
    height_factor = np.array([2.77, 3.15, 2.57, 1.81, 1.55])

    # parameter amplitudes
    amp_press = np.array([0.00, -3.75, -2.25, -1.75, -0.50])
    amp_temp = np.array([0.00, 7.00, 11.00, 15.00, 14.50])
    amp_wvp = np.array([0.00, 8.85, 7.24, 5.36, 3.39])
    amp_lr = np.array([0.00, 0.00025, 0.00032, 0.00081, 0.00062])
    amp_hf = np.array([0.00, 0.33, 0.46, 0.74, 0.30])

    if latitude > 0.:  # northern hemisphere
        phase = doy - 28.0
    else:  # southern hemisphere
        phase = doy - 211.0
    cos_term = -np.cos(phase * 2. * np.pi / 365.25)
    abs_lat = np.abs(latitude)
    if abs_lat < 15.0:
        weight = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    elif abs_lat < 30.0:
        weight = np.array([(30. - abs_lat) / 15., (abs_lat - 15.) / 15.,
                           0.0, 0.0, 0.0])
    elif abs_lat < 45.0:
        weight = np.array([0.0, (45. - abs_lat) / 15., (abs_lat - 30.) / 15.,
                           0.0, 0.0])
    elif abs_lat < 60.0:
        weight = np.array([0.0, 0.0, (60. - abs_lat) / 15.,
                           (abs_lat - 45.) / 15., 0.0])
    elif abs_lat < 75.0:
        weight = np.array([0.0, 0.0, 0.0, (75. - abs_lat) / 15.,
                           (abs_lat - 60.) / 15.])
    else:
        weight = np.array([0.0, 0.0, 0.0, 0.0, 1.0])

    res_pressure = (np.dot(weight, pressures) + np.dot(weight, amp_press) *
                    cos_term)
    res_temp = (np.dot(weight, temperatures) + np.dot(weight, amp_temp) *
                cos_term)
    res_wvp = (np.dot(weight, water_vapor_press) + np.dot(weight, amp_wvp) *
               cos_term)
    res_lr = np.dot(weight, lapse_rate) + np.dot(weight, amp_lr) * cos_term
    res_hf = np.dot(weight, height_factor) + np.dot(weight, amp_hf) * cos_term

    return (res_pressure, res_temp, res_wvp, res_lr, res_hf)


def get_saturation_pressure(temperature):
    """ Get the saturation vapor pressure for a specific temperature.
    The function uses an approximation for the saturation water
    vapour pressure (or equilibrium vapor pressure)
    found in http://acc.igs.org/tropo/wetpp.f

    >>> '%.10f' % get_saturation_pressure(299.7634)
    '34.9690400850'

    Input arguments:
    temperature      temperature in Kelvin

    Return values:
    sat_pressure     saturation pressure in millibars
    """
    tk = np.float64(temperature)
    t0 = np.float64(273.15)
    tc = tk - t0
    # The conversion of the parameter -5.3 to float32 precision is important to
    # get agreement with the FORTRAN code.
    sat_pressure = (6.11 * (tk/t0) ** (np.float32(-5.3)) *
                    np.exp(25.2 * tc / tk))

    # sat_pressure = 0.01 * np.exp(1.2378847e-5 * temperature * temperature -
    #                              1.9121316e-2 * temperature + 33.93711047 -
    #                              6.3431645e3 / temperature)
    return sat_pressure


def zenith_delay_saastamoinen(pressure, rel_humidity, temperature, latitude,
                              height, do_gravity_correction_wet=True):
    """ Get the hydrostatic and wet zenith delay according to the
        Saastamoinen model.
    The model is for example explained in
    Saastamoinen, J. “Atmospheric Correction for the Troposphere and
    Stratosphere in Radio Ranging Satellites.”
    In The Use of Artificial Satellites for Geodesy,
    edited by Soren W. Henriksen, Armandoncini, and Bernard H. Chovitz, 247–51.
    American Geophysical Union, 1972. https://doi.org/10.1029/GM015p0247.

    The numerical factors are taken from
    Davis, J. L., T. A. Herring, I. I. Shapiro, A. E. E. Rogers,
    and G. Elgered.
    “Geodesy by Radio Interferometry: Effects of Atmospheric Modeling Errors
    on Estimates of Baseline Length.”
    Radio Science 20, no. 6 (November 1, 1985): 1593–1607.
    https://doi.org/10.1029/RS020i006p01593.

    Note on heights:
    the height parameter, according to the original papers should here be the
    height above the geoid. However, the implementations from Geo++ and Davis
    require the ellipsoidal height.
    Therefore, ellipsoidal height should be used here. It is only used to
    calculate the gravity correction and 100 m (neglected) geoid undulation
    correspond to 0.07 mm ZTD difference and can safely be neglected.

    Input argument:
    pressure          barometric pressure in mbar
    rel_humidity      relative humidity (between 0 and 1)
    temperature       temperature in K
    latitude          geodetic latitude of the point were the delay is
                      estimated (degrees)
    height            elliptic height
    do_gravity_correction_wet  For the wet part, no gravity correction is used
                               in the geo++ sources.
                               The reason for this is unclear but to get the
                               same results, this is also
                               done here, but switchable.
    Return values:
    zhd               zenith hydrostatic delay (m)
    zwd               zenith wet delay (m)
    """
    gravity_correction = (1.0 - 0.00266 *
                          np.cos(2.0 * latitude / 180.0 * np.pi) -
                          0.00000028 * height)
    sat_pressure = get_saturation_pressure(temperature)
    zhd = 0.0022768 * pressure / gravity_correction  # zenith hydrostatic delay
    if do_gravity_correction_wet:
        zwd = (0.0022768 * (1255.0 / temperature + 0.05) * rel_humidity *
               sat_pressure / gravity_correction)
    else:
        zwd = (0.0022768 * (1255.0 / temperature + 0.05) * rel_humidity *
               sat_pressure)
    return (zhd, zwd)


def get_model_troposphere(llh, doy, f_dbg=None):
    """ Compute the hydrostatic and wet delays according to the Geo++ model

    Calls the UNB3 weather model to compute surface weather data and afterwards
    extrapolates these parameters to the station height.
    With the parameters at station height, the Saastamoinen model is fed
    to retreive the wet and hydrostatic delays.

    Input arguments:
    llh               station coordinates latitude, longitude, height
                      in degrees, degrees, meter
    doy               day of year (between 0 and 366)
    f_dbg             output file (object) to debug used model parameters

    Return values:
    delays            tupel with zenith hydrostatic and wet delay (m,m)
    """
    orthometric_height = llh[2] - compute_geoid_hgt(llh[0], llh[1])
    if f_dbg:
        print('H={:.4f}'.format(orthometric_height), file=f_dbg)

    weather = get_unb3_weather(llh[0], doy)
    temp_at_station = weather[1] - weather[3] * orthometric_height
    press_at_station = (weather[0] * (temp_at_station / weather[1]) **
                        (9.80665 / 287.0537625 / weather[3]))
    wvp_at_station = (weather[2] * (temp_at_station / weather[1]) **
                      (9.80665 / 287.0537625 / weather[3] *
                       (1.0 + weather[4])))
    sat_pressure = get_saturation_pressure(temp_at_station)
    rel_humidity = wvp_at_station / sat_pressure
    # Apply mask to relative humidity, i.e., it has to lie in [0, 1]
    if rel_humidity > 1.0:
        rel_humidity = 1.0e0
    if f_dbg:
        print('T(H)={:.5f}'.format(temp_at_station), file=f_dbg)

    delays = zenith_delay_saastamoinen(press_at_station, rel_humidity,
                                       temp_at_station, llh[0], llh[2])
    return delays


def compute_gmf(doy, dlat, dlon, dhgt, zd):
    """ This subroutine determines the Global Mapping Functions GMF
        Reference: Boehm, J., A.E. Niell, P. Tregoning, H. Schuh (2006),
        Global Mapping Functions (GMF):
        A new empirical mapping function based on numerical weather model data,
        Geoph. Res. Letters, Vol. 33, L07304, doi:10.1029/2005GL025545.

        input data
        ----------
        dmjd: modified julian date --> now doy
        dlat: ellipsoidal latitude in radians
        dlon: longitude in radians
        dhgt: height in m
        zd:   zenith distance in radians

        output data
        -----------
        gmfh: hydrostatic mapping function
        gmfw: wet mapping function

        Johannes Boehm, 2005 August 30

        ref 2006 Aug. 14: recursions for Legendre polynomials (O. Montenbruck)
        ref 2011 Jul. 21: latitude -> ellipsoidal latitude (J. Boehm)
        ref 2020 Jul. 6: adapted to Python with doy as input instead of
                          dmjd (F. Darugna)
    """
    ah_mean = [+1.2517e+02, +8.503e-01, +6.936e-02, -6.760e+00, +1.771e-01,
               +1.130e-02, +5.963e-01, +1.808e-02, +2.801e-03, -1.414e-03,
               -1.212e+00, +9.300e-02, +3.683e-03, +1.095e-03, +4.671e-05,
               +3.959e-01, -3.867e-02, +5.413e-03, -5.289e-04, +3.229e-04,
               +2.067e-05, +3.000e-01, +2.031e-02, +5.900e-03, +4.573e-04,
               -7.619e-05, +2.327e-06, +3.845e-06, +1.182e-01, +1.158e-02,
               +5.445e-03, +6.219e-05, +4.204e-06, -2.093e-06, +1.540e-07,
               -4.280e-08, -4.751e-01, -3.490e-02, +1.758e-03, +4.019e-04,
               -2.799e-06, -1.287e-06, +5.468e-07, +7.580e-08, -6.300e-09,
               -1.160e-01, +8.301e-03, +8.771e-04, +9.955e-05, -1.718e-06,
               -2.012e-06, +1.170e-08, +1.790e-08, -1.300e-09, +1.000e-10]

    bh_mean = [+0.000e+00, +0.000e+00, +3.249e-02, +0.000e+00, +3.324e-02,
               +1.850e-02, +0.000e+00, -1.115e-01, +2.519e-02, +4.923e-03,
               +0.000e+00, +2.737e-02, +1.595e-02, -7.332e-04, +1.933e-04,
               +0.000e+00, -4.796e-02, +6.381e-03, -1.599e-04, -3.685e-04,
               +1.815e-05, +0.000e+00, +7.033e-02, +2.426e-03, -1.111e-03,
               -1.357e-04, -7.828e-06, +2.547e-06, +0.000e+00, +5.779e-03,
               +3.133e-03, -5.312e-04, -2.028e-05, +2.323e-07, -9.100e-08,
               -1.650e-08, +0.000e+00, +3.688e-02, -8.638e-04, -8.514e-05,
               -2.828e-05, +5.403e-07, +4.390e-07, +1.350e-08, +1.800e-09,
               +0.000e+00, -2.736e-02, -2.977e-04, +8.113e-05, +2.329e-07,
               +8.451e-07, +4.490e-08, -8.100e-09, -1.500e-09, +2.000e-10]

    ah_amp = [-2.738e-01, -2.837e+00, +1.298e-02, -3.588e-01, +2.413e-02,
              +3.427e-02, -7.624e-01, +7.272e-02, +2.160e-02, -3.385e-03,
              +4.424e-01, +3.722e-02, +2.195e-02, -1.503e-03, +2.426e-04,
              +3.013e-01, +5.762e-02, +1.019e-02, -4.476e-04, +6.790e-05,
              +3.227e-05, +3.123e-01, -3.535e-02, +4.840e-03, +3.025e-06,
              -4.363e-05, +2.854e-07, -1.286e-06, -6.725e-01, -3.730e-02,
              +8.964e-04, +1.399e-04, -3.990e-06, +7.431e-06, -2.796e-07,
              -1.601e-07, +4.068e-02, -1.352e-02, +7.282e-04, +9.594e-05,
              +2.070e-06, -9.620e-08, -2.742e-07, -6.370e-08, -6.300e-09,
              +8.625e-02, -5.971e-03, +4.705e-04, +2.335e-05, +4.226e-06,
              +2.475e-07, -8.850e-08, -3.600e-08, -2.900e-09, +0.000e+00]

    bh_amp = [+0.000e+00, +0.000e+00, -1.136e-01, +0.000e+00, -1.868e-01,
              -1.399e-02, +0.000e+00, -1.043e-01, +1.175e-02, -2.240e-03,
              +0.000e+00, -3.222e-02, +1.333e-02, -2.647e-03, -2.316e-05,
              +0.000e+00, +5.339e-02, +1.107e-02, -3.116e-03, -1.079e-04,
              -1.299e-05, +0.000e+00, +4.861e-03, +8.891e-03, -6.448e-04,
              -1.279e-05, +6.358e-06, -1.417e-07, +0.000e+00, +3.041e-02,
              +1.150e-03, -8.743e-04, -2.781e-05, +6.367e-07, -1.140e-08,
              -4.200e-08, +0.000e+00, -2.982e-02, -3.000e-03, +1.394e-05,
              -3.290e-05, -1.705e-07, +7.440e-08, +2.720e-08, -6.600e-09,
              +0.000e+00, +1.236e-02, -9.981e-04, -3.792e-05, -1.355e-05,
              +1.162e-06, -1.789e-07, +1.470e-08, -2.400e-09, -4.000e-10]

    aw_mean = [+5.640e+01, +1.555e+00, -1.011e+00, -3.975e+00, +3.171e-02,
               +1.065e-01, +6.175e-01, +1.376e-01, +4.229e-02, +3.028e-03,
               +1.688e+00, -1.692e-01, +5.478e-02, +2.473e-02, +6.059e-04,
               +2.278e+00, +6.614e-03, -3.505e-04, -6.697e-03, +8.402e-04,
               +7.033e-04, -3.236e+00, +2.184e-01, -4.611e-02, -1.613e-02,
               -1.604e-03, +5.420e-05, +7.922e-05, -2.711e-01, -4.406e-01,
               -3.376e-02, -2.801e-03, -4.090e-04, -2.056e-05, +6.894e-06,
               +2.317e-06, +1.941e+00, -2.562e-01, +1.598e-02, +5.449e-03,
               +3.544e-04, +1.148e-05, +7.503e-06, -5.667e-07, -3.660e-08,
               +8.683e-01, -5.931e-02, -1.864e-03, -1.277e-04, +2.029e-04,
               +1.269e-05, +1.629e-06, +9.660e-08, -1.015e-07, -5.000e-10]

    bw_mean = [+0.000e+00, +0.000e+00, +2.592e-01, +0.000e+00, +2.974e-02,
               -5.471e-01, +0.000e+00, -5.926e-01, -1.030e-01, -1.567e-02,
               +0.000e+00, +1.710e-01, +9.025e-02, +2.689e-02, +2.243e-03,
               +0.000e+00, +3.439e-01, +2.402e-02, +5.410e-03, +1.601e-03,
               +9.669e-05, +0.000e+00, +9.502e-02, -3.063e-02, -1.055e-03,
               -1.067e-04, -1.130e-04, +2.124e-05, +0.000e+00, -3.129e-01,
               +8.463e-03, +2.253e-04, +7.413e-05, -9.376e-05, -1.606e-06,
               +2.060e-06, +0.000e+00, +2.739e-01, +1.167e-03, -2.246e-05,
               -1.287e-04, -2.438e-05, -7.561e-07, +1.158e-06, +4.950e-08,
               +0.000e+00, -1.344e-01, +5.342e-03, +3.775e-04, -6.756e-05,
               -1.686e-06, -1.184e-06, +2.768e-07, +2.730e-08, +5.700e-09]

    aw_amp = [+1.023e-01, -2.695e+00, +3.417e-01, -1.405e-01, +3.175e-01,
              +2.116e-01, +3.536e+00, -1.505e-01, -1.660e-02, +2.967e-02,
              +3.819e-01, -1.695e-01, -7.444e-02, +7.409e-03, -6.262e-03,
              -1.836e+00, -1.759e-02, -6.256e-02, -2.371e-03, +7.947e-04,
              +1.501e-04, -8.603e-01, -1.360e-01, -3.629e-02, -3.706e-03,
              -2.976e-04, +1.857e-05, +3.021e-05, +2.248e+00, -1.178e-01,
              +1.255e-02, +1.134e-03, -2.161e-04, -5.817e-06, +8.836e-07,
              -1.769e-07, +7.313e-01, -1.188e-01, +1.145e-02, +1.011e-03,
              +1.083e-04, +2.570e-06, -2.140e-06, -5.710e-08, +2.000e-08,
              -1.632e+00, -6.948e-03, -3.893e-03, +8.592e-04, +7.577e-05,
              +4.539e-06, -3.852e-07, -2.213e-07, -1.370e-08, +5.800e-09]

    bw_amp = [+0.000e+00, +0.000e+00, -8.865e-02, +0.000e+00, -4.309e-01,
              +6.340e-02, +0.000e+00, +1.162e-01, +6.176e-02, -4.234e-03,
              +0.000e+00, +2.530e-01, +4.017e-02, -6.204e-03, +4.977e-03,
              +0.000e+00, -1.737e-01, -5.638e-03, +1.488e-04, +4.857e-04,
              -1.809e-04, +0.000e+00, -1.514e-01, -1.685e-02, +5.333e-03,
              -7.611e-05, +2.394e-05, +8.195e-06, +0.000e+00, +9.326e-02,
              -1.275e-02, -3.071e-04, +5.374e-05, -3.391e-05, -7.436e-06,
              +6.747e-07, +0.000e+00, -8.637e-02, -3.807e-03, -6.833e-04,
              -3.861e-05, -2.268e-05, +1.454e-06, +3.860e-07, -1.068e-07,
              +0.000e+00, -2.658e-02, -1.947e-03, +7.131e-04, -3.506e-05,
              +1.885e-07, +5.792e-07, +3.990e-08, +2.000e-08, -5.700e-09]
    # Define pi
    pi = 3.14159265359
    # degree n and order m
    nmax = 9
    # unit vector
    x = np.cos(dlat) * np.cos(dlon)
    y = np.cos(dlat) * np.sin(dlon)
    z = np.sin(dlat)
    # Legendre polynomials
    V = np.zeros((nmax + 1, nmax + 1))
    W = np.zeros((nmax + 1, nmax + 1))
    V[0, 0] = 1
    W[0, 0] = 0
    V[1, 0] = z * V[0, 0]
    W[1, 0] = 0

    for n in range(2, nmax + 1):
        V[n, 0] = ((2 * n - 1) * z * V[n - 1, 0] - (n - 1) * V[n - 2, 0]) / n
        W[n, 0] = 0

    for m in range(1, nmax + 1):
        V[m, m] = (2 * m - 1) * (x*V[m - 1, m - 1] - y * W[m - 1, m - 1])
        W[m, m] = (2 * m - 1) * (x*W[m - 1, m - 1] + y * V[m - 1, m - 1])
        if (m < nmax):
            V[m + 1, m] = (2 * m + 1) * z * V[m, m]
            W[m + 1, m] = (2 * m + 1) * z * W[m, m]
        for n in range(m + 2, nmax + 1):
            V[n, m] = ((2 * n - 1) * z * V[n - 1, m] -
                       (n + m - 1) * V[n - 2, m]) / (n - m)
            W[n, m] = ((2 * n - 1) * z * W[n - 1, m] -
                       (n + m - 1) * W[n - 2, m]) / (n - m)
    # (1) hydrostatic mf
    bh = 0.0029
    c0h = 0.062
    if dlat < 0:  # southern hemisphere
        phh = pi
        c11h = 0.007
        c10h = 0.002
    else:                # northern hemisphere
        phh = 0
        c11h = 0.005
        c10h = 0.001
    ch = c0h + ((np.cos(doy / 365.25 * 2.0 * pi + phh) + 1) *
                c11h / 2.0 + c10h) * (1.0 - np.cos(dlat))

    ahm = 0
    aha = 0
    i = 0
    for n in range(0, nmax + 1):
        for m in range(0, n + 1):
            ahm = ahm + (ah_mean[i] * V[n, m] + bh_mean[i] * W[n, m])
            aha = aha + (ah_amp[i] * V[n, m] + bh_amp[i] * W[n, m])
            i += 1
    ah = (ahm + aha * np.cos(doy / 365.25 * 2.0 * np.pi)) * 1e-5

    sine = np.sin(pi / 2.0 - zd)
    beta = bh / (sine + ch)
    gamma = ah / (sine + beta)
    topcon = (1.0 + ah / (1.0 + bh / (1.0 + ch)))
    gmfh = topcon / (sine + gamma)

    # height correction for hydrostatic mapping function from Niell (1996)
    # in order to reduce the coefficients to sea level
    a_ht = 2.53e-5
    b_ht = 5.49e-3
    c_ht = 1.14e-3
    hs_km = dhgt / 1.0e3

    beta = b_ht / (sine + c_ht)
    gamma = a_ht / (sine + beta)
    topcon = (1.0 + a_ht / (1.0 + b_ht / (1.0 + c_ht)))
    ht_corr_coef = 1.0 / sine - topcon / (sine + gamma)
    ht_corr = ht_corr_coef * hs_km
    gmfh = gmfh + ht_corr

    # (2) wet mf
    bw = 0.00146
    cw = 0.04391
    awm = 0
    awa = 0
    i = 0
    for n in range(0, nmax + 1):
        for m in range(0, n + 1):
            awm = awm + (aw_mean[i] * V[n, m] + bw_mean[i] * W[n, m])
            awa = awa + (aw_amp[i] * V[n, m] + bw_amp[i] * W[n, m])
            i += 1
    aw = (awm + awa * np.cos(doy / 365.25 * 2.0 * np.pi)) * 1e-5
    beta = bw / (sine + cw)
    gamma = aw / (sine + beta)
    topcon = (1.0 + aw / (1.0 + bw / (1.0 + cw)))
    gmfw = topcon / (sine + gamma)

    return [gmfh, gmfw]


if __name__ == "__main__":
    doctest.testmod()
