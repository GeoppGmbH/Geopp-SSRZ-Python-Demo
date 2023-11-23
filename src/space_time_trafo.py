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
"""
import numpy as np
import math
from numpy import linalg as LA

# -----------------------------------------------------------------------------
#                                       Classes
# -----------------------------------------------------------------------------


class GCFormat:
    """ Class to express an epoch in Gregorian Calendar format
    """

    def __init__(self):
        self.year = 0
        self.month = 0
        self.day = 0
        self.hour = 0
        self.min = 0
        self.sec = 0.0


class JDFormat:
    """ Class to express an epoch in Julian Date format
    """

    def __init__(self):
        self.jd = 0.0
        self.jd_frac = 0.0


class GPSFormat:
    """ Class to express an epoch in GPS format
    """

    def __init__(self):
        self.week = 0
        self.sow = 0.0


class Constants:
    """ Class of constants used in the SSRZ Python demo
    """
    # Earth parameters
    re = 6.37e6                   # [m]
    WGS84_a = 6378137             # [m]
    WGS84_f = 1 / 298.257223564
    omega_e = 7.29211514670e-5    # [rad/s]
    # speed of light
    c = 2.99792458e8              # [m/s]
    # seconds in a day, week
    day_seconds = 86400.0     # [s]
    week_seconds = 604800.0   # [s]
    # gravitational constant
    mu_gps = 398600.50e9     # [m^3/s^2]
    mu_gal = 3.986004418e14  # [m^3/s^2]
    mu_glo = 3.986004418e14       # [m^3/s^2]
    # relativistic clock correction parameter
    F_gps = -4.442807633e-10
    F_gal = -4.442807309e-10
    # Julian date variables
    jdmjd0 = 2400000.5e0
    mjd1950 = 33282e0
    MjdAtGpsRef = 10969
    # Time difference offsets
    tai_gps = 19.0e0 / day_seconds
    # Leap seconds between UTC and GPS
    utc2gps_ls = 18.0e0   # [s]


# =============================================================================
#                          Relative azimuth elevation
# =============================================================================
def compute_az_el(sat_xyz, rec_xyz, lat, lon):
    """ Relative azimuth and elevation computation
    Reference:
        "Satellite Orbits", Montenbruck & Gill, chapter 6.2 pages 211-212
    """
    R = np.array([[-np.sin(lon),
                   +np.cos(lon),
                   +0.0],
                  [-np.sin(lat) * np.cos(lon),
                   -np.sin(lat) * np.sin(lon),
                   +np.cos(lat)],
                  [+np.cos(lat) * np.cos(lon),
                   +np.cos(lat) * np.sin(lon),
                   +np.sin(lat)]])

    s = np.dot(R, sat_xyz - rec_xyz)
    azimuth = np.arctan2(s[0], s[1])
    elevation = np.arctan2(s[2], np.sqrt(s[0] ** 2 + s[1] ** 2))
    # Correct for negative azimuth values
    if azimuth < 0:
        azimuth += 2. * np.pi

    return np.array([azimuth, elevation])


def ell2cart(lat, long, height):
    """ Coordinates transformation using WGS84 ellipsoid definition
    Input:
        - lat   : ellipsoidal latitude
        - lon   : ellipsoidal longitude
        - height: ellipsoidal height
    Output:
        - [x,y,z] vector of geocentric-cartesian coordinates
    Reference:
        - "Satellite Geodesy", Seeber
    """
    h = height
    lat = np.radians(lat)
    long = np.radians(long)
    a = Constants().WGS84_a
    f = Constants().WGS84_f
    N_bar = a / np.sqrt(1 - f * (2 - f) * (np.sin(lat)) ** 2)
    # Cartesian coordinates:
    x = (N_bar + h) * np.cos(lat) * np.cos(long)
    y = (N_bar + h) * np.cos(lat) * np.sin(long)
    z = (((1 - f) ** 2) * N_bar + h) * np.sin(lat)
    cart = np.array([x, y, z])
    return cart


def jd2gc_format(jd_time):
    """ Method to represent an epoch time in Gregorian Calendar format given
        the epoch expressed in Julian Date format.
        Reference: IAU-SOFA
    """
    # Minimum and maximum allowed JD
    jd_min, jd_max = -68569.5e0, 1e9

    # check if date is acceptable
    dj = jd_time.jd + jd_time.jd_frac
    if ((dj < jd_min) | (dj > jd_max)):
        raise ValueError("Date is not acceptable")
    else:
        # Copy the date, big then small, and re-align to midnight
        if jd_time.jd >= jd_time.jd_frac:
            d1 = jd_time.jd
            d2 = jd_time.jd_frac
        else:
            d1 = jd_time.jd_frac
            d2 = jd_time.jd
        d2 = d2 - 0.5e0
        # Separate day and fraction
        f1 = np.mod(d1, 1e0)
        f2 = np.mod(d2, 1e0)
        f = np.mod(f1 + f2, 1e0)
        if f < 0e0:
            f += 1e0
        d = round(d1 - f1) + round(d2 - f2) + round(f1 + f2 - f)
        jd = round(d) + 1
        # Express day in Gregorian Calendar
        l = jd + 68569
        # l = int(l)
        n = (4 * l) // 146097
        # n = round(n)
        l = l - (146097 * n + 3) // 4
        # l = round(l)
        ii = (4000 * (l+1)) // 1461001
        # ii = round(ii)
        l = l - (1461 * ii) // 4 + 31
        # l = round(l)
        k = (80 * l) // 2447
        iday = l - (2447 * k) // 80
        # id = round(id)
        l = k // 11
        # l = round(l)
        im = k + 2 - 12 * l
        # im = round(im)
        iy = 100 * (n - 49) + ii + l
        # iy = int(iy)
        # fraction of day
        fd = f
        # Compute time of date
        hour = fd * 24.0
        ihour = int(hour)
        fmin = (hour - ihour) * 60.0
        imin = int(fmin)
        sec = (fmin - imin) * 60.0
        # Save into GC format structure
        gc_time = GCFormat()
        gc_time.year = iy
        gc_time.month = im
        gc_time.day = iday
        gc_time.hour = ihour
        gc_time.min = imin
        gc_time.sec = sec

    return gc_time


def gpsTime2y_doy_hms(week, gpsTime):
    """ Function to convert GPS time to year, doy and hour minutes seconds
    """
    iepy = 1980
    iepd = 6
    # Hour of the day
    ss = math.fmod(gpsTime, Constants().day_seconds)
    hh = np.floor(ss / 3600)
    # -------------------- Conversion to time of year ------------------------
    # Number of days since iepy
    ndy = week * 7 + np.floor((gpsTime - ss) /
                              Constants().day_seconds +
                              0.5) + iepd - 1
    # Number of years since iepy
    ny = np.floor(ndy / 365.25)
    # Current year
    iy = np.floor(iepy + ny)
    # Day of the year
    DOY = np.floor(ndy - ny * 365.25 + 1)
    # Minutes
    ss = math.fmod(ss, 3600)
    mm = np.floor(ss / 60)
    # Seconds
    ss = math.fmod(ss, 60)
    return iy, DOY, hh, mm, ss


def gps_time_from_y_doy_hms(year, doy, hh, mm, sec):
    """
        Compute gps time from date as year, doy, hours, minutes, seconds
    """
    iepy = 1980
    iepd = 6
    if year < 100:
        if year > 80:
            year += 1900
        else:
            year += 2000
    ndy = ((year-iepy) * 365 + doy - iepd + ((year - 1901) / 4) -
           ((iepy - 1901) / 4))
    week = (ndy / 7)
    time = (round((week - int(week)) * 7) *
            Constants().day_seconds + hh * 3600.0e0 +
            mm * 60.0e0 + sec)
    # check if seconds are greater than a week
    if time >= Constants().week_seconds:
        week += 1
        time -= Constants().week_seconds
    else:
        pass
    return int(week), time


def date_to_doy(year: int, mon: int, day: int):
    """
        Compute day of the year from year, month and day of the month
    """
    leap = 0
    if np.mod(year, 4) == 0:
        leap = 1
    doy = (mon - 1) * 30 + mon / 2 + day
    if mon > 2:
        doy = doy - 2 + leap
    if ((mon > 8) & (np.mod(mon, 2) != 0)):
        doy = doy + 1
    return int(doy)


def doy_to_date(year: int, doy: int) -> int:
    """
        Compute date as year, month, day of the month from year and day of year
    """
    mon = np.int(doy/30) + 1
    dom = -1
    while dom < 0:
        dom = doy - date_to_doy(year, mon, 0)
        mon -= 1
    mon += 1
    if dom == 0:
        mon -= 1
        if ((mon == 11) | (mon == 4) | (mon == 6) | (mon == 9)):
            dom = 30
        elif mon == 2:
            if np.mod(year, 4) == 0:
                dom = 29
            else:
                dom = 28
        else:
            dom = 31
    return [int(year), int(mon), int(dom)]


def glo_time2gps_time(glo_day, glo_time, glo_year, ls):
    # Moscow day -> UTC day
    if glo_time >= 10800.0:
        utc_time = glo_time - 10800.0
    else:
        utc_time = glo_time - 10800.0 + Constants().day_seconds
    # compute gregorian date
    [day, mon, year] = glo_time2greg_day(glo_day, glo_year)
    # compute day of the year
    doy = date_to_doy(year, mon, day)
    # get gps week and time
    [gps_week, gps_time] = gps_time_from_y_doy_hms(year, doy, 0, 0, utc_time +
                                                   ls)
    return gps_week, gps_time


def glo_time2greg_day(nt, n4):
    """
        Compute Gregorian day from glonass time.
        Reference: GLONASS ICD
    """
    JD0 = 1461.0 * (n4 - 1.0) + nt + 2450082.5
    # julian date for the current date
    JDN = JD0 + 0.5
    a = JDN + 32044.0
    b = int((4.0 * a + 3.0) / 146097.0)
    c = a - int(146097.0 * b / 4.0)
    d = int((4 * c + 3.0) / 1461.0)
    e = c - int(1461.0 * d / 4)
    m = int((5 * e + 2.0) / 153.0)
    day = e - int((153.0 * m + 2.0) / 5) + 1
    mon = m + 3.0 - 12.0 * int(m / 10)
    year = 100.0 * b + d - 4800.0 + int(m / 10)
    return day, mon, year


def get_ls_from_date(year, month):
    """
        Function to get leap seconds since 1999. After 31st of December 1998
        the leap seconds were 14.
        The function covers the updates after that time.
    """
    if ((year > 2005) & (year < 2009)):
        ls = 14
    elif ((year >= 2009) & (year < 2012)):
        ls = 15
    elif year == 2012:
        if month < 7:
            ls = 15
        else:
            ls = 16
    elif ((year > 2012) & (year < 2015)):
        ls = 16
    elif year == 2015:
        if month < 7:
            ls = 16
        else:
            ls = 17
    elif year == 2016:
        if month < 7:
            ls = 17
        else:
            ls = 18
    elif year > 2016:
        ls = 18
    else:
        ls = 14
    return ls


# -----------------------------------------------------------------------------
#                        GPS to Gregorian Calendar format
# -----------------------------------------------------------------------------
def gps2gc_format(gps: GPSFormat) -> GCFormat:
    # Calculate JD
    jd_date = JDFormat()
    jd_date.jd = (Constants.jdmjd0 + Constants.mjd1950 +
                  Constants.MjdAtGpsRef + (gps.week - 1) * 7)
    sec = gps.sow / Constants.day_seconds
    daysInWeek = int(sec)
    jd_date.jd = jd_date.jd + daysInWeek
    jd_date.jd_frac = sec - daysInWeek
    # Transform from jd to gregorian calendar format
    gc_time = jd2gc_format(jd_date)
    return gc_time


# =============================================================================
#               From XYZ coord to spherical lat, lon, radial
# =============================================================================
def compute_xyz2sph(xyz_m):
    """ From coord to spherical lat, lon , radial
    """
    radial_m = LA.norm(xyz_m[0:2])
    lat_rad = np.arctan2(xyz_m[2], radial_m)
    lon_rad = np.arctan2(xyz_m[1], xyz_m[0])
    return np.array([lat_rad, lon_rad, radial_m])


# =============================================================================
#                    From spherical lat, lon to xyz spherical
# =============================================================================
def compute_sph2xyz(r, lat, lon):
    """ From spherical lat, lon to xyz spherical
    """
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.array([x, y, z])


# =============================================================================
#    From XYZ coord to spherical lat, lon, height, spherical radius as optional
#    parameters
# =============================================================================
def compute_xyz2sph_llh(xyz_m, re_m=None):
    """ From coord to spherical lat, lon , height over sphere with radius re
    """
    [lat_rad, lon_rad, radial_m] = compute_xyz2sph(xyz_m)
    if re_m is None:
        re_m = Constants().re
    height_m = radial_m-re_m

    return np.array([lat_rad, lon_rad, height_m])


# ---------------------------------------------------------------------------------------
#              From cartesian Earth fixed to local, topocentric coordinates
# ------------------------------------------------------------------------------------
def cart2enu(xyz, llh0):
    """
    Computes the local coordinates of a target point xyz w.r.t. an observer at
    ellipsoidal (WGS84) location llh0.
    """
    lat = llh0[0]
    lon = llh0[1]
    hgt = llh0[2]
    xyz0 = ell2cart(lat, lon, hgt)
    dxyz = xyz - xyz0
    # Compute rotation matrix
    R1 = get_r1(-(np.pi / 2.0 - np.deg2rad(lat)))
    R3 = get_r3(-(np.deg2rad(lon) + np.pi / 2.0))
    R = np.dot(R1, R3)
    # Compute local coordinates
    enu = np.dot(R, dxyz)
    return enu


def diff_time_s(w0, t0, w1, t1):
    """
        Compute time difference in seconds considering
        week and seconds of week.
    """
    dt = ((w1 - w0) * 86400 + (t1 - t0))
    return dt


def get_r1(theta):
    """ Method to get the rotation matrix around the x-axis of angle theta
        counterclockwise.
    """
    R1 = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(theta), -np.sin(theta)],
                   [0.0, np.sin(theta), np.cos(theta)]])
    return R1


def get_r2(theta):
    """ Method to get the rotation matrix around the y-axis of angle theta
        counterclockwise.
    """
    R2 = np.array([[np.cos(theta), 0.0, -np.sin(theta)],
                   [0.0, 1.0, 0.0],
                   [np.sin(theta), 0.0, np.cos(theta)]])
    return R2


def get_r3(theta):
    """ Method to get the rotation matrix around the z-axis of angle theta
        counterclockwise.
    """
    R3 = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                   [np.sin(theta), np.cos(theta), 0.0],
                   [0.0, 0.0, 1.0]])
    return R3