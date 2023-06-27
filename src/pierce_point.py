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
"""
import space_time_trafo as trafo
from ephemeris import Orbit
import numpy as np
import math
import numpy.linalg as LA

# *************************************************************************** #
#                              Ancillary data                                 #
# *************************************************************************** #
re = trafo.Constants().re                   # [m]
omega_zero_dot = trafo.Constants().omega_e  # earth rotation rate rad/s
c = trafo.Constants().c                     # speed of light

# =============================================================================
#                           XYZ sat SPIN correction
# =============================================================================


def compute_spin_corr(sat_xyz, rec_xyz):
    """ Compute satellite spin
    """
    rho = np.sqrt((sat_xyz[0] - rec_xyz[0]) ** 2 +
                  (sat_xyz[1] - rec_xyz[1]) ** 2 +
                  (sat_xyz[2] - rec_xyz[2]) ** 2)
    sat_spin = np.zeros((1, 3))
    sat_spin[0, 0] = (sat_xyz[0] +
                      (sat_xyz[1] * omega_zero_dot *
                      (rho / c)))
    sat_spin[0, 1] = (sat_xyz[1] - (sat_xyz[0] * omega_zero_dot *
                      (rho / c)))
    sat_spin[0, 2] = sat_xyz[2]

    return sat_spin[0]


# =============================================================================
#                            Pierce Point computation
# =============================================================================
def compute_pp(lat, lon, height, el, az, t, layer_h):
    """ Pierce Point computation method
    """
    # ***************** Spherical Earth's central angle ***************** #
    # angle between rover position and the projection of the pierce point
    # to the spherical Earth surface
    tmp = ((re + height) /
           (re + layer_h) * np.cos(el))

    psi_pp = np.pi / 2.0 - el - np.arcsin(tmp)
    # ******************* Latitude and Longitude PP ********************* #
    tmp = np.tan(psi_pp) * np.cos(az)
    ctg_lat = 1.0 / np.tan(lat)
    # Latitude
    phi_pp = (np.arcsin(np.sin(lat) * np.cos(psi_pp) +
              np.cos(lat) * np.sin(psi_pp) * np.cos(az)))

    ang = np.arcsin(np.sin(psi_pp) * np.sin(az) / np.cos(phi_pp))
    # Longitude
    if (((lat >= 0) & (+tmp > ctg_lat)) | ((lat < 0) & (-tmp > ctg_lat))):
        lambda_pp = lon + np.pi - ang
    else:
        lambda_pp = lon + ang

    sun_shift = math.fmod((t - 50400.0) * np.pi / 43200.0, 2.0 * np.pi)
    lon_s = math.fmod(lambda_pp + sun_shift, 2.0 * np.pi)
    slant_factor = 1.0 / np.sin(el + psi_pp)

    return np.array([psi_pp, lambda_pp, phi_pp, slant_factor,
                     sun_shift, lon_s])

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

# =============================================================================
#                     From XYZ coord to spherical lat, lon, height
#                           XYZ sat SPIN correction
# =============================================================================


def compute_xyz2sph(xyz):
    """ From coord to spherical lat, lon , height
    """
    height_s = LA.norm(xyz) - re
    p = LA.norm(xyz[0:2])
    lat_s = np.arctan2(xyz[2], p)
    lon_s = np.arctan2(xyz[1], xyz[0])

    return np.array([lat_s, lon_s, height_s])