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

import pierce_point as pp
import space_time_trafo as trafo
from ephemeris import Orbit

""" SSRZ Pierce Point Origin
    ***************************************************************************
    Description:
    the module contains the pierce point origin (PPO) class and methodes that
    are required to compute the state space influence (SSI) of the global (gsi)
    and regional STEC (rsi) model. The pierce point origin is stored
    seperately from the decoded states as the update of the PPO can be
    different from the update intervals of gsi and rsi SSR parameters
    (coefficients).
    ***************************************************************************
    References:
       - Geo++ SSRZ documentation v1.1.2
"""


class PiercePointOrigin:
    """ Class to store and update pierce point origin of the regional stec
        model.
    """

    def __init__(self, sat, gpo_lat, gpo_lon, gpo_hgt, gpo_update, layer_hgt):
        # static parameters
        self.sat = sat
        self.gpo_lat = gpo_lat
        self.gpo_lon = gpo_lon
        self.gpo_hgt = gpo_hgt
        self.gpo_update = gpo_update
        self.layer_hgt = layer_hgt * 1e3
        self.gpo_xyz = trafo.ell2cart(gpo_lat, gpo_lon, gpo_hgt)
        # dynamic parameters
        self.last_update_week = 0
        self.last_update_epoch = 0.0
        self.phi_ppo = None
        self.lambda_ppo = None
        self.height_sph = None

    # =========================================================================
    #                        Get Pierce Point Parameters
    # =========================================================================
    def get_ppo(self):
        return self.phi_ppo, self.lambda_ppo, self.height_sph

    # =========================================================================
    #   Compute new Pierce Point Origin (only based on time and ephemeris)
    # =========================================================================
    def compute_pierce_point_origin(self, week, epoch, gnss, ephemeris):
        orbit_ppo = Orbit(gnss, self.gpo_xyz, epoch, week)
        sat_clock = 'corrected'
        sv_xyz = orbit_ppo.compute_state_vector(ephemeris, sat_clock)
        # apply spin correction
        sat_spin = pp.compute_spin_corr(sv_xyz, self.gpo_xyz)
        # compute spherical coordinates
        [lat_sph,
         lon_sph,
         height_sph] = pp.compute_xyz2sph(self.gpo_xyz)
        # compute az and el using spherical coordinates
        [az, el] = trafo.compute_az_el(
            sat_spin, self.gpo_xyz, lat_sph, lon_sph)

        # compute pierce point
        [psi_ppo, lambda_ppo,
         phi_ppo, sf_ppo,
         sun_shift_ppo, lon_s_ppo] = pp.compute_pp(lat_sph, lon_sph,
                                                   height_sph, el, az, epoch,
                                                   self.layer_hgt)
        # Save objects
        self.phi_ppo = phi_ppo
        self.lambda_ppo = lambda_ppo
        self.height_sph = height_sph
        self.last_update_epoch = epoch
        self.last_update_week = week

        return self.get_ppo()

    # =========================================================================
    #      Get Regional STEC Pierce Point Origin latitude and longitute
    # =========================================================================
    def get_rsi_ppo(self, week, epoch, gnss, ephemeris):
        # Time interval
        dt = ((week - self.last_update_week) * 604800 +
              (epoch - self.last_update_week))
        dt = self.gpo_update + 1  # WORK-AROUND!!! to ensure correct PPO usage
        # check if week epoch are out of update length
        if (dt > self.gpo_update):
            # find last required PPO computation epoch
            ppo_epoch = epoch - (epoch % self.gpo_update)
            ppo_week = week
            # Check epoch
            if (ppo_epoch < 0):
                ppo_epoch = ppo_epoch + 604800
                ppo_week = week - 1
            # Compute ppo
            self.compute_pierce_point_origin(ppo_week, ppo_epoch, gnss,
                                             ephemeris)

        return self.get_ppo()

    # =========================================================================
    #         Get Regional STEC Pierce Point Origin latitude and longitute
    # =========================================================================
    def get_gsi_ppo(self, epoch, sv_xyz):
        # Check epoch
        if epoch - self.last_update_epoch <= self.gpo_update:
            self.last_update_epoch = epoch
            # Transform to spherical coordinates
            [self.phi_ppo, self.lambda_ppo,
             self.height_sph] = pp.compute_xyz2sph(sv_xyz)
        # Get PPO
        self.get_ppo()
