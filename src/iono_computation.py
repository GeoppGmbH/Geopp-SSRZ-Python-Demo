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
import space_time_trafo as trafo
import interp_module as interp
import pierce_point as pp
import pierce_point_origin as ppo


class IonoComputation:
    def __init__(self, ssr, epoch, isys, isat, state, rec, system, prn, f1,
                 iono_type, week=None, ephemeris=None, md=None,
                 el_ellips=None, gsi_ppo=None, rsi_ppo=None):
        """
        Class with methods to compute ionospheric influence on a receiver
        location for a particular satellite and for a specific frequency.    
        ***********************************************************************
        Description:
        the class IonoComputation includes methods to compute the STEC
        for a satellite + frequency for a specific receiver location.
        It takes in input the ssrz ionospheric message.
        Depending on the iono type, different functional models
        are considered (either spherical harmonics or chebyshev).
        The satellite and receiver positions are passed to the PiercePoint class,
        which computes pierce point parameters, but for the gridded iono.
        The gridded iono (iono type 'gri') is interpolated using a IDW method
        using the 'interp_module' module.
        Input:
            - epoch     : epoch considered for the computation
            - state     : satellite state vector
            - rec       : receiver position
            - system    : GNSS system involved
            - ID        : ID of the satellite considered
            - f1        : frequency considered
            - iono_type : ionosphere type (e.g. 'gvi' or 'rsi')
            - isys      : index of gnss system to consider
            - isat      : satellite index to consider for the gnss system isys
            - week      : GPS week
            - ephemeris : set of ephemeris from navigation data
            - md        : metadata
            - el_ellips : elippsoidal elevation
            - gsi_ppo   : global satellite-dependent PPO

        """
        if md is None:
            self.md = None
        else:
            self.md = md
        self.system = system
        self.prn = prn
        self.ssr = ssr
        self.ephemeris = ephemeris
        self.epoch = epoch
        self.sv_xyz = state[0:3]
        self.rec_xyz = rec['cartesian']
        self.rec_llh = rec['ellipsoidal']
        self.re = trafo.Constants().re
        self.omega_e = trafo.Constants().omega_e     # [rad/s]
        self.system = system
        self.gsi_ppo = gsi_ppo
        self.rsi_ppo = rsi_ppo
        self.layers = 1       # Currently, hard-coded value
        self.stec_corr_f1 = 0
        self.stec = 0
        self.strg = ''
        # select the type of iono parameter to compute
        if iono_type == 'gvi':
            iono = ssr
            # global ionosphere based on spherical harmonics
            for ll in range(iono.n_layer):  # always 1 here
                self.height = iono.hgt
                self.sh_deg = iono.deg
                self.sh_ord = iono.ord
                self.c = iono.cos_coeff
                self.s = iono.sin_coeff

                [lat_sph, lon_sph, height_sph,
                 el, az, psi_pp,
                 lambda_pp, phi_pp,
                 sf, sun_shift, lon_s,
                 p_nm, p_cos, p_sin, m, n,
                 vtec] = self.compute_global_iono()
                stec = vtec * sf
                self.stec += stec
                self.stec_corr_f1 += 40.3 * 1.0e16 / (f1 * f1) * stec
                self.strg = "".join([self.strg, '### SV pos/vel for SV ', prn, ' at ',
                                     f'{epoch}',
                                     ': ', '{:16.4f}'.format(state[0]), '   ',
                                     '{:16.4f}'.format(state[1]), '   ',
                                     '{:16.4f}'.format(
                                         state[2]), ' [m]', '   ',
                                     '{:9.4f}'.format(state[3]), '   ',
                                     '{:9.4f}'.format(state[4]), '   ',
                                     '{:9.4f}'.format(
                                         state[5]), ' [m/s]', '\n',
                                     'PPt at t=',
                                     f'{epoch}', '(sun shift= ',
                                     '{:11.8f}'.format(
                                         sun_shift * 180 / np.pi),
                                     ' deg)', ' \n',
                                     'PPt from Ref phi_R= ',
                                     '{:11.8f}'.format(lat_sph * 180 / np.pi),
                                     ' lam_R=',
                                     '{:11.8f}'.format(lon_sph * 180 / np.pi),
                                     ' rE+hR= ',
                                     '{:10.3f}'.format(height_sph + 6370000),
                                     '(spherical!)', '\n',
                                     'PPt from Ref to SV at elev= ',
                                     '{:11.8f}'.format(el * 180 / np.pi),
                                     ' azim ',
                                     '{:11.8f}'.format(az * 180 / np.pi),
                                     '(spherical!)', '\n',
                                     'PPt psi_pp= ',
                                     '{:11.8f}'.format(psi_pp * 180 / np.pi),
                                     ' phi_pp ',
                                     '{:11.8f}'.format(phi_pp * 180 / np.pi),
                                     ' lam_pp ',
                                     '{:11.8f}'.format(
                                         lambda_pp * 180 / np.pi),
                                     ' lon_S ',
                                     '{:11.8f}'.format(lon_s * 180 / np.pi),
                                     ' rE+hI: ',
                                     '{:10.3f}'.format(
                                         self.height * 1000 + 6370000),
                                     '\n'
                                     'Pnm : '])
                # Lagrange Polynomials
                for o in range(len(p_nm)):
                    m_ind = int(m[o])
                    n_ind = int(n[o])
                    self.strg = "".join([self.strg, 'P(', f'{n_ind}', ',',
                                         f'{m_ind}', ')=',
                                         '{:7.4f}'.format(p_nm[o]), '; '])
                # Cosines
                self.strg = "".join([self.strg, '\n', 'Pcos: '])
                for o in range(len(p_cos)):
                    m_ind = int(m[o])
                    n_ind = int(n[o])
                    self.strg = "".join([self.strg, 'P(', f'{n_ind}', ',',
                                         f'{m_ind}', ')=',
                                         '{:7.4f}'.format(p_cos[o]), '; '])
                # Sines
                self.strg = "".join([self.strg, '\n', 'Psin: '])
                for o in range(len(p_sin)):
                    m_ind = int(m[o])
                    n_ind = int(n[o])
                    self.strg = "".join([self.strg, 'P(', f'{n_ind}', ',',
                                         f'{m_ind}', ')=',
                                         '{:7.4f}'.format(p_sin[o]), '; '])
                self.strg = "".join([self.strg, '\n',
                                     'Sum VTEC=',
                                     '{:6.3f}'.format(vtec),
                                     '[TECU]', ',', ' sf=',
                                     '{:6.3f}'.format(sf),
                                     ',', 'STEC=',
                                     '{:6.3f}'.format(stec),
                                     '[TECU]', '\n',
                                     'SSR_VTEC: SV', prn,
                                     ' Have SSR VTEC Iono slant influence: ',
                                     '{:6.3f}'.format(stec), '[TECU]',
                                     '{:6.3f}'.format(self.stec_corr_f1),
                                     '[m-L1]', '\n'])
        # ------------------------Global or Regional STEC ---------------------
        elif ((iono_type == 'gsi') | (iono_type == 'rsi')):
            # global or regional satellite-dependent ionosphere
            layer_hgt = ssr.layer_hgt
            if ssr.n_order > 0:
                max_order = ssr.n_order - 1
            else:
                max_order = ssr.n_order
            if iono_type == 'gsi':
                a_coeff = ssr.gsi_coeff
                gpo = None
            else:
                a_coeff = ssr.coeff
                gpo = [ssr.gpo_lat, ssr.gpo_lon, ssr.gpo_hgt]
            nc_max = ssr.tot_coeff
            # PPO phi and lambda
            if not self.rsi_ppo:
                self.rsi_ppo = ppo.PiercePointOrigin(prn, gpo[0], gpo[1],
                                                     gpo[2], ssr.gpo_update,
                                                     layer_hgt)
            # get phi and lambda of the currently valid ppo
            [phi_ppo, lambda_ppo,
             height] = self.rsi_ppo.get_rsi_ppo(week, epoch, system,
                                                ephemeris)
            lon_ppo = lambda_ppo  # lon_s_ppo
            # get PP coordinates
            [psi_pp, lon_pp,
             phi_pp, sf_pp,
             sun_shift_pp,
             lon_s_pp] = self.compute_receiver_pierce_point(layer_hgt)
            # time difference between current and last PPO update epoch
            dt = ((week - self.rsi_ppo.last_update_week) *
                  604800) + (epoch - self.rsi_ppo.last_update_epoch)
            # compute dN and dE based on phi and lambda of PPO and PP and dt
            [dN_pp, dE_pp] = self.compute_dne_ppo_pp(phi_ppo, lon_ppo,
                                                     phi_pp, lon_pp,
                                                     dt, 'rsi')
            vtec = self.compute_sat_vtec(a_coeff, max_order, nc_max, isys,
                                         isat, dN_pp, dE_pp)
            # compute stec
            self.stec = sf_pp * vtec
            # compute stec for f1 frequency
            self.stec_corr_f1 = 40.3e16 / (f1 * f1) * self.stec
        # ---------------------------- Global STEC ----------------------------
        elif iono_type == 'gri':
            # gridded ionosphere
            if len(self.ssr.grid_values) == 0.0:
                vtec = np.nan
            else:
                vtec = self.compute_gri(isys, isat)
            # the gridded ionosphere is defined at the station height.
            # As a consequence, the slant factor is 1/sin(el)
            sf = 1.0 / np.sin(el_ellips)
            iono_pp = self.compute_receiver_pierce_point(400)
            sf = iono_pp[3]
            self.stec = sf * vtec
            self.stec_corr_f1 = 40.3e16 / (f1 * f1) * self.stec

    def __str__(self):
        return self.strg

# =============================================================================
#                       Legendre polynomials computation
# =============================================================================
    def compute_legendre_poly(self, deg, order, lat_pp, lon_s):
        """ Recursive Legendre polynomials computation
        """
        x = np.sin(lat_pp)
        # ------ Calculate Legendre polynomials with recursive algorithm ---- #
        max_val = np.max([order, deg])
        nmax = int(max_val + 1)
        p = np.zeros((nmax, nmax))
        p[0][0] = 1.0
        for m in range(1, nmax, 1):
            p[m][m] = (2 * m - 1) * np.sqrt((1 - x * x)) * p[m - 1][m - 1]

        for m in range(1, nmax - 1, 1):
            p[m + 1][m] = (2 * x + 1) * x * p[m][m]

        for m in range(0, nmax, 1):
            for n in range(m + 1, nmax, 1):
                p[n][m] = 1 / (n - m) * ((2 * n - 1) * x * p[n - 1][m] -
                                         (n + m - 1) * p[n - 2][m])
        # ----------- Compute associated Legendre polynomials Nnm ----------- #
        p_nm = []
        p_cos = []
        p_sin = []

        m_ind = []
        n_ind = []
        # Loop
        for n in range(0, nmax, 1):
            mmax = int(np.min([n, order])) + 1
            for m in range(0, mmax, 1):
                s2 = (((2 * n + 1) * math.factorial(n - m)) /
                      (math.factorial(n + m)))
                if (m == 0):
                    n_nm = np.sqrt(1 * s2) * p[n][m]
                else:
                    n_nm = np.sqrt(2 * s2) * p[n][m]
                # Ploynomials computation
                p_nm = np.append(p_nm, n_nm)
                p_cos = np.append(p_cos, n_nm * np.cos(m * lon_s))
                p_sin = np.append(p_sin, n_nm * np.sin(m * lon_s))
                m_ind = np.append(m_ind, m)
                n_ind = np.append(n_ind, n)

        return (p_nm, p_cos, p_sin, m_ind, n_ind)

# =============================================================================
#                           VTEC computation
# =============================================================================
    def compute_vtec_legendre(self, order, degree, c_nm, s_nm, p_cos, p_sin):
        """Computation of the VTEC
        """
        # Initialize variables
        vtec = 0
        nmax = int(degree + 1)
        ii = 0
        # Loop
        for n in range(0, nmax, 1):
            mmax = int(np.min([n, order])) + 1
            for m in range(0, mmax, 1):
                tot_1 = 0
                tot_2 = 0
                if m == 0:
                    tot_1 = c_nm[m][0][n] * p_cos[ii]
                else:
                    tot_2 = (c_nm[m][0][n] * p_cos[ii] +
                             s_nm[m - 1][0][n - 1] *
                             p_sin[ii])
                # Update VTEC and index
                vtec += tot_1 + tot_2
                ii += 1

        return vtec

# =============================================================================
#                        Global ionospheric corrections
# =============================================================================
    def compute_global_iono(self):
        """ Computation of IONO correction per satellite
        """
        # Initialize variables
        c_nm = []
        s_nm = []
        c_print = np.array(self.c)
        s_print = np.array(self.s)
        order = self.sh_ord
        deg = self.sh_deg
        nc = len(self.c)
        ns = len(self.s)
        h = self.height * 1e3
        index = 0
        # Loop over order for cosines
        for ii in range(int(order) + 1):
            c_nm.append([])
            if index < nc:
                c2append = np.concatenate((np.zeros(ii),
                                           c_print[index:index +
                                                   (int(deg) +
                                                    1 - ii)]), axis=0)
                c_nm[ii].append(c2append)
            index = index + (int(deg) + 1 - ii)
        # Loop over order for sines
        index = 0
        for jj in range(int(order)):
            s_nm.append([])
            if index < ns:
                s2append = np.concatenate((np.zeros(jj),
                                           s_print[index:index +
                                           (int(deg) - jj)]), axis=0)
                s_nm[jj].append(s2append)
            index = index + (int(deg) - jj)
        # ******************************************************************* #
        #                                                                     #
        #                       Pierce Point computation                      #
        #                                                                     #
        # ******************************************************************* #
        # apply spin correction
        xyz_spin = pp.compute_spin_corr(self.sv_xyz, self.rec_xyz)
        # compute spherical coordinates
        [lat_sph, lon_sph, height_sph] = pp.compute_xyz2sph(self.rec_xyz)
        # compute az and el using spherical coordinates
        [az, el] = trafo.compute_az_el(xyz_spin, self.rec_xyz, lat_sph, lon_sph)
        # compute pierce point
        [psi_pp, lambda_pp,
         phi_pp, sf,
         sun_shift, lon_s] = pp.compute_pp(lat_sph, lon_sph, height_sph,
                                           el, az, self.epoch, h)
        # compute Legendre polynomials
        [p_nm, p_cos, p_sin,
         m, n] = IonoComputation.compute_legendre_poly(self, deg, order,
                                                       phi_pp, lon_s)
        # compute vtec
        vtec = IonoComputation.compute_vtec_legendre(self, order, deg, c_nm,
                                                     s_nm, p_cos, p_sin)

        return (lat_sph, lon_sph, height_sph, el, az, psi_pp,
                lambda_pp, phi_pp,
                sf, sun_shift, lon_s, p_nm, p_cos, p_sin, m, n, vtec)

    # =========================================================================
    #                  Satellite-depenendent ionospheric vtec
    # =========================================================================
    def compute_sat_vtec(self, a_coeff, max_order, nc_max, idx_gnss, idx_sat,
                         dN, dE):
        """
            Computation of satellite dependent VTEC for both gsi and rsi.
            The computation is based on  Chebyshev polynomials.
            The a_coeff in input are not considered to be already in
            the correct order of application.
            Recursive formulation for Chebyshev polynomials
            if ii == 0:
                t = 1
            elif ii == 1:
                t = x
            elif ii == 2:
                t = 2 * x ** 2 - 1
            else:
                t = 2 * x * cheb_poly[ii-1] - cheb_poly[ii-2]
        """
        # define the list of ii and jj in the coeff of the ssrz corrections
        if nc_max == 1:
            ii_list = np.array([0])
            jj_list = np.array([0])
        elif nc_max == 3:
            ii_list = np.array([0, 0, 1])
            jj_list = np.array([0, 1, 0])
        elif nc_max == 4:
            ii_list = np.array([0, 0, 1, 1])
            jj_list = np.array([0, 1, 0, 1])
        elif nc_max == 5:
            ii_list = np.array([0, 0, 1, 0, 2])
            jj_list = np.array([0, 1, 0, 2, 0])
        elif nc_max == 6:
            ii_list = np.array([0, 0, 1, 0, 1, 2])
            jj_list = np.array([0, 1, 0, 2, 1, 0])
        vtec = 0
        # initialize chebyshev polynomials
        cp_dN = []
        cp = []
        nc = 0  # initialize number of coefficients
        for ii in range(max_order + 1):
            cp_dN.append([])
            cp_dE = []
            cp.append([])
            for jj in range(max_order + 1):
                if ((ii + jj <= max_order) & (nc <= nc_max)):
                    cp[ii].append([])
                    cp_dE.append([])
                    cp_dN[ii] = self.compute_chebyshev_poly(ii, dN, cp_dN)
                    cp_dE[jj] = self.compute_chebyshev_poly(jj, dE, cp_dE)
                    # find the correct index for the a coefficient
                    kk = np.where(((ii_list == ii) & (jj_list == jj)))[0][0]
                    try:
                        a_ij = a_coeff[kk][idx_gnss][idx_sat]
                    except IndexError:
                        print(kk, idx_gnss, idx_sat)
                    vtec += a_ij * cp_dN[ii] * cp_dE[jj]
                    cp[ii][jj] = cp_dN[ii] * cp_dE[jj]
                    nc += 1
        return vtec

    # =========================================================================
    #                       Method for recursive Cheb poly
    # =========================================================================
    def compute_chebyshev_poly(self, ii, x, cheb_poly):
        """ Recursive formulation for Chebyshev polynomials
        """
        if ii == 0:
            t = 1
        elif ii == 1:
            t = x
        elif ii == 2:
            t = 2 * x ** 2 - 1
        else:
            t = 2 * x * cheb_poly[ii - 1] - cheb_poly[ii - 2]

        return t

    # =========================================================================
    #                          Compute Receiver Pierce Point
    # =========================================================================
    def compute_receiver_pierce_point(self, layer_hgt):
        """ Computation of Pierce Point between receiver antenna (rec_xyz) and
            and satellite (sv_xyz) at layer height (layer_hgt).
        """
        # height needs to be in meter, but the input h is in km
        h = layer_hgt * 1e3
        # apply spin correction
        xyz_spin = pp.compute_spin_corr(self.sv_xyz, self.rec_xyz)
        # compute spherical coordinates
        [lat_sph, lon_sph, height_sph] = pp.compute_xyz2sph(self.rec_xyz)
        # compute az and el using spherical coordinates
        [az, el] = trafo.compute_az_el(xyz_spin, self.rec_xyz, lat_sph, lon_sph)
        # compute pierce point
        [psi_pp, lambda_pp,
         phi_pp, sf_pp,
         sun_shift_pp, lon_s_pp] = pp.compute_pp(lat_sph, lon_sph, height_sph,
                                                 el, az, self.epoch, h)
        lon_pp = lambda_pp  # lon_s_pp
        return psi_pp, lon_pp, phi_pp, sf_pp, sun_shift_pp, lon_s_pp

    # =========================================================================
    #                       Compute dN and dE from PPO and PP
    # =========================================================================
    def compute_dne_ppo_pp(self, phi_ppo, lon_ppo, phi_pp, lon_pp,
                           dt, iono_type):
        """
        Method to compute NOrth, East differences from PPO and PP
        """
        # correct for Earth rotation in time dt
        lon_pp += (dt / 43200.0) * np.pi
        # get unit vector of pierce point
        e_pp = trafo.compute_sph2xyz(1.0, phi_pp, lon_pp)
        # get unit vector of pierce point origin
        e_ppo = trafo.compute_sph2xyz(1.0, phi_ppo, lon_ppo)
        # North Pole unit vector
        e_np = np.array([0., 0., 1.])
        n = np.cross(e_np, e_ppo) / LA.norm(np.cross(e_np, e_ppo))
        # Distance npp of the rover pierce point PP to the meridian plane
        # through PPO
        n_pp = np.dot(n, e_pp)
        # Angular distance of the PP from the meridian plane through PPO
        dE_pp_t = np.arcsin(n_pp)
        # Negative longitude of PP in the transversal system
        e_f = (e_pp - n_pp * n) / LA.norm(e_pp - n_pp * n)
        n_bar = np.cross(e_f, e_ppo) / LA.norm(np.cross(e_f, e_ppo))
        # consider the sign change of the latitude at the equator
        if np.dot(n_bar, n) > 0:
            dN_pp_t = +np.arcsin(LA.norm(np.cross(e_f, e_ppo)))
        else:
            dN_pp_t = -np.arcsin(LA.norm(np.cross(e_f, e_ppo)))
        if iono_type == 'rsi':
            scale_dist = 6.37
            dN_pp = dN_pp_t * scale_dist
            dE_pp = dE_pp_t * scale_dist
        elif iono_type == 'gsi':
            # it uses a stereographic projection of the transversal system
            dN_pp = 2 * np.tan(dN_pp_t / 2)
            dE_pp = 2 * np.tan(dE_pp_t / 2)

        return dN_pp, dE_pp

    # =========================================================================
    #              Compute gridded ionosphere using IDW interpolation
    # =========================================================================
    def compute_gri(self, ii, jj):
        """ Method to compute the influence of the gridded ionosphere
            for the user location through interpolation based on IDW.
        """
        # get gridded ionosphere
        data = self.ssr.grid_values[ii][jj]
        # get grid llh from metadata
        grid_list = self.ssr.md_grid.grid_blk_list
        lat = []
        lon = []
        hgt = []
        for ii in range(len(grid_list)):  # loop over the number of grids
            chain_blk = grid_list[ii].chain_blk
            for jj in range(len(chain_blk)):  # loop over the number of chains
                lat = np.append(lat, chain_blk[jj].lat)
                lon = np.append(lon, chain_blk[jj].lon)
                hgt = np.append(hgt, chain_blk[jj].hgt)
        # define rerefence for the local coordinates
        lat0 = self.rec_llh[0]
        lon0 = self.rec_llh[1]
        # query point for the interpolation
        latq = self.rec_llh[0]
        lonq = self.rec_llh[1]
        hei0 = self.rec_llh[2]
        [x0, y0, z0] = trafo.ell2cart(lat0, lon0, hei0)
        E = []
        N = []
        lat_deg = []
        lon_deg = []
        for ii in range(len(lat)):
            lat_pt = np.rad2deg(lat[ii])
            lon_pt = np.rad2deg(lon[ii])
            lat_deg.append(lat_pt)
            lon_deg.append(lon_pt)
            [x, y, z] = trafo.ell2cart(lat_pt, lon_pt, hei0)
            dx = x - x0
            dy = y - y0
            dz = z - z0
            north = (-dx * np.sin(np.deg2rad(lat_pt)) *
                     np.cos(np.deg2rad(lon_pt)) -
                     dy * np.sin(np.deg2rad(lat_pt)) *
                     np.sin(np.deg2rad(lon_pt)) +
                     dz * np.cos(np.deg2rad(lat_pt)))
            east = (-dx * np.sin(np.deg2rad(lon_pt)) +
                    dy * np.cos(np.deg2rad(lon_pt)))
            E = np.append(E, east)
            N = np.append(N, north)
        # define the query point in the local system
        [xq, yq, zq] = trafo.ell2cart(latq, lonq, 0.0)
        dxq = xq - x0
        dyq = yq - y0
        dzq = zq - z0
        northq = (-dxq * np.sin(np.deg2rad(latq)) * np.cos(np.deg2rad(lonq)) -
                  dyq * np.sin(np.deg2rad(latq)) * np.sin(np.deg2rad(lonq)) +
                  dzq * np.cos(np.deg2rad(latq)))
        eastq = (-dxq * np.sin(np.deg2rad(lonq)) +
                 dyq * np.cos(np.deg2rad(lonq)))
        query = np.array([[eastq, northq]])
        # grid creation
        grid_en = np.array([[E[ii], N[ii]] for ii in range(len(E))])
        # directions
        directions = np.array([[E[i] - query[0][0],
                                N[i] - query[0][1]] for i in range(len(E))])
        distances = []
        for ii in range(len(directions)):
            distances = np.append(distances, np.linalg.norm(directions[ii]))
        # compute 2D interpolation
        values = np.array([[data[ii]] for ii in range(len(data))])
        interpolation = interp.Interpolator2D(grid_en, values)
        interpolation.IDW(query)
        vtec_interp = interpolation.results

        return vtec_interp
