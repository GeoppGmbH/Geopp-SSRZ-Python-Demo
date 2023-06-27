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
    ***************************************************************************
    Description:
    the tropospheric ratio w.r.t. the model is computed using the coefficients
    read from the SSRZ messages. The ratio is then multiplied by the
    model (UNB3) and the mapping function (GVMF) to get the slant tropospheric
    delay.
    ***************************************************************************
    Remark:
    concerning the grid troposphere, an inverse distance weighting (IDW)
    method is considered  as simple reference. The user can choose more
    sophisticated methods depending on the application.
"""

from ast import Try
import numpy as np
import tropo_model
import space_time_trafo as trafo
import interp_module as interp


class TropoComputation:
    """ Class to compute tropospheric delay based on the SSR messages.
        UNB3 model, Vienna Global Mapping Functions.

    """

    def __init__(self, epoch, ssr, rec_llh, el, doy, tropo_type, comp,
                 comp_index, md=None, f_dbg=None):
        """
            *******************************************************************
            Class of methods to compute tropospheric influences for a receiver
            location of a specific tropospheric component (e.g., dry).

            Input:
                - epoch      : epoch considered for the computation
                - ssr        : decoded ssr messages
                - rec        : receiver position ellipsoidal coordinates
                               (lat, lon, hgt)
                - cc         : index of the component considered
                - el         : elevation [rad]
                - doy        : day of year
                - tropo_type : type of tropospheric correction
                - comp       : tropospheric component (e.g., dry)
                - comp_index : tropo component index
        """
        if md is not None:
            self.md = md
        else:
            self.md = None
        self.ssr = ssr
        self.rec_llh = rec_llh
        self.cc = comp_index
        self.doy = doy
        # Mapping improvement extension
        if comp == "m":
            self.tropo = self.compute_mapping_improvement(el)
            return
        # compute the UNB3 model with 2 components: dry[0] and wet[1]
        doy = int(doy)
        model = tropo_model.get_model_troposphere(rec_llh, doy, f_dbg)
        self.model_dry_zenith = model[0]
        self.model_wet_zenith = model[1]
        # apply global mapping function
        zd = np.pi/2 - el
        lat_rad = np.deg2rad(self.rec_llh[0])
        lon_rad = np.deg2rad(self.rec_llh[1])
        [gmfh, gmfw] = tropo_model.compute_gmf(doy, lat_rad, lon_rad,
                                               self.rec_llh[2],
                                               zd)
        self.gmfh = gmfh
        self.gmfw = gmfw

        dgb_tp_h = '{:>3s}'.format("key")
        dgb_tp_h += ',{:>8s}'.format("epoch[s]")
        dgb_tp_h += ',{:>12s},{:>12s},{:>12s}'.format(
            "lat[deg]", "lon[deg]", "h[m]")
        dgb_tp_h += ',{:>12s},{:>10s},{:>1s}'.format("el[deg]", "doy", "c")
        dgb_tp_h += ',{:>9s}'.format("ZTDmodel")
        dgb_tp_h += ',{:>9s}'.format("gmf")
        dgb_tp_h += ',{:>9s}'.format("STDmodel")
        dgb_tp_h += ',{:>9s}'.format("scaleFa")
        dgb_tp_h += ',{:>9s}'.format("SSI [m]")

        dgb_tp = '{:3s}'
        dgb_tp += ',{:8.1f}'
        dgb_tp += ',{:12.7f},{:12.7f},{:12.5f}'
        dgb_tp += ',{:12.8f},{:10.5f},{:1s}'
        dgb_tp += ',{:9.5}'
        dgb_tp += ',{:9.5}'
        dgb_tp += ',{:9.5}'
        dgb_tp += ',{:9.5}'
        dgb_tp += ',{:9.5}'

        model_tot_slant = model[0] * gmfh + model[1] * gmfw
        self.model_tot_slant = model_tot_slant
        if tropo_type == 'rt':
            # compute regional tropo effect
            tropo_ratio = self.compute_rt()
        elif tropo_type == 'grt':
            # compute grid regional tropo effect
            tropo_ratio = self.compute_grt()
        elif tropo_type is None:
            return
        self.scale_factor = tropo_ratio
        # compute the value in meters
        if comp == 'd':
            self.tropo = gmfh * model[0] * tropo_ratio
        elif comp == 'w':
            self.tropo = gmfw * model[1] * tropo_ratio
            if f_dbg is not None and tropo_type == "grt":
                print(dgb_tp.format('GRT', epoch,
                                    rec_llh[0], rec_llh[1], rec_llh[2],
                                    np.rad2deg(el), doy, comp,
                                    model[1],
                                    gmfw,
                                    model[1]*gmfw,
                                    tropo_ratio,
                                    self.tropo), file=f_dbg)
        elif comp == "t":
            self.tropo = model_tot_slant * tropo_ratio

    def compute_rt(self):
        """ Method to compute the regional troposphere
        """
        # ground point coordinates
        gpo_llh = self.ssr.gpo_llh
        # compute lat, lon, hgt difference between receiver and ground point
        dlat = np.deg2rad(self.rec_llh[0] - gpo_llh[0])
        dlon = np.deg2rad(self.rec_llh[1] - gpo_llh[1])
        dhgt = self.rec_llh[2] - gpo_llh[2]
        # define horizontal and vertical correlation lengths
        corr_hor = 500e3  # [m]
        corr_ver = 2e3    # [m]
        # scale for the correlation length and the longitude
        # according to the latitude
        dN = dlat * 6378135.0 / corr_hor
        dE = dlon * np.cos(np.deg2rad(gpo_llh[0])) * 6378135.0 / corr_hor
        dh = dhgt / corr_ver
        # load coefficients
        a_coeff = self.ssr.coeff[self.cc]
        # Consider the metadata tag for different coefficient pattern
        if self.ssr.md_tag == 2:
            # ------------------------- Box pattern ---------------------------
            # set max order per lat, lon, hgt
            max_order_lat = self.ssr.max_order_lat
            max_order_lon = self.ssr.max_order_lon
            max_order_hgt = self.ssr.max_order_hgt
            # define the list of ii and jj in the coefficient of
            # the ssrz corrections
            # based on Table 3.6 of the SSRZ document and Figure 3.
            # example: a00      a10      a01      a20      a11      a02
            ll_list = np.array([0, 0, 1, 1, 2, 2])
            mm_list = np.array([0, 1, 0, 1, 0, 1])
            # adjust the lists to the order for latitute and longitude
            ll_list_adj = ll_list[np.where((ll_list <= max_order_lat) &
                                           (mm_list <= max_order_lon))]
            mm_list_adj = mm_list[np.where((ll_list <= max_order_lat) &
                                           (mm_list <= max_order_lon))]
            rt = 0
            nn = 0  # number of parameters
            # initialize chebyshev polynomials
            # Remark: the construction is assuming that max_order_hgt is 1
            for kk in range(max_order_hgt):
                cp_dlat = []
                for ll in range(max_order_lat):
                    cp_dlat.append([])
                    cp_dlon = []
                    for mm in range(max_order_lon):
                        if ((ll + mm <= max_order_lat + max_order_lon) &
                            (nn <= max_order_hgt * max_order_lat *
                             max_order_lon)):
                            cp_dlon.append([])
                            cp_dlat[ll] = self.compute_chebyshev_poly(ll, dN,
                                                                      cp_dlat)
                            cp_dlon[mm] = self.compute_chebyshev_poly(mm, dE,
                                                                      cp_dlon)
                            # find the correct index for the a coefficient
                            ii = np.where(((ll_list_adj == ll) &
                                           (mm_list_adj == mm)))[0][0]
                            a_lm = a_coeff[ii]
                            rt += a_lm * (dh ** kk) * cp_dlat[ll] * cp_dlon[mm]
                            nn += 1
        else:
            # ----------------------- Triangle pattern ------------------------
            max_order_hor = self.ssr.max_hor[self.cc]
            max_order_hgt = self.ssr.max_hgt[self.cc]
            # define the list of ii and jj in the coefficient of
            # the ssrz corrections
            # based on Table 3.6 of the SSRZ document and Figure 3.
            # example: a00      a10      a01      a20      a11      a02
            ll_list = np.array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3])
            mm_list = np.array([0, 1, 0, 2, 1, 0, 3, 2, 1, 0])
            # adjust the lists to the order for latitute and longitude
            ll_list_adj = ll_list[np.where((ll_list <= max_order_hor) &
                                           (mm_list <= max_order_hor))]
            mm_list_adj = mm_list[np.where((ll_list <= max_order_hor) &
                                           (mm_list <= max_order_hor))]
            rt = 0
            nn = 0  # number of parameters
            # initialize chebyshev polynomials
            # Remark: the construction is assuming that max_order_hgt is 1
            for kk in range(max_order_hgt + 1):
                cp_dlat = []
                for ll in range(max_order_hor + 1):
                    cp_dlat.append([])
                    cp_dlon = []
                    for mm in range(max_order_hor + 1 - ll):
                        cp_dlon.append([])
                        cp_dlat[ll] = self.compute_chebyshev_poly(ll, dN,
                                                                  cp_dlat)
                        cp_dlon[mm] = self.compute_chebyshev_poly(mm, dE,
                                                                  cp_dlon)
                        # find the correct index for the a coefficient
                        ii = np.where(((ll_list_adj == ll) &
                                       (mm_list_adj == mm)))[0][0]
                        a_lm = a_coeff[ii]
                        rt += a_lm * (dh ** kk) * cp_dlat[ll] * cp_dlon[mm]
                        nn += 1
        return rt

    def compute_chebyshev_poly(self, ii, x, cheb_poly):
        """ Recursive formulation for Chebyshev polynomials.
        Input:
            ii: index
            x : value
            cheb_poly: polynomial
        Output:
            computed ii value of the polynomial
        """
        if ii == 0:
            t = 1
        elif ii == 1:
            t = x
        elif ii == 2:
            t = 2 * x ** 2 - 1
        else:
            t = 2 * x * cheb_poly[ii-1] - cheb_poly[ii-2]

        return t

    def compute_grt(self):
        """ Method to compute the influence of the gridded troposphere
            for the user location through interpolation based on IDW.
            Be aware that the grid is a ratio relative to the Saastamoinen
            model, therefore independent from the height.
        """
        # get gridded troposphere
        data = self.ssr.grid_values[self.cc]
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
        hei0 = self.rec_llh[2]
        # query point for the interpolation
        latq = self.rec_llh[0]
        lonq = self.rec_llh[1]
        heiq = self.rec_llh[2]
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
                     np.cos(np.deg2rad(lon_pt)) - dy *
                     np.sin(np.deg2rad(lat_pt)) *
                     np.sin(np.deg2rad(lon_pt)) +
                     dz * np.cos(np.deg2rad(lat_pt)))
            east = (-dx * np.sin(np.deg2rad(lon_pt)) +
                    dy * np.cos(np.deg2rad(lon_pt)))
            E = np.append(E, east)
            N = np.append(N, north)
        # define the query point in the local system
        [xq, yq, zq] = trafo.ell2cart(latq, lonq, heiq)
        dxq = xq - x0
        dyq = yq - y0
        dzq = zq - z0
        northq = (-dxq * np.sin(np.deg2rad(latq)) *
                  np.cos(np.deg2rad(lonq)) - dyq *
                  np.sin(np.deg2rad(latq)) *
                  np.sin(np.deg2rad(lonq)) + dzq *
                  np.cos(np.deg2rad(latq)))
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
        tropo_ratio_interp = interpolation.results

        return tropo_ratio_interp

    def compute_mapping_improvement(self, elevation):
        """
            The method computes the mapping improvement for a given elevation.
        """
        try:
            max_order_hor = self.ssr.max_hor[self.cc]
            max_order_hgt = self.ssr.max_hgt[self.cc]
        except AttributeError:
            print("Warning: metadata message is missing. MF set to NaN.")
            mfi = np.nan
            return mfi
        max_el = self.md.md_gr.md_block.reg_tropo.blk[0].cff[self.cc].max_el
        # Transform to rad
        max_el = np.deg2rad(max_el)
        if elevation > max_el:
            mfi = 0.0
        else:
            mfi = 0.0
            coeff = self.ssr.coeff[self.cc]
            # Normalized elevation
            epsilon = (max_el - elevation) / max_el
            # Scaled difference between rover and ground point height
            hgt_gpo = self.md.md_gr.md_block.reg_tropo.blk[0].hgt_gpo
            hgt = self.rec_llh[2]
            dh = (hgt - hgt_gpo) * 1.0e-3
            # define the list of ii and jj in the coefficient of
            # the ssrz corrections
            # based on Table 3.6 of the SSRZ document and Figure 3.
            # example: a00      a10      a01      a20      a11      a02
            p_list = np.array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3])
            q_list = np.array([0, 1, 0, 2, 1, 0, 3, 2, 1, 0])
            for p in range(max_order_hgt + 1):
                for q in range(max_order_hor + 1):
                    # find the correct index for the a coefficient
                    ii = np.where(((p_list == p) &
                                   (q_list == q)))[0][0]
                    mfi += (coeff[ii] * (1.0 - np.cos(epsilon * np.pi / 2.0)) *
                            epsilon ** q * dh ** p)
        return mfi
