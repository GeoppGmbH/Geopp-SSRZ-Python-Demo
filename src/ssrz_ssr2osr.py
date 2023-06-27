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
    Collections of classes to translate SSR parameters in OSR for a user
    location.
    ***************************************************************************
"""
import numpy as np
from numpy import linalg as LA
import space_time_trafo as trafo
import iono_computation
import tropo_computation
import pierce_point as pp
from ephemeris import Orbit
import solid_earth_tides as tides
from datetime import datetime, timedelta


class ssr2osr:
    def __init__(self, ssr, md, week, epoch, nav_data, receiver,
                 f_out_iono, f_dbg, do_not_use_gnss=[], csv_out=False,
                 do_sed=True, sed_perm=False):
        """
            Class to compute the SSR influence for a user position using
            decoded SSRZ messages.

            Input:
                - ssr: ssr parameters at the considered epoch
                - md:  decoded metadata
                - week: GPS week of the computation
                - epoch: GPS TOW of the computation
                - nav_data: navigation data acquired from navigation RINEX
                - receiver: receiver WGS84 ellipsoidal coordinates and
                            cartesian coordinates
                - f_out_iono: debug file for global vtec computation
                - f_dbg: debug file
                - do_not_use_gnss: list of constellations not to be used
                - csv_out: if True, output SSI in csv format
                - sed_perm: if True, zero_tide tides in solid Earth tides
            Output:
                callable objects for the following SSR influences:
                - orbit
                - clock
                - code bias
                - phase bias
                - global ionosphere
                - global satellite-dependent ionosphere
                - regional ionosphere
                - grid ionosphere
                - regional troposphere
                - grid troposphere
                - shapiro effect
                - wind-up effect

            *******************************************************************
            Description:
            first, the satellite state vector is computed passing the ephemeris
            message to the class Orbit. Second, the ssr influence on the user
            position is computed for each satellite for all the components by
            calling the classes OrbCorr, ClockCorr, CodeBias, PhaseBias,
            Ionosphere, Troposphere, ShapiroEffect and WindUp.
            The __str__ method can be used to print the content of the
            message in a human readable format.
            *******************************************************************
        """
        # compute doy
        [iy, doy, hh, mm, ss] = trafo.gpsTime2y_doy_hms(week, epoch)
        # Transform epoch from gps to Gregorian calendar format
        epoch_gps = trafo.GPSFormat()
        epoch_gps.week = week
        epoch_gps.sow = epoch
        epoch_gc = trafo.gps2gc_format(epoch_gps)
        # compute doy with fractional part
        doy = doy + ((hh * 3600.0 + mm * 60.0 + ss) / 84600.0)
        self.week = week
        # satellites
        self.sats = []
        # receiver coordinates
        xyz = receiver['cartesian']
        lat = receiver['ellipsoidal'][0]
        lon = receiver['ellipsoidal'][1]
        height = receiver['ellipsoidal'][2]
        self.rec = receiver['cartesian']
        # ****************************
        # Variable initialization
        # ****************************
        # components initialization
        self.orb = []
        self.clk = []
        self.clk_hr = []
        self.cbias = []
        self.pbias = []
        self.gvi = []
        self.gsi = []
        self.gri = []
        self.rsi = []
        self.gt = []
        self.rt = []
        self.grt = []
        self.wup = []
        self.shap = []
        self.el = []
        self.az = []
        # satellite state vector
        self.sat_state = []
        # pierce point origins
        self.gsi_ppo = []
        self.rsi_ppo = []
        # string to output
        self.strg = ''
        # output type (csv flag)
        self.csv_out = csv_out
        # ****************************
        # loop over gnss considering timing blocks and satellite groups of the
        # low rate msg
        # ****************************
        ii = 0
        for bb in range(len(ssr.lr)):
            for sg in range(len(ssr.lr[bb])):
                lr_msg = ssr.lr[bb][sg]
                for kk in range(len(lr_msg.sv_array)):
                    ii = kk
                    self.sats.append([])
                    self.sats[ii] = lr_msg.sv_array[ii]
                    # append gnss component
                    self.orb.append([])
                    self.clk.append([])
                    self.clk_hr.append([])
                    self.cbias.append([])
                    self.pbias.append([])
                    self.gvi.append([])
                    self.gsi.append([])
                    self.gri.append([])
                    self.rsi.append([])
                    self.gt.append([])
                    self.rt.append([])
                    self.grt.append([])
                    self.wup.append([])
                    self.shap.append([])
                    self.sat_state.append([])
                    self.el.append([])
                    self.az.append([])
                    self.gsi_ppo.append([])
                    self.rsi_ppo.append([])
                    # list of satellite per gnss
                    sat_list = lr_msg.sv_array[kk]
                    if len(sat_list) == 0:
                        continue
                    gnss = sat_list[0][0]
                    # find the corresponding high rate timing block and
                    # satellite group
                    n_g_h = md.sat_gr.sat_md_blk.n_g_hr
                    for hh in range(n_g_h):
                        if gnss in md.sat_gr.sat_md_blk.hr_block[hh].gnss:
                            hr_sat_gr_idx = hh
                            break
                    n_timing_h = md.md_gr.md_block.hr_md_block.n_timing
                    for hh in range(n_timing_h):
                        if md.md_gr.md_block.hr_md_block.hr_timing_block[hh].bit_mask[hr_sat_gr_idx] == 1:
                            hr_t_blk_idx = hh
                            break
                    hr_msg = ssr.hr[hr_t_blk_idx][hr_sat_gr_idx]
                    # chech if the time tag of the hr is consistent with lr
                    if hr_msg.time_tag_15 < lr_msg.time_tag_15:
                        return
                    # find gnss index for hr msg
                    for hh in range(len(hr_msg.sv_array)):
                        if len(hr_msg.sv_array[hh]) > 0:
                            if hr_msg.sv_array[hh][0][0] == gnss:
                                kk_hr = hh  # gnss index for hr msg
                                break
                        else:
                            continue
                    if gnss in do_not_use_gnss:
                        continue
                    # ****************************
                    # loop over satellites
                    # ****************************
                    for jj in range(len(sat_list)):
                        sat = sat_list[jj]
                        # append satellite component per gnss
                        self.orb[ii].append([])
                        self.clk[ii].append([])
                        self.clk_hr[ii].append([])
                        self.cbias[ii].append([])
                        self.pbias[ii].append([])
                        self.gvi[ii].append([])
                        self.gsi[ii].append([])
                        self.gri[ii].append([])
                        self.rsi[ii].append([])
                        self.gt[ii].append([])
                        self.rt[ii].append([])
                        self.grt[ii].append([])
                        self.wup[ii].append([])
                        self.shap[ii].append([])
                        self.sat_state[ii].append([])
                        self.el[ii].append([])
                        self.az[ii].append([])
                        self.gsi_ppo[ii].append([])
                        self.rsi_ppo[ii].append([])
                        #######################################################
                        # Orbit/Clock/Bias
                        #######################################################
                        # get ephemeris for the correct satellite
                        prn = int(sat[1:])
                        iode = lr_msg.be_iod[ii][jj]
                        ephemeris = nav_data.get_ephemeris(gnss, prn, iode,
                                                           epoch)
                        if ephemeris is None:
                            print('WARNING: no ephemeris data for' + sat)
                            if jj == len(sat_list) - 1:
                                ii += 1  # increase system index
                            continue

                        if ((gnss == 'G') | (gnss == 'E') | (gnss == 'J')):
                            ls = 0
                        elif gnss == 'C':
                            ls = 14
                        else:
                            ls = None  # e.g. for GLO it is computed from eph
                            if jj == len(sat_list) - 1:
                                ii += 1  # increase system index
                                continue
                        orbit_p = Orbit(gnss, xyz, epoch, week, ls)
                        # compute satellite state vector correcting for
                        # the satellite clock, light time and Sagnac effect
                        sat_clock = True
                        light_time = True
                        sagnac_eff = True
                        sat_state = orbit_p.compute_state_vector(ephemeris,
                                                                 sat_clock,
                                                                 light_time,
                                                                 sagnac_eff)
                        self.sat_state[ii][jj] = sat_state
                        # Compute geometric range [m]
                        rho = np.sqrt((sat_state[0] - self.rec[0]) ** 2 +
                                      (sat_state[1] - self.rec[1]) ** 2 +
                                      (sat_state[2] - self.rec[2]) ** 2)
                        # Save satellite coordinates at transmission time
                        # (no application of Sagnac effect)
                        sat_state_no_sagnac = orbit_p.compute_state_vector(
                            ephemeris, sat_clock, light_time, False)
                        # frequency to be considered for correction
                        # computation as an example
                        # GPS/QZSS(L1), GLONASS(L1), Galileo(E1) and Beidou(2I)
                        if gnss == 'R':
                            ch = ephemeris.freq_num
                            fr = (1602 + ch * 9.0 / 16.0) * 1e6
                        elif gnss == 'C':
                            fr = 1561.098 * 1e6
                        else:
                            ch = 0
                            fr = 2.0 * 77.0 * 10.23 * 1e6
                        # wavelength
                        wavelength = trafo.Constants().c / fr  # [m]
                        # compute orbit and clock obs line corrections
                        if np.any(ssr.lr):
                            self.orb[ii][jj] = OrbCorr(lr_msg, hr_msg,
                                                       sat_state,
                                                       self.rec, kk, kk_hr,
                                                       jj).corr
                            dt = epoch - ssr.tt.gps_tow
                            clock_corr = ClockCorr(lr_msg, hr_msg, dt, kk,
                                                   kk_hr, jj)
                            self.clk[ii][jj] = clock_corr.corr
                            self.clk_hr[ii][jj] = clock_corr.corr_hr
                        else:
                            self.orb[ii][jj] = np.nan
                            self.clk[ii][jj] = np.nan
                        orb_out = self.make_output_format(self.orb[ii][jj])
                        clk_out = self.make_output_format(self.clk[ii][jj])
                        clk_hr_out = self.make_output_format(
                            self.clk_hr[ii][jj])
                        clk_tot_out = self.make_output_format(self.clk[ii][jj] +
                                                              self.clk_hr[ii][jj])

                        # code and phase bias
                        if np.any(ssr.lr):
                            # selection of the output signal
                            if ((gnss == 'G') | (gnss == 'J') | (gnss == 'R') |
                                                                (gnss == 'E')):
                                signal = '1C'
                            elif gnss == 'C':
                                signal = '2I'
                            self.cbias[ii][jj] = CodeBias(lr_msg, kk, jj, gnss,
                                                          signal).corr
                            self.pbias[ii][jj] = PhaseBias(lr_msg, kk, jj,
                                                           gnss, signal).corr
                            # compute wind up effect
                            self.wup[ii][jj] = WindUp(sat_state, fr, self.rec,
                                                      lat, lon).corr
                        else:
                            self.cbias[ii][jj] = np.nan
                            self.pbias[ii][jj] = np.nan
                            self.wup[ii][jj] = np.nan
                        if self.csv_out:
                            cbias = -self.cbias[ii][jj]
                            pbias = (-self.pbias[ii][jj] * trafo.Constants().c
                                     / fr)  # [m]
                        else:
                            cbias = self.cbias[ii][jj]
                            pbias = (self.pbias[ii][jj] * trafo.Constants().c
                                     / fr)  # [m]
                        if csv_out:
                            cbias_out = self.make_output_format(-cbias)
                            pbias_out = self.make_output_format(-pbias)
                        else:
                            cbias_out = self.make_output_format(cbias)
                            pbias_out = self.make_output_format(pbias)

                        #######################################################
                        #                   Wind-Up/Relativity
                        #######################################################
                        if self.csv_out:
                            wup = self.wup[ii][jj] / wavelength
                        else:
                            wup = self.wup[ii][jj]
                        wup_out = self.make_output_format(wup)

                        # compute relativistic shapiro effect
                        self.shap[ii][jj] = ShapiroEffect(sat_state[0:3],
                                                          self.rec).corr
                        shap_out = self.make_output_format(self.shap[ii][jj])
                        #######################################################
                        #                         Iono
                        #######################################################
                        iono_tot_tecu = 0
                        iono_tot_meters = 0
                        # compute global ionosphere
                        if (np.any(ssr.gvi)):
                            iono_type = 'gvi'
                            gvi_msg = ssr.gvi
                            # check time tag w.r.t. lr
                            if gvi_msg.time_tag_15 >= lr_msg.time_tag_15:
                                glo_vtec = Ionosphere(gvi_msg, epoch, kk, jj,
                                                      sat_state,
                                                      receiver, gnss, sat,
                                                      fr, iono_type,
                                                      dbg_out=f_out_iono)
                                self.gvi[ii][jj] = glo_vtec.corr
                                iono_tot_tecu += glo_vtec.corr
                                iono_tot_meters += glo_vtec.corr_f1
                            else:
                                self.gvi[ii][jj] = np.nan
                        else:
                            self.gvi[ii][jj] = np.nan
                        gvi_out = self.make_output_format(self.gvi[ii][jj])
                        # satellite-dependent global ionosphere
                        if len(lr_msg.gsi_coeff) > 0:
                            iono_type = 'gsi'
                            glo_stec = Ionosphere(lr_msg, epoch, kk, jj,
                                                  sat_state,
                                                  receiver, gnss, sat,
                                                  fr, iono_type,)
                            self.gsi[ii][jj] = glo_stec.corr
                            iono_tot_tecu += glo_stec.corr
                            iono_tot_meters += glo_stec.corr_f1
                        else:
                            self.gsi[ii][jj] = np.nan
                        gsi_out = self.make_output_format(self.gsi[ii][jj])
                        # satellite-dependent regional ionosphere
                        if (np.any(ssr.rsi[bb][sg])):
                            rsi_msg = ssr.rsi[bb][sg]
                            # check the time tag w.r.t. low rate
                            if rsi_msg.time_tag_15 >= lr_msg.time_tag_15:
                                iono_type = 'rsi'
                                # gnss is the RINEX Code of the system
                                # sat is a string ('G12' or 'C02')
                                reg_iono = Ionosphere(ssr=rsi_msg, epoch=epoch,
                                                      week=week, ephemeris=ephemeris,
                                                      isys=ii, isat=jj, state=sat_state,
                                                      rec=receiver, gnss=gnss, sat=sat,
                                                      fr=fr, iono_type=iono_type,
                                                      rsi_ppo=self.rsi_ppo[ii][jj])
                                self.rsi[ii][jj] = reg_iono.corr
                                self.rsi_ppo[ii][jj] = reg_iono.rsi_ppo

                                iono_tot_tecu += reg_iono.corr
                                iono_tot_meters += reg_iono.corr_f1
                            else:
                                self.rsi[ii][jj] = np.nan
                        else:
                            self.rsi[ii][jj] = np.nan
                        rsi_out = self.make_output_format(self.rsi[ii][jj])
                        # ellipsoidal elevation
                        [self.az[ii][jj],
                         self.el[ii][jj]] = trafo.compute_az_el(sat_state[0:3],
                                                                self.rec,
                                                                np.deg2rad(
                                                                    lat),
                                                                np.deg2rad(lon))
                        # grid ionosphere
                        if (np.any(ssr.gri[bb][sg])):
                            # get data for the closest available grid
                            gri_msg = get_closest_grid(corr_list=ssr.gri[bb][sg],
                                                       llh0=[lat, lon, height])
                            if gri_msg is None:
                                self.gri[ii][jj] = np.nan
                            else:
                                # check the time tag w.r.t. low rate
                                if gri_msg.time_tag_15 >= lr_msg.time_tag_15:
                                    iono_type = 'gri'
                                    grid_iono = Ionosphere(gri_msg, epoch, kk, jj,
                                                           sat_state,
                                                           receiver, gnss, sat,
                                                           fr, iono_type,
                                                           md=md,
                                                           el=self.el[ii][jj])
                                    self.gri[ii][jj] = grid_iono.corr
                                    iono_tot_tecu += grid_iono.corr
                                    iono_tot_meters += grid_iono.corr_f1
                                else:
                                    self.gri[ii][jj] = np.nan
                        else:
                            self.gri[ii][jj] = np.nan
                        gri_out = self.make_output_format(self.gri[ii][jj])

                        # total iono output
                        iono_tot_m_out = self.make_output_format(
                            iono_tot_meters)
                        iono_tot_out = self.make_output_format(iono_tot_tecu)

                        #######################################################
                        # Tropo
                        #######################################################
                        tropo_tot_corr_meters = 0
                        # Global tropo
                        gt = 0.0  # Global tropo is assumed to be zero
                        gt_out = self.make_output_format(gt)
                        # regional troposphere
                        self.rt[ii][jj] = 0.0
                        mfi = 0.0e0
                        ssr_rt = ssr.rt
                        if (np.any(ssr_rt)):
                            for cc in range(len(ssr_rt.components)):
                                comp = ssr.rt.components[cc]
                                # check the time tag w.r.t. low rate
                                if ssr_rt.time_tag_15 >= lr_msg.time_tag_15:
                                    tropo_type = 'rt'
                                    if comp == "m":
                                        mfi = Troposphere(epoch, ssr_rt,
                                                          receiver['ellipsoidal'],
                                                          self.el[ii][jj],
                                                          doy,
                                                          comp, cc,
                                                          tropo_type,
                                                          md,
                                                          f_dbg=f_dbg
                                                          ).corr
                                    else:
                                        self.rt[ii][jj] += Troposphere(epoch, ssr_rt,
                                                                       receiver['ellipsoidal'],
                                                                       self.el[ii][jj],
                                                                       doy,
                                                                       comp, cc,
                                                                       tropo_type,
                                                                       None,
                                                                       f_dbg=f_dbg
                                                                       ).corr
                                        tropo_tot_corr_meters += self.rt[ii][jj]
                                else:
                                    self.rt[ii][jj] = np.nan
                        else:
                            self.rt[ii][jj] = np.nan
                        rt_out = self.make_output_format(self.rt[ii][jj])
                        # gridded troposphere
                        self.grt[ii][jj] = .0
                        # get closest available grid
                        ssr_grt = get_closest_grid(corr_list=ssr.grt,
                                                   llh0=[lat, lon, height])
                        if (np.any(ssr_grt)):
                            for cc in range(len(ssr_grt.components)):
                                comp = ssr_grt.components[cc]
                                tropo_type = 'grt'
                                self.grt[ii][jj] += Troposphere(epoch, ssr_grt,
                                                                receiver['ellipsoidal'],
                                                                self.el[ii][jj],
                                                                doy,
                                                                comp, cc,
                                                                tropo_type,
                                                                md,
                                                                f_dbg=f_dbg
                                                                ).corr
                                tropo_tot_corr_meters += self.grt[ii][jj]
                        else:
                            self.grt[ii][jj] = np.nan
                        grt_out = self.make_output_format(self.grt[ii][jj])

                        tropo_corr_out = self.make_output_format(
                            tropo_tot_corr_meters)

                        # The models above compute the additive _corrections_
                        # to the nominal tropo model (UNB3).
                        # Evaluate the model to compute the total tropo delay.
                        tropo_model = tropo_computation.TropoComputation(
                            epoch, None, receiver['ellipsoidal'], self.el[ii][jj],
                            doy, None, None, None, f_dbg=f_dbg)
                        # Mapping function improvement
                        mfi_meters = tropo_model.model_tot_slant * mfi
                        mfi_out = self.make_output_format(mfi_meters)
                        std_meters = (tropo_model.model_tot_slant +
                                      tropo_tot_corr_meters +
                                      mfi_meters)
                        std_out = self.make_output_format(std_meters)
                        # Model output variables
                        model_zhd = tropo_model.model_dry_zenith
                        model_zhd_out = self.make_output_format(model_zhd)
                        model_zwd = tropo_model.model_wet_zenith
                        model_zwd_out = self.make_output_format(model_zwd)
                        model_shd = (tropo_model.model_dry_zenith *
                                     tropo_model.gmfh)
                        model_shd_out = self.make_output_format(model_shd)
                        model_swd = (tropo_model.model_wet_zenith *
                                     tropo_model.gmfw)
                        model_swd_out = self.make_output_format(model_swd)
                        #######################################################
                        # Solid Earth Tides
                        #######################################################
                        date_time = datetime(year=epoch_gc.year,
                                             month=epoch_gc.month,
                                             day=epoch_gc.day,
                                             hour=epoch_gc.hour,
                                             minute=epoch_gc.min,
                                             second=int(epoch_gc.sec),
                                             microsecond=int((epoch_gc.sec -
                                                              int(epoch_gc.sec))
                                                             * 1.0e6))
                        # UTC to GPS time by applying leap seconds
                        date_time_utc = (date_time -
                                         timedelta(seconds=trafo.Constants().utc2gps_ls))
                        if do_sed:
                            tides_neu = tides.compute_solid_tides(date_time=date_time_utc,
                                                                  lat=lat, lon=lon,
                                                                  zero_tide=sed_perm)
                        else:
                            tides_neu = np.zeros(3)
                        tides_north = self.make_output_format(tides_neu[0])
                        tides_east = self.make_output_format(tides_neu[1])
                        tides_up = self.make_output_format(tides_neu[2])
                        # elevation and azimuth
                        el_deg = np.rad2deg(self.el[ii][jj])
                        az_deg = np.rad2deg(self.az[ii][jj])
                        el_out = self.make_output_format(el_deg)
                        az_out = self.make_output_format(az_deg)
                        self.epoch = epoch
                        # print output only for satellites with
                        # elevation larger than 0
                        if self.el[ii][jj] >= 0:
                            if csv_out:
                                strg_sat = "".join(['{:4.0f}'.format(self.week),
                                                    ',',
                                                    '{:8.1f}'.format(
                                                        self.epoch),
                                                    ',',
                                                    f'{sat}', ',',
                                                    signal, ',',
                                                    '{:11.9f}'.format(
                                                        wavelength),
                                                    ',', az_out, ',',
                                                    el_out, ',',
                                                    '{:19.9f}'.format(rho),
                                                    ',',
                                                    '{:19.9f}'.format(
                                                        sat_state_no_sagnac[0]),
                                                    ',',
                                                    '{:19.9f}'.format(
                                                        sat_state_no_sagnac[1]),
                                                    ',',
                                                    '{:19.9f}'.format(
                                                        sat_state_no_sagnac[2]),
                                                    ',',
                                                    clk_out, ',',
                                                    clk_hr_out, ',',
                                                    orb_out, ',',
                                                    cbias_out, ',',
                                                    pbias_out, ',',
                                                    iono_tot_m_out, ',',
                                                    iono_tot_out, ',',
                                                    gvi_out, ',',
                                                    gsi_out, ',',
                                                    rsi_out, ',',
                                                    gri_out, ',',
                                                    std_out, ',',
                                                    gt_out, ',',
                                                    rt_out, ',',
                                                    grt_out, ',',
                                                    model_zhd_out, ',',
                                                    model_zwd_out, ',',
                                                    model_shd_out, ',',
                                                    model_swd_out, ',',
                                                    wup_out, ',',
                                                    shap_out, ',',
                                                    tides_north, ',',
                                                    tides_east, ',',
                                                    tides_up, ',',
                                                    mfi_out, '\n'])
                                # Remove format spaces
                                self.strg = "".join([self.strg,
                                                     '  ', strg_sat.replace(' ', '')])
                            else:
                                self.strg = "".join([self.strg, '   ',
                                                     '{:8.0f}'.format(
                                                         self.week),
                                                     '   ',
                                                     '{:8.4f}'.format(
                                                         self.epoch),
                                                     '    ',
                                                     f'{sat}', '    ',
                                                     el_out, '  ',
                                                     az_out, '  ',
                                                     orb_out, '   ',
                                                     clk_tot_out, '    ',
                                                     clk_out, '  ',
                                                     clk_hr_out, '    ',
                                                     iono_tot_m_out, '      ',
                                                     iono_tot_out, '         ',
                                                     gvi_out, '         ',
                                                     gsi_out, '            ',
                                                     rsi_out, '      ',
                                                     gri_out, '         ',
                                                     std_out, '     ',
                                                     tropo_corr_out, '    ',
                                                     rt_out, '         ',
                                                     grt_out, '      ',
                                                     shap_out, '   ',
                                                     cbias_out, '   ',
                                                     pbias_out, '   ',
                                                     wup_out, '\n'])

    def __str__(self):
        return self.strg

    def __repr__(self):
        return ('OSR objects: week, epoch, sats, orb, clk, clk_hr, ' +
                'cbias, pbias, gsi, rsi, gri, grt, rt, el, az, wup, shap')

    def make_output_format(self, value):
        """
            Method to prepare the right format for the output. If the variable
            is empty then the output will be n/a.
        """
        if ((value is None) | (np.isnan(value))):
            if self.csv_out:
                value_out = '{:8.4f}'.format(0.0)
            else:
                value_out = '{:8s}'.format('    n/a')
        else:
            if np.size(value) == 0:
                if self.csv_out:
                    value_out = '{:8.4f}'.format(0.0)
                else:
                    value_out = '{:8s}'.format('    n/a')
            else:
                value_out = '{:9.5f}'.format(value)
        return value_out

    @classmethod
    def get_header(cls):
        return ("".join(['#****************************************',
                '*****************************************',
                         '*****************************************',
                         '*****************************************',
                         '*********************************', '\n',
                         '# Satellite elevation and azimuth are output in [deg],',
                         '\n',
                         '# ionospheric output is reported in [TECU] but ',
                         'the total ionosphere, reported in [m]. ', '\n',
                         '# All the other parameters are in [m].', '\n',
                         '# Frequencies used for wup, ',
                         'total iono impact, code and phase bias are L1, G1, E1, ',
                         'B1-2,', '\n',
                         '# respectively for ',
                         'GPS/QZSS(1C), GLONASS(1C), Galileo(1C) and Beidou(2I).',
                         '\n',
                         '# For the gridded components, the interpolation method ',
                         'used is IDW. ', '\n',
                         '# IDW is used as an example, the user should select ',
                         'an appropriate interpolation technique.', '\n',
                         '# GPS week    GPS TOW       SV       elev    azim      ',
                         'sv_orb     sv_clk_tot  sv_clk  sv_clk_hr     iono_tot(m)  ',
                         'iono_tot(TECU)  iono_glo_vtec(GVI) iono_glo_sat(GSI)  ',
                         'iono_reg(RSI) iono_grid(GRI)   ',
                         'tropo_tot(m) tropo_corr  tropo_reg(RT) tropo_grid(GRT)   ',
                         'shapiro    cbias      phbias     wind-up ',
                         '\n',
                         '# ---------------------------------------',
                         '-----------------------------------------',
                         '-----------------------------------------',
                         '-----------------------------------------',
                         '------------------------------------------- ', '\n']))

    @classmethod
    def get_header_csv(cls):
        return ("".join(['# Week,time,SAT,signal,wavelength(m),azimuth(deg),',
                'elevation(deg),range(m),X(m),Y(m),Z(m),LowRate clock (m),',
                         'HighRate clock(m),orbit(m),CodeBias(m),PhaseBias(m),',
                         'STEC total(m),STEC total(TECU),STEC GVI(TECU),',
                         'STEC GSI(TECU),STEC RSI(TECU),STEC GRI(TECU),tropo(m),',
                         'GT(m),RT(m),GRT(m),modelZTDdry(m),modelZTDwet(m),',
                         'modelSTDdry(m),modelSTDwet(m),windup(cyc),relativity(m),',
                         'solidEarthTide dN(m),solidEarthTide dE(m),',
                         'solidEarthTide dU(m)',
                         'mappingImprovement(m)']))


# =============================================================================
# OSR orbit corrections
# =============================================================================
class OrbCorr:
    """ Orbit correction computation.
        Input:
            - ssr low rate corrections
            - satellite state vector at the transmission time
            - gnss index
            - index of desired satellite
        Output:
            orbit correction along the line of sight
    """

    def __init__(self, lr_msg, hr_msg, state_tr, rec, ii, ii_hr, jj):
        # orbit corrections
        # low rate
        dr = lr_msg.rad[ii][jj]
        dt = lr_msg.atr[ii][jj]
        dc = lr_msg.ctr[ii][jj]
        # high rate
        if hr_msg is not None:
            if hr_msg.rad is not None:
                dr += hr_msg.rad[ii_hr][jj]
        # compute radial, along-track and cross-track satellite coordinates
        sat_tr = state_tr[0:3]
        vel_tr = state_tr[3:]
        alo = vel_tr / LA.norm(vel_tr)
        crs = (np.cross(sat_tr, vel_tr) /
               LA.norm(np.cross(sat_tr, vel_tr)))
        rad = np.cross(alo, crs)
        # compute rotation matrix to radial, along-track and cross-track
        # coordinates
        R = np.array([rad, alo, crs])
        delta_o = np.array([dr, dt, dc])
        delta_x = np.dot(np.transpose(R), delta_o)
        sight = (rec - sat_tr) / LA.norm(rec - sat_tr)
        self.corr = np.dot(delta_x, sight)


# =============================================================================
# OSR clock corrections
# =============================================================================
class ClockCorr:
    """ Orbit correction computation.
        Input:
            - ssr clock corrections in [mm], [mm/s], [mm/s**2]
            - delta time w.r.t. the received msg of corrections
            - index of the desired satellite
        Output:
            clock correction along the line of sight in [m]
    """

    def __init__(self, lr_msg, hr_msg, dt, ii, ii_hr, jj):
        c0 = lr_msg.c0[ii][jj]
        c1 = lr_msg.c1[ii][jj]
        if hr_msg is not None:
            if hr_msg.clk is not None:
                c0_hr = hr_msg.clk[ii_hr][jj]
            else:
                c0_hr = None
        else:
            c0_hr = None
        self.corr = c0 + c1 * dt
        self.corr_hr = c0_hr


# =============================================================================
# OSR Code Bias
# =============================================================================
class CodeBias:
    """Class to get code bias from decoded RTCM-SSR message
        Input:
            - SSRZ low rate msg
            - index of the desired gnss
            - index of the desired satellite
            - tracking mode of the desired signal
        Output:
            - code bias correction for the desired satellite and tracking mode
    """

    def __init__(self, lr_msg, ii, jj, gnss, signal):
        # get reference signal
        try:
            ref_sgn = lr_msg.ref_signal[gnss][0]
        except KeyError:
            ref_sgn = 'n/a'
        if signal == ref_sgn:
            self.corr = 0
        else:
            # find signal index
            # it reads the pb_signals and remove the first that is the ref_sgn
            try:
                sgn_idx = np.where(lr_msg.pb_signals[gnss] == signal)[0][0] - 1
                self.corr = lr_msg.cb[ii][sgn_idx][jj]
            except KeyError:
                self.corr = np.nan


# =============================================================================
# OSR Phase Bias
# =============================================================================
class PhaseBias:
    """Class to get phase bias from decoded RTCM-SSR message
        Input:
            - SSRZ low rate msg
            - index of the desired gnss
            - index of the desired satellite
            - tracking mode of the desired signal
        Output:
            - code bias correction for the desired satellite and tracking mode
    """

    def __init__(self, lr_msg, ii, jj, gnss, signal):
        # find signal index
        try:
            sgn_idx = np.where(lr_msg.pb_signals[gnss] == signal)[0][0]
            self.corr = lr_msg.pb[ii][jj][sgn_idx]
        except KeyError:
            # GNSS not available
            self.corr = np.nan


# =============================================================================
#    Ionosphere influence
# =============================================================================
class Ionosphere:
    """
        It passes the input values to the IonoComputation class for computing
        ionospheric influences for a user position.
    """

    def __init__(self, ssr, epoch, isys, isat, state, rec, gnss, sat, fr,
                 iono_type, ppo=None, md=None, el=None, gsi_ppo=None,
                 rsi_ppo=None, dbg_out=None, week=None, ephemeris=None):
        iono_influence = iono_computation.IonoComputation(ssr, epoch, isys,
                                                          isat, state, rec,
                                                          gnss, sat, fr,
                                                          iono_type, md=md,
                                                          el_ellips=el,
                                                          gsi_ppo=gsi_ppo,
                                                          rsi_ppo=rsi_ppo,
                                                          week=week,
                                                          ephemeris=ephemeris)
        # Global vtec
        if iono_type == 'gvi':
            print(iono_influence, file=dbg_out)
        self.corr = iono_influence.stec   # [TECU]
        self.corr_f1 = iono_influence.stec_corr_f1  # [m]
        # Global slant
        if iono_type == 'gsi':
            self.gsi_ppo = iono_influence.gsi_ppo
        # Regional slant
        if iono_type == 'rsi':
            self.rsi_ppo = iono_influence.rsi_ppo


# =============================================================================
#    Troposphere influence
# =============================================================================
class Troposphere:
    """
        It passes the input values to the TropoComputation class for computing
        the regional tropospheric influence for the user position.
    """

    def __init__(self, epoch, ssr, rec_llh, el, doy, component,
                 component_index, tropo_type, md=None, f_dbg=None):
        tropo_influence = tropo_computation.TropoComputation(epoch, ssr,
                                                             rec_llh, el, doy,
                                                             tropo_type,
                                                             component,
                                                             component_index,
                                                             md, f_dbg)
        self.corr = tropo_influence.tropo


# =============================================================================
#                         Shapiro effect
# =============================================================================
class ShapiroEffect:
    """ Class to compute the shapiro effect.
        Input:
                - satellite coordinates at the transmission time
                - receiver coordinates
        Output:
                - shapiro effect correction [m]
        Reference:
            Teunissen, P. and Montenbruck, O. Springer Handbook of GNSS,
            Springer, 2017
    """

    def __init__(self, sat_tr, rec):
        c = trafo.Constants().c
        mu = trafo.Constants().mu_gps
        self.corr = (2.0 * mu / c ** 2 *
                     np.log((np.linalg.norm(sat_tr) +
                             np.linalg.norm(rec) +
                             np.linalg.norm(sat_tr - rec)) /
                            (np.linalg.norm(sat_tr) + np.linalg.norm(rec) -
                             np.linalg.norm(sat_tr - rec))))


# =============================================================================
#                                   wind up correction
# =============================================================================
class WindUp:
    def __init__(self, state, fr, rec, lat, lon):
        """ Function to compute the wind up effect
            Input:
                - sat state vector in ECEF
                - rec coord in ECEF
                - ellip lat, long of the rec
                - the frequency of the signals considered
            Output:
                - phase wind up correction for the input frequency
        """
        sat = state[0:3]
        vel = state[3:]

        lam = trafo.Constants().c / fr
        diff = rec - sat
        k = diff / LA.norm(diff)

        # from deg to rad for lat, lon
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)

        # correction for Eart rotation
        vel[0] = vel[0] - trafo.Constants().omega_e * sat[1]
        vel[1] = vel[1] + trafo.Constants().omega_e * sat[0]

        # ee, en, eu unit vecotrs in ENU ref frame
        ee = np.array([-np.sin(lon), +np.cos(lon), +0.0])
        en = np.array([-np.cos(lon) * np.sin(lat),
                       -np.sin(lon) * np.sin(lat),
                       +np.cos(lat)])

        # Computation of the ex, ey, ez unit vectors
        ez = -sat / LA.norm(sat)
        ey = -np.cross(sat, vel) / LA.norm(np.cross(sat, vel))
        ex = np.cross(ey, ez)

        # yaw angle rotation
        # in SSRZ the yaw angle is assumed to be 0
        yaw = 0
        R = np.array([[+np.cos(yaw), np.sin(yaw), 0.0],
                      [-np.sin(yaw), np.cos(yaw), 0.0],
                      [0.0, 0.0, 1.0]])

        e_xyz = np.array([ex, ey, ez])
        e_Rxyz = np.dot(R, e_xyz)
        ex = e_Rxyz[0]
        ey = e_Rxyz[1]
        ez = e_Rxyz[2]

        # Effective dipole for the satellite
        flag = 'sat'
        D_sat = WindUp.compute_eff_dipole(k, ex, ey, flag)

        # Effective dipole for the receiver
        flag = 'rec'
        D_rec = WindUp.compute_eff_dipole(k, ee, en, flag)

        # Wind up computation
        gamma = np.dot(k, np.cross(D_sat, D_rec))

        omega = np.arccos(np.dot(D_sat, D_rec) /
                          (LA.norm(D_sat) * LA.norm(D_rec)))
        omega = -omega / (2.0 * np.pi)

        if gamma < 0:
            omega = -omega
        # Correction for specific wavelength
        self.corr = omega * lam

    @staticmethod
    def compute_eff_dipole(k, ex, ey, flag):
        """ Computation of the effective dipole for the phase wind up

            Input:
                    - k     : dist unit vector for sat-rec in ECEF
                    - ex, ey: unit vectors in the x,y plane.
                      If the satellite is considered
                          then x = t and y = -n directions
                          (ref to radial-track-normal ref frame);
                      if the receiver is considered x = E and y = N
                            (North-East-Up ref frame)
                    - flag  : if 'sat' the formula
                              for the satellite is considered,
                             if 'rec' the formula
                             for the receiver is considered

            Output:
                    - D: effective dipole

                Formulas:
                    --> sat: D = ex - k*dot(k,ex) - cross(k,ey)
                    --> rec: D = ex - k*dot(k,ex) + cross(k,ey)

                Reference:
                    Springer Handbook for GNSS, Teunissen & Montenbruck,
                    chap.19 pag. 570
        """

        if flag == 'sat':
            D = ex - k * np.dot(k, ex) - np.cross(k, ey)
        elif flag == 'rec':
            D = ex - k * np.dot(k, ex) + np.cross(k, ey)

        return D

# =============================================================================
#                 Get closest available grid corrections method
# =============================================================================


def get_closest_grid(corr_list, llh0):
    """ 
        Method to retrieve closest grid to query point.
        Grids with all nan values are excluded
    """
    # Remove empty grids
    corr_no_empty_list = list(filter(None, corr_list))
    # Compute closest distance to query location
    corr_out = None
    min_distance = 1.0e10
    for ii in range(len(corr_no_empty_list)):
        corr = corr_no_empty_list[ii]
        distance_list = []
        for chain in corr.chains:
            for jj in range(len(chain.lat)):
                lat = np.rad2deg(chain.lat[jj])
                lon = np.rad2deg(chain.lon[jj])
                hgt = chain.hgt[jj]
                # transform to xyz
                xyz = trafo.ell2cart(lat, lon, hgt)
                # transform to topocentric enu
                enu = trafo.cart2enu(xyz, llh0=llh0)
                distance_list.append(np.linalg.norm(enu))
        # Calculate minimum distance of the grid from the query point
        distance = np.nanmin(np.array(distance_list))
        # Save this distance as the minimum distance for the current grid
        if distance < min_distance:
            min_distance = distance
            corr_out = corr
    return corr_out