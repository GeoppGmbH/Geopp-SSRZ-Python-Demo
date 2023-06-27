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
   ****************************************************************************
   Group of classes to create ephemeris objects.
   Input : decoded ephemeris message
   Output: object oriented ephemeris message
   ****************************************************************************
   Description:
   the Ephemeris class is initialized with the objects: systems (GNSSs of
   received ephemeris messages), gps, glo, gal, bds and qzs. When the ephemeris
   message of a new system is received, the system is appended to the systems
   object by the method add_system.

   The GNSS objects, e.g. gps, are defined by the GNSS class,
   which has the objects sat, sat_epochs and eph. sat is a list of satellite ID
   with received ephemeris, sat_epochs has, for each satellite of sat,
   the list of epoch of received ephemeris. eph is the content of the
   ephemeris which is defined by the class StateAcc if the GNSS is GLONASS,
   by the class Elements if not.

   Every time a new message is read, the GNSS object, e.g. gps, is updated by
   using the methods add_sv of the Satellite class to add a new satellite
   and add_epoch of the Epochs class to update the epochs and add a new decoded
   ephemeris message.

   A method to get the closest in time ephemeris of a specific satellite is
   included in the Ephemeris class: get_closest_epo.
"""
import numpy as np
import space_time_trafo as trafo
from numpy import linalg as LA
import scipy


class Elements:
    def __init__(self, dec_msg):
        """ Class of orbital elements
        """
        self.sat_id = dec_msg.sat_id
        self.root_a = dec_msg.root_a
        self.dn = dec_msg.dn
        self.ecc = dec_msg.ecc
        self.i0 = dec_msg.i0
        self.idot = dec_msg.idot
        self.omega_0 = dec_msg.omega_0
        self.omega_dot = dec_msg.omega_dot
        self.omega = dec_msg.omega
        self.m0 = dec_msg.m0
        self.crc = dec_msg.crc
        self.crs = dec_msg.crs
        self.cuc = dec_msg.cuc
        self.cus = dec_msg.cus
        self.cic = dec_msg.cic
        self.cis = dec_msg.cis
        self.af0 = dec_msg.af_zero
        self.af1 = dec_msg.af_one
        self.af2 = dec_msg.af_two
        self.toe = dec_msg.toe
        self.toc = dec_msg.toc
        self.week = dec_msg.week

    def __repr__(self):
        return "Orbital elements of sat " + self.sat_id + " at epoch " + str(self.toe)


class StateAcc:
    def __init__(self, dec_msg):
        self.sat_id = dec_msg.sat_id
        # Position
        self.xn = dec_msg.xn
        self.yn = dec_msg.yn
        self.zn = dec_msg.zn
        # Velocity
        self.dxn = dec_msg.dxn
        self.dyn = dec_msg.dyn
        self.dzn = dec_msg.dzn
        # Acceleration
        self.ddxn = dec_msg.ddxn
        self.ddyn = dec_msg.ddyn
        self.ddzn = dec_msg.ddzn
        # Time
        self.tb = dec_msg.tb
        self.nt = dec_msg.nt
        self.n4 = dec_msg.n4
        # Freq
        self.gamma = dec_msg.gamma
        self.tau = dec_msg.tau
        self.tau_c = dec_msg.tau_c
        self.ch = dec_msg.freq

    def __repr__(self):
        return (
            "State vector and acc of sat " +
            self.sat_id + " at epoch " + str(self.tb)
        )


class Epochs:
    def __init__(self, dec_msg, epochs=None, ephemeris=None):
        if epochs is None:
            self.epochs = []
        else:
            self.epochs = epochs

        if ephemeris is None:
            self.ephemeris = []
        else:
            self.ephemeris = ephemeris
        self.dec_msg = dec_msg

    def add_epo(self, epo):
        if epo not in self.epochs:
            self.epochs = np.append(self.epochs, epo)

            if self.dec_msg.gnss_short == "R":
                self.ephemeris = np.append(
                    self.ephemeris, StateAcc(self.dec_msg))
            else:
                self.ephemeris = np.append(
                    self.ephemeris, Elements(self.dec_msg))


class Satellite:
    def __init__(self, satellites=None):
        if satellites is None:
            self.prn = []
        else:
            self.prn = satellites

    def add_sv(self, sv):
        if sv not in self.prn:
            self.prn = np.append(self.prn, sv)


class GNSS:
    def __init__(
        self, dec_msg=None, epo=None, sv=None, eph=None, epochs=None, satellites=None
    ):
        if eph is None:
            self.eph = {}
        else:
            self.eph = eph

        if epochs is None:
            self.sat_epochs = {}
        else:
            self.sat_epochs = epochs

        sat = Satellite(satellites)

        try:
            epoch_list = Epochs(dec_msg, self.sat_epochs[sv], self.eph[sv])
        except KeyError:
            epoch_list = Epochs(dec_msg)

        if dec_msg is None:
            self.sat = []
        else:
            sat.add_sv(sv)
            self.sat = sat.prn
            epoch_list.add_epo(epo)
            self.sat_epochs[sv] = epoch_list.epochs
            self.eph[sv] = epoch_list.ephemeris

    def __repr__(self):
        return "GNSS ephemeris class with objects: sat, sat_epochs[sv]," + "eph[sv]"


class Ephemeris:
    def __init__(self):
        self.systems = []
        self.gps = GNSS()
        self.glo = GNSS()
        self.gal = GNSS()
        self.bds = GNSS()
        self.qzs = GNSS()

    def add_ephemeris_msg(self, dec_msg, system):
        if system == "G":
            self.gps = GNSS(
                dec_msg,
                dec_msg.toe,
                dec_msg.sat_id,
                self.gps.eph,
                self.gps.sat_epochs,
                self.gps.sat,
            )
        elif system == "R":
            self.glo = GNSS(
                dec_msg,
                dec_msg.tb,
                dec_msg.sat_id,
                self.glo.eph,
                self.glo.sat_epochs,
                self.glo.sat,
            )
        elif system == "E":
            self.gal = GNSS(
                dec_msg,
                dec_msg.toe,
                dec_msg.sat_id,
                self.gal.eph,
                self.gal.sat_epochs,
                self.gal.sat,
            )
        elif system == "C":
            self.bds = GNSS(
                dec_msg,
                dec_msg.toe,
                dec_msg.sat_id,
                self.bds.eph,
                self.bds.sat_epochs,
                self.bds.sat,
            )
        elif system == "J":
            self.qzs = GNSS(
                dec_msg,
                dec_msg.toe,
                dec_msg.sat_id,
                self.qzs.eph,
                self.qzs.sat_epochs,
                self.qzs.sat,
            )

    def __repr__(self):
        return "Ephemeris class with objects: systems, gps, glo, gal, bds," + "qzs"

    def add_system(self, system):
        if system not in self.systems:
            self.systems = np.append(self.systems, system)

    def get_closest_epo(self, epo, gnss, sv):
        try:
            index = np.where(
                np.abs(gnss.sat_epochs[sv] - epo)
                == np.min(np.abs(gnss.sat_epochs[sv] - epo))
            )
            closest_eph = gnss.eph[sv][index[0]]
        except KeyError:
            # In this case, there is no ephemeris for that satellite
            closest_eph = []
        return closest_eph


# =============================================================================
#                  satellite state vector computation
# =============================================================================
class Orbit:
    """
    Class to compute satellite state vector at a certain epoch.
    It is defined by three methods:
        - compute_state_vector, function to compute the state vector
        - propagate_state, it propagates the initial state vector for the
                            desired epoch, used with GLONASS satellites
        - propagate_orbit_elements, it propagates the orbt elements
                                    from ephemeris for the desired epoch
    """

    def __init__(self, gnss, receiver_xyz, epoch, week, ls=None):
        self.gnss = gnss
        self.receiver_xyz = receiver_xyz
        self.ls = ls
        self.epoch = epoch
        self.week = week
        self.c = trafo.Constants().c  # speed of light [m/s]
        self.omega_E = trafo.Constants().omega_e  # Earth rotation rate [rad/s]

        # copied from ssrz_ssr2osr.py
        if self.ls is None:
            if (gnss == "G") | (gnss == "E") | (gnss == "J"):
                self.ls = 0
            elif gnss == "C":
                self.ls = 14

    def compute_state_vector(
        self,
        ephemeris,
        sat_clock_corrected=False,
        light_time_corrected=True,
        sagnac_corrected=True,
    ):
        """
        State vector computation. It includes both propagation of
        state vector (GLONASS) and orbit elements.
        """
        if self.gnss == "R":
            # Info from navigation message:
            tb = ephemeris.tb
            # compute the russian day (4-year cycle) at ephemeris time
            [iy, doy, hh, mm, ss] = trafo.gpsTime2y_doy_hms(
                self.week, ephemeris.toc)
            if np.mod(iy, 4) != 0:
                doy = doy + np.mod(iy, 4) * 365 + 1
            Rday = doy  # day in four years cycle
            gamma = ephemeris.gamma  # RINEX conventions
            tau = -ephemeris.tau  # RINEX conventions
            dt_tau_c = 0
            # get leap seconds from ephemeris
            ls = ephemeris.ls
            # Initialize variables
            radial = 20e6  # [m] distance first guess
            radial_last = 0
            dt_sv = 10
            dt_sv_last = 0
            state_0 = np.zeros(6)
            t_xsv = 0
            d_xsv = 0
            td_xsv = 0
            epsilon = 1.0e-2
            # Get initial ytate vector
            state_0 = np.array(
                [
                    ephemeris.xn,
                    ephemeris.yn,
                    ephemeris.zn,
                    ephemeris.dxn,
                    ephemeris.dyn,
                    ephemeris.dzn,
                ]
            )
            # get the luni-solar acc from ephemeris
            luni_solar = np.array(
                [ephemeris.ddxn, ephemeris.ddyn, ephemeris.ddzn])
            # Light time first guess
            dtt = radial / self.c
            if light_time_corrected:
                # Iteration for final time of iteration for GLONASS
                while np.abs(radial - radial_last) > 0.00001:
                    if sat_clock_corrected:
                        tf = self.epoch + dt_sv - radial / self.c - ls
                    else:
                        tf = self.epoch - radial / self.c - ls
                    radial_last = radial
                    dt_sv_last = dt_sv
                    [state, dts, t_day, i_day, t0] = self.compute_glonass_orbit(
                        tf,
                        Rday,
                        tb,
                        tau,
                        gamma,
                        dt_tau_c,
                        epsilon,
                        luni_solar,
                        d_xsv,
                        td_xsv,
                        state_0,
                    )
                    coord = state[0:3]
                    vel = state[3:6]
                    radial = LA.norm(coord - self.receiver_xyz)  # [m]
                    t_xsv = t_day
                    d_xsv = i_day
                    td_xsv = t0
                    dt_sv = dts
                    # Light time
                    tau = radial / self.c
            else:
                if sat_clock_corrected:
                    tf = self.epoch + dt_sv - ls
                else:
                    tf = self.epoch - ls
                [state, dts, t_day, i_day, t0] = self.compute_glonass_orbit(
                    tf,
                    Rday,
                    tb,
                    tau,
                    gamma,
                    dt_tau_c,
                    epsilon,
                    luni_solar,
                    d_xsv,
                    td_xsv,
                    state_0,
                )
                coord = state[0:3]
                vel = state[3:6]
                tau = 0.0e0
            state = np.append(coord, vel)

        else:
            radial = 0
            radial_last = 20e6  # [m] first guess about sat - rec distance
            dt_sv = ephemeris.af0
            dt_sv_last = 1.0e2
            state_0 = np.zeros(6)
            t_xsv = 0
            # Light time first guess
            if light_time_corrected:
                # Iteration for final time of iteration
                while (np.abs(radial - radial_last) > 1e-4) | (
                    np.abs(dt_sv - dt_sv_last) > 0.1 * 1e-7
                ):
                    if sat_clock_corrected:
                        tf = self.epoch + dt_sv - radial / self.c - self.ls
                    else:
                        tf = self.epoch - radial / self.c - self.ls

                    radial_last = radial
                    dt_sv_last = dt_sv
                    [state, t_xsv_old, dt_sv] = Orbit.propagate_orbit_elements(
                        self, ephemeris, tf, t_xsv, state_0
                    )

                    radial = LA.norm(
                        np.array([state[0], state[1], state[2]]) -
                        self.receiver_xyz
                    )
                    t_xsv = t_xsv_old
                    state_0 = state
                    # Light time
                    tau = radial / self.c
            else:
                tf = self.epoch
                if sat_clock_corrected:
                    tf = self.epoch + dt_sv - self.ls
                else:
                    tf = self.epoch - self.ls
                [state, t_xsv_old, dt_sv] = Orbit.propagate_orbit_elements(
                    self, ephemeris, tf, t_xsv, state_0
                )
                tau = 0

        # Correct for Sagnac effect (Rotation of Earth ECI - ECEF rotation)
        if sagnac_corrected:
            omega_tau = self.omega_E * tau
            cos_omega = np.cos(omega_tau)
            sin_omega = np.sin(omega_tau)
            state_out = state
            state_out[0] = cos_omega * state[0] + sin_omega * state[1]
            state_out[1] = -sin_omega * state[0] + cos_omega * state[1]
            state_out[2] = state[2]
        else:
            state_out = state

        return state_out

    def compute_glonass_orbit(
        self,
        tf,
        Rday,
        tb,
        tau,
        gamma,
        dt_tau_c,
        epsilon,
        luni_solar,
        d_xsv,
        td_xsv,
        state_0,
    ):
        """
        GLONASS orbit computation
        """
        i_day = int((tf + 10800.0 + 0.005) / trafo.Constants().day_seconds)
        t_day = tf + 10800.0 - i_day * trafo.Constants().day_seconds

        # With correction for Moskow time
        [iy, DOY, hh, mm, ss] = trafo.gpsTime2y_doy_hms(
            self.week, tf + 10800.0)

        if np.mod(iy, 4) != 0:
            DOY = DOY + np.mod(iy, 4) * 365 + 1
        i_day = DOY  # day in four years cycle

        tk = (i_day - Rday) * trafo.Constants().day_seconds + (t_day - tb)
        while tk <= -trafo.Constants().day_seconds / 2:
            tk = tk + trafo.Constants().day_seconds
        while tk > trafo.Constants().day_seconds / 2:
            tk = tk - trafo.Constants().day_seconds

        # calculate actual clock drift and bias
        dts = -(tau - gamma * tk)
        if np.abs(dt_tau_c) < 1.0:  # plausibilty check for TauC
            dts = dts - dt_tau_c

        t_day = t_day - dts
        tk = (i_day - Rday) * trafo.Constants().day_seconds + (
            t_day - tb
        )  # with correct t_day
        while tk <= -trafo.Constants().day_seconds / 2:
            tk = tk + trafo.Constants().day_seconds
            t_day = t_day + trafo.Constants().day_seconds
        while tk > trafo.Constants().day_seconds / 2:
            tk = tk - trafo.Constants().day_seconds
            t_day = t_day - trafo.Constants().day_seconds

        dt = (i_day - d_xsv) * trafo.Constants().day_seconds + (t_day - td_xsv)
        dts = -(tau - gamma * tk)

        if np.abs(dt) <= epsilon:
            coord = np.array(
                [
                    state_0[0] + state_0[3] * dt,
                    state_0[1] + state_0[4] * dt,
                    state_0[2] + state_0[5] * dt,
                ]
            )
            vel = np.array([state_0[3], state_0[4], state_0[5]])
            state = np.array(
                [coord[0], coord[1], coord[2], vel[0], vel[1], vel[2]])
            t0 = td_xsv + (d_xsv - i_day) * trafo.Constants().day_seconds
        else:
            if dt > 900:
                d_xsv = Rday
                td_xsv = tb
            t0 = td_xsv + (d_xsv - i_day) * trafo.Constants().day_seconds
            state = Orbit.propagate_state(
                self, t0, state_0 * 1e3, luni_solar * 1e3, t_day
            )
            if np.size(state) > 6:
                state = state[-1, :]  # last integration time
            else:
                pass

        return state, dts, t_day, i_day, t0

    def propagate_state(self, t0, y0, parameters, tf):
        """
        Function to propagate the state vector of a satellite from initial
        conditions t0, y0 to the desired time tf. The maximum time step
        used by the integrator is hard coded to 30 seconds.
        Reference: GLONASS ICD, General Description of Code Division
        Multiple Access Signal System. Edition 1.0, Moskow, 2016
        """

        # differential function
        def f(t, y):
            ax_ls = parameters[0]  #
            ay_ls = parameters[1]  # --> luni-solar accelerations [m/s^2]
            az_ls = parameters[2]  #

            # PZ90 data
            gm = trafo.Constants().mu_glo  # [m^3/s^2] gravitational constant
            ae = 6378136.0  # [m]
            J2_0 = 1082625.75e-9  # second degree zonal coefficient of normal potential
            oe = 7.2921151467e-5  # [rad/s] Earth mean angular velocity

            r = np.sqrt(y[0] ** 2 + y[1] ** 2 + y[2] ** 2)

            v_x = y[3]
            v_y = y[4]
            v_z = y[5]

            r2 = r**2
            r3 = r * r2
            r5 = r3 * r2

            # J20 factors
            J2_factor_1 = (
                -1.5e0 * J2_0 * gm * ae * ae /
                r5 * (1.0 - 5.0 * y[2] ** 2 / r2)
            )
            J2_factor_3 = (
                -1.5e0 * J2_0 * gm * ae * ae /
                r5 * (3.0 - 5.0 * y[2] ** 2 / r2)
            )

            a_x = (
                -gm / r3 * y[0]
                + J2_factor_1 * y[0]
                + oe**2 * y[0]
                + 2 * oe * y[4]
                + ax_ls
            )
            a_y = (
                -gm / r3 * y[1]
                + J2_factor_1 * y[1]
                + oe**2 * y[1]
                - 2 * oe * y[3]
                + ay_ls
            )
            a_z = -gm / r3 * y[2] + J2_factor_3 * y[2] + az_ls

            return [v_x, v_y, v_z, a_x, a_y, a_z]

        integrator = scipy.integrate.RK45(
            fun=f, t0=t0, y0=y0, t_bound=tf, max_step=30.0, atol=1e-12
        )
        while integrator.status != "finished":
            integrator.step()
        sol = integrator.y
        return sol

    def propagate_orbit_elements(self, kepler, time, t_xsv, state_0):
        """References:
        - GPS, Galileo, QZSS, Beidou ICDs
        - Springer Handbook of GNSS, Teunissen Montenbruck 2017
        - R. Marson, S. Lagrasta, F. Malvolti, T.S.V. Tiburtina:
          Fast generation of precision orbit ephemeris, Proc.
          ION ITM 2011, San Diego (ION, Virginia 2011) pp. 565–
          576
        """

        if (self.gnss == "G") | (self.gnss == "J"):
            F = trafo.Constants().F_gps
            mu = trafo.Constants().mu_gps
        elif (self.gnss == "E") | (self.gnss == "C"):
            F = trafo.Constants().F_gal
            mu = trafo.Constants().mu_gal

        # Input data from ephemeris message
        a = kepler.root_a * kepler.root_a
        dn = kepler.dn
        e = kepler.ecc
        i0 = kepler.i0
        di_dt = kepler.idot
        Omega0 = kepler.omega_0
        Omega_dot = kepler.omega_dot
        omega = kepler.omega
        M0 = kepler.m0
        Crc = kepler.crc
        Crs = kepler.crs
        Cuc = kepler.cuc
        Cus = kepler.cus
        Cic = kepler.cic
        Cis = kepler.cis
        af0 = kepler.af0
        af1 = kepler.af1
        af2 = kepler.af2
        toe = kepler.toe
        toc = kepler.toc
        wn = kepler.week

        # Define tolerance for linear propagation
        epsilon = 1e-3
        # Clock corrections
        delta_clock = time - toc  # time
        while delta_clock <= -trafo.Constants().week_seconds / 2.0:
            delta_clock = delta_clock + trafo.Constants().week_seconds

        while delta_clock >= trafo.Constants().week_seconds / 2.0:
            delta_clock = delta_clock - trafo.Constants().week_seconds

        dtsdot = af1 + af2 * delta_clock  # drift
        dts = af0 + (af1 + af2 * delta_clock) * delta_clock  # bias

        # Period
        tol = 10.0 * 1e-3  # [s]
        dT = wn * trafo.Constants().week_seconds + time
        dt = np.abs(time - t_xsv)
        T = time - toe - dts

        if np.abs(T) > trafo.Constants().week_seconds / 2.0:
            T = (
                np.mod(T, trafo.Constants().week_seconds / 2.0)
                - np.sign(T) * trafo.Constants().week_seconds / 2.0
            )

        # Mean motion
        n = np.sqrt(mu / (a**3.0)) + dn

        # Mean anomaly
        M = M0 + n * T

        # Derivative Mean Anomaly
        dM = n

        # Eccentric anomaly : M = E - e*sin(E)
        i_max = 10
        toll = 0.5 * 1e-11
        E0 = M
        i = 0
        diff = 2.0 * toll

        while (i < i_max) & (diff >= toll):
            i = i + 1
            E = M + e * np.sin(E0)
            diff = np.abs(E - E0)
            E0 = E

        # Correction of clock for the relativistic effect
        dtr = F * e * kepler.root_a * np.sin(E)
        saver = 1.0 - e * np.cos(E)
        dfr = F * e * kepler.root_a * np.cos(E) * n / saver
        if self.gnss == "C":
            dtr = -2.0 * mu**0.5 / (self.c**2) * e * kepler.root_a * np.sin(E)
        dts = dts + dtr
        dtsdot = dtsdot + dfr

        # Derivative eccentric anomaly
        dE = dM / (1.0 - e * np.cos(E))

        # True anomaly
        theta = np.arctan2(
            (np.sqrt(1.0 - e**2.0) * np.sin(E)) / (1.0 - e * np.cos(E)),
            (np.cos(E) - e) / (1.0 - e * np.cos(E)),
        )

        # Derivative true anomaly
        dtheta = (dE * np.sqrt(1.0 - e**2.0)) / (1.0 - e * np.cos(E))

        # Argument of latitude
        u_bar = omega + theta

        # Derivative argument of latitude
        du_bar = dtheta

        # Periodic corrections
        delta_r = Crs * np.sin(2.0 * u_bar) + Crc * np.cos(2.0 * u_bar)
        delta_u = Cus * np.sin(2.0 * u_bar) + Cuc * np.cos(2.0 * u_bar)
        delta_i = Cis * np.sin(2.0 * u_bar) + Cic * np.cos(2.0 * u_bar)

        # Deriv of corr terms for radius vector module and angular anomaly
        ddelta_r = 2 * du_bar * \
            (Crs * np.cos(2.0 * u_bar) - Crc * np.sin(2.0 * u_bar))
        ddelta_u = 2 * du_bar * \
            (Cus * np.cos(2.0 * u_bar) - Cuc * np.sin(2.0 * u_bar))
        ddelta_i = 2 * du_bar * \
            (Cis * np.cos(2.0 * u_bar) - Cic * np.sin(2.0 * u_bar))

        # Perturbed radius, argument of latitude and inclination
        r = a * (1.0 - e * np.cos(E)) + delta_r
        u = u_bar + delta_u
        i = i0 + di_dt * T + delta_i

        # Derivative of radius vector module and inclination
        dr = a * e * dE * np.sin(E) + ddelta_r
        d_i = di_dt + ddelta_i

        # Greenwich longitude of the ascending node
        lambda_Om = Omega0 + (Omega_dot - self.omega_E) * \
            T - self.omega_E * toe

        # Rate of change of angular offset (derivative of lambda_Om)
        dOmega = Omega_dot - self.omega_E

        # Rotation matrices (left handed rotations -> clockwise of -angle)
        R3 = np.array(
            [
                [np.cos(lambda_Om), -np.sin(lambda_Om), 0.0],
                [np.sin(lambda_Om), np.cos(lambda_Om), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        R1 = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(i), -np.sin(i)],
                [0.0, np.sin(i), +np.cos(i)],
            ]
        )

        # Satellite coordinates on Orbital Plane
        xp = r * np.cos(u)
        yp = r * np.sin(u)

        # Earth-fixed position
        r_ITRF = np.dot(np.dot(R3, R1), np.array([xp, yp, 0.0]))

        # Derivative of satellite coordinates on Orbital Plane
        dxp = dr * np.cos(u) - r * (du_bar + ddelta_u) * np.sin(u)
        dyp = dr * np.sin(u) + r * (du_bar + ddelta_u) * np.cos(u)

        # Rotation matrix for velocity computation
        dR1 = d_i * np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, -np.sin(i), -np.cos(i)],
                [0.0, +np.cos(i), -np.sin(i)],
            ]
        )

        dR3 = dOmega * np.array(
            [
                [-np.sin(lambda_Om), -np.cos(lambda_Om), 0.0],
                [+np.cos(lambda_Om), -np.sin(lambda_Om), 0.0],
                [+0.0, +0.0, +0.0],
            ]
        )

        # Velocity
        v_ITRF = np.dot(np.dot(R3, R1), np.array([dxp, dyp, 0.0])) + np.dot(
            np.dot(dR3, R1) + np.dot(R3, dR1), np.array([xp, yp, 0.0])
        )

        if np.abs(dT) <= tol:
            r_ITRF = r_ITRF + v_ITRF * dT

        # State vector
        state = np.array(
            [r_ITRF[0], r_ITRF[1], r_ITRF[2], v_ITRF[0], v_ITRF[1], v_ITRF[2]]
        )

        # Linear propagation of the orbit
        if dt < epsilon:
            state = state_0 + np.array(
                [state_0[3] * dt, state_0[4] * dt, state_0[5] * dt, 0, 0, 0]
            )

        return [state, time, dts]
