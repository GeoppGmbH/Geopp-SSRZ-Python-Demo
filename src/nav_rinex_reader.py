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
    Module to read rinex nav files.
    ***************************************************************************
    Description:
    the module contains classes and methods to read RINEX navigation files.
    The constructor of the 'nav_rinex_reader' module expects
    RINEX navigation file as a input text file. The 'EphemerisList' class is
    the main class with objects the GNSSs (e.g. gps). After reading the header
    by using the 'RinexNavHeader' class the GNSS objects are filled accordingly
    to the file content. The method 'get_ephemeris' can be used
    to obtain the set of ephemeris for a requested IOD or the closest set
    to a given epoch.
    ***************************************************************************
    Reference:
        RINEX Version 3.04 ftp://ftp.igs.org/pub/data/format/rinex304.pdf
"""
import numpy as np
import datetime
import space_time_trafo as trafo

MAX_GPS_SATELLITES = 64


# =============================================================================
#                     Rinex navigation data class list
# =============================================================================
class EphemerisList:
    """ EphemerisList class to keep a list of ephemeris data from rinex files.
    It should keep a sub-list for each satellite.
    """

    def __init__(self):
        self.gps = [None] * MAX_GPS_SATELLITES
        self.glo = [None] * MAX_GPS_SATELLITES
        self.gal = [None] * MAX_GPS_SATELLITES
        self.bds = [None] * MAX_GPS_SATELLITES
        self.qzs = [None] * MAX_GPS_SATELLITES

    def import_rinex_nav_file(self, file_name, nav_type_gal=None):
        with open(file_name, 'r') as rinex_file:
            self.header = RinexNavHeader(rinex_file)
            self.header.parse_header()
            self.fill_from_file(rinex_file, nav_type_gal=nav_type_gal)

    def fill_from_file(self, file_object, t_start=None, t_stop=None,
                       nav_type_gal=None):
        for line in file_object:
            if line[0] == 'G':
                gps_sat_prn = int(line[1:3])
                if self.gps[gps_sat_prn-1] is None:
                    self.gps[gps_sat_prn-1] = EphemerisListSatGPS(gps_sat_prn)
                self.gps[gps_sat_prn-1].add_new_eph_from_file(line,
                                                              file_object)
            if line[0] == 'R':
                glo_sat_prn = int(line[1:3])
                if self.glo[glo_sat_prn-1] is None:
                    self.glo[glo_sat_prn-1] = EphemerisListSatGLO(glo_sat_prn)
                self.glo[glo_sat_prn-1].add_new_eph_from_file(line,
                                                              file_object)
            if line[0] == 'E':
                gal_sat_prn = int(line[1:3])
                if self.gal[gal_sat_prn-1] is None:
                    self.gal[gal_sat_prn-1] = EphemerisListSatGal(gal_sat_prn)
                self.gal[gal_sat_prn-1].add_new_eph_from_file(line,
                                                              file_object,
                                                              nav_type_gal)

            if line[0] == 'C':
                bds_sat_prn = int(line[1:3])
                if self.bds[bds_sat_prn-1] is None:
                    self.bds[bds_sat_prn-1] = EphemerisListSatBDS(bds_sat_prn)
                self.bds[bds_sat_prn-1].add_new_eph_from_file(line,
                                                              file_object)

            if line[0] == 'J':
                qzs_sat_prn = int(line[1:3])
                if self.qzs[qzs_sat_prn-1] is None:
                    self.qzs[qzs_sat_prn-1] = EphemerisListSatQZS(qzs_sat_prn)
                self.qzs[qzs_sat_prn-1].add_new_eph_from_file(line,
                                                              file_object)

    def get_ephemeris(self, gnss, prn, iode, epoch):
        """
            It gets the correct block of ephemeris checking the iode
            if provided, otherwise providing the block of ephemeris with the
            epoch closest to the requested epoch.
        """
        if gnss == 'G':
            ephemeris = self.gps[prn-1]
        elif gnss == 'R':
            ephemeris = self.glo[prn-1]
        elif gnss == 'E':
            ephemeris = self.gal[prn-1]
        elif gnss == 'C':
            ephemeris = self.bds[prn-1]
        elif gnss == 'J':
            ephemeris = self.qzs[prn-1]

        if ephemeris is None:
            return None

        list_of_epochs = np.array(ephemeris.list_of_sat_epochs)
        # Check for the same iode
        idx = 0
        if iode is None:
            idx = np.where(np.abs(list_of_epochs - epoch) ==
                           np.min(np.abs(list_of_epochs - epoch)))[0][0]
        else:
            if (gnss == "G") or (gnss == "E") or (gnss == "J"):
                while ephemeris.list_of_ephemeris[idx].iode != iode:
                    idx += 1
            elif gnss == "R":
                tb_min = ephemeris.list_of_ephemeris[idx].tb / 900 # [min]
                while tb_min != iode:
                    idx += 1
                    tb_min = ephemeris.list_of_ephemeris[idx].tb / 900 # [min]
            elif gnss == "C":
                while np.mod(ephemeris.list_of_ephemeris[idx].toc / 720, 240) != iode:
                    idx += 1
            else:
                print("".join(["Warning: ", gnss, " not implemented yet."]))

        eph_out = ephemeris.list_of_ephemeris[idx]
        return eph_out

    def __repr__(self):
        strg = ('Class of objects containing the GNSS ephemeris ' +
                'and the header of the rinex file. ' +
                'The ephemeris are organized in list of prn numbers. ' +
                'The objects of the GNSS are gps, gal, ' +
                'glo, qzs, bds. ' +
                'Example: gps[0] gives a list of available ' +
                'ephemeris for the G01 satellite.')
        return strg


class EphemerisListSatGPS:
    def __init__(self, sat_id):
        self.list_of_ephemeris = []
        self.list_of_sat_epochs = []
        self.sat_prn = 'G' + str(sat_id)

    def add_new_eph_from_file(self, line, file_object):
        new_ephemeris = EphemerisGPS()
        new_ephemeris.fill_from_file(line, file_object)
        self.list_of_ephemeris.append(new_ephemeris)
        # Add ephemeris epoch
        # If available consider transmission time
        # otherwise, consider toe
        if new_ephemeris.transmission_time != 0.9999e9:
            # Transmission time is available
            self.list_of_sat_epochs.append(new_ephemeris.transmission_time)
        else:
            self.list_of_sat_epochs.append(new_ephemeris.toe)

    def __repr__(self):
        strg = ('It contains three objects: sat_prn, list_of_ephemeris ' +
                'and list_of_sat_epochs.')
        return strg


class EphemerisListSatGal:
    def __init__(self, sat_id):
        self.list_of_ephemeris = []
        self.sat_prn = 'E' + str(sat_id)
        self.list_of_sat_epochs = []

    def add_new_eph_from_file(self, line, file_object, nav_type=None):
        new_ephemeris = EphemerisGal()
        new_ephemeris.fill_from_file(line, file_object)
        if nav_type is None:
            self.list_of_ephemeris.append(new_ephemeris)
            # Add ephemeris epoch
            # If available consider transmission time
            # otherwise, consider toe
            if new_ephemeris.transmission_time != 0.9999e9:
                # Transmission time is available
                self.list_of_sat_epochs.append(new_ephemeris.transmission_time)
            else:
                self.list_of_sat_epochs.append(new_ephemeris.toe)
        else:
            if new_ephemeris.nav_type == nav_type:
                self.list_of_ephemeris.append(new_ephemeris)
                # Add ephemeris epoch
                # If available consider transmission time
                # otherwise, consider toe
                if new_ephemeris.transmission_time != 0.9999e9:
                    # Transmission time is available
                    self.list_of_sat_epochs.append(
                        new_ephemeris.transmission_time)
                else:
                    self.list_of_sat_epochs.append(new_ephemeris.toe)
            else:
                pass

    def __repr__(self):
        strg = ('It contains three objects: sat_prn, list_of_ephemeris ' +
                'and list_of_sat_epochs.')
        return strg


class EphemerisListSatGLO:
    def __init__(self, sat_id):
        self.list_of_ephemeris = []
        self.sat_prn = 'R' + str(sat_id)
        self.list_of_sat_epochs = []

    def add_new_eph_from_file(self, line, file_object):
        new_ephemeris = EphemerisGLO()
        new_ephemeris.fill_from_file(line, file_object)
        self.list_of_ephemeris.append(new_ephemeris)
        self.list_of_sat_epochs.append(new_ephemeris.frame_time)

    def __repr__(self):
        strg = ('It contains three objects: sat_prn, list_of_ephemeris ' +
                'and list_of_sat_epochs.')
        return strg


class EphemerisListSatBDS:
    def __init__(self, sat_id):
        self.list_of_ephemeris = []
        self.sat_prn = 'C' + str(sat_id)
        self.list_of_sat_epochs = []

    def add_new_eph_from_file(self, line, file_object):
        new_ephemeris = EphemerisBDS()
        new_ephemeris.fill_from_file(line, file_object)
        self.list_of_ephemeris.append(new_ephemeris)
        # Add ephemeris epoch
        # If available consider transmission time
        # otherwise, consider toe
        if new_ephemeris.transmission_time != 0.9999e9:
            # Transmission time is available
            self.list_of_sat_epochs.append(new_ephemeris.transmission_time)
        else:
            self.list_of_sat_epochs.append(new_ephemeris.toe)

    def __repr__(self):
        strg = ('It contains three objects: sat_prn, list_of_ephemeris ' +
                'and list_of_sat_epochs.')
        return strg


class EphemerisListSatQZS:
    def __init__(self, sat_id):
        self.list_of_ephemeris = []
        self.list_of_sat_epochs = []
        self.sat_prn = 'J' + str(sat_id)

    def add_new_eph_from_file(self, line, file_object):
        new_ephemeris = EphemerisGPS()
        new_ephemeris.fill_from_file(line, file_object)
        self.list_of_ephemeris.append(new_ephemeris)
        # Add ephemeris epoch
        # If available consider transmission time
        # otherwise, consider toe
        if new_ephemeris.transmission_time != 0.9999e9:
            # Transmission time is available
            self.list_of_sat_epochs.append(new_ephemeris.transmission_time)
        else:
            self.list_of_sat_epochs.append(new_ephemeris.toe)

    def __repr__(self):
        strg = ('It contains three objects: sat_prn, list_of_ephemeris ' +
                'and list_of_sat_epochs.')
        return strg


class RinexNavHeader:
    def __init__(self, file_object):
        self.file_object = file_object

    def parse_header(self):
        first_line = self.file_object.readline()
        if not first_line.endswith('RINEX VERSION / TYPE\n'):
            raise IOError

        for line in self.file_object:
            if line.find('END OF HEADER') > -1:
                break


class EphemerisGPS:
    def __init__(self):
        pass

    def fill_from_file(self, line, file_object):
        self.sat = line[0:3]
        self.year = int(line[4:8])
        self.month = int(line[9:11])
        self.day = int(line[12:14])
        self.hour = int(line[15:17])
        self.minute = int(line[18:20])
        self.second = int(line[21:23])
        self.dt_obj = datetime.datetime(self.year, self.month,
                                        self.day, self.hour,
                                        self.minute, int(self.second),
                                        int((self.second -
                                             int(self.second)) * 1e6))
        doy = trafo.date_to_doy(self.year, self.month, self.day)
        [gps_week, self.toc] = trafo.gps_time_from_y_doy_hms(self.year, doy,
                                                             self.hour,
                                                             self.minute,
                                                             self.second)
        self.af0 = float(line[23:42])
        self.af1 = float(line[42:61])
        self.af2 = float(line[61:80])

        line = file_object.readline()
        self.iode = float(line[4:23])
        self.crs = float(line[23:42])
        self.dn = float(line[42:61])
        self.m0 = float(line[61:80])

        line = file_object.readline()
        self.cuc = float(line[4:23])
        self.ecc = float(line[23:42])
        self.cus = float(line[42:61])
        self.root_a = float(line[61:80])

        line = file_object.readline()
        self.toe = float(line[4:23])
        self.cic = float(line[23:42])
        self.omega_0 = float(line[42:61])
        self.cis = float(line[61:80])

        line = file_object.readline()
        self.i0 = float(line[4:23])
        self.crc = float(line[23:42])
        self.omega = float(line[42:61])
        self.omega_dot = float(line[61:80])

        line = file_object.readline()
        self.idot = float(line[4:23])
        self.codes_on_l2 = float(line[23:42])
        self.week = float(line[42:61])
        self.l2p_data_flag = float(line[61:80])

        line = file_object.readline()
        self.sv_accuracy = float(line[4:23])
        self.sv_health = float(line[23:42])
        self.tgd = float(line[42:61])
        self.iodc = float(line[61:80])

        line = file_object.readline()
        # Adjust transmission time
        if float(line[4:23]) < 0:
            self.transmission_time = float(line[4:23]) + 604800.0e0
        elif float(line[4:23]) > 604800.0e0:
            self.transmission_time = float(line[4:23]) - 604800.0e0
        else:
            self.transmission_time = float(line[4:23])
        try:
            self.fit_interval = float(line[23:42])
        except ValueError:
            pass  # fit interval not reported

    def __repr__(self):
        strg = ('Class with objects the ephemeris information of ' +
                'the satellite.' +
                ' More, the date and time is accessible through dt_obj.')
        return strg


class EphemerisGLO:
    def __init__(self):
        pass

    def fill_from_file(self, line, file_object):
        self.sat = line[0:3]

        self.year = int(line[4:8])
        self.month = int(line[9:11])
        self.day = int(line[12:14])
        self.hour = int(line[15:17])
        self.minute = int(line[18:20])
        self.second = int(line[21:23])
        self.dt_obj = datetime.datetime(self.year, self.month,
                                        self.day, self.hour,
                                        self.minute, int(self.second),
                                        int((self.second -
                                             int(self.second)) * 1e6))
        doy = trafo.date_to_doy(self.year, self.month, self.day)
        [gps_week, self.toc] = trafo.gps_time_from_y_doy_hms(self.year, doy,
                                                             self.hour,
                                                             self.minute,
                                                             self.second)
        # compute leap seconds
        self.ls = trafo.get_ls_from_date(self.year, self.month)
        # read remaining lines
        self.tau = float(line[23:42])
        self.gamma = float(line[42:61])
        self.frame_time = float(line[61:80])

        line = file_object.readline()
        self.xn = float(line[4:23])
        self.dxn = float(line[23:42])
        self.ddxn = float(line[42:61])
        self.health = float(line[61:80])

        line = file_object.readline()
        self.yn = float(line[4:23])
        self.dyn = float(line[23:42])
        self.ddyn = float(line[42:61])
        self.freq_num = float(line[61:80])

        line = file_object.readline()
        self.zn = float(line[4:23])
        self.dzn = float(line[23:42])
        self.ddzn = float(line[42:61])
        self.age = float(line[61:80])

        # computation of tb reference time for the ephemeris
        # with correction for Moskow time
        mow_time = self.toc + 3 * 3600  # seconds of week (Moskow time)
        self.nd = int(mow_time / (3600 * 24))  # nday in the week (Moskow time)
        self.tb = mow_time - self.nd * 3600 * 24  # seconds of day (Moskow time)

    def __repr__(self):
        strg = ('Class with satellite ephemeris information as objects.' +
                'Main objects are: xn, yn, zn, dxn, dyn, dzn, tb (UTC)' +
                ' Date and time are accessible through dt_obj.')
        return strg


class EphemerisGal:
    def __init__(self):
        pass

    def fill_from_file(self, line, file_object):
        self.sat = line[0:3]
        self.year = int(line[4:8])
        self.month = int(line[9:11])
        self.day = int(line[12:14])
        self.hour = int(line[15:17])
        self.minute = int(line[18:20])
        self.second = int(line[21:23])
        self.dt_obj = datetime.datetime(self.year, self.month,
                                        self.day, self.hour,
                                        self.minute, int(self.second),
                                        int((self.second -
                                            int(self.second)) * 1e6))
        doy = trafo.date_to_doy(self.year, self.month, self.day)
        [gps_week, self.toc] = trafo.gps_time_from_y_doy_hms(self.year, doy,
                                                             self.hour,
                                                             self.minute,
                                                             self.second)
        self.af0 = float(line[23:42])
        self.af1 = float(line[42:61])
        self.af2 = float(line[61:80])

        line = file_object.readline()
        self.iode = float(line[4:23])
        self.crs = float(line[23:42])
        self.dn = float(line[42:61])
        self.m0 = float(line[61:80])

        line = file_object.readline()
        self.cuc = float(line[4:23])
        self.ecc = float(line[23:42])
        self.cus = float(line[42:61])
        self.root_a = float(line[61:80])

        line = file_object.readline()
        self.toe = float(line[4:23])
        self.cic = float(line[23:42])
        self.omega_0 = float(line[42:61])
        self.cis = float(line[61:80])

        line = file_object.readline()
        self.i0 = float(line[4:23])
        self.crc = float(line[23:42])
        self.omega = float(line[42:61])
        self.omega_dot = float(line[61:80])

        line = file_object.readline()
        self.idot = float(line[4:23])
        self.data_sources = int(float(line[23:42]))
        if (self.data_sources & (1 << 1)):
            self.nav_type = "INAV-E1b"
        elif (self.data_sources & (1 << 2)):
            self.nav_type = "FNAV"
        elif (self.data_sources & (1 << 3)):
            self.nav_type = "INAV-E5b"
        else:
            self.nav_type = "Unknown"
        self.week = float(line[42:61])

        line = file_object.readline()
        self.sisa = float(line[4:23])
        self.sv_health = float(line[23:42])
        self.bgd_e5a = float(line[42:61])
        self.bgd_e5b = float(line[61:80])

        line = file_object.readline()
        # Adjust transmission time
        if float(line[4:23]) < 0:
            self.transmission_time = float(line[4:23]) + 604800.0e0
        elif float(line[4:23]) > 604800.0e0:
            self.transmission_time = float(line[4:23]) - 604800.0e0
        else:
            self.transmission_time = float(line[4:23])

    def __repr__(self):
        strg = ('Class satellite ephemeris as objects.' +
                ' Date and time are accessible through dt_obj.')
        return strg


class EphemerisQZS:
    def __init__(self):
        pass

    def fill_from_file(self, line, file_object):
        self.sat = line[0:3]
        # Date & time
        self.year = int(line[4:8])
        self.month = int(line[9:11])
        self.day = int(line[12:14])
        self.hour = int(line[15:17])
        self.minute = int(line[18:20])
        self.second = int(line[21:23])
        self.dt_obj = datetime.datetime(self.year, self.month,
                                        self.day, self.hour,
                                        self.minute, int(self.second),
                                        int((self.second -
                                             int(self.second)) * 1e6))
        doy = trafo.date_to_doy(self.year, self.month, self.day)
        [gps_week, self.toc] = trafo.gps_time_from_y_doy_hms(self.year,
                                                             doy, self.hour,
                                                             self.minute,
                                                             self.second)
        self.af0 = float(line[23:42])
        self.af1 = float(line[42:61])
        self.af2 = float(line[61:80])

        line = file_object.readline()
        self.iode = float(line[4:23])
        self.crs = float(line[23:42])
        self.dn = float(line[42:61])
        self.m0 = float(line[61:80])

        line = file_object.readline()
        self.cuc = float(line[4:23])
        self.ecc = float(line[23:42])
        self.cus = float(line[42:61])
        self.root_a = float(line[61:80])

        line = file_object.readline()
        self.toe = float(line[4:23])
        self.cic = float(line[23:42])
        self.omega_0 = float(line[42:61])
        self.cis = float(line[61:80])

        line = file_object.readline()
        self.i0 = float(line[4:23])
        self.crc = float(line[23:42])
        self.omega = float(line[42:61])
        self.omega_dot = float(line[61:80])

        line = file_object.readline()
        self.idot = float(line[4:23])
        self.codes_on_l2 = float(line[23:42])
        self.week = float(line[42:61])
        self.l2p_data_flag = float(line[61:80])

        line = file_object.readline()
        self.sv_accuracy = float(line[4:23])
        self.sv_health = float(line[23:42])
        self.tgd = float(line[42:61])
        self.iodc = float(line[61:80])

        line = file_object.readline()
        # Adjust transmission time
        if float(line[4:23]) < 0:
            self.transmission_time = float(line[4:23]) + 604800.0e0
        elif float(line[4:23]) > 604800.0e0:
            self.transmission_time = float(line[4:23]) - 604800.0e0
        else:
            self.transmission_time = float(line[4:23])
        self.fit_interval = float(line[23:42])

    def __repr__(self):
        strg = ('Class with satellite ephemeris as objects.' +
                ' Date and time are accessible through dt_obj.')
        return strg


class EphemerisBDS:
    def __init__(self):
        pass

    def fill_from_file(self, line, file_object):
        self.sat = line[0:3]

        self.year = int(line[4:8])
        self.month = int(line[9:11])
        self.day = int(line[12:14])
        self.hour = int(line[15:17])
        self.minute = int(line[18:20])
        self.second = int(line[21:23])
        self.dt_obj = datetime.datetime(self.year, self.month,
                                        self.day, self.hour,
                                        self.minute, int(self.second),
                                        int((self.second -
                                             int(self.second)) * 1e6))
        doy = trafo.date_to_doy(self.year, self.month, self.day)
        [gps_week, self.toc] = trafo.gps_time_from_y_doy_hms(self.year,
                                                             doy, self.hour,
                                                             self.minute,
                                                             self.second)
        self.af0 = float(line[23:42])
        self.af1 = float(line[42:61])
        self.af2 = float(line[61:80])

        line = file_object.readline()
        self.aode = float(line[4:23])
        self.crs = float(line[23:42])
        self.dn = float(line[42:61])
        self.m0 = float(line[61:80])

        line = file_object.readline()
        self.cuc = float(line[4:23])
        self.ecc = float(line[23:42])
        self.cus = float(line[42:61])
        self.root_a = float(line[61:80])

        line = file_object.readline()
        self.toe = float(line[4:23])
        self.cic = float(line[23:42])
        self.omega_0 = float(line[42:61])
        self.cis = float(line[61:80])

        line = file_object.readline()
        self.i0 = float(line[4:23])
        self.crc = float(line[23:42])
        self.omega = float(line[42:61])
        self.omega_dot = float(line[61:80])

        line = file_object.readline()
        self.idot = float(line[4:23])
        self.week = float(line[42:61])

        line = file_object.readline()
        self.sv_accuracy = float(line[4:23])
        self.sv_h1 = float(line[23:42])
        self.tgd1 = float(line[42:61])
        self.tgd2 = float(line[61:80])

        line = file_object.readline()
        self.transmission_time = float(line[4:23])
        self.aodc = float(line[23:42])

    def __repr__(self):
        strg = ('Class with satellite ephemeris as objects.' +
                ' Date and time are accessible through dt_obj.')
        return strg
