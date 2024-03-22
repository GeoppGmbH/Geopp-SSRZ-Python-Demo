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
    (at your option) any LATer version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    aLONg with this program.  If not, see <https://www.gnu.org/licenses/>.
    ---------------------------------------------------------------------------
    Test script to use the SSRZ Python Demonstrator
    Input:
    - path       : SSRZ binary file path
    - f_in       : SSRZ binary file (i.e. "name.rtc" or "name.bin")
    - NAV_FILE   : RINEX 3.04 navigation file (e.g. BRDC*.rnx)
    - LAT        : user ellipsoidal LATitude [deg]
    - LON        : user ellipsoidal LONgitude [deg]
    - HEIGHT     : user ellipsoidal HEIGHT [m]
    - decoded_out: flag to request decoded msg in txt file output (0/1)
    - OUT_FOLDER : desired folder for the output, if not provided, i.e.==None,
                   the output folder will be 'path//SSRZ_demo//'
    - DECODE_ONLY: decode SSRZ messages without OSR conversion
    - GPSWEEK_START      :
      SOW_START          : set start time for OSR conversion. If both values are set
                           to zero the first decoded epoch will be used
    - GPSWEEK_END        :
      SOW_END            : set end time for OSR conversion. If both values are set to
                           zero the OSR conversion runs until the last decoded epoch
    - do_not_use_msg     : this list indicates which messages should be skipped from
                           decoding (and OSR conversion). Example 'ZM007' or 'RT'
    - do_not_use_gnss    : exclude GNSS ('G', 'R', 'E',...) from OSR conversion
    - print_ssi_after_msg: set a message ('GVI', 'RT',...) after which the OSR
                           conversion shall start. If this is not set, OSR
                           conversion and SSI output will be performed after
                           each received/decoded SSRZ message
    - CSV_OUT            : enables csv output with ssi after RT message
    - DO_SED             : enables solid Earth tides computation
    - ZERO_TIDE          : flag to consider zero tide system for solid Earth tides.
                           Default is a "conventional tide free" system,
                           in conformance with the IERS Conventions.
    Output:
    - name_ssr.txt    : txt file of the decoded SSRZ message
    - name_ssi.txt or
    - name_ssi.csv    : txt file with SSR influence on user location
    - name_gvi.txt    : txt file with computed global vtec ionospheric
                        parameters, e.g. pierce point
    - ssr             : decoded SSRZ parameters class
    - ssi             : class of computed ssr influence
"""
import os
import do_ssrz_demo


# =============================================================================
#                         Input and output file names
# =============================================================================
full_path = os.path.realpath(__file__)
current_dir = os.path.dirname(full_path)
PATH = os.path.join(current_dir, '..', 'test_data')
os.chdir(PATH)
SSRZ_FILE = 'FRA1060i.ssz'
NAV_FILE = "BRDC00IGS_R_20230600000_01D_MN.rnx"
OUT_FOLDER = None
LAT = 52.5
LON = 9.5
HEIGHT = 100.0
# Date-time settings
GPSWEEK_START = 0
SOW_START = 0
GPSWEEK_END = 0
SOW_END = 0

# decode only flag
DECODE_ONLY = 0

# skip SSRZ message(s)
do_not_use_msg = []

# skip GNSS
do_not_use_gnss = []

# print ssi after reception of message
print_ssi_after_msg = ["GT"]

# csv format for ssi output
CSV_OUT = True

# Solid Earth Tides
DO_SED = True
ZERO_TIDE = True
# =============================================================================
#                                Call
# =============================================================================
[ssr, ssi] = do_ssrz_demo.ssrz_demo(SSRZ_FILE, [LAT, LON, HEIGHT], NAV_FILE,
                                    DECODE_ONLY, OUT_FOLDER,
                                    GPSWEEK_START, SOW_START,
                                    GPSWEEK_END, SOW_END, do_not_use_msg,
                                    do_not_use_gnss, print_ssi_after_msg,
                                    CSV_OUT, DO_SED, ZERO_TIDE)
