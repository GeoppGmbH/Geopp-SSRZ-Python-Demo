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
    Test script to use the SSRZ Python Demonstrator
    Input:
    - path       : SSRZ binary file path
    - f_in       : SSRZ binary file (i.e. "name.rtc" or "name.bin")
    - nav_file   : RINEX 3.04 navigation file (e.g. BRDC*.rnx)
    - lat        : user ellipsoidal latitude [deg]
    - lon        : user ellipsoidal longitude [deg]
    - height     : user ellipsoidal height [m]
    - decoded_out: flag to request decoded msg in txt file output (0/1)
    - out_folder : desired folder for the output, if not provided, i.e.==None,
                   the output folder will be 'path//SSRZ_demo//'
    - decode_only: decode SSRZ messages without OSR conversion
    - gpsweek_start      :
      sow_start          : set start time for OSR conversion. If both values are set
                           to zero the first decoded epoch will be used
    - gpsweek_end        :
      sow_end            : set end time for OSR conversion. If both values are set to
                           zero the OSR conversion runs until the last decoded epoch
    - do_not_use_msg     : this list indicates which messages should be skipped from
                           decoding (and OSR conversion). Example 'ZM007' or 'RT'
    - do_not_use_gnss    : exclude GNSS ('G', 'R', 'E',...) from OSR conversion
    - print_ssi_after_msg: set a message ('GVI', 'RT',...) after which the OSR
                           conversion shall start. If this is not set, OSR
                           conversion and SSI output will be performed after
                           each received/decoded SSRZ message
    - csv_out            : enables csv output with ssi after RT message
    - do_sed             : enables solid Earth tides computation
    - zero_tide          : flag to consider zero tide system for solid Earth tides.
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
import do_ssrz_demo
import os

# =============================================================================
#                         Input and output file names
# =============================================================================
full_path = os.path.realpath(__file__)
current_dir = os.path.dirname(full_path)
path = os.path.join(current_dir, '..', 'test_data')
os.chdir(path)
ssrz_file = 'FRA1060i.ssz'
nav_file = "BRDC00IGS_R_20230600000_01D_MN.rnx"
out_folder = None
lat = 52.5
lon = 9.5
height = 100.0
# Date-time settings
gpsweek_start = 0
sow_start = 0
gpsweek_end = 0
sow_end = 0

# decode only flag
decode_only = 0

# skip SSRZ message(s)
do_not_use_msg = []

# skip GNSS
do_not_use_gnss = []

# print ssi after reception of message
print_ssi_after_msg = []

# csv format for ssi output
csv_out = True

# Solid Earth Tides
do_sed = True
zero_tide = True
# =============================================================================
#                                Call
# =============================================================================
[ssr, ssi] = do_ssrz_demo.ssrz_demo(ssrz_file, [lat, lon, height], nav_file,
                                    decode_only, out_folder,
                                    gpsweek_start, sow_start,
                                    gpsweek_end, sow_end, do_not_use_msg,
                                    do_not_use_gnss, print_ssi_after_msg,
                                    csv_out, do_sed, zero_tide)
