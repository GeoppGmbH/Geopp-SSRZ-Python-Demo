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
    Script to use the SSRZ Python Demonstrator
    starting a GUI interface.
    Input:
    - path       : SSRZ binary file path
    - f_in       : SSRZ binary file (e.g. "name.ssz", "name.bin")
    - nav_in     : RINEX 3.04 navigation file (e.g. BRDC*.rnx)
    - lat        : user ellipsoidal latitude [deg]
    - lon        : user ellipsoidal longitude [deg]
    - height     : user ellipsoidal height [m]
    - out_folder : desired folder for the output, if not provided, i.e.==None,
                   the output folder is 'path//SSRZ_demo//'

    Output:
    - name_ssr.txt    : txt file of the decoded SSRZ message
    - name_ssi.txt    : txt file with SSR influence on user location
    - name_ion.txt    : txt file with computed global ionospheric parameters,
                        e.g. pierce point
    - ssr         : decoded SSRZ
    - ssi         : computed ssr influence for user's location parameters
    ***************************************************************************
"""
import do_ssrz_demo
import tkinter as tk
import os

# =============================================================================
# Title & Logo
# =============================================================================
window = tk.Tk()
window.title('Geo++ SSRZ Python Demo v2.2')
window.geometry('740x350')
window.grid()
# Setting it up
full_path = os.path.realpath(__file__)
current_dir = os.path.dirname(full_path)
img = tk.PhotoImage(file=current_dir + '\\..\\docs\\images\\geopp_logo.png')
# Displaying it
tk.Label(window, image=img).grid(row=0, column=2, columnspan=2, rowspan=2,
                                 sticky='w', padx=5, pady=5)
# =============================================================================
#                         Input and output entries
# =============================================================================
# Insert path to rtc file
first_row = tk.Frame()
first_row.grid(row=0, column=0, sticky='w')
lbl_path = tk.Label(first_row, text="Path to ssrz binary file:    ")
lbl_path.pack(side='left')
txt_path = tk.Entry(first_row, width=40)
txt_path.pack(side='left')
# Insert ssrz file name
second_row = tk.Frame()
second_row.grid(row=1, column=0, sticky='w')
lbl_file = tk.Label(second_row, text="SSRZ file name:                 ")
lbl_file.pack(side='left')
txt_file = tk.Entry(second_row, width=20)
txt_file.pack(side='left')
# Insert ephemeris file name
third_row = tk.Frame()
third_row.grid(row=2, column=0, sticky='w')
lbl_eph = tk.Label(third_row, text="RINEX Nav file :                 ")
lbl_eph.pack(side='left')
txt_eph = tk.Entry(third_row, width=20)
txt_eph.pack(side='left')
# Insert output folder (optional)
fourth_row = tk.Frame()
fourth_row.grid(row=3, column=0, sticky='w')
lbl_out = tk.Label(fourth_row, text="Output folder (optional):  ")
lbl_out.pack(side='left')
txt_out = tk.Entry(fourth_row, width=40)
txt_out.pack(side='left')
# set latitude of rover point
fifth_row = tk.Frame()
fifth_row.grid(row=4, column=0, sticky='w')
lbl_lat = tk.Label(fifth_row, text="Ellips. lat. [deg] ")
lbl_lat.pack(side='left')
txt_lat = tk.Entry(fifth_row, width=20)
txt_lat.pack(side='left')
# set longitude of rover point
sixth_row = tk.Frame()
sixth_row.grid(row=5, column=0, sticky='w')
lbl_lon = tk.Label(sixth_row, text="Ellips. lon. [deg]")
lbl_lon.pack(side='left')
txt_lon = tk.Entry(sixth_row, width=20)
txt_lon.pack(side='left')
# set height of rover point
seventh_row = tk.Frame()
seventh_row.grid(row=6, column=0, sticky='w')
lbl_hei = tk.Label(seventh_row, text="Ellips. hei. [m]    ")
lbl_hei.pack(side='left')
txt_hei = tk.Entry(seventh_row, width=20)
txt_hei.pack(side='left')
# set GPS Week start
eighth_row = tk.Frame()
eighth_row.grid(row=7, column=0, sticky='w')
lbl_week_s = tk.Label(eighth_row, text="GPS Week start [-]  (optional)")
lbl_week_s.pack(side='left')
txt_week_s = tk.Entry(eighth_row, width=20)
txt_week_s.pack(side='left')
# set GPS Week end
nineth_row = tk.Frame()
nineth_row.grid(row=8, column=0, sticky='w')
lbl_week_e = tk.Label(nineth_row, text="GPS Week end  [-]  (optional)")
lbl_week_e.pack(side='left')
txt_week_e = tk.Entry(nineth_row, width=20)
txt_week_e.pack(side='left')
# set GPS Seconds of Week start
tenth_row = tk.Frame()
tenth_row.grid(row=9, column=0, sticky='w')
lbl_sow_s = tk.Label(tenth_row, text="GPS SoW start [s]   (optional)")
lbl_sow_s.pack(side='left')
txt_sow_s = tk.Entry(tenth_row, width=20)
txt_sow_s.pack(side='left')
# set GPS Seconds of Week end
eleventh_row = tk.Frame()
eleventh_row.grid(row=10, column=0, sticky='w')
lbl_sow_e = tk.Label(eleventh_row, text="GPS SoW end  [s]   (optional)")
lbl_sow_e.pack(side='left')
txt_sow_e = tk.Entry(eleventh_row, width=20)
txt_sow_e.pack(side='left')
# set csv for ssi output
twelveth_row = tk.Frame()
twelveth_row.grid(row=11, column=0, sticky='w')
lbl_csv = tk.Label(
    twelveth_row, text="SSI as CSV (1 or 0, 1 defeult)   (optional)")
lbl_csv.pack(side='left')
txt_csv = tk.Entry(twelveth_row, width=20)
txt_csv.pack(side='left')
# =============================================================================
# Read input class
# =============================================================================


class read_input():
    def __init__(self):
        if txt_path.get()[-1] == '\\':
            self.path = txt_path.get()
        else:
            self.path = txt_path.get() + '\\'
        self.file = txt_file.get()
        if len(txt_eph.get()) == 0:
            self.nav_file = None
        else:
            self.nav_file = txt_eph.get()
        if len(txt_lat.get()) == 0:
            lat = 52.5
            lon = 9.5
            hei = 100
        else:
            lat = float(txt_lat.get())
            lon = float(txt_lon.get())
            hei = float(txt_hei.get())

        self.rover_coord = [lat, lon, hei]

        out_folder = txt_out.get()
        if len(out_folder) == 0:
            self.out_folder = None
        else:
            if out_folder[-1] == '\\':
                self.out_folder = out_folder
            else:
                self.out_folder = out_folder + '\\'
        # Date time settings
        if len(txt_sow_s.get()) == 0:
            self.gpsweek_start = 0
            self.sow_start = 0
            self.gpsweek_end = 0
            self.sow_end = 0
        else:
            self.gpsweek_start = int(txt_week_s.get())
            self.sow_start = float(txt_sow_s.get())
            self.gpsweek_end = int(txt_week_s.get())
            self.sow_end = float(txt_sow_s.get())
        # CSV for SSI option
        if len(txt_csv.get()) == 0:
            self.csv_out = True
        else:
            self.csv_out = int(txt_csv.get())

# =============================================================================
# Decode only
# =============================================================================


def decode_msg():
    inputs = read_input()
    decode_only = 1
    [ssr, ssi] = do_ssrz_demo.ssrz_demo(inputs.path + inputs.file,
                                        inputs.rover_coord,
                                        inputs.path + inputs.nav_file,
                                        decode_only,
                                        inputs.out_folder
                                        )
    return ssr


# =============================================================================
# Compute SSR influence on rover position
# =============================================================================
def compute_ssi():
    inputs = read_input()
    decode_only = 0
    # skip SSRZ message(s)
    do_not_use_msg = []

    # skip GNSS
    do_not_use_gnss = []

    # print ssi after reception of message
    print_ssi_after_msg = []
    # Solid Earth Tides
    do_sed = False
    zero_tide = True
    [ssr, ssi] = do_ssrz_demo.ssrz_demo(inputs.path + inputs.file,
                                        inputs.rover_coord,
                                        inputs.path + inputs.nav_file,
                                        decode_only,
                                        inputs.out_folder,
                                        inputs.gpsweek_start,
                                        inputs.sow_start,
                                        inputs.gpsweek_end,
                                        inputs.sow_end,
                                        do_not_use_msg, do_not_use_gnss,
                                        print_ssi_after_msg,
                                        inputs.csv_out,
                                        do_sed, zero_tide)
    return [ssr, ssi]


# =============================================================================
# Buttons
# =============================================================================
# Decode SSRZ messages
btn_dec = tk.Button(window, text='Decode SSRZ', command=decode_msg)
btn_dec.grid(row=4, column=1, sticky='w')
# Compute SSR2OSR
btn_osr = tk.Button(window, text='Compute SSR influence' + '\n' +
                    'for user location', command=compute_ssi)
btn_osr.grid(row=6, column=1, sticky='w')


# =============================================================================
#                                Main loop
# =============================================================================
window.mainloop()
