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
    ***************************************************************************
    Description:
    decoding the message is performed by finding the RTCM preamble in the byte
    stream (in this version SSRZ uses the RTCM framing).
    Then the message length is decoded and the CRC sum check is
    computed (Ref. Numerical Recipes, Press, W. H. et al., 3rd edition,
    cap. 22.4).
    After verification of the CRC the complete message passes to
    the 'ssrz_decoder' module for decoding.
    RINEX navigation data are read using the module 'nav_rinex_reader'.
    If the 'dec_only' flag is not 1 and the navigation file is provided,
    the SSR influence for a rover location is computed using the module
    'ssrz_ssr2osr'. The SSR influence computation is thought to be calculated
    for each epoch with an high-rate message update.
"""
import os
import errno
import cbitstruct as bitstruct
import crcmod
import numpy as np
import ssrz_decoder
import space_time_trafo as trafo
import ssrz_ssr
import ssrz_ssr2osr
import nav_rinex_reader as nav_reader


# =============================================================================
#           check if SSI should be computed after reception of msg
# =============================================================================
def check_print_ssi_after_msg(dec_msg, print_ssi_after_msg:list):
    """
    It checks the list print_ssi_after_msg. If specific message are considered
    the output will be reported only if those messages have been decoded.

    :param dec_msg: decoded msg
    :param print_ssi_after_msg: list of code names. So far, only GVI,
                                RT, and GT are supported.
    """
    p_ssi = True  # Process SSI flag
    if 'GVI' in print_ssi_after_msg:
        if not isinstance(dec_msg, ssrz_decoder.IonoGloVtec):
            p_ssi = False
    if 'RT' in print_ssi_after_msg:
        if not isinstance(dec_msg, ssrz_decoder.TropoReg):
            p_ssi = False
    if 'GT' in print_ssi_after_msg:
        if not isinstance(dec_msg, ssrz_decoder.TropoGrid):
            p_ssi = False
    return p_ssi


# =============================================================================
#                                  SSRZ demo
# =============================================================================
def ssrz_demo(ssrz_file, user_llh, nav_file=None, dec_only=None,
              out_folder=None,
              week_start=0, time_start=0, week_end=0, time_end=0,
              do_not_use_msg=[], do_not_use_gnss=[],
              print_ssi_after_msg=["GT"],
              csv_out=False, do_sed=True, zero_tide=False):
    """
    Decoding of SSRZ message and computing influence from SSR components on
    user position.

    Input:
    - ssrz_file   : complete path of the SSRZ binary file
    - user_llh    : ellipsoidal coordinates of the user position,
                    lat[deg], lon[deg], height [m] considering WGS84
    - nav_file    : RINEX navigation file
    - dec_only    : if 1, the demo works as decoder, i.e. the SSR influence on
                    rover position is not performed, but the decoding of
                    SSRZ messages
    - out_folder  : desired folder for the output, if not provided, i.e.==None,
                   the output folder will be 'path//SSRZ_demo//'
    - gpsweek_start      :
      sow_start          : set start time for OSR conversion. If both values
                           are set to zero the first decoded epoch
                           will be used
    - gpsweek_end        :
      sow_end            : set end time for OSR conversion.
                           If both values are set to zero the OSR conversion
                           runs until the last decoded epoch
    - do_not_use_msg     : this list indicates which messages should be
                           skipped from decoding (and OSR conversion).
                           Example: 'ZM007' or 'RT'
    - do_not_use_gnss    : exclude GNSS ('G', 'R', 'E',...) from OSR conversion
    - print_ssi_after_msg: set a message ('GVI', 'RT',...) after which the OSR
                           conversion shall start. If this is not set, OSR
                           conversion and SSI output will be performed after
                           each received/decoded SSRZ message
    - csv_out            : enables csv output with ssi after RT message
                           (hard-coded)
    - do_sed             : compute solid Earth tides by using Milbert's code
    - zero_tide          : flag to consider zero tide system for
                           solid Earth tides.
                           Default is a "conventional tide free" system,
                           in conformance with the IERS Conventions.

    Output:
        - printed decoded ssrz messages
        - printed SSR influence for user location if required
        - printed global vtec ionosphere if required
        - ssr : class of decoded SSRZ parameters
        - ssi : class of computed ssr influence for the user's location
                parameters
    ***************************************************************************
    """
    # ------------------------- open output files -----------------------------
    if out_folder is None:
        full_path = os.path.realpath(__file__)
        current_dir = os.path.dirname(full_path)
        out_folder = os.path.join(current_dir, '..', 'test_data', 'SSRZ_demo')
    else:
        pass
    try:
        os.makedirs(out_folder)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    header = "".join(['#------------------------------------------------------------',
                      '---', '\n',
                      '# Geo++ SSRZ Python Demonstrator v2.2 ', '\n',
                      '# Copyright (C) 2020 Francesco Darugna <fd@geopp.de>  ',
                      'Geo++ GmbH', '\n',
                      '#                    Jannes B. Wübbena <jw@geopp.de>  ',
                      'Geo++ GmbH', '\n',
                      '#------------------------------------------------------------',
                      '---'])
    _, basename = os.path.split(ssrz_file)
    filename, _ = os.path.splitext(basename)
    dec_out_path = os.path.join(out_folder, filename + '_ssr.txt')
    dec_out = open(dec_out_path, 'w', encoding="utf-8")
    print(header, file=dec_out)
    if ((dec_only == 0) & (nav_file is not None)):
        if csv_out:
            ssi_out = open(
                os.path.join(out_folder, filename + '_ssi.csv'),
                'w', encoding="utf-8"
            )
        else:
            ssi_out = open(
                os.path.join(out_folder, filename + '_ssi.txt'),
                'w', encoding="utf-8"
            )
        print(header, file=ssi_out)
        if csv_out:
            print(ssrz_ssr2osr.ssr2osr.get_header_csv(), file=ssi_out)
        else:
            print(ssrz_ssr2osr.ssr2osr.get_header(), file=ssi_out)
        ssi_dbg = None
        gvi_out = open(
            os.path.join(out_folder, filename + '_gvi.txt'),
            'w', encoding="utf-8"
        )
        print(header, file=gvi_out)
    else:
        ssi_out = None
        gvi_out = None
    # ---------------------------- Read ephemerides ---------------------------
    if nav_file is not None:
        # read navigation data
        nav_data = nav_reader.EphemerisList()
        # In SSRZ Galileo INAV clock is chosen as a reference
        nav_data.import_rinex_nav_file(nav_file, nav_type_gal="INAV-E1b")
    # --------------------------  Input ssrz_data -----------------------------
    receiver = {}
    user_xyz = trafo.ell2cart(user_llh[0], user_llh[1], user_llh[2])
    receiver['ellipsoidal'] = np.array(user_llh)
    receiver['cartesian'] = np.array(user_xyz)
    # Read data
    with open(ssrz_file, 'rb') as f:
        data = f.read()
    # ---------------------- Loop over the whole message ----------------------
    # Description of the preamble
    preamble = b'\xd3'
    # initialization of ephemeris, metadata and ssr variables
    ssr = ssrz_ssr.SSRObjects()
    dec_md = None
    dec_corr = None
    ssi = []
    t_count = 0      # counting time for the updates
    osr = None  # initialize ssr2osr computation
    types_list = []  # list of the message types contained in the ssrz file
    rec_lrm = False  # Low Rate Message (LRM) has been received flag
    # Start loop
    ii = 0
    while ii <= len(data):
        if data[ii:ii + 1] == preamble:
            frame_header = data[ii:ii + 3]
            # Getting RTCM header consisting of preamble (8 bit),
            # reserved bits (6 bit) and messange length (10) bit
            # Remark: this might change in future versions
            try:
                frame_header_unpack = bitstruct.unpack('u8u6u10', frame_header)
                msg_len = frame_header_unpack[2]

                # check CRC for the completeness of the message
                msg_complete = data[ii:ii + 6 + msg_len]
                # create the function for CRC-24Q
                crc_fun = crcmod.crcmod.mkCrcFun(0x1864CFB, rev=False,
                                                 initCrc=0x000000,
                                                 xorOut=0x000000)
                # compute crc value for the complete msg,
                # if correctly received,
                # it should be 0
                crc_value = crc_fun(msg_complete)
            except TypeError:
                break
            if crc_value == 0:
                msg_content = data[ii + 3:ii + 3 + msg_len]
                # decode msg
                try:
                    read_msg = ssrz_decoder.SSRZDecoder(msg_content, msg_len,
                                                        dec_md, dec_corr,
                                                        rec_lrm,
                                                        do_not_use_msg)
                except NotImplementedError:
                    # Skip non-SSRZ messages.
                    ii += msg_len + 6
                    continue
                # update decoded metadata
                dec_md = read_msg.metadata
                # update decoded corrections
                dec_corr = read_msg.corrections
                # extract the decoded message
                dec_msg = read_msg.dec_msg
                if dec_msg is not None:
                    # update decoded metadata
                    dec_md = read_msg.metadata
                    # update decoded corrections
                    dec_corr = read_msg.corrections
                    print(read_msg, file=dec_out)
                    # print(read_msg)
                    # extract the message type
                    msg_type = "".join([str(read_msg.msg_type), '.',
                                        str(read_msg.msg_subtype), '.',
                                        str(int(read_msg.ssrz_msg_type[-3:]))])

                    # Check if msg is a Low Rate Message
                    if read_msg.ssrz_msg_type == 'ZM002':
                        rec_lrm = True
                    # ---- Print out msg only if the metadata was decoded -----
                    if ((dec_md.md_gr is not None) & (dec_corr is not None)):
                        # get the low rate update and high rate update
                        # low rate
                        # number of timing blocks
                        n_time = dec_md.md_gr.md_block.lr_md_block.n_timing
                        # update interval
                        ui_lr = dec_md.md_gr.md_block.lr_md_block.lr_timing_block[
                            n_time-1].ssrz_timing_block.ui
                        # high rate
                        # number of timing blocks
                        n_time = dec_md.md_gr.md_block.hr_md_block.n_timing
                        # update interval
                        ui_hr = dec_md.md_gr.md_block.hr_md_block.hr_timing_block[
                            n_time-1].ssrz_timing_block.ui
                        # read time tag
                        if dec_corr.tt is not None:
                            epoch = dec_corr.tt.gps_tow
                            week = dec_corr.tt.gps_week
                        else:
                            pass
                        # ------------------- Print message -------------------
                        # print out msg if the low-rate msg was decoded
                        # and it is not either new metadata or time msg
                        # Warning: time offsets are not considered yet
                        if ((int(read_msg.ssrz_msg_type[-3:]) < 11) &
                            (int(read_msg.ssrz_msg_type[-3:]) != 9) &
                            (int(read_msg.ssrz_msg_type[-3:]) != 8) &
                            (dec_corr.lr is not None) &
                           (dec_corr.tt is not None)):
                            tt_15 = dec_msg.time_tag_15
                            if t_count == 0:
                                t0 = tt_15
                                try:
                                    epoch0 = epoch
                                except UnboundLocalError:
                                    print('Time tag message missing')
                            # ---------- Save and compute SSR influence -------
                            # save ssr and compute ssr influence for a
                            # given location
                            ssr.add_epoch(epoch, dec_md, dec_corr)
                            t_count += 1
                            # check if there are ephemeris and hr already
                            # decoded
                            # Remark. We'll just assume that
                            # ZM007 (RT) is always present and always the last
                            # in the stream, and generate when that comes in.
                            if ((nav_file is not None) and
                                (np.any(dec_corr.hr)) and
                               (dec_only != 1)):
                                p_ssi = True  # Process SSI flag
                                p_ssi = check_print_ssi_after_msg(
                                    dec_msg, print_ssi_after_msg
                                )
                                if (not rec_lrm):
                                    p_ssi = False
                                    break
                                if week_start > 0 and week_end > 0:
                                    dt_start = trafo.diff_time_s(week_start,
                                                                 time_start,
                                                                 week, epoch)
                                    dt_end = trafo.diff_time_s(week_end,
                                                               time_end,
                                                               week, epoch)
                                    if dt_start < 0:
                                        p_ssi = False
                                    if dt_end > 0:
                                        p_ssi = False
                                        break
                                if p_ssi:
                                    osr = ssrz_ssr2osr.ssr2osr(
                                        ssr.corrections[-1], dec_md, week,
                                        epoch, nav_data, receiver,
                                        gvi_out, ssi_dbg, do_not_use_gnss,
                                        csv_out, do_sed, zero_tide
                                    )
                                    ssi.append(osr)
                                    print(osr, file=ssi_out)
                                    tt_15_old = tt_15
                    else:
                        msg_type = None
                    # save msg type that has been decoded
                    if msg_type is not None:
                        types_list = np.append(types_list, msg_type)
                # update index
                ii = ii + msg_len + 6
            else:
                ii = ii + 1
        else:
            ii = ii + 1
    print('### Decoded SSR message types:\n' +
          str(np.unique(types_list)) + ' ###')
    dec_out.close()
    if ssi_out is not None:
        ssi_out.close()
        if ssi_dbg is not None:
            ssi_dbg.close()
        gvi_out.close()
    return [ssr, ssi]
