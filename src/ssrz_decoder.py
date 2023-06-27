"""
   ----------------------------------------------------------------------------
   Copyright (C) 2020 Francesco Darugna <fd@geopp.de>  Geo++ GmbH,
                      Jannes B. Wübbena <jw@geopp.de>  Geo++ GmbH.

   A list of all the historical SSRZ Python Demonstrator contributors in
   CREDITS.info

   The first author has received funding from the European Union's Horizon 2020
   research and innovation programme under the Marie Sklodowska-Curie Grant
   Agreement No 722023.
   ----------------------------------------------------------------------------

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
    ----------------------------------------------------------------------------
    ***************************************************************************
    SSRZ message decoder multi class file
    Input:
    - message: data message of the a single type message
    - type_len: message type length
    - dec_metadata: decoded ssrz metadata msg (if available)
    - dec_corrections: decoded ssrz correction msg (if available)
    Output:
    - decoded message: decoded data message for the input type
    ***************************************************************************
    Description:
    the constructor of the ssrz_decoder class expects the SSRZ message content
    as a byte object, without the SSRZ message frame. The message will be
    decoded and be accessed by the dec_msg member (defined by the SSRZ message
    type) of the created ssrz_decoder object. The '__str__' method can be used
    to print the content of the message in a human readable format.
    The decoding of the messages takes advantage the SSRZ structure
    by utilizing the 'ssrz_blocks' and 'ssrz_fields' modules.
    Each message is described by a class, which has the methods '__str__'
    and '__repr__' to print the content in a human readble format and to get
    information about their objects, respectively.
    The metadata and corrections messages are updated epoch by epoch by using
    the module 'ssrz_messages_temp'.
    ***************************************************************************
    List of SSRZ messages considered:
        - 4090.7.11 satellite metadata  ZM011
        - 4090.7.12 metadata            ZM012
        - 4090.7.13 grid metadata       ZM013
        - 4090.7.1 high-rate correction          ZM001
        - 4090.7.2 low-rate correction           ZM002
        - 4090.7.3 gridded iono correction       ZM003
        - 4090.7.4 gridded tropo correction      ZM004
        - 4090.7.5 sat regional iono correction  ZM005
        - 4090.7.6 global iono vtec  correction  ZM006
        - 4090.7.7 regional tropo correction     ZM007
        - 4090.7.9 time tag message              ZM009
    References:
       - Geo++ SSRZ documentation:
           Geo++ State Space Representation Format (SSRZ)
           Document Version 1.1.2
"""

import cbitstruct as bitstruct
import ssrz_blocks as blocks
import ssrz_fields as fields
import ssrz_messages_temp as ssrz_msg
import numpy as np


class SSRZDecoder:
    def __init__(self, message, type_len, dec_metadata=None,
                 dec_corrections=None, haveReceivedLowRateMessage=True,
                 do_not_use_msg=[]):
        # inizialization of metadata
        if dec_metadata is None:
            # no metadata has been decoded yet
            self.metadata = ssrz_msg.Metadata()
            self.corrections = None
        else:
            self.metadata = dec_metadata
        # inizialization of corrections starting when md are decoded
        if self.metadata.md_gr is None:
            if dec_corrections is None:
                # no corrections has been decoded yet
                self.corrections = ssrz_msg.Corrections()
            else:
                self.corrections = dec_corrections
        else:
            if dec_corrections.lr is None:
                # no corrections has been decoded yet
                self.corrections = ssrz_msg.Corrections(self.metadata)
            else:
                self.corrections = dec_corrections
        self.msg = message
        self.type_len = type_len

        # RTCM message type.
        self.msg_type = bitstruct.unpack('u12', message)[0]
        if self.msg_type != 4090:
            raise NotImplementedError('RTCM message type %d not supported.' %
                                      self.msg_type)

        # Geo++ Message Subtype
        self.msg_subtype = bitstruct.unpack('u12u4', message)[1]
        if self.msg_subtype != 7:
            raise NotImplementedError('Geo++ (4090) subtype %d not supported.'
                                      % self.msg_subtype)
        self.dec_msg = None
        # the message type can be removed now
        msg_content = message[2:]
        # SSRZ Message Type Number Indicator zdf002
        [zdf002, unpacked_bits] = fields.zdf002(msg_content)
        if ((zdf002 == 0)
           and (self.metadata.md_gr is not None)
           and (self.corrections.lr is not None)
           and (haveReceivedLowRateMessage)):
            self.ssrz_msg_type = 'ZM001'
            self.dec_msg = CorrHr(msg_content, unpacked_bits, self.metadata,
                                  self.corrections)
            # update the corrections for the high rate messages
            # if the lr was not None and valid
            if self.dec_msg.no_lr == 0:
                t_blk_index = self.dec_msg.t_blk_index
                sat_group_index = self.dec_msg.sat_group_index
                self.corrections.update(hr=self.dec_msg, t_blk=t_blk_index,
                                        sat_gr=sat_group_index)
            else:
                self.dec_msg = None
        elif ((zdf002 == 1) & (self.metadata.md_gr is not None)):
            self.ssrz_msg_type = 'ZM002'
            self.dec_msg = CorrLr(msg_content, unpacked_bits, self.metadata)
            t_blk_index = self.dec_msg.t_blk_index
            sat_group_index = self.dec_msg.sat_group_index
            # update the corrections for the low rate messages
            self.corrections.update(lr=self.dec_msg, t_blk=t_blk_index,
                                    sat_gr=sat_group_index)
            haveReceivedLowRateMessage = True
        elif ((zdf002 == 2) and (self.metadata.md_gr is not None)
              and (len(self.metadata.grid_gr) != 0)
              and (haveReceivedLowRateMessage)
              and ('ZM003' not in do_not_use_msg)
              and ('GRI' not in do_not_use_msg)):
            self.ssrz_msg_type = 'ZM003'
            self.dec_msg = IonoGrid(msg_content, unpacked_bits, self.metadata,
                                    self.corrections)
            # update the corrections for the iono grid
            # if the lr was not None and valid
            if self.dec_msg.no_lr == 0:
                t_blk_index = self.dec_msg.t_blk_index
                sat_group_index = self.dec_msg.sat_group_index
                # Check if gridded iono has been initialize for
                # multiple grids
                gri_temp = self.corrections.gri[t_blk_index][sat_group_index]
                if len(gri_temp) != len(self.metadata.grid_gr):
                    for ii in range(len(self.metadata.grid_gr) - len(gri_temp)):
                        gri_temp.append([])
                    self.corrections.update(gri=gri_temp,
                                            t_blk=t_blk_index,
                                            sat_gr=sat_group_index)
                else:
                    pass
                self.corrections.update(gri=self.dec_msg, t_blk=t_blk_index,
                                        sat_gr=sat_group_index,
                                        grid_number=self.dec_msg.grid_number)
            else:
                self.dec_msg = None
        elif ((zdf002 == 3) & (self.metadata.md_gr is not None)
              and (len(self.metadata.grid_gr) != 0)
              and ('ZM004' not in do_not_use_msg)
              and ('GRT' not in do_not_use_msg)):
            self.ssrz_msg_type = 'ZM004'
            self.dec_msg = TropoGrid(msg_content, unpacked_bits, self.metadata)
            # Check if gridded iono has been initialize for
            # multiple grids
            grt_temp = self.corrections.grt
            if len(grt_temp) != len(self.metadata.grid_gr):
                for ii in range(len(self.metadata.grid_gr) - len(grt_temp)):
                    grt_temp.append([])
                self.corrections.update(grt=grt_temp)
            else:
                pass
            # update the corrections for the tropo grid
            self.corrections.update(grt=self.dec_msg,
                                    grid_number=self.dec_msg.grid_number)
        elif ((zdf002 == 4) and (self.metadata.md_gr is not None)
              and (haveReceivedLowRateMessage)
              and ('ZM005' not in do_not_use_msg)
              and ('RSI' not in do_not_use_msg)):
            self.ssrz_msg_type = 'ZM005'
            self.dec_msg = IonoSatReg(msg_content, unpacked_bits,
                                      self.metadata, self.corrections)
            # update the corrections for the rsi messages
            # if the lr was not None and valid
            if self.dec_msg.no_lr == 0:
                t_blk_index = self.dec_msg.t_blk_index
                sat_group_index = self.dec_msg.sat_group_index
                self.corrections.update(rsi=self.dec_msg, t_blk=t_blk_index,
                                        sat_gr=sat_group_index)
            else:
                self.dec_msg = None
        elif ((zdf002 == 5) & (self.metadata.md_gr is not None)
              and ('ZM006' not in do_not_use_msg)
              and ('GVI' not in do_not_use_msg)):
            self.ssrz_msg_type = 'ZM006'
            self.dec_msg = IonoGloVtec(msg_content, unpacked_bits,
                                       self.metadata, self.corrections)
            # update the corrections for the tropo correction message
            self.corrections.update(gvi=self.dec_msg)
        elif ((zdf002 == 6) & (self.metadata.md_gr is not None)
              and ('ZM007' not in do_not_use_msg)
              and ('RT' not in do_not_use_msg)):
            self.ssrz_msg_type = 'ZM007'
            self.dec_msg = TropoReg(msg_content, unpacked_bits,
                                    self.metadata, self.corrections)
            # update the corrections for the tropo correction message
            self.corrections.update(rt=self.dec_msg)
        elif ((zdf002 == 7) & (self.metadata.md_gr is not None)):
            self.ssrz_msg_type = 'ZM008'
            self.dec_msg = QixBias(msg_content, unpacked_bits, self.metadata)
            # update the corrections for the tropo correction message
            self.corrections.update(qix=self.dec_msg)
        elif ((zdf002 == 8) & (self.metadata.md_gr is not None)):
            self.ssrz_msg_type = 'ZM009'
            self.dec_msg = TimeTag(msg_content, unpacked_bits)
            # update the corrections for the tropo correction message
            self.corrections.update(tt=self.dec_msg)
        elif zdf002 == 10:
            self.ssrz_msg_type = 'ZM011'
            self.dec_msg = MetaSatGroup(msg_content, unpacked_bits,
                                        self.metadata)
            # update the metadata for the satellite group data
            self.metadata.update(sat_gr=self.dec_msg)
        elif zdf002 == 11:
            self.ssrz_msg_type = 'ZM012'
            self.dec_msg = MetaMsg(msg_content, unpacked_bits, self.metadata)
            # update the metadata for the metadata group
            self.metadata.update(md_gr=self.dec_msg)
        elif zdf002 == 12:
            self.ssrz_msg_type = 'ZM013'
            self.dec_msg = MetaGrid(msg_content, unpacked_bits, self.metadata)
            self.metadata.update(grid_gr=self.dec_msg)
        else:
            self.dec_msg = None
            self.ssrz_msg_type = None

    # =========================================================================
    #                              Printing method
    # =========================================================================
    def __str__(self):
        # ******************************************************************* #
        #                                                                     #
        #                SSRZ High Rate Correction Message ZM001              #
        #                                                                     #
        # ******************************************************************* #
        if self.ssrz_msg_type == 'ZM001':
            strg = ('### SSRZ - ' + str(self.msg_type) + '.' +
                    str(self.msg_subtype) + '.' +
                    str(int(self.ssrz_msg_type[-3:])) +
                    ' High Rate Correction Message ' +
                    '<' + str(self.ssrz_msg_type) + '>' + '\n')
            strg += self.dec_msg.__str__()
        # ******************************************************************* #
        #                                                                     #
        #                SSRZ Low Rate Correction Message ZM002               #
        #                                                                     #
        # ******************************************************************* #
        elif self.ssrz_msg_type == 'ZM002':
            strg = ('### SSRZ - ' + str(self.msg_type) + '.' +
                    str(self.msg_subtype) + '.' +
                    str(int(self.ssrz_msg_type[-3:])) +
                    ' Low Rate Correction Message ' +
                    '<' + str(self.ssrz_msg_type) + '>' + '\n')
            if self.dec_msg.strg_nan is None:
                strg += self.dec_msg.__str__()
            else:
                strg += self.dec_msg.strg_nan
        # ******************************************************************* #
        #                                                                     #
        #          SSRZ Gridded Ionosphere Correction Message (ZM003)         #
        #                                                                     #
        # ******************************************************************* #
        elif self.ssrz_msg_type == 'ZM003':
            strg = ('### SSRZ - ' + str(self.msg_type) + '.' +
                    str(self.msg_subtype) + '.' +
                    str(int(self.ssrz_msg_type[-3:])) +
                    ' Gridded Ionosphere Correction Message ' +
                    '<' + str(self.ssrz_msg_type) + '>' + '\n')
            strg += self.dec_msg.__str__()
        # ******************************************************************* #
        #                                                                     #
        #          SSRZ Gridded Troposphere Correction Message (ZM004)        #
        #                                                                     #
        # ******************************************************************* #
        elif self.ssrz_msg_type == 'ZM004':
            strg = ('### SSRZ - ' + str(self.msg_type) + '.' +
                    str(self.msg_subtype) + '.' +
                    str(int(self.ssrz_msg_type[-3:])) +
                    ' Gridded Troposphere Correction Message ' +
                    '<' + str(self.ssrz_msg_type) + '>' + '\n')
            strg += self.dec_msg.__str__()
        # ******************************************************************* #
        #                                                                     #
        #      SSRZ Satellite dependent regional ionosphere  message ZM005    #
        #                                                                     #
        # ******************************************************************* #
        elif self.ssrz_msg_type == 'ZM005':
            strg = ('### SSRZ - ' + str(self.msg_type) + '.' +
                    str(self.msg_subtype) + '.' +
                    str(int(self.ssrz_msg_type[-3:])) +
                    ' Satellite Dependent Regional Ionosphere ' +
                    'Correction Message ' +
                    '<' + str(self.ssrz_msg_type) + '>' + '\n')
            if self.dec_msg.strg_nan is None:
                strg += self.dec_msg.__str__()
            else:
                strg += self.dec_msg.strg_nan
        # ******************************************************************* #
        #                                                                     #
        #           SSRZ Global VTEC Ionosphere Correction Message ZM006      #
        #                                                                     #
        # ******************************************************************* #
        elif self.ssrz_msg_type == 'ZM006':
            strg = ('### SSRZ - ' + str(self.msg_type) + '.' +
                    str(self.msg_subtype) + '.' +
                    str(int(self.ssrz_msg_type[-3:])) +
                    ' Global VTEC Ionosphere ' +
                    'Correction Message ' +
                    '<' + str(self.ssrz_msg_type) + '>' + '\n')
            strg += self.dec_msg.__str__()
        # ******************************************************************* #
        #                                                                     #
        #             SSRZ Regional troposphere correction message    ZM007   #
        #                                                                     #
        # ******************************************************************* #
        elif self.ssrz_msg_type == 'ZM007':
            strg = ('### SSRZ - ' + str(self.msg_type) + '.' +
                    str(self.msg_subtype) + '.' +
                    str(int(self.ssrz_msg_type[-3:])) +
                    ' Regional Troposphere ' +
                    'Correction Message ' +
                    '<' + str(self.ssrz_msg_type) + '>' + '\n')
            strg += self.dec_msg.__str__()
        # ******************************************************************* #
        #                                                                     #
        #                   SSRZ QIX Bias correction message    ZM008         #
        #                                                                     #
        # ******************************************************************* #
        elif self.ssrz_msg_type == 'ZM008':
            strg = ('### SSRZ - ' + str(self.msg_type) + '.' +
                    str(self.msg_subtype) + '.' +
                    str(int(self.ssrz_msg_type[-3:])) +
                    ' QIX Bias ' +
                    'Correction Message ' +
                    '<' + str(self.ssrz_msg_type) + '>' + '\n')
            strg += self.dec_msg.__str__()
        # ******************************************************************* #
        #                                                                     #
        #                 SSRZ Time tag definition message    ZM009           #
        #                                                                     #
        # ******************************************************************* #
        elif self.ssrz_msg_type == 'ZM009':
            strg = ('### SSRZ - ' + str(self.msg_type) + '.' +
                    str(self.msg_subtype) + '.' +
                    str(int(self.ssrz_msg_type[-3:])) +
                    ' Time Tag Message ' +
                    '<' + str(self.ssrz_msg_type) + '>' + '\n')
            strg += self.dec_msg.__str__()
        # ******************************************************************* #
        #                                                                     #
        #               SSRZ Satellite Group Definition Message (ZM011)       #
        #                                                                     #
        # ******************************************************************* #
        elif self.ssrz_msg_type == 'ZM011':
            strg = ('### SSRZ - ' + str(self.msg_type) + '.' +
                    str(self.msg_subtype) + '.' +
                    str(int(self.ssrz_msg_type[-3:])) +
                    ' Satellite Group Definition Message ' +
                    '<' + str(self.ssrz_msg_type) + '>' + '\n' +
                    'Metadata tag                       : ' +
                    str(self.dec_msg.md_tag) + '\n')
            if self.dec_msg.md_tag == 1:
                strg += self.dec_msg.sat_md_blk.__str__()
            else:
                strg += 'Data not available'
            strg += '\n'
        # ******************************************************************* #
        #                                                                     #
        #                  SSRZ Metadata Message (ZM012)                      #
        #                                                                     #
        # ******************************************************************* #
        elif self.ssrz_msg_type == 'ZM012':
            strg = ('### SSRZ - ' + str(self.msg_type) + '.' +
                    str(self.msg_subtype) + '.' +
                    str(int(self.ssrz_msg_type[-3:])) +
                    ' SSRZ Metadata Message ' +
                    '<' + str(self.ssrz_msg_type) + '>' + '\n' +
                    'Metadata IOD                       : ' +
                    str(self.dec_msg.iodm) + '\n')
            # sat md block
            try:
                self.dec_msg.md_block.sat_md_block
                strg += 'Satellited MD block' + '\n'
                strg += self.dec_msg.md_block.sat_md_block.__str__()
            except AttributeError:
                pass
            try:
                # SSRZ high rate metadata message block
                self.dec_msg.md_block.hr_md_block
                strg += 'High rate block' + '\n'
                strg += self.dec_msg.md_block.hr_md_block.__str__()
            except AttributeError:
                pass
            try:
                # SSRZ low rate metadata message block
                if self.dec_msg.md_block.lr_md_block.__str__() != 'None':
                    strg += 'Low rate block' + '\n'
                    strg += self.dec_msg.md_block.lr_md_block.__str__()
            except AttributeError:
                pass
            try:
                # SSRZ Gridded Iono Correction Metadata Message Block ZMB003
                strg += self.dec_msg.md_block.grid_iono_block.__str__()
            except AttributeError:
                pass
            try:
                # SSRZ Gridded Tropo Correction Metadata Message Block ZMB004
                strg += self.dec_msg.md_block.grid_tropo_block.__str__()
            except AttributeError:
                pass
            try:
                # SSRZ Satellite dependent Regional Ionosphere
                # Correction Metadata Message Block ZMB005
                strg += self.dec_msg.md_block.sat_reg_iono_block.__str__()
            except AttributeError:
                pass
            try:
                # SSRZ Global VTEC Ionosphere
                # Correction Metadata Message Block ZMB006
                strg += self.dec_msg.md_block.glo_vtec.__str__()
            except AttributeError:
                pass
            try:
                # SSRZ Regional Troposphere
                # Correction Metadata Message Block ZMB007
                strg += self.dec_msg.md_block.reg_tropo.__str__()
            except AttributeError:
                pass
            try:
                # SSRZ QIX bias
                # Correction Metadata Message Block ZMB008
                strg += self.dec_msg.md_block.qix_bias.__str__()
            except AttributeError:
                pass
        # ******************************************************************* #
        #                                                                     #
        #                  SSRZ Grid definition Message (ZM013)               #
        #                                                                     #
        # ******************************************************************* #
        elif self.ssrz_msg_type == 'ZM013':
            strg = ('### SSRZ - ' + str(self.msg_type) + '.' +
                    str(self.msg_subtype) + '.' +
                    str(int(self.ssrz_msg_type[-3:])) +
                    ' SSRZ Metadata Message ' +
                    '<' + str(self.ssrz_msg_type) + '>' + '\n')
            strg += self.dec_msg.grid_block.__str__()
        else:
            strg = ''
        return strg


# =============================================================================
#                                  METADATA
# =============================================================================
# *************************************************************************** #
#                                                                             #
#                  SSRZ Satellite Group Definition Message (ZM011)            #
#                                                                             #
# *************************************************************************** #
class MetaSatGroup:
    def __init__(self, msg, unpacked_bits, md):
        # ssrz metadata tag zdf020
        [self.md_tag, unpacked_bits] = fields.zdf020(msg, unpacked_bits)
        if self.md_tag == 1:
            # only in this case there is the metadata block
            # satellite group definition metadata block zmb011
            self.sat_md_blk = blocks.zmb011(msg, unpacked_bits, self.md_tag)
        else:
            return

    def __str__(self):
        strg = ('Metadata satellite group message' + '\n' +
                'Metadata tag: ' + str(self.md_tag) + '\n' +
                self.sat_md_blk.__str__())
        return strg

    def __repr__(self):
        strg = ('Metadata satellite group class with objects: md_tag and ' +
                'sat_md_blk')
        return strg


# *************************************************************************** #
#                                                                             #
#                      SSRZ Metadata Message (ZM012)                          #
#                                                                             #
# *************************************************************************** #
class MetaMsg:
    def __init__(self, msg, unpacked_bits, md):
        # ZDF005 issue of metadata, necessary for the correct association
        [self.iodm, unpacked_bits] = fields.zdf005(msg, unpacked_bits)
        # start of metadata block ZMB012
        self.md_block = blocks.zmb012(msg, unpacked_bits, md)

    def __str__(self):
        strg = ('Metadata message' + '\n' +
                'Issue of metadata: ' + str(self.iodm) + '\n' +
                self.md_block.__str__())
        return strg

    def __repr__(self):
        strg = ('Metadata message class with objects: iodm and ' +
                'md_block')
        return strg


# *************************************************************************** #
#                                                                             #
#             SSRZ Grid Definition Metadata Message (ZM013)                   #
#                                                                             #
# *************************************************************************** #
class MetaGrid:
    def __init__(self, msg, unpacked_bits, md):
        # ssrz metadata tag zdf020
        [self.md_tag, unpacked_bits] = fields.zdf020(msg, unpacked_bits)
        # grid definition message block
        self.grid_block = blocks.zmb013(msg, unpacked_bits, self.md_tag)

    def __str__(self):
        strg = ('Metadata grid definition message' + '\n' +
                'Metadata tag: ' + str(self.md_tag) + '\n' +
                self.grid_block.__str__())
        return strg

    def __repr__(self):
        strg = ('Metadata grid class with objects: md_tag and ' +
                'grid_block')
        return strg


# =============================================================================
#                                CORRECTIONS
# =============================================================================
# *************************************************************************** #
#                                                                             #
#                      SSRZ High Rate Correction Message ZM001                #
#                                                                             #
# *************************************************************************** #
class CorrHr:
    def __init__(self, msg, unpacked_bits, md, corr):
        # ssrz 15 minutes time tag zdf050
        [self.time_tag_15, unpacked_bits] = fields.zdf050(msg, unpacked_bits)
        # number of high rate satellite groups from metadata
        try:
            # try to check first in the satellite block of the metadata
            n_g = md.md_gr.sat_md_blk.n_g_hr
        except AttributeError:
            # check in the satellite block of the satellite group
            n_g = md.sat_gr.sat_md_blk.n_g_hr
        # satellite group list mask zdf016
        [self.sv_gr_bit_mask, unpacked_bits] = fields.zdf016(msg,
                                                             unpacked_bits,
                                                             n_g)
        # find the corresponding satellite group
        sat_group_index = np.where(self.sv_gr_bit_mask == 1)[0][0]
        self.sat_group_index = sat_group_index
        # find the corresponding timing block
        for ii in range(md.md_gr.md_block.hr_md_block.n_timing):
            t_blk = md.md_gr.md_block.hr_md_block.hr_timing_block[ii]
            if t_blk.bit_mask[sat_group_index] == 1:
                self.t_blk_index = ii
        self.gnss_hr_md = ''
        for ii in range(n_g):
            if self.sv_gr_bit_mask[ii] == 1:
                self.gnss_hr_md += md.sat_gr.sat_md_blk.hr_block[ii].gnss
        # Remark:
        # the satellite group has to be already been transmitted in the
        # low rate and it has to be valid, i.e. the validity of the
        # low rate msg has to be checked for the correct timing block
        # and satellite group.

        # find the corresponding sat group in the correct timing block
        lr_t_blk_idx = []
        lr_sat_group_idx = []
        for ii in range(len(corr.lr)):
            for jj in range(len(corr.lr[ii])):
                lr_sv_gr_bit_mask = corr.lr[ii][jj].sv_gr_bit_mask
                if lr_sv_gr_bit_mask[jj] == 1:
                    gnss_lr = md.sat_gr.sat_md_blk.lr_block[jj].gnss
                    if ((gnss_lr in self.gnss_hr_md) &
                        (self.time_tag_15 - corr.lr[ii][jj].time_tag_15 <
                         md.md_gr.md_block.lr_md_block.lr_timing_block[ii].ssrz_timing_block.ui)):
                        lr_t_blk_idx.append(ii)
                        lr_sat_group_idx.append(jj)
        lr_t_blk_idx = np.unique(lr_t_blk_idx)
        if len(lr_t_blk_idx) == 0:
            # if there is no common sat group return and set the flag saying
            # that it was not possible to decode the hr message
            self.no_lr = 1
            return
        else:
            self.no_lr = 0
        # at this point it is possible to assume that the hr corr can be
        # decoded for the same satellites of the lr corr
        self.sv_array = []
        for ii in range(len(lr_t_blk_idx)):
            for jj in range(len(lr_sat_group_idx)):
                self.sv_array.append([])
                self.sv_array = corr.lr[ii][jj].sv_array
        # ssrz high rate clock corrections
        # read related metadata
        # number of rice blocks
        n_clk_rb = md.md_gr.md_block.hr_md_block.n_rb_clk
        # c0 resolution
        c_hr_res0 = md.md_gr.md_block.hr_md_block.clk_res
        # ssrz satellite parameter rice block definition
        clk_sat_rb = md.md_gr.md_block.hr_md_block.rb_clk
        # ssrz compressed satellite parameter block
        # for high rate clock correction c0
        para_id = 0   # id for c0 parameter
        sat_parameter_blk = blocks.zdb003(msg, unpacked_bits, n_clk_rb,
                                          clk_sat_rb, self.sv_array,
                                          para_id, c_hr_res0)
        self.clk = sat_parameter_blk.sat_p
        unpacked_bits = sat_parameter_blk.unpacked_bits
        # ssrz high rate orbit corrections
        # check if the number of parameters is larger than 0 in the md
        n_orb_p = md.md_gr.md_block.hr_md_block.orb_p
        if n_orb_p == 1:
            # ssrz satellite parameter rice block definition
            orb_sat_rb = md.md_gr.md_block.hr_md_block.rb_orb
            # number of rice blocks
            n_orb_rb = md.md_gr.md_block.hr_md_block.n_rb_orb
            # radial orbit component
            rad_res0 = md.md_gr.md_block.hr_md_block.rad_res
            para_id = 0
            sat_parameter_blk = blocks.zdb003(msg, unpacked_bits, n_orb_rb,
                                              orb_sat_rb, self.sv_array,
                                              para_id, rad_res0)
            self.rad = sat_parameter_blk.sat_p
        else:
            self.rad = None

    def __str__(self):
        strg = ('{:>4}'.format('Time') + '  {:^5}'.format('Sat') +
                '{:^10}'.format('C0[m]'))
        if self.rad is not None:
            strg += '{:^9}'.format('Rad[m]')
        strg += '\n'

        time = '{:>4.1f}'.format(self.time_tag_15)
        # loop over GNSS:
        for ii in range(len(self.sv_array)):
            sat_list = self.sv_array[ii]
            if len(sat_list) < 1:
                continue
            # loop over satellites:
            for jj in range(len(sat_list)):
                sat = '  {:>3}'.format(sat_list[jj])
                c0 = '  {:+6.4f}'.format(self.clk[ii][jj])
                strg += time + sat + c0
                if self.rad is not None:
                    rad = '  {:+7.4f}'.format(self.rad[ii][jj])
                    strg += rad
                strg += '\n'
        return strg

    def __repr__(self):
        strg = ('High rate corrections class with objects: time_tag_15, ' +
                'sv_gr_bit_mask, sv_array,' +
                ' clk, rad. clk and rad are organized' +
                ' following sv_array per gnss and satellite indexing')
        return strg


# *************************************************************************** #
#                                                                             #
#                        SSRZ Low Rate Correction Message ZM002               #
#                                                                             #
# *************************************************************************** #
class CorrLr:
    def __init__(self, msg, unpacked_bits, md):
        if md.md_gr.md_block.lr_md_block.zdf020 == 2:
            # ssrz 15 minutes time tag zdf050
            [self.time_tag_15, unpacked_bits] = fields.zdf050(msg,
                                                              unpacked_bits)
        # ssrz metadata iod zdf005
        [self.iod, unpacked_bits] = fields.zdf005(msg, unpacked_bits)
        # ssrz metadata announcement bit zdf006
        [self.md_ann_bit, unpacked_bits] = fields.zdf006(msg, unpacked_bits)
        if md.md_gr.md_block.lr_md_block.zdf020 == 1:
            # ssrz 15 minutes time tag zdf050
            [self.time_tag_15, unpacked_bits] = fields.zdf050(msg,
                                                              unpacked_bits)
        # number of low rate satellite groups from metadata
        try:
            # try to check first in the satellite block of the metadata
            n_g = md.md_gr.sat_md_blk.n_g_lr
        except AttributeError:
            # check in the satellite block of the satellite group
            n_g = md.sat_gr.sat_md_blk.n_g_lr
        # satellite group list mask zdf016
        [self.sv_gr_bit_mask, unpacked_bits] = fields.zdf016(msg,
                                                             unpacked_bits,
                                                             n_g)
        # find the corresponding satellite group
        sat_group_index = np.where(self.sv_gr_bit_mask == 1)[0][0]
        self.sat_group_index = sat_group_index
        # find the corresponding timing block
        for ii in range(md.md_gr.md_block.lr_md_block.n_timing):
            t_blk = md.md_gr.md_block.lr_md_block.lr_timing_block[ii]
            if t_blk.bit_mask[sat_group_index] == 1:
                self.t_blk_index = ii
        # find the gnss involved in the satellite group
        gnss = md.sat_gr.sat_md_blk.lr_block[sat_group_index].gnss
        # find the max sat per gnss
        max_id = md.sat_gr.sat_md_blk.lr_block[sat_group_index].max_sat_id
        # satellite bit mask zdf018
        [self.sv_array, unpacked_bits] = fields.zdf018(msg, unpacked_bits,
                                                       gnss, max_id)
        # read bits length fro iod from metadata message
        iod_length = md.md_gr.md_block.lr_md_block.iod_def_block.nb_iod_gnss
        # ssrz be iod zdf309
        [self.be_iod, unpacked_bits] = fields.zdf309(msg, unpacked_bits,
                                                     self.sv_array,
                                                     iod_length)
        # ssrz low rate clock corrections
        # read related metadata
        # number of clock components
        n_clk_components = md.md_gr.md_block.lr_md_block.n_lr_clk
        # number of rice blocks
        n_clk_rb = md.md_gr.md_block.lr_md_block.n_rb_clk
        # c0 resolution
        c0_res0 = md.md_gr.md_block.lr_md_block.c0_res
        # ssrz satellite parameter rice block definition
        clk_sat_rb = md.md_gr.md_block.lr_md_block.rb_clk
        # ssrz compressed satellite parameter block
        # for low rate clock correction c0
        para_id = 0   # id for c0 parameter
        try:
            sat_parameter_blk = blocks.zdb003(msg, unpacked_bits, n_clk_rb,
                                              clk_sat_rb, self.sv_array,
                                              para_id, c0_res0)
            self.strg_nan = None
        except ValueError:
            self.strg_nan = 'SSRZ data are not available' + '\n'
            return
        self.c0 = sat_parameter_blk.sat_p
        unpacked_bits = sat_parameter_blk.unpacked_bits
        if n_clk_components > 1:
            c1_res0 = md.md_gr.md_block.lr_md_block.c1_res0
            # for low rate clock correction c1
            para_id = 1   # id for c0 parameter
            sat_parameter_blk = blocks.zdb003(msg, unpacked_bits, n_clk_rb,
                                              clk_sat_rb, self.sv_array,
                                              para_id, c1_res0)
            self.c1 = sat_parameter_blk.sat_p
            unpacked_bits = sat_parameter_blk.unpacked_bits
        else:
            self.c1 = []
            for ii in range(len(self.sv_array)):
                self.c1.append([])
                for jj in range(len(self.sv_array[ii])):
                    self.c1[ii].append(0)
        # ssrz low rate orbit corrections
        n_orb_p = md.md_gr.md_block.lr_md_block.orb_p
        if n_orb_p == 3:
            # ssrz satellite parameter rice block definition
            orb_sat_rb = md.md_gr.md_block.lr_md_block.rb_orb
            # number of rice blocks
            n_orb_rb = md.md_gr.md_block.lr_md_block.n_rb_orb
            # radial orbit component
            rad_res0 = md.md_gr.md_block.lr_md_block.rad_res
            para_id = 0
            sat_parameter_blk = blocks.zdb003(msg, unpacked_bits, n_orb_rb,
                                              orb_sat_rb, self.sv_array,
                                              para_id, rad_res0)
            self.rad = sat_parameter_blk.sat_p
            unpacked_bits = sat_parameter_blk.unpacked_bits
            # along-track orbit component
            atr_res0 = md.md_gr.md_block.lr_md_block.atr_res
            para_id = 1
            sat_parameter_blk = blocks.zdb003(msg, unpacked_bits, n_orb_rb,
                                              orb_sat_rb, self.sv_array,
                                              para_id, atr_res0)
            self.atr = sat_parameter_blk.sat_p
            unpacked_bits = sat_parameter_blk.unpacked_bits
            # cross-track orbit component
            ctr_res0 = md.md_gr.md_block.lr_md_block.ctr_res
            para_id = 2
            sat_parameter_blk = blocks.zdb003(msg, unpacked_bits, n_orb_rb,
                                              orb_sat_rb, self.sv_array,
                                              para_id, ctr_res0)
            self.ctr = sat_parameter_blk.sat_p
            unpacked_bits = sat_parameter_blk.unpacked_bits
        else:
            self.rad = None
            self.atr = None
            self.ctr = None

        # ssrz low rate velocity corrections
        n_vel_p = md.md_gr.md_block.lr_md_block.n_vel
        if n_vel_p == 3:
            # ssrz satellite parameter rice block definition
            vel_sat_rb = md.md_gr.md_block.lr_md_block.rb_vel
            # number of rice blocks
            n_vel_rb = md.md_gr.md_block.lr_md_block.n_rb_vel
            # radial orbit velocity component
            rad_vel_res0 = md.md_gr.md_block.lr_md_block.rad_vel_res
            para_id = 0
            sat_parameter_blk = blocks.zdb003(msg, unpacked_bits, n_vel_rb,
                                              vel_sat_rb, self.sv_array,
                                              para_id, rad_vel_res0)
            self.vel_rad = sat_parameter_blk.sat_p
            unpacked_bits = sat_parameter_blk.unpacked_bits
            # along-track orbit velocity component
            atr_vel_res0 = md.md_gr.md_block.lr_md_block.atr_vel_res
            para_id = 1
            sat_parameter_blk = blocks.zdb003(msg, unpacked_bits, n_vel_rb,
                                              vel_sat_rb, self.sv_array,
                                              para_id, atr_vel_res0)
            self.vel_atr = sat_parameter_blk.sat_p
            unpacked_bits = sat_parameter_blk.unpacked_bits
            # cross-track orbit velocity component
            ctr_vel_res0 = md.md_gr.md_block.lr_md_block.ctr_vel_res
            para_id = 2
            sat_parameter_blk = blocks.zdb003(msg, unpacked_bits, n_vel_rb,
                                              vel_sat_rb, self.sv_array,
                                              para_id, ctr_vel_res0)
            self.vel_ctr = sat_parameter_blk.sat_p
            unpacked_bits = sat_parameter_blk.unpacked_bits
        else:
            self.vel_rad = []
            self.vel_atr = []
            self.vel_ctr = []
            for ii in range(len(self.sv_array)):
                self.vel_rad.append([])
                self.vel_atr.append([])
                self.vel_ctr.append([])
                for jj in range(len(self.sv_array[ii])):
                    self.vel_rad[ii].append(0)
                    self.vel_atr[ii].append(0)
                    self.vel_ctr[ii].append(0)
        # pb_signals from metadata
        self.pb_signals = md.md_gr.md_block.lr_md_block.pb_signals_blk.signals
        # ssrz low rate code bias corrections
        # reference signals are dictionaries for gnss
        self.ref_signal = md.md_gr.md_block.lr_md_block.cb_ref_signals_blk.signals
        # number of rice blocks
        n_cb_rb = md.md_gr.md_block.lr_md_block.n_rb_cb
        # default resolution
        cb_res0 = md.md_gr.md_block.lr_md_block.cb_res
        # code bias rice block
        cb_rb_list = md.md_gr.md_block.lr_md_block.rb_cb
        sig_bias_blk = blocks.zdb004(msg, unpacked_bits, n_cb_rb,
                                     cb_rb_list, self.sv_array, cb_res0,
                                     self.pb_signals, self.ref_signal)
        self.cb = sig_bias_blk.sig_b
        unpacked_bits = sig_bias_blk.unpacked_bits
        # ssrz low rate phase bias corrections
        # default resolution
        pb_res = md.md_gr.md_block.lr_md_block.pb_res
        # number of pb bits
        pb_n_bits = md.md_gr.md_block.lr_md_block.pb_bl
        phase_bias_blk = blocks.zdb032(msg, unpacked_bits, self.sv_array,
                                       pb_res, pb_n_bits, self.pb_signals)
        # the phase bias is organized per gnss per satellite per signal
        self.pb = phase_bias_blk.pb
        self.conti = phase_bias_blk.conti
        unpacked_bits = phase_bias_blk.unpacked_bits
        # ssrz low rate satellite dependent global ionosphere correction
        # ionospheric correction flag
        self.iono_flag = md.md_gr.md_block.lr_md_block.iono_flag
        if self.iono_flag == 1:
            # ionospheric resolution from metadata
            iono_res = md.md_gr.md_block.lr_md_block.iono_res
            # ssrz compressed satellite dependent coefficients block for
            # the satellite dependent global ionsphere corrections
            self.tot_coeff = int(md.md_gr.md_block.lr_md_block.model.int_mp[0])
            # get layer height from metadata [km]
            self.layer_hgt = float(
                md.md_gr.md_block.lr_md_block.model.flt_mp[0])
            # number of orders m + l
            self.n_order = md.md_gr.md_block.lr_md_block.n_order
            # number of rice blocks
            n_rb = md.md_gr.md_block.lr_md_block.coeff_blk.n_rb
            # default bin size
            coeff_rb = md.md_gr.md_block.lr_md_block.coeff_blk.coeff_rb
            try:
                sat_glo_iono_blk = blocks.zdb008(msg, unpacked_bits,
                                                 self.sv_array,
                                                 n_rb, coeff_rb, iono_res,
                                                 self.tot_coeff)
                self.strg_nan = None
            except ValueError:
                self.strg_nan = 'SSRZ data are not available' + '\n'
                return
            # sat glo iono coefficients
            self.gsi_coeff = sat_glo_iono_blk.sat_p
            unpacked_bits = sat_glo_iono_blk.unpacked_bits
        else:
            self.gsi_coeff = []

    def __str__(self):
        strg = ('{:>4}'.format('Time') + '  {:^5}'.format('Sat') +
                '{:^6}'.format('IODe') + '{:^8}'.format('C0[m]') +
                '{:^8}'.format('C1[m]') +
                '{:^10}'.format('Rad[m]') + '{:^9}'.format('ATr[m]') +
                '{:^10}'.format('CTr[m]') + '{:^9}'.format('dRad[m/s]') +
                '{:^10}'.format('dATr[m/s]') + '{:^9}'.format('dCTr[m/s]') +
                ' {:^4}'.format('Sig') + '{:^9}'.format('CB[m]') +
                '{:^10}'.format('PB[cyc]') + ' {:^3}'.format('ODI') +
                '  {:^4}'.format('Sig') +
                '{:^9}'.format('CB[m]') + '{:^10}'.format('PB[cyc]') +
                ' {:^3}'.format('ODI') + ' ... \n')
        time = '{:>4.1f}'.format(self.time_tag_15)
        # loop over GNSS:
        for ii in range(len(self.sv_array)):
            sat_list = self.sv_array[ii]
            if len(sat_list) >= 1:
                gnss = sat_list[0][0]
                try:
                    sig_list = self.pb_signals[gnss]
                except KeyError:
                    continue
                try:
                    ref_signal = self.ref_signal[gnss][0]
                except KeyError:
                    ref_signal = 'n/a'
            else:
                continue
            # loop over satellites:
            for jj in range(len(sat_list)):
                sat = '  {:>3}'.format(sat_list[jj])
                iode = '  {:>3d}'.format(int(self.be_iod[ii][jj]))
                c0 = '  {:+7.3f}'.format(self.c0[ii][jj])
                c1 = '  {:+5.3f}'.format(self.c1[ii][jj])
                strg += time + sat + iode + c0 + c1
                if self.rad[ii][jj] is not None:
                    rad = '  {:>+8.4f}'.format(self.rad[ii][jj])
                    atr = '  {:>+8.4f}'.format(self.atr[ii][jj])
                    ctr = '  {:>+8.4f}'.format(self.ctr[ii][jj])
                    v_rad = '  {:>+7.4f}'.format(self.vel_rad[ii][jj])
                    v_atr = '  {:>+7.4f}'.format(self.vel_atr[ii][jj])
                    v_ctr = '  {:>+7.4f}'.format(self.vel_ctr[ii][jj])
                    strg += rad + atr + ctr + v_rad + v_atr + v_ctr + ' '
                # biases per signal
                for kk in range(len(sig_list)):
                    sig = '  {:>2}'.format(sig_list[kk])
                    if ref_signal == 'n/a':
                        cb_kk = kk
                    else:
                        cb_kk = kk - 1
                    if sig_list[kk] == ref_signal:
                        cb = '  {:>+8.4f}'.format(0)
                    else:
                        self.cb[ii][cb_kk][jj]
                        if (((np.sign(self.cb[ii][cb_kk][jj]) == -1) &
                             (self.cb[ii][cb_kk][jj] == 0)) |
                           (np.isnan(self.cb[ii][cb_kk][jj]))):
                            cb = '  {:>7}'.format('n/a')
                        else:
                            cb = '  {:>+8.4f}'.format(self.cb[ii][cb_kk][jj])
                    if self.pb[ii][jj][kk] == 0:
                        pb = '  {:>8}'.format('n/a')
                        conti = ' {:>3}'.format('n/a')
                    else:
                        pb = '  {:>+8.4f}'.format(self.pb[ii][jj][kk])
                        conti = ' {:>3}'.format(self.conti[ii][jj][kk])
                    strg += ' ' + sig + cb + pb + conti
                strg += '\n'
        if self.iono_flag == 1:
            strg += 'Satellite-dependent Global Ionosphere (GSI)' + '\n'
            strg += '{:>4}'.format('Time') + '  {:^5}'.format('Sat')
            a_lm_list = ['a00', 'a01', 'a10', 'a02', 'a11', 'a20', 'a03',
                         'a12', 'a21', 'a30']
            for ii in range(len(self.gsi_coeff)):
                strg += '  {:^7}'.format(a_lm_list[ii])
            strg += '\n'
            time = '{:>4.1f}'.format(self.time_tag_15)
            # loop over gnss
            for ii in range(len(self.sv_array)):
                sat_list = self.sv_array[ii]
                if len(sat_list) == 0:
                    continue
                # loop over satellites
                for jj in range(len(sat_list)):
                    sat = '  {:>3}'.format(sat_list[jj])
                    strg += time + sat
                    # loop over coeff
                    for kk in range(len(self.gsi_coeff)):
                        cff = '  {:>+7.3f}'.format(self.gsi_coeff[kk][ii][jj])
                        strg += cff
                    strg += '\n'
        return strg

    def __repr__(self):
        strg = ('Low rate corrections class with main objects: sv_array, ' +
                'cb, pb, rad, atr, ctr, vel_rad, vel_atr, vel_ctr, c0, c1, ' +
                'ref_signal, pb_signals, gsi_coeff. ' +
                'Orbit and clock corrections ' +
                '(e.g. rad, c0) are organized per GNSS per satellite. ' +
                'cb is organized per gnss, signal and satellite, while ' +
                'pb is organized per gnss, satellite and signal.')
        return strg


# *************************************************************************** #
#                                                                             #
#           SSRZ Gridded Ionosphere Correction Message    ZM003               #
#                                                                             #
# *************************************************************************** #
class IonoGrid:
    def __init__(self, msg, unpacked_bits, md, corr):
        # ssrz 15 minutes time tag zdf050
        [self.time_tag_15, unpacked_bits] = fields.zdf050(msg, unpacked_bits)
        # ssrz grid id
        [self.grid_id, unpacked_bits] = fields.zdf105(msg, unpacked_bits)
        # ssrz grid iod
        [self.grid_iod, unpacked_bits] = fields.zdf106(msg, unpacked_bits)
        # low rate metadata
        md_lr = md.md_gr.md_block.lr_md_block
        # iono grid metadata
        md_gri = md.md_gr.md_block.grid_iono_block
        # number of high rate satellite groups from metadata
        try:
            # try to check first in the satellite block of the metadata
            n_g = md.md_gr.sat_md_blk.n_g_lr
        except AttributeError:
            # check in the satellite block of the satellite group
            n_g = md.sat_gr.sat_md_blk.n_g_lr
        # satellite group list mask zdf016
        [self.sv_gr_bit_mask, unpacked_bits] = fields.zdf016(msg,
                                                             unpacked_bits,
                                                             n_g)
        # satellite group considered
        sat_group_idx = np.where(self.sv_gr_bit_mask == 1)[0][0]
        # find the corresponding low rate corrections if already transmitted
        # and valid
        timing_idx = 0
        for ii in range(len(corr.lr)):
            if len(corr.lr[ii]) > sat_group_idx:
                lr_corr = corr.lr[ii][sat_group_idx]
                timing_idx == ii
        self.sat_group_index = sat_group_idx
        self.t_blk_index = timing_idx
        # check available satellite in low rate message
        # grid update interval
        if isinstance(md_gri, blocks.zmb003_1):
            grid_ui = md_gri.gri_time_blk[timing_idx].ssrz_timing_block.ui
        elif isinstance(md_gri, blocks.zmb003_2):
            grid_ui = md_gri.time_para.blk_list[timing_idx].upd_off_blk.t_upd
        else:
            print("Warning: please consider only zmb003-1 and zmb003-2")
            self.no_lr = 1
            return
        if ((self.time_tag_15 - lr_corr.time_tag_15 >
             md_lr.lr_timing_block[timing_idx].ssrz_timing_block.ui - 1) |
            (self.time_tag_15 - lr_corr.time_tag_15 >
             grid_ui - 1)):
            self.no_lr = 1
            return
        else:
            self.no_lr = 0
        # at this point it is possible to assume that the hr corr can be
        # decoded for the same satellites of the lr corr
        self.sv_array = lr_corr.sv_array
        # grid metadata
        md_grid_gr_list = md.grid_gr
        # find the correct grid
        grid_idx = None
        for ii in range(len(md_grid_gr_list)):
            if grid_idx is None:
                pass
            else:
                break
            md_grid = md_grid_gr_list[ii].grid_block
            igrd = 0
            while ((grid_idx is None) and (igrd < md_grid.n_grids)):
                if ((md_grid.grid_blk_list[igrd].id == self.grid_id)
                        and (md_grid.grid_blk_list[igrd].iod == self.grid_iod)):
                    grid_idx = igrd
                    break
                else:
                    igrd += 1
        self.grid_number = ii  # Save the corresponding grid number
        self.md_grid = md_grid
        if grid_idx is None:
            self.n_pts = 0
            self.grid_values = []
            self.sv_array = []
            return
        # grid chains
        self.chains = md_grid.grid_blk_list[grid_idx].chain_blk
        # compute tot number of points
        self.n_pts = 0
        for ii in range(len(self.chains)):
            self.n_pts += self.chains[ii].n_pts
        # get iono grid metadata from zmb003
        # resolution  gr_iono_res
        res0 = md_gri.gr_iono_res
        # bin size parameter bin_size
        p0 = md_gri.bin_size
        # gridded data block
        self.grid_values = []
        # predictor flag
        self.pred_flag = []
        # loop over gnss
        for ii in range(len(self.sv_array)):
            self.grid_values.append([])
            self.pred_flag.append([])
            if len(self.sv_array[ii]) == 0:
                continue
            else:
                sat_list = self.sv_array[ii]
                for jj in range(len(sat_list)):
                    self.grid_values[ii].append([])
                    self.pred_flag[ii].append([])
                    # ssrz compressed grid_valuesdded data block for
                    # grid_valuesdded ionosphere corrections
                    chain_block = blocks.zdb006(msg, unpacked_bits,
                                                self.chains, p0, res0)

                    self.grid_values[ii][jj] = chain_block.chain
                    self.pred_flag[ii][jj] = chain_block.pred_flag
                    unpacked_bits = chain_block.unpacked_bits

    def __str__(self):
        strg = ('{:>4}'.format('Time') + '  {:^5}'.format('ID') +
                '{:^6}'.format('IODG') + '{:^5}'.format('Sat') +
                '{:^8}'.format('PredFl'))
        for ii in range(self.n_pts):
            strg += '{:^9}'.format('gr_pt')
        strg += '\n'
        strg += '{:30s}'.format('                             ')
        for ii in range(self.n_pts):
            strg += '{:^9.0f}'.format(ii)
        strg += '\n'

        time = '{:>4.1f}'.format(self.time_tag_15)
        grid_values_id = '{:>4.0f}'.format(self.grid_id)
        grid_values_iod = '{:>5.0f}'.format(self.grid_iod)

        # loop over GNSS:
        for ii in range(len(self.sv_array)):
            sat_list = self.sv_array[ii]
            # loop over satellites:
            for jj in range(len(sat_list)):
                sat = '    {:>3}'.format(sat_list[jj])
                values = self.grid_values[ii][jj]
                pred_flag = '  {:>3}'.format(self.pred_flag[ii][jj][0])
                strg += (time + grid_values_id + grid_values_iod + sat +
                         pred_flag + '   ')
                for val in values:
                    if np.isnan(val):
                        grid_values = '  {:>7}'.format('n/a')
                    else:
                        grid_values = '  {:+7.4f}'.format(val)
                    strg += grid_values
                strg += '\n'
        return strg

    def __repr__(self):
        strg = ('Gridded ionosphere corrections class' +
                ' with main objects: ' +
                'sv_array, n_pts, time_tag_15, grid_values_id, ' +
                'grid_values_iod, grid_values.' +
                ' The grid values are contained in ' +
                'grid_values and organized ' +
                'per GNSS per satellite.')
        return strg


# *************************************************************************** #
#                                                                             #
#           SSRZ Gridded Troposphere Correction Message    ZM004              #
#                                                                             #
# *************************************************************************** #
class TropoGrid:
    def __init__(self, msg, unpacked_bits, md):
        # ssrz 15 minutes time tag zdf050
        [self.time_tag_15, unpacked_bits] = fields.zdf050(msg, unpacked_bits)
        # ssrz grid id
        [self.grid_id, unpacked_bits] = fields.zdf105(msg, unpacked_bits)
        # ssrz grid iod
        [self.grid_iod, unpacked_bits] = fields.zdf106(msg, unpacked_bits)
        # grid metadata
        md_grid_gr_list = md.grid_gr
        # find the correct grid
        grid_idx = None
        for ii in range(len(md_grid_gr_list)):
            if grid_idx is None:
                pass
            else:
                break
            md_grid = md_grid_gr_list[ii].grid_block
            igrd = 0
            while ((grid_idx is None) and (igrd < md_grid.n_grids)):
                if ((md_grid.grid_blk_list[igrd].id == self.grid_id)
                        and (md_grid.grid_blk_list[igrd].iod == self.grid_iod)):
                    grid_idx = igrd
                    break
                else:
                    igrd += 1
        self.grid_number = ii  # Save the corresponding grid number
        self.md_grid = md_grid
        if grid_idx is None:
            self.n_pts = 0
            self.grid_values = []
            self.sv_array = []
            return
        # grid chains
        self.chains = md_grid.grid_blk_list[grid_idx].chain_blk
        # compute tot number of points
        self.n_pts = 0
        for ii in range(len(self.chains)):
            self.n_pts += self.chains[ii].n_pts
        # tropo grid metadata
        md_grt = md.md_gr.md_block.grid_tropo_block
        # get tropo grid metadata from zmb003
        self.components = md_grt.components
        # list of resolutions per components
        res0_list = md_grt.res
        # bin size parameter list per component bin_size
        p0_list = md_grt.p0
        # gridded data block
        self.grid_values = []
        # predictor flag
        self.pred_flag = []
        # loop over components
        for ii in range(len(self.components)):
            res0 = res0_list[ii]
            p0 = p0_list[ii]
            self.grid_values.append([])
            self.pred_flag.append([])
            chain_block = blocks.zdb006(msg, unpacked_bits, self.chains,
                                        p0, res0)

            self.grid_values[ii] = chain_block.chain
            self.pred_flag[ii] = chain_block.pred_flag
            unpacked_bits = chain_block.unpacked_bits

    def __str__(self):
        strg = ('{:>4}'.format('Time') + '  {:^5}'.format('ID') +
                '{:^6}'.format('IODe') + '{:^5}'.format('Comp') +
                '{:^8}'.format('PredFl'))
        for ii in range(self.n_pts):
            strg += '{:^10}'.format('gr_pt')
        strg += '\n'
        strg += '{:30s}'.format('                             ')
        for ii in range(self.n_pts):
            strg += '{:^10.0f}'.format(ii)
        strg += '\n'

        time = '{:>4.1f}'.format(self.time_tag_15)
        grid_id = '{:>4.0f}'.format(self.grid_id)
        grid_iod = '{:>5.0f}'.format(self.grid_iod)

        # loop over component:
        for ii in range(len(self.components)):
            comp = '    {:>3}'.format(self.components[ii])
            values = self.grid_values[ii]
            pred_flag = '  {:>3}'.format(self.pred_flag[ii][0])
            strg += time + grid_id + grid_iod + comp + pred_flag + '   '
            for val in values:
                if np.isnan(val):
                    grt = '  {:>7}'.format('n/a')
                else:
                    grt = '  {:+7.5f}'.format(val)
                strg += grt
            strg += '\n'
        return strg

    def __repr__(self):
        strg = ('Gridded troposphere corrections class with main objects: ' +
                'n_pts, time_tag_15, grid_id, grid_iod, gri.' +
                ' The gridded values are contained in gri and organized ' +
                'per tropospheric component.')
        return strg


# *************************************************************************** #
#                                                                             #
# SSRZ Satellite dependent regional ionosphere correction message    ZM005    #
#                                                                             #
# *************************************************************************** #
class IonoSatReg:
    def __init__(self, msg, unpacked_bits, md, corr):
        # number of low rate satellite groups from metadata
        try:
            # first try to check  in the satellite block of the metadata
            n_g = md.md_gr.sat_md_blk.n_g_lr
        except AttributeError:
            # check in the satellite block of the satellite group
            n_g = md.sat_gr.sat_md_blk.n_g_lr
        # ssrz 15 minutes time tag zdf050
        [self.time_tag_15, unpacked_bits] = fields.zdf050(msg, unpacked_bits)
        # satellite group list mask zdf016
        [self.sv_gr_bit_mask, unpacked_bits] = fields.zdf016(msg,
                                                             unpacked_bits,
                                                             n_g)
        # satellite group considered
        sat_group_idx = np.where(self.sv_gr_bit_mask == 1)[0][0]
        # find the corresponding low rate corrections if already transmitted
        # and valid
        timing_idx = 0
        for ii in range(len(corr.lr)):
            if len(corr.lr[ii]) > sat_group_idx:
                lr_corr = corr.lr[ii][sat_group_idx]
                timing_idx == ii
        self.sat_group_index = sat_group_idx
        self.t_blk_index = timing_idx
        # check available satellite in low rate message
        if ((self.time_tag_15 - lr_corr.time_tag_15 >
             md.md_gr.md_block.lr_md_block.lr_timing_block[timing_idx].ssrz_timing_block.ui - 1)):
            self.no_lr = 1
            return
        else:
            self.no_lr = 0
        # at this point it is possible to assume that the hr corr can be
        # decoded for the same satellites of the lr corr
        self.sv_array = lr_corr.sv_array
        # use metadata for metadatatag=1
        if md.md_gr.md_block.sat_reg_iono_block.md_tag == 1:
            # total number of coefficients from metadata
            self.tot_coeff = int(
                md.md_gr.md_block.sat_reg_iono_block.model_blk.int_mp[0])
            # get layer height from metadata [km]
            self.layer_hgt = float(
                md.md_gr.md_block.sat_reg_iono_block.model_blk.flt_mp[0])
            # get ground point origin from metadata [deg]
            self.gpo_lat = float(
                md.md_gr.md_block.sat_reg_iono_block.model_blk.flt_mp[1])
            # get ground point origin from metadata [deg]
            self.gpo_lon = float(
                md.md_gr.md_block.sat_reg_iono_block.model_blk.flt_mp[2])
            # get ground point origin from metadata [m]
            self.gpo_hgt = float(
                md.md_gr.md_block.sat_reg_iono_block.model_blk.flt_mp[3])
            # get ground point origin update interval [s]
            self.gpo_update = float(
                md.md_gr.md_block.sat_reg_iono_block.model_blk.int_mp[2])
        # ssrz compressed satellite dependent regional ionosphere coefficients
        # block zdb008
        # number of orders m + l
        self.n_order = md.md_gr.md_block.sat_reg_iono_block.n_order
        # number of rice blocks
        n_rb = md.md_gr.md_block.sat_reg_iono_block.coeff_blk.n_rb
        # default bin size
        coeff_rb = md.md_gr.md_block.sat_reg_iono_block.coeff_blk.coeff_rb
        # resolution
        res0 = md.md_gr.md_block.sat_reg_iono_block.sat_reg_res
        try:
            sat_reg_iono_blk = blocks.zdb008(msg, unpacked_bits,
                                             self.sv_array, n_rb, coeff_rb,
                                             res0, self.tot_coeff)
            self.strg_nan = None
        except ValueError:
            self.strg_nan = 'SSRZ data are not available' + '\n'
            return
        # sat reg iono coefficients
        self.coeff = sat_reg_iono_blk.sat_p
        unpacked_bits = sat_reg_iono_blk.unpacked_bits

    def __str__(self):
        strg = '{:>4}'.format('Time') + '  {:^5}'.format('Sat')
        a_lm_list = ['a00', 'a01', 'a10', 'a02', 'a11', 'a20', 'a03',
                     'a12', 'a21', 'a30']
        for ii in range(len(self.coeff)):
            strg += '  {:^7}'.format(a_lm_list[ii])
        strg += '\n'
        time = '{:>4.1f}'.format(self.time_tag_15)
        # loop over gnss
        for ii in range(len(self.sv_array)):
            sat_list = self.sv_array[ii]
            if len(sat_list) == 0:
                continue
            # loop over satellites
            for jj in range(len(sat_list)):
                sat = '  {:>3}'.format(sat_list[jj])
                strg += time + sat
                # loop over coeff
                for kk in range(len(self.coeff)):
                    coeff = '  {:>+7.3f}'.format(self.coeff[kk][ii][jj])
                    strg += coeff
                strg += '\n'
        return strg

    def __repr__(self):
        strg = ('Regional ionosphere corrections class with main objects: ' +
                'time_tag_15, sv_array, n_order, coeff. The coefficients ' +
                'are organized per GNSS ii per satellite jj  as follows: ' +
                'coeff[ii][jj]. The vector of coefficients follows the order' +
                'of the SSRZ documentation.')

        return strg


# *************************************************************************** #
#                                                                             #
#            SSRZ Global VTEC ionosphere correction message    ZM006          #
#                                                                             #
# *************************************************************************** #
class IonoGloVtec:
    def __init__(self, msg, unpacked_bits, md, corr):
        [self.time_tag_15, unpacked_bits] = fields.zdf050(msg, unpacked_bits)
        # ssrz vtec flag
        [self.vtec_flag, unpacked_bits] = fields.zdf330(msg, unpacked_bits)
        # ssrz correction block
        # fetching the metadata
        md_gvi = md.md_gr.md_block.glo_vtec
        # resolution
        self.res = md_gvi.res
        # total number of corrections Nvtec
        self.n_vtec = int(md_gvi.model_blk.int_mp[0])
        # number of ionospheric layers
        self.n_layer = int(md_gvi.model_blk.int_mp[1])  # always one
        # degree of spherical harmonics
        self.deg = int(md_gvi.model_blk.int_mp[2])
        # order of spherical harmonics
        self.ord = int(md_gvi.model_blk.int_mp[3])
        # height of ionospheric layer
        self.hgt = float(md_gvi.model_blk.flt_mp[1])  # [km]
        # ssrz global vtec bin size indicator
        [zdf331, unpacked_bits] = fields.zdf331(msg, unpacked_bits)
        p = 10 + zdf331
        # compute coefficients with zdb001
        self.cos_coeff = []
        self.sin_coeff = []
        # number of cosine
        self.n_cos = int((self.deg + 1) * (self.deg + 2) / 2 -
                         (self.deg - self.ord) * (self.deg - self.ord + 1) / 2)
        # number of cosine
        self.n_sin = int((self.deg + 1) * (self.deg + 2) / 2 -
                         (self.deg - self.ord) * (self.deg - self.ord + 1) / 2
                         - (self.deg + 1))

        for ii in range(self.n_cos):
            coeff_block = blocks.zdb062(msg, unpacked_bits, p)
            self.cos_coeff = np.append(self.cos_coeff,
                                       coeff_block.m * self.res)
            unpacked_bits = coeff_block.unpacked_bits
        for jj in range(self.n_sin):
            coeff_block = blocks.zdb062(msg, unpacked_bits, p)
            self.sin_coeff = np.append(self.sin_coeff,
                                       coeff_block.m * self.res)
            unpacked_bits = coeff_block.unpacked_bits

    def __str__(self):
        strg = 'Total number of coefficients: ' + str(self.n_vtec) + '\n'
        index = 0
        for ii in range(int(self.ord) + 1):
            if index < self.n_cos:
                strg += ('C' + f'{ii}' + '[TECU]' +
                         ': ' + str(self.cos_coeff[index: index +
                                                   (int(self.deg) +
                                                    1 - ii)]) +
                         '\n')
            index += (int(self.deg) + 1 - ii)

        index = 0
        for jj in range(int(self.ord)):
            if index < self.n_sin:
                strg += ('S' + f'{jj+1}' + '[TECU]' +
                         ': ' + str(self.sin_coeff[index: index +
                                                   (int(self.deg) +
                                                    1 - (jj + 1))]) + '\n')
            index += int(self.deg - jj)
        return strg

    def __repr__(self):
        strg = ('Global ionosphere VTEC corrections.')
        return strg


# *************************************************************************** #
#                                                                             #
#              SSRZ regional troposphere correction message   ZM007           #
#                                                                             #
# *************************************************************************** #
class TropoReg:
    def __init__(self, msg, unpacked_bits, md, corr):
        # metadata tag
        self.md_tag = md.md_gr.md_block.reg_tropo.zdf020
        [self.time_tag_15, unpacked_bits] = fields.zdf050(msg, unpacked_bits)
        # initialize coefficients
        self.coeff = []
        # regional tropo metadata
        if (self.md_tag == 2):
            md_tropo = md.md_gr.md_block.reg_tropo.blk
            # number of components from metadata
            self.components = md_tropo.trans_comp
            n_comp = len(self.components)
            # starting elevation (max) for mapping improvements
            self.map_el = float(md_tropo.model_blk.flt_mp[0])
            # save ground point origin coordinates
            self.gpo_llh = [float(md_tropo.model_blk.flt_mp[1]),
                            float(md_tropo.model_blk.flt_mp[2]),
                            float(md_tropo.model_blk.flt_mp[3])]
            # max order latitute
            self.max_order_lat = int(md_tropo.model_blk.int_mp[3])
            self.max_order_lon = int(md_tropo.model_blk.int_mp[4])
            self.max_order_hgt = int(md_tropo.model_blk.int_mp[5])
            self.n_para_max = (self.max_order_lat *
                               self.max_order_lon *
                               self.max_order_hgt)
            for ii in range(n_comp):
                self.coeff.append([])
                if self.components[ii] == "m":
                    n_para_max = int(md_tropo.model_blk.int_mp[6])
                else:
                    n_para_max = self.n_para_max
                # resolution of the component
                res = md_tropo.res[ii]
                # coefficient blocks
                rb_list = md_tropo.cff[ii].coeff_rb

                # ssrz compressed coefficients block
                if self.components[ii] == "m":
                    comp_coeff = blocks.zdb009(msg, unpacked_bits, rb_list,
                                               n_para_max, res)
                else:
                    comp_coeff = blocks.zdb007(msg, unpacked_bits, rb_list,
                                               n_para_max, res)
                self.coeff[ii] = comp_coeff.coeff
                unpacked_bits = comp_coeff.unpacked_bits
        else:
            mreg_flag = 1
            while mreg_flag == 1:
                # Remark: currently, the demo expects to work with only one
                # region.
                # Region ID r_id
                [r_id, unpacked_bits] = fields.zdf027(msg, unpacked_bits)
                md_tropo = md.md_gr.md_block.reg_tropo.blk[r_id - 1]
                # number of components from metadata
                self.components = md_tropo.trans_comp
                # save ground point origin coordinates
                self.gpo_llh = [float(md_tropo.lat_gpo),
                                float(md_tropo.lon_gpo),
                                float(md_tropo.hgt_gpo)]
                self.max_hor = []
                self.max_hgt = []
                n_comp = len(self.components)
                for ii in range(n_comp):
                    self.max_hor.append(md_tropo.max_hor[ii])
                    self.max_hgt.append(md_tropo.max_hgt[ii])
                    self.coeff.append([])
                    tro_bas = md_tropo.cff[ii]
                    n_para_max = tro_bas.n_coeff
                    rb_list = tro_bas.blk.coeff_rb
                    res = md_tropo.res[ii]
                    # ssrz compressed coefficients block
                    if self.components[ii] == "m":
                        comp_coeff = blocks.zdb009(msg, unpacked_bits, rb_list,
                                                   n_para_max, res)
                    else:
                        comp_coeff = blocks.zdb007(msg, unpacked_bits, rb_list,
                                                   n_para_max, res)
                    self.coeff[ii] = comp_coeff.coeff
                    unpacked_bits = comp_coeff.unpacked_bits
                # Update multi region flag
                mreg_flag = md_tropo.mreg_flag

    def __str__(self):
        strg = '{:>4}'.format('Time') + '  {:^5}'.format('Comp')
        strg += '\n'
        time = '{:>4.1f}'.format(self.time_tag_15)
        # loop over components
        for ii in range(len(self.components)):
            strg += time + '{:^5}'.format(self.components[ii])
            for jj in range(len(self.coeff[ii])):
                coeff = '  {:>+7.6f}'.format(self.coeff[ii][jj])
                strg += coeff
            strg += '\n'
        return strg

    def __repr__(self):
        strg = ('Regional troposphere corrections class with main objects: ' +
                'time_tag_15, components, max_order_lat, max_order_lon, ' +
                'max_order_hgt, coeff. The coefficients ' +
                'are organized per component ii as follows: ' +
                'coeff[ii]. The vector of coefficients follows the order' +
                'of the SSRZ documentation, but accordingly to a non-squared' +
                'lat-lon description of the coefficients defined by the ' +
                'max_order_lat and max_order_lon.')

        return strg


# *************************************************************************** #
#                                                                             #
#                      SSRZ QIX Bias correction message ZM008                 #
#                                                                             #
# *************************************************************************** #
class QixBias:
    def __init__(self, msg, unpacked_bits, md):
        # ssrz qix metadata flag
        [self.qix_md_flag, unpacked_bits] = fields.zdf320(msg, unpacked_bits)
        if self.qix_md_flag == 1:
            # metadata tag
            [self.md_tag, unpacked_bits] = fields.zdf020(msg, unpacked_bits)
            if ((self.md_tag == 1) | (self.md_tag == 2)):
                self.qix_md = blocks.zmb008(msg, unpacked_bits, self.md_tag)
                unpacked_bits = self.qix_md.unpacked_bits
        else:
            self.qix_md = md.md_gr.md_block.qix_bias
            self.md_tag = self.qix_md.zdf020
        if ((self.md_tag == 1) | (self.md_tag == 2)):
            # gnss id bit mask
            [self.gnss, unpacked_bits] = fields.zdf012(msg, unpacked_bits)
            self.max_sat_id = []
            for ii in range(len(self.gnss)):
                [max_sat, unpacked_bits] = fields.zdf013(msg, unpacked_bits)
                self.max_sat_id = np.append(self.max_sat_id, int(max_sat))

            self.sat_group = {}
            for gg in range(len(self.gnss)):
                syst = self.gnss[gg]
                self.sat_group[syst] = []
                for ii in range(int(self.max_sat_id[gg])):
                    ii += 1  # to avoid prn = 0
                    # satellite group bit mask per gnss zdf015
                    [bit_mask,
                        unpacked_bits] = fields.zdf015(msg, unpacked_bits)
                    if bit_mask == 0:
                        continue
                    else:
                        if ii < 10:
                            prn = '0' + str(ii)
                        else:
                            prn = str(ii)
                        self.sat_group[syst] = np.append(self.sat_group[syst],
                                                         syst + prn)
            # code bias
            if ((self.md_tag == 1) | ((self.md_tag == 2) &
               (self.qix_md.qix_cb_flag == 1))):
                # ssrz compressed signal bias block for qix coed bias
                # corrections
                n_cb_rb = self.qix_md.n_cb_rb
                cb_rb_list = self.qix_md.cb_rb_list
                res0 = self.qix_md.cb_res
                biases = blocks.zdb004_qix(msg, unpacked_bits,
                                           n_cb_rb, cb_rb_list,
                                           self.gnss,
                                           self.sat_group, res0)
                self.qix_cb = biases.sig_b
                self.qix_cb_signals = biases.sig_list

                if ((self.md_tag == 2) & (self.qix_md.qix_cb_flag == 1) &
                   (self.qix_md.qix_pb_flag == 0)):
                    self.qix_pb = np.array(biases.sig_b, copy=True)
                    for ii in range(len(self.qix_pb)):
                        for jj in range(len(self.qix_pb[ii])):
                            self.qix_pb[ii][jj] = self.qix_pb[ii][jj] * 0
            # phase bias
            if ((self.md_tag == 1) | ((self.md_tag == 2) &
               (self.qix_md.qix_pb_flag == 1))):
                # ssrz compressed signal bias block for qix coed bias
                # corrections
                n_pb_rb = self.qix_md.n_pb_rb
                pb_rb_list = self.qix_md.pb_rb_list
                res0 = self.qix_md.pb_res
                biases = blocks.zdb004_qix(msg, unpacked_bits,
                                           n_pb_rb, pb_rb_list,
                                           self.gnss,
                                           self.sat_group, res0)
                self.qix_pb = biases.sig_b
                self.qix_pb_signals = biases.sig_list
                if ((self.md_tag == 2) & (self.qix_md.qix_cb_flag == 0) &
                   (self.qix_md.qix_pb_flag == 1)):
                    self.qix_cb = biases.sig_b
                    for ii in range(len(self.qix_pb)):
                        for jj in range(len(self.qix_pb[ii])):
                            self.qix_cb[ii][jj] = self.qix_pb[ii][jj] * 0

    def __str__(self):
        if self.qix_md_flag == 0:
            strg = 'QIX bias n/a'
        else:
            strg = ''
        if ((self.md_tag == 1) | ((self.md_tag == 2) &
           (self.qix_md.qix_cb_flag == 1))):
            # code bias
            strg = 'QIX Code bias' + '\n'
            # loop over GNSS:
            for ii in range(len(self.gnss)):
                for jj in range(len(self.qix_cb_signals[ii])):
                    sig = self.qix_cb_signals[ii][jj]
                    strg += '    {:>6}'.format(str(sig))
                strg += '\n'
                gnss = self.gnss[ii]
                sat_list = self.sat_group[gnss]
                if len(sat_list) < 1:
                    continue
                # loop over satellites:
                for jj in range(len(sat_list)):
                    strg += str(sat_list[jj])
                    # biases per signal
                    for kk in range(len(self.qix_cb_signals[ii])):
                        if (((np.sign(self.qix_cb[ii][jj][kk]) == -1) &
                             (self.qix_cb[ii][jj][kk] == 0)) |
                                (np.isnan(self.qix_cb[ii][jj][kk]))):
                            cb = '  {:>7}'.format('n/a')
                        else:
                            cb = '  {:>+6.5f}'.format(self.qix_cb[ii][jj][kk] *
                                                      1e-3)
                        strg += ' ' + cb
                    strg += '\n'
        else:
            strg += 'QIX Code Bias n/a'
        # phase bias
        if ((self.md_tag == 1) | ((self.md_tag == 2) &
           (self.qix_md.qix_pb_flag == 1))):
            strg = 'QIX Phase bias [m]' + '\n'
            # loop over GNSS:
            for ii in range(len(self.gnss)):
                gnss = self.gnss[ii]
                sat_list = self.sat_group[gnss]
                if len(sat_list) < 1:
                    continue
                for jj in range(len(self.qix_pb_signals[ii])):
                    sig = self.qix_pb_signals[ii][jj]
                    strg += ' {:>6}'.format(str(sig))
                strg += '\n'
                # loop over satellites:
                for jj in range(len(sat_list)):
                    # biases per signal
                    for kk in range(len(self.qix_pb_signals[ii])):
                        if (((np.sign(self.qix_pb[ii][jj][kk]) == -1) &
                             (self.qix_pb[ii][jj][kk] == 0)) |
                                (np.isnan(self.qix_pb[ii][jj][kk]))):
                            pb = '  {:>7}'.format('n/a')
                        else:
                            pb = '  {:>+6.5f}'.format(self.qix_pb[ii][jj][kk] *
                                                      1e-3)
                        strg += ' ' + pb
                    strg += '\n'
        else:
            strg += 'QIX Phase Bias n/a'
        return strg

    def __repr__(self):
        strg = 'QIX bias class with main objects: qix_md_flag, qix_cb, qix_pb'
        return strg


# *************************************************************************** #
#                                                                             #
#                        SSRZ time tag message ZM009                          #
#                                                                             #
# *************************************************************************** #
class TimeTag:
    def __init__(self, msg, unpacked_bits):
        # ssrz time tag definition
        [self.time_tag_def, unpacked_bits] = fields.zdf230(msg, unpacked_bits)
        if self.time_tag_def == 0:
            [self.hour_thirty_s, unpacked_bits] = fields.zdf231(msg,
                                                                unpacked_bits)
        elif self.time_tag_def == 1:
            [self.hour_five_s, unpacked_bits] = fields.zdf232(msg,
                                                              unpacked_bits)
        elif self.time_tag_def == 2:
            [self.time_tag_15, unpacked_bits] = fields.zdf050(msg,
                                                              unpacked_bits)
        elif self.time_tag_def == 3:
            [self.hour_one_s, unpacked_bits] = fields.zdf233(msg,
                                                             unpacked_bits)
        elif self.time_tag_def == 4:
            [self.day_one_s, unpacked_bits] = fields.zdf234(msg,
                                                            unpacked_bits)
        elif self.time_tag_def == 5:
            [self.gps_tow, unpacked_bits] = fields.zdf052(msg,
                                                          unpacked_bits)
        elif self.time_tag_def == 6:
            [self.gps_week, unpacked_bits] = fields.zdf051(msg,
                                                           unpacked_bits)
            [self.gps_tow, unpacked_bits] = fields.zdf052(msg,
                                                          unpacked_bits)

    def __str__(self):
        strg = 'Time tag definition: ' + str(self.time_tag_def) + '\n'
        if self.time_tag_def == 0:
            strg += ('1hour-30seconds time tag [s]: ' +
                     str(self.hour_thirty_s) + '\n')
        elif self.time_tag_def == 1:
            strg += ('1hour-5seconds time tag [s]: ' +
                     str(self.hour_five_s) + '\n')
        elif self.time_tag_def == 2:
            strg += ('15 minutes time tag [s]: ' +
                     str(self.time_tag_15) + '\n')
        elif self.time_tag_def == 3:
            strg += ('1hour-1seconds time tag [s]: ' +
                     str(self.hour_one_s) + '\n')
        elif self.time_tag_def == 4:
            strg += ('1day-1seconds time tag [s]: ' +
                     str(self.day_one_s) + '\n')
        elif self.time_tag_def == 5:
            strg += ('GPS time of the week [s]: ' +
                     str(self.gps_tow) + '\n')
        elif self.time_tag_def == 6:
            strg += ('GPS week                : ' +
                     str(self.gps_week) + '\n')
            strg += ('GPS time of the week [s]: ' +
                     str(self.gps_tow) + '\n')
        return strg

    def __repr__(self):
        strg = ('Time tag message. The objects depend on the object ' +
                'time_tag_def. For example, if time_tag_def == 6, ' +
                'the additional objects are: gps_week and gps_tow.')
        return strg
