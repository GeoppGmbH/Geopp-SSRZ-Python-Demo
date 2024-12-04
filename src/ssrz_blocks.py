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
    SSRZ data-block multi class file
    ***************************************************************************
    Description:
    the module contains classes and methods to decode the SSRZ blocks.
    The constructor of the 'ssrz_blocks' module expects
    specific parts of the SSRZ message content as a byte object.
    Each block class receives in input the message and the list of already
    unpacked bits. In output, each block provides the decoded content of the
    block and the unpacked bits after decoding the block.
    ***************************************************************************

    References:
        - Geo++ SSRZ documentation v1.1.2
    """
import numpy as np
import ssrz_fields as fields


# =============================================================================
#                       Ancillary classes and methods
# =============================================================================
def compute_order_list(np):
    """ Compute the length of the list of m+l combined order given
        the total number of satellite parameters. For example if the output is
        3 it means that the maximum m+l = 2.
    """
    # max number per column/row considering that the number needs to be the
    # same for the two directions (row and column)
    nmax = round(np/2 + 1e-12)
    kk = 0
    ml_list = []
    for mm in range(nmax):
        kk += 1
        if kk + nmax <= np:
            for ll in range(nmax):
                kk += 1
                ml_list.append(mm + ll)
    return len(ml_list)


def compute_ncoeff_from_order_triangle(lm):
    """ Compute the number of coeff for a given maximum order l + m,
        considering the triangle configuration.
    """
    n_coeff = 0
    for ord in range(lm + 1):
        n_coeff += ord + 1
    return n_coeff


class MakeGrid:
    """ Method to reconstruct the grid of the grid metadata. The main idea is
        to go through a chain following a list of baselines and reconstruct
        a grid of points.
        Reference:
            Geo++ State Space Representation Format (SSRZ)
            Document version 1.0
            2020-04-19
    """

    def __init__(self, r1, r2, ss, ds, use_base_flag, pos_flag,
                 add_left_flag, add_right_flag, res):
        # parameters initialization
        lat = np.array([r1[0], r2[0]])
        lon = np.array([r1[1], r2[1]])
        kk = 0  # index defining the baseline to use
        r1_list = [[r1[0], r1[1]]]
        r2_list = [[r2[0], r2[1]]]
        for ii in range(len(ss)):
            # check if the current baseline should be used or one of the next
            # triangle
            if use_base_flag[ii] == 0:
                kk += 1
            r1 = r1_list[kk]
            r2 = r2_list[kk]
            r3 = self.compute_triangle(r1, r2, ss[ii], ds[ii], pos_flag[ii])
            lat3 = r3[0]
            lon3 = r3[1]
            lat = np.append(lat, lat3)
            lon = np.append(lon, lon3)
            # update baseline list
            [r1_list, r2_list] = self.add_baseline(r1, r2, r3,
                                                   add_left_flag[ii],
                                                   add_right_flag[ii],
                                                   pos_flag[ii], r1_list,
                                                   r2_list)
        self.lat = lat
        self.lon = lon

    def compute_triangle(self, r1, r2, ss, ds, point_pos):
        """ Function to complete the triangle computing r3 given r1 and r2.
        """
        s12 = np.linalg.norm(np.array(r2) - np.array(r1))
        t12 = np.arctan2(r2[1] - r1[1], r2[0] - r1[0])
        s13 = s12 + ss - ds/2
        s23 = s13 + ds
        alpha = np.arccos((s13 ** 2 + s12 ** 2 - s23 ** 2) / (2 * s13 * s12))
        if point_pos == 1:
            t3 = t12 - alpha
        else:
            t3 = t12 + alpha
        r3 = np.array(r1) + np.array([s13 * np.cos(t3), s13 * np.sin(t3)])
        # transform to python list
        r3 = [r3[0], r3[1]]
        return r3

    def add_baseline(self, r1, r2, r3, add_left, add_right, point_pos,
                     r1_list, r2_list):
        """ Function to add a new baseline. The direction of the baseline
            depends on the point position flag for both the addition
            on the left and on the right. Please refer to the corresponding
            fields and the documentation.
        """
        if ((add_left == 1) & (add_right == 0)):
            if point_pos == 1:
                # r3 is on the left of s12 --> s13 has to be used
                r1_list.append(r1)
                r2_list.append(r3)

            else:
                # r3 is on the right of s12 --> s23 has to be used
                r1_list.append(r2)
                r2_list.append(r3)

        elif ((add_left == 0) & (add_right == 1)):
            if point_pos == 1:
                # r3 is on the left of s12 --> s23 has to be used
                r1_list.append(r3)
                r2_list.append(r2)
            else:
                # r3 is on the right of s12 --> s13 has to be used
                r1_list.append(r3)
                r2_list.append(r1)
        elif ((add_left == 1) & (add_right == 1)):
            if point_pos == 1:
                r1_list.append(r1)
                r2_list.append(r3)
                r1_list.append(r3)
                r2_list.append(r2)
            else:
                r1_list.append(r2)
                r2_list.append(r3)
                r1_list.append(r3)
                r2_list.append(r1)

        return r1_list, r2_list


class GridDefinition:
    """ This class handles the grid definition block. It has a list of grid
        blocks and a method to add new grid blocks.
    """

    def __init__(self) -> None:
        self.n_grids = 0
        self.blockList = []

    def add_grid_block(self, block) -> None:
        """
            Method to add a new grid to the grid block list. The approach is
            the following. If the new block has a new ID it is added to the
            block list. If the new block has the same ID of a block already
            included by the block list but a new IOD, it replaces the previous
            block.
        """
        # find index of blocklist's block with same ID of the input block
        idx = [np.where(self.blockList[ii].id == block.id)
               for ii in range(len(self.blockList))]
        if np.size(idx) == 0:
            self.n_grids += 1
            self.blockList.append(block)
        else:
            if (self.blockList[idx].iod != block.iod):
                # Replace old block with the new one
                self.blockList[idx] = block
            else:
                pass


# =============================================================================
#                               DATA BLOCKS (DB)
# =============================================================================
# *************************************************************************** #
#                                                                             #
#                      SSRZ Rice-encoded integer value (ZDB001)               #
#                                                                             #
# *************************************************************************** #
class zdb001:
    def __init__(self, msg, unpacked_bits, p):
        # sign bit zdf303
        [s, unpacked_bits] = fields.zdf303(msg, unpacked_bits)
        # rice quotient zdf 304
        [q, unpacked_bits] = fields.zdf304(msg, unpacked_bits)
        # rice remainder zdf305
        [r, unpacked_bits] = fields.zdf305(msg, unpacked_bits, p)
        # value m = s(2**p * q + r)
        if np.isnan(r):
            self.m = np.nan   # mark the number as not valid
        elif ((s == -1) & ((2.0 ** p * q + r) == 0)):
            self.m = np.nan
        else:
            self.m = s * (2.0 ** p * q + r)
        self.unpacked_bits = unpacked_bits


# *************************************************************************** #
#                                                                             #
#                       SSRZ Rice Block (ZDB002)                              #
#                                                                             #
# *************************************************************************** #
class zdb002:
    def __init__(self, msg, unpacked_bits, n_p, p0):
        # scale factor indicator zdf301
        [scale_factor, unpacked_bits] = fields.zdf301(msg, unpacked_bits)
        # ssrz bin size indicator zdf302
        [b, unpacked_bits] = fields.zdf302(msg, unpacked_bits)
        # construction of the bin size parameter
        p = b + p0
        rice_val = []
        for ii in range(n_p):
            integer = zdb001(msg, unpacked_bits, p)
            unpacked_bits = integer.unpacked_bits
            rice_val = np.append(rice_val, integer.m)
        # Save self values
        self.a = scale_factor
        self.int = rice_val
        self.unpacked_bits = unpacked_bits


# *************************************************************************** #
#                                                                             #
#          SSRZ Compressed Satellite Parameter Block (ZDB003)                 #
#                                                                             #
# *************************************************************************** #
class zdb003:
    """ It is a sequence of SSRZ rice blocks (zdb002) containing encoded values
        (e.g. for all satellites) of one satellite-dependent parameter
        (e.g. satellite clock C0).
    """

    def __init__(self, msg, unpacked_bits, n_rice_blocks, rb_list, sat_array,
                 para_id, x0):
        # para_id is the indicator number of the parameter
        # considered in the block
        self.sat_p = []
        for ii in range(n_rice_blocks):
            # consider the metadata rice block
            md_rb = rb_list[ii]
            gnss_rb = md_rb.gnss
            p0 = md_rb.bins[para_id]
            n_para = 0
            for jj in range(len(sat_array)):
                sat_list = sat_array[jj]
                if len(sat_list) < 1:
                    continue
                gnss = sat_list[0][0]
                # check if the metadata for the corrections is in this rb
                if gnss not in gnss_rb:
                    continue
                n_para += len(sat_list)
            if n_para == 0:
                continue
            rice_block = zdb002(msg, unpacked_bits, n_para, p0)
            unpacked_bits = rice_block.unpacked_bits
            # integer value
            n = rice_block.int
            # resolution
            dx = x0 * 2 ** rice_block.a
            # save the values in the array
            for ii in range(n_rice_blocks):
                last_index_n = 0
                for jj in range(len(sat_array)):
                    self.sat_p.append([])
                    sat_list = sat_array[jj]
                    if len(sat_list) < 1:
                        continue
                    gnss = sat_list[0][0]
                    # check if the metadata for the corrections is in this rb
                    if gnss not in gnss_rb:
                        continue
                    self.sat_p[jj] = n[last_index_n:
                                       last_index_n + len(sat_list)] * dx
                    last_index_n += len(sat_list)

        self.unpacked_bits = unpacked_bits


# *************************************************************************** #
#                                                                             #
#                 SSRZ Compressed Signal bias Block (ZDB004)                  #
#                                                                             #
# *************************************************************************** #
class zdb004:
    """ It is a sequence of SSRZ rice blocks (zdb002) containing encoded values
        (e.g. for all satellites and signals).
    """

    def __init__(self, msg, unpacked_bits, n_rice_blocks, rb_list, sat_array,
                 x0, pb_signals, ref_signal=None):
        # para_id is the indicator number of the parameter
        # considered in the block
        self.sig_b = [[] for ii in range(len(sat_array))]
        for ii in range(n_rice_blocks):
            # consider the metadata rice block
            md_rb = rb_list[ii]
            p0 = md_rb.bin_size
            # number of parameters per rice block
            n_para = 0
            for jj in range(len(sat_array)):
                sat_list = sat_array[jj]
                if len(sat_list) < 1:
                    continue
                gnss = sat_list[0][0]
                # check if gnss is in the rice block
                if gnss not in md_rb.bit_mask_blk.gnss:
                    continue
                signals = md_rb.bit_mask_blk.signals[gnss]
                n_para += len(sat_list) * len(signals)
            if n_para == 0:
                continue
            rice_block = zdb002(msg, unpacked_bits, n_para, p0)
            # integer value
            n = rice_block.int
            # resolution
            dx = x0 * 2 ** rice_block.a
            unpacked_bits = rice_block.unpacked_bits
            last_index_n = 0
            for jj in range(len(sat_array)):
                sat_list = sat_array[jj]
                if len(sat_list) < 1:
                    continue
                gnss = sat_list[0][0]
                # list of pb signals
                try:
                    sig_list = pb_signals[gnss]
                except KeyError:
                    continue
                # reference signal
                try:
                    ref_sig = ref_signal[gnss][0]
                    # list of cb signals without reference
                    sig_list_cb = np.delete(sig_list,
                                            np.where(sig_list == ref_sig))
                except KeyError:
                    sig_list_cb = sig_list
                for ss in range(len(sig_list_cb)):
                    self.sig_b[jj].append([])
                # check if gnss is in the rice block
                if gnss not in md_rb.bit_mask_blk.gnss:
                    continue
                signals = md_rb.bit_mask_blk.signals[gnss]
                for sig in signals:
                    kk = np.where(sig_list_cb == sig)[0][0]
                    biases = n[last_index_n:last_index_n + len(sat_list)] * dx
                    self.sig_b[jj][kk] = biases
                    last_index_n += len(sat_list)

        self.unpacked_bits = unpacked_bits


# *************************************************************************** #
#                                                                             #
#                 SSRZ Compressed Signal bias Block for qix (ZDB004)          #
#                                                                             #
# *************************************************************************** #
class zdb004_qix:
    """ It is a sequence of SSRZ rice blocks (zdb002) containing encoded values
        (e.g. for all satellites and signals).
    """

    def __init__(self, msg, unpacked_bits, n_rice_blocks, rb_list, gnss_list,
                 sat_array, x0):
        # para_id is the indicator number of the parameter
        # considered in the block
        self.sig_b = [[] for ii in range(len(sat_array))]
        self.sig_list = [[] for ii in range(len(sat_array))]
        for ii in range(n_rice_blocks):
            # consider the metadata rice block
            md_rb = rb_list[ii]
            p0 = md_rb.bin_size
            # number of parameters per rice block
            n_para = 0
            for jj in range(len(sat_array)):
                gnss = gnss_list[jj]
                sat_list = sat_array[gnss]
                if len(sat_list) < 1:
                    continue
                # check if gnss is in the rice block
                if gnss not in md_rb.bit_mask_blk.gnss:
                    continue
                self.sig_list[jj].append([])
                signals = md_rb.bit_mask_blk.signals[gnss]
                self.sig_list[jj] = signals
                n_para += len(sat_list) * len(signals)
            if n_para == 0:
                continue
            rice_block = zdb002(msg, unpacked_bits, n_para, p0)
            # integer value
            n = rice_block.int
            # resolution
            dx = x0 * 2 ** rice_block.a
            unpacked_bits = rice_block.unpacked_bits
            last_index_n = 0
            for jj in range(len(sat_array)):
                gnss = gnss_list[jj]
                sat_list = sat_array[gnss]
                if len(sat_list) < 1:
                    continue
                gnss = sat_list[0][0]
                # check if gnss is in the rice block
                if gnss not in md_rb.bit_mask_blk.gnss:
                    continue
                signals = md_rb.bit_mask_blk.signals[gnss]
                # loop over satellites
                for kk in range(len(sat_list)):
                    self.sig_b[jj].append([])
                    biases = n[last_index_n:last_index_n + len(signals)] * dx
                    self.sig_b[jj][kk] = biases
                    last_index_n += len(signals)
        self.unpacked_bits = unpacked_bits


# *************************************************************************** #
#                                                                             #
#                  SSRZ Compressed Chain Data Block (ZDB005)                  #
#                                                                             #
# *************************************************************************** #
class zdb005:
    def __init__(self, msg, unpacked_bits, n_pts_chain, p0):
        # ssrz predictor flag zdf312
        [self.pred_flag, unpacked_bits] = fields.zdf315(msg, unpacked_bits)
        if self.pred_flag == 0:
            # there is only one rice block containing N_pts_chain encoded
            # values
            # ssrz rice block zdb002
            block = zdb002(msg, unpacked_bits, n_pts_chain, p0)
            self.blk_all = block
            self.blk_pred = np.nan
            unpacked_bits = block.unpacked_bits
        elif self.pred_flag == 1:
            # two rice blocks follow
            # not encoded with the gridded-data predictor
            block = zdb002(msg, unpacked_bits, 1, p0)
            unpacked_bits = block.unpacked_bits
            # encoded with the gridded-data predictor algorithm
            block_pred = zdb002(msg, unpacked_bits, n_pts_chain - 1, p0)
            self.blk_first = block
            self.blk_pred = block_pred
            unpacked_bits = block_pred.unpacked_bits
        self.unpacked_bits = unpacked_bits


# *************************************************************************** #
#                                                                             #
#                  SSRZ Compressed Gridded Data Block (ZDB006)                #
#                                                                             #
# *************************************************************************** #
class zdb006:
    def __init__(self, msg, unpacked_bits, chains, p0, x0):
        chain = []
        self.pred_flag = []
        for ii in range(len(chains)):
            n_pts_chain = chains[ii].n_pts
            # predictor list from metadata
            md_grid_pred = chains[ii].predictor_list
            block = zdb005(msg, unpacked_bits, n_pts_chain, p0)
            self.pred_flag.append(block.pred_flag)
            # possibly predictor values
            if block.pred_flag == 0:
                # resolution
                dx = x0 * 2.0 ** block.blk_all.a
                chain_block = block.blk_all.int * dx
            elif block.pred_flag == 1:
                # resolution
                dx1 = x0 * 2.0 ** block.blk_first.a
                a = block.blk_first.int[0] * dx1
                rice_values = np.array([a])
                chain_block = np.array([a])
                if n_pts_chain == 1:
                    break
                # apply predictor
                dx = x0 * 2.0 ** block.blk_pred.a
                rice_values = np.append(rice_values, block.blk_pred.int * dx)
                b = a + rice_values[0]
                chain_block = np.append(chain_block, b)
                c = a + rice_values[1]
                chain_block = np.append(chain_block, c)
                estimated_values = chain_block
                for jj in range(n_pts_chain - 3):
                    kk = jj + 3
                    ia = kk - (md_grid_pred[jj].da + 1)
                    ib = kk - (md_grid_pred[jj].db + 1)
                    ic = kk - (md_grid_pred[jj].dc + 1)
                    predicted_value = (estimated_values[ib] +
                                       estimated_values[ic] -
                                       estimated_values[ia])
                    if np.isnan(rice_values[kk]):
                        val = rice_values[kk]
                        estimated_values = np.append(estimated_values,
                                                     predicted_value)
                    else:
                        val = (rice_values[kk] + predicted_value)
                        estimated_values = np.append(estimated_values, val)
                    chain_block = np.append(chain_block, val)

            chain = np.append(chain, chain_block)
            unpacked_bits = block.unpacked_bits
        self.chain = chain
        self.unpacked_bits = unpacked_bits


# *************************************************************************** #
#                                                                             #
#                  SSRZ Compressed Coefficients Block (ZDB007)                #
#                                                                             #
# *************************************************************************** #
class zdb007:
    def __init__(self, msg, unpacked_bits, rb_list, n_para_max, x0):
        self.coeff = []
        n_para_tot = 0  # total number of parameters considering all rb
        for ii in range(len(rb_list)):
            rb = rb_list[ii]
            # order bit mask
            order_bit_mask = np.array(rb.order_bit_mask)
            # define number of parameters
            n_para_order = 0
            for pp in range(len(order_bit_mask)):
                if order_bit_mask[pp] == 1:
                    n_para_order += pp + 1
            if n_para_order + n_para_tot > n_para_max:
                n_para = n_para_max - n_para_tot
            else:
                n_para = n_para_order
            n_para_tot += n_para
            # p0
            p0 = rb.p0
            if n_para == 0:  # no rice block is transmitted
                continue
            rice_block = zdb002(msg, unpacked_bits, n_para, p0)
            unpacked_bits = rice_block.unpacked_bits
            # integer value
            n = rice_block.int
            # resolution
            dx = x0 * 2 ** rice_block.a
            self.coeff = np.append(self.coeff, n * dx)
        self.unpacked_bits = unpacked_bits


# *************************************************************************** #
#                                                                             #
#      SSRZ Compressed Satellite-dependent Coefficient Block (ZDB008)         #
#                                                                             #
# *************************************************************************** #
class zdb008:
    def __init__(self, msg, unpacked_bits, sat_array, n_rice_blocks, rb_list,
                 x0, n_coeff_max):
        # para_id is the indicator number of the parameter
        # considered in the block
        self.sat_p = []
        last_index_n = 0
        n_coeff_tot = 0  # cumulative tot number of coeff for the rb
        for ii in range(n_rice_blocks):
            # consider the metadata rice block for the coefficients
            md_rb = rb_list[ii]
            p0 = md_rb.p0
            # bit mask for the attributes (i.e. which m+l order is considered)
            attributes = np.array(md_rb.order_bit_mask)
            n_coeff = 0
            # calculation of the number of coefficients per rice block based
            # on the attribute related to the order
            for lm in range(len(attributes)):
                if attributes[lm] == 1:
                    if lm + 1 + n_coeff_tot > n_coeff_max:
                        n_coeff += n_coeff_max - n_coeff_tot
                    else:
                        n_coeff += lm + 1
            n_coeff_tot += n_coeff
            n_para = 0
            for jj in range(len(sat_array)):
                sat_list = sat_array[jj]
                if len(sat_list) < 1:
                    continue
                n_para += len(sat_list) * n_coeff
            if n_para == 0:  # no rice block is transmitted
                continue
            rice_block = zdb002(msg, unpacked_bits, n_para, p0)
            unpacked_bits = rice_block.unpacked_bits
            # integer value
            n = rice_block.int
            # resolution
            dx = x0 * 2 ** rice_block.a
            # save the values in the array
            for jj in range(n_coeff):
                self.sat_p.append([])
                for kk in range(len(sat_array)):
                    self.sat_p[jj].append([])
                    sat_list = sat_array[kk]
                    if len(sat_list) < 1:
                        continue
                    self.sat_p[jj][kk] = n[last_index_n:
                                           last_index_n + len(sat_list)] * dx
                    last_index_n += len(sat_list)

        self.unpacked_bits = unpacked_bits


# *************************************************************************** #
#                                                                             #
#                    SSRZ Compressed Parameter Block (ZDB009)                 #
#                                                                             #
# *************************************************************************** #
class zdb009:
    """
        It is a sequence of SSRZ rice blocks including encoded parameters.
        The included parameters are per SSRZ rice block and are given in
        ascending order of SSRZ parameter list bit mask (ZDF047), which is
        an attribute in the corresponding SSRZ compressed parameter data
        block definition (ZDB027).
    """

    def __init__(self, msg, unpacked_bits, rb_list, n_para_max, x0):
        self.coeff = []
        n_para_tot = 0  # total number of parameters considering all rb
        for ii in range(len(rb_list)):
            rb = rb_list[ii]
            # order bit mask
            bit_mask = np.array(rb.bit_mask)
            # define number of parameters
            n_para = 0
            n_para_tot = 0
            for pp in range(len(bit_mask)):
                if bit_mask[pp] == 1:
                    n_para += 1
            if n_para + n_para_tot > n_para_max:
                n_para = n_para_max - n_para_tot
            else:
                pass
            n_para_tot += n_para
            # p0
            p0 = rb.p0
            if n_para == 0:  # no rice block is transmitted
                continue
            rice_block = zdb002(msg, unpacked_bits, n_para, p0)
            unpacked_bits = rice_block.unpacked_bits
            # integer value
            n = rice_block.int
            # resolution
            dx = x0 * 2 ** rice_block.a
            self.coeff = np.append(self.coeff, n * dx)
        self.unpacked_bits = unpacked_bits

    def __repr__(self):
        strg = ('Class of compressed parameter with main objects: coeff ' +
                'and unpacked bits')
        return strg


# *************************************************************************** #
#                                                                             #
#                  SSRZ Compressed Chain Block (ZDB010)                       #
#                                                                             #
# *************************************************************************** #
class zdb010:
    def __init__(self, msg, unpacked_bits, zdf020, res, hgt_res):
        """ The SSRZ compressed chain block includes all parameters to
            reconstruct the N_pts_chain grid points per chain
        """
        # number of grid points per chain zdf095
        [zdf095, unpacked_bits] = fields.zdf095(msg, unpacked_bits)
        self.n_pts = zdf095 + 1
        # gid bin size parameter to decode the lat and long values
        # of the first chain point
        [self.p0, unpacked_bits] = fields.zdf096(msg, unpacked_bits)
        # riced encoded integer value zdb001
        # lat first chain point
        rice_int_lat0 = zdb001(msg, unpacked_bits, self.p0)
        lat0 = rice_int_lat0.m * res
        unpacked_bits = rice_int_lat0.unpacked_bits
        # lon first chain point
        rice_int_lon0 = zdb001(msg, unpacked_bits, self.p0)
        lon0 = rice_int_lon0.m * res
        # compute the first positioning vector of the grid
        r1 = np.array([lat0, lon0])
        unpacked_bits = rice_int_lon0.unpacked_bits
        # grid bin size to rice decode the following
        # 2 (n_pts_chain -1) parameters
        self.p_dl = -1  # set as default; although not used
        if self.n_pts > 1:
            [self.p_dl, unpacked_bits] = fields.zdf096(msg, unpacked_bits)
            ds = []
            ss = []
            for ii in range(2 * (self.n_pts - 1)):
                rice_int_val = zdb001(msg, unpacked_bits, self.p_dl)
                unpacked_bits = rice_int_val.unpacked_bits
                if ii == 0:
                    dlat = rice_int_val.m * res
                    lat1 = lat0 + dlat      # lat of 2nd of 1st baseline
                elif ii == 1:
                    dlon = rice_int_val.m * res
                    lon1 = lon0 + dlon      # lon of 2nd point of 1st baseline
                elif ((ii > 1) & np.mod(ii, 2) == 0):
                    ds = np.append(ds, rice_int_val.m * res)
                elif ((ii > 1) & np.mod(ii, 2) != 0):
                    ss = np.append(ss, rice_int_val.m * res)
        # compute the second positioning vector of the grid to calculate the
        # first baseline
            r2 = np.array([lat1, lon1])
            # use baseline flag zdf 097
            b_flag_list = []
            for ii in range(self.n_pts - 2):
                [b_flag, unpacked_bits] = fields.zdf097(msg, unpacked_bits)
                b_flag_list.append(b_flag)
            # point position flag zdf098
            p_flag_list = []
            for ii in range(self.n_pts - 2):
                [p_flag, unpacked_bits] = fields.zdf098(msg, unpacked_bits)
                p_flag_list.append(p_flag)
            # add baseline left flag zdf099
            l_flag_list = []
            for ii in range(self.n_pts - 2):
                [l_flag, unpacked_bits] = fields.zdf099(msg, unpacked_bits)
                l_flag_list.append(l_flag)
            # add baseline right flag zdf100
            r_flag_list = []
            for ii in range(self.n_pts - 2):
                [r_flag, unpacked_bits] = fields.zdf100(msg, unpacked_bits)
                r_flag_list.append(r_flag)
            # compute the complete lat, lon grid
            grid = MakeGrid(r1, r2, ss, ds, b_flag_list,
                            p_flag_list, l_flag_list, r_flag_list, res)
            self.lat = grid.lat
            self.lon = grid.lon
            # print(self.__str__)
        else:  # n_pts<=1
            self.lat = np.array([lat0])
            self.lon = np.array([lon0])
        if zdf020 == 3:
            # height flag zdf 101
            [hgt_flag, unpacked_bits] = fields.zdf101(msg, unpacked_bits)
            if hgt_flag == 1:
                # grid point height resolution
                [hgt_res, unpacked_bits] = fields.zdf102(msg, unpacked_bits)
        else:
            hgt_flag = None
        if (((zdf020 == 3) & (hgt_flag == 1)) |
           ((zdf020 == 4) & (hgt_res is not None))):
            # grid bin size parameter zdf096
            [p_hgt0, unpacked_bits] = fields.zdf096(msg, unpacked_bits)
            # rice encoded integer value absolute height for the first point
            rice_int_val = zdb001(msg, unpacked_bits, p_hgt0)
            unpacked_bits = rice_int_val.unpacked_bits
            hgt0 = rice_int_val.m * hgt_res
            height = np.array([hgt0])
            if (self.n_pts > 1):
                # bin size parameter for height differences zdf096
                [p_hgt, unpacked_bits] = fields.zdf096(msg, unpacked_bits)
                # rice encoded integer value height differences
                dh_list = []
                for ii in range(self.n_pts - 1):
                    rice_int_val = zdb001(msg, unpacked_bits, p_hgt)
                    unpacked_bits = rice_int_val.unpacked_bits
                    dh = rice_int_val.m * hgt_res
                    dh_list.append(dh)
                    hgt1 = hgt0 + dh
                    height = np.append(height, hgt1)
                    hgt0 = hgt1
        self.hgt = height
        # gridded data predictior points flag zdf103
        [self.grid_pred_p_flag,
         unpacked_bits] = fields.zdf103(msg, unpacked_bits)
        self.predictor_list = []
        if self.grid_pred_p_flag == 1:
            for ii in range(self.n_pts - 3):
                # ssrz predictor point indicator block
                predictor = zdb011(msg, unpacked_bits)
                self.predictor_list.append(predictor)
                unpacked_bits = predictor.unpacked_bits
        self.unpacked_bits = unpacked_bits

    def __str__(self):
        strg = ('     Number of grid points: ' + str(self.n_pts) + '\n' +
                '     Bin size parameter of the first chain point: ' +
                str(self.p0) + '\n' +
                '     Bin size parameter of following 2(n_pts_chain -1)' +
                ' parameters: ' +
                str(self.p_dl) + '\n' +
                '     ' + '{:>4}'.format('pt_id') +
                ' {:>10}'.format('lat [rad]') + ' {:>10}'.format('lon [rad]') +
                ' {:>10}'.format('hgt [m]') + '\n')
        for ii in range(self.n_pts):
            lat = '   {:>8.7f}'.format(self.lat[ii])
            lon = '   {:>8.7f}'.format(self.lon[ii])
            hgt = '   {:>7.3f}'.format(self.hgt[ii])
            strg += '    {:>4}'.format(ii) + '  ' + lat + lon + hgt + '\n'
        return strg

    def __repr__(self):
        strg = ('Class of compressed chain block with main objects: n_pts, ' +
                'lat, lon, hgt. Lat, lon, hgt are ellipsoidal coordinates ' +
                ' described as vectors with n_pts components')
        return strg


# *************************************************************************** #
#                                                                             #
#                  SSRZ Predictor Point indicator Block (ZDB011)              #
#                                                                             #
# *************************************************************************** #
class zdb011:
    def __init__(self, msg, unpacked_bits):
        # predictor point indicator for poin da
        [self.da, unpacked_bits] = fields.zdf104(msg, unpacked_bits)
        # predictor pint indicator for point db
        [self.db, unpacked_bits] = fields.zdf104(msg, unpacked_bits)
        # predictor pint indicator for point dc
        [self.dc, unpacked_bits] = fields.zdf104(msg, unpacked_bits)
        self.unpacked_bits = unpacked_bits


# *************************************************************************** #
#                                                                             #
#                  SSRZ Compressed Gridded Data Block (ZDB012)                #
#                                                                             #
# *************************************************************************** #
class zdb012:
    def __init__(self, msg, unpacked_bits, zdf020, dx, hgt_res):
        self.zdf020 = zdf020
        # grid id zdf105
        [self.id, unpacked_bits] = fields.zdf105(msg, unpacked_bits)
        # grid iod zdf106
        [self.iod, unpacked_bits] = fields.zdf106(msg, unpacked_bits)
        if zdf020 == 3:
            # order part of the grid point coordinate resolution zdf093
            [self.m, unpacked_bits] = fields.zdf092(msg, unpacked_bits)
            # integer part of the grid point coordinate resolution
            [self.n, unpacked_bits] = fields.zdf093(msg, unpacked_bits)
            # final resolution in km is:
            self.dx = self.n * 10 ** (-1 * (self.m + 2))
            self.hgt_res = None
        else:
            self.dx = dx
            self.hgt_res = hgt_res

        # number of chains per grid
        [zdf094, unpacked_bits] = fields.zdf094(msg, unpacked_bits)
        self.n_chain = zdf094 + 1

        self.chain_blk = []
        for ii in range(self.n_chain):
            # ssrz compressed chain block zdb010
            compressed_chain_blk = zdb010(msg, unpacked_bits, zdf020, self.dx,
                                          hgt_res)
            self.chain_blk.append(compressed_chain_blk)
            unpacked_bits = compressed_chain_blk.unpacked_bits

        self.unpacked_bits = unpacked_bits

    def __str__(self):
        strg = ('   Grid ID:  ' + str(self.id) + '\n' +
                '   Grid IOD: ' + str(self.iod) + '\n')
        if self.zdf020 == 3:
            strg += ('   Order part of the grid point coord res: ' +
                     str(self.m) +
                     '\n' +
                     '   Integer part of the grid point coord res: ' +
                     str(self.n) +
                     '\n')
        strg += '   Grid resolution [km]: ' + str(self.dx) + '\n'
        if self.zdf020 == 4:
            strg += '   Height resolution: ' + str(self.hgt_res)

        strg += '   Number of chains: ' + str(self.n_chain) + '\n'
        for ii in range(self.n_chain):
            strg += '    # Chain n.° ' + str(ii + 1) + ': \n'
            strg += self.chain_blk[ii].__str__()
        return strg

    def __repr__(self):
        strg = ('Class of compressed gridded data block with objects: ' +
                ' id, iod, n_chain, dx, hgt_res, chain_blk. Chain_blk is an ' +
                'array of compressed chain blocks')
        return strg


# *************************************************************************** #
#                                                                             #
#                  SSRZ Satellite Group Definition Block (ZDB017)             #
#                                                                             #
# *************************************************************************** #
class zdb017:
    def __init__(self, msg, unpacked_bits):
        # gnss id bit mask zdf012
        self.gnss = fields.zdf012(msg, unpacked_bits)[0]
        # update read bits
        unpacked_bits = fields.zdf012(msg, unpacked_bits)[1]
        # max sat id/prn per gnss and group zdf013
        self.max_sat_id = []
        for g in range(len(self.gnss)):
            self.max_sat_id = np.append(self.max_sat_id,
                                        fields.zdf013(msg, unpacked_bits)[0])
            # update read bits
            unpacked_bits = fields.zdf013(msg, unpacked_bits)[1]
        self.max_sat_id = self.max_sat_id.astype('int')
        # satellite group mode zdf014
        self.sat_group_mode = fields.zdf014(msg, unpacked_bits)[0]
        # update read bits
        unpacked_bits = fields.zdf014(msg, unpacked_bits)[1]
        if self.sat_group_mode == 0:
            self.sat_group = {}
            for g in range(len(self.gnss)):
                syst = self.gnss[g]
                self.sat_group[syst] = []
                for ii in range(self.max_sat_id[g]):
                    ii += 1  # to avoid prn = 0
                    # satellite group bit mask per gnss zdf015
                    bit_mask = fields.zdf015(msg, unpacked_bits)[0]
                    # update read bits
                    unpacked_bits = fields.zdf015(msg, unpacked_bits)[1]
                    if bit_mask == 0:
                        continue
                    else:
                        if ii < 10:
                            prn = '0' + str(ii)
                        else:
                            prn = str(ii)
                        self.sat_group[syst] = np.append(self.sat_group[syst],
                                                         syst + prn)
        elif self.sat_group_mode == 6:
            self.sat_group = {}
            for g in range(len(self.gnss)):
                syst = self.gnss[g]
                self.sat_group[syst] = []
                for ii in range(self.max_sat_id[g]):
                    ii += 1  # to avoid prn = 0
                    if ii < 10:
                        prn = '0' + str(ii)
                    else:
                        prn = str(ii)
                    self.sat_group[syst] = np.append(self.sat_group[syst],
                                                     syst + prn)
        # save read bits
        self.unpacked_bits = unpacked_bits


# *************************************************************************** #
#                                                                             #
#                       SSRZ Timing Block (ZDB018)                            #
#                                                                             #
# *************************************************************************** #
class zdb018:
    """ It includes length and offset of SSR update interval
    """

    def __init__(self, msg, unpacked_bits):
        # length of ssr update interval zdf053
        [self.ui, unpacked_bits] = fields.zdf053(msg, unpacked_bits)  # [s]
        # offset of ssr update interval zdf054
        [self.offset, self.unpacked_bits] = fields.zdf054(msg,
                                                          unpacked_bits)  # [s]

    def __str__(self):
        strg = ('    SSR Update Interval (UI) [s]:  ' + str(self.ui) + '\n' +
                '    Offset of the UI         [s]:  ' + str(self.offset) +
                '\n')
        return strg

    def __repr__(self):
        strg = ('Timing block class with objects: ui and offset.')
        return strg


# *************************************************************************** #
#                                                                             #
#             SSRZ Satellite dependent Timing Block (ZDB019)                  #
#                                                                             #
# *************************************************************************** #
class zdb019:
    """SSRZ satellite dependent timing block ZDB019: it includes
       the update and offset intervals of a satellite-dependent
       SSR parameter sets and in addition indicators of the
       related satellite groups.
    """

    def __init__(self, msg, unpacked_bits, n_g):
        # timing block zdb033
        # ssrz timing block zdb018
        self.ssrz_timing_block = zdb018(msg, unpacked_bits)
        # update unpacked bits
        unpacked_bits = self.ssrz_timing_block.unpacked_bits
        # ssrz satellite group list bit mask zdf 016
        [self.bit_mask, self.unpacked_bits] = fields.zdf016(msg,
                                                            unpacked_bits,
                                                            n_g)

    def __str__(self):
        strg = self.ssrz_timing_block.__str__()
        strg += ('    Satellite group list bit mask: ' +
                 str(self.bit_mask) + '\n')
        return strg

    def __repr__(self):
        strg = ('Satellite timing block with objects: ssrz_timing_block, ' +
                'bit_mask')
        return strg


# *************************************************************************** #
#                                                                             #
#                  SSRZ Signal Bit Mask Block (ZDB020)                       #
#                                                                             #
# *************************************************************************** #
class zdb020:
    """It is used to indicate signals of all systems
    """

    def __init__(self, msg, unpacked_bits):
        # SSRZ GNSS ID Bit mask zdf012
        [self.gnss, unpacked_bits] = fields.zdf012(msg, unpacked_bits)
        self.signals = {}
        for system in self.gnss:
            if system == 'n/a':
                continue
            else:
                [sig_list, unpacked_bits] = fields.zdf019(msg, unpacked_bits,
                                                          system)
                self.signals[system] = sig_list
        self.unpacked_bits = unpacked_bits

    def __str__(self):
        strg = '    Signal list: ' + '\n'
        for system in self.gnss:
            strg += '        ' + system + ':'
            if system == 'n/a':
                strg += 'n/a'
                continue
            for sig in self.signals[system]:
                if sig != self.signals[system][-1]:
                    strg += sig + ','
                else:
                    strg += sig
            strg += '\n'
        return strg

    def __repr__(self):
        strg = ('Signal block with objects: gnss, signals, unpacked_bits')
        return strg


# *************************************************************************** #
#                                                                             #
#               SSRZ Satellite Parameter Rice Block Definition (ZDB021)       #
#                                                                             #
# *************************************************************************** #
class zdb021:
    """ It includes the system bit mask as attribute and the default bin size
        parameter to define the rice block metadata
    """

    def __init__(self, msg, unpacked_bits, n_par):
        # SSRZ GNSS ID Bit mask zdf012
        [self.gnss, unpacked_bits] = fields.zdf012(msg, unpacked_bits)
        self.bins = []
        for ii in range(n_par):
            # Default bin size parameter zdf043
            [bin_size, unpacked_bits] = fields.zdf043(msg, unpacked_bits)
            self.bins = np.append(self.bins, bin_size)
        self.unpacked_bits = unpacked_bits

    def __str__(self):
        strg = ('  SSRZ Satellite Parameter Rice Block\n' +
                '     GNSS ID :' + self.gnss + '\n' +
                '     Default Bin Size per parameter ' + '\n')
        for ii in range(len(self.bins)):
            strg += ('       Default Bin Size (p0):' +
                     str(int(self.bins[ii])) + '\n')
        return strg

    def __repr__(self):
        strg = ('Satellite parameter rice block class with objects: ' +
                'gnss, bins, unpacked_bits')
        return strg


# *************************************************************************** #
#                                                                             #
#               SSRZ Signal Rice Block Definition (ZDB022)                    #
#                                                                             #
# *************************************************************************** #
class zdb022:
    """ It includes GNSS iD and signal ID bit masks as attributes and the
        Default Bin Size Parameter to define the Rice Block Metadata
    """

    def __init__(self, msg, unpacked_bits):
        # ssrz signal bit mask block
        self.bit_mask_blk = zdb020(msg, unpacked_bits)
        unpacked_bits = self.bit_mask_blk.unpacked_bits
        # default bin size parameter
        [self.bin_size, unpacked_bits] = fields.zdf046(msg, unpacked_bits)
        self.unpacked_bits = unpacked_bits

    def __str__(self):
        strg = self.bit_mask_blk.__str__()
        strg += ('      Default Bin Size Parameter (p0): ' +
                 str(self.bin_size) + '\n')
        return strg

    def __repr__(self):
        strg = ('Signal rice block definition class with objects: ' +
                'bit_mask_blk, bin_size, unpacked_bits')
        return strg


# *************************************************************************** #
#                                                                             #
#       SSRZ Compressed Coefficient Data Block Definition (ZDB023)            #
#                                                                             #
# *************************************************************************** #
class zdb023:
    """ It includes the default bin size parameter values and attributes
        required to define the rice blocks the corresponding ssrz compressed
        coefficient block.
    """

    def __init__(self, msg, unpacked_bits, nb_model):
        # number of rice blocks zdf042
        [self.n_rb, unpacked_bits] = fields.zdf042(msg, unpacked_bits)
        self.coeff_rb = []
        for bb in range(self.n_rb):
            # ssrz coefficient rice block definition zdb024
            crb = zdb024(msg, unpacked_bits, nb_model)
            self.coeff_rb = np.append(self.coeff_rb, crb)
            unpacked_bits = crb.unpacked_bits
        self.unpacked_bits = unpacked_bits

    def __repr__(self):
        strg = ('Compressed coefficients block class with objects: ' +
                'n_rb, coeff_rb, unpacked_bits')
        return strg


# *************************************************************************** #
#                                                                             #
#               SSRZ Coefficient Rice Block Definition (ZDB024)               #
#                                                                             #
# *************************************************************************** #
class zdb024:
    """ It includes the coefficient order bit mask and default bin size
        parameter values
    """

    def __init__(self, msg, unpacked_bits, nb_model):
        # coefficient order bit mask zdf116
        [self.order_bit_mask, unpacked_bits] = fields.zdf116(msg,
                                                             unpacked_bits,
                                                             nb_model)
        # default bin size parameter
        [self.p0, unpacked_bits] = fields.zdf043(msg, unpacked_bits)
        self.unpacked_bits = unpacked_bits

    def __repr__(self):
        strg = ('Coefficient rice block class with objects: ' +
                'order_bit_mask, p0, unpacked_bits')
        return strg


# *************************************************************************** #
#                                                                             #
#            SSRZ Global VTEC Coeff Rice Block Definition (ZDB026)            #
#                                                                             #
# *************************************************************************** #
class zdb026:
    """ Global VTEC coefficient Rice Block Definition
    """

    def __init__(self, msg, unpacked_bits, nb_model):
        # default bin size parameter
        [self.p0, unpacked_bits] = fields.zdf043(msg, unpacked_bits)
        # parameter list bit mask zdf116
        [self.par_bit_mask, unpacked_bits] = fields.zdf116(msg,
                                                           unpacked_bits,
                                                           nb_model)
        self.unpacked_bits = unpacked_bits

    def __repr__(self):
        strg = ('Coefficient rice block class with objects: ' +
                'par_bit_mask, p0, unpacked_bits')
        return strg


# *************************************************************************** #
#                                                                             #
#          SSRZ Compressed Parameter Data Block Definition (ZDB027)           #
#                                                                             #
# *************************************************************************** #
class zdb027:
    """ It includes the attribute SSRZ parameter list bit mask and the default
        bin size parameter value of the Rice Block definitions.
    """

    def __init__(self, msg, unpacked_bits, n_para):
        # number of rice blocks zdf042
        [self.n_rb, unpacked_bits] = fields.zdf042(msg, unpacked_bits)
        self.coeff_rb = []
        for rb in range(self.n_rb):
            blk = zdb028(msg, unpacked_bits, n_para)
            self.coeff_rb.append(blk)
            unpacked_bits = blk.unpacked_bits
        self.unpacked_bits = unpacked_bits

    def __repr__(self):
        strg = ('Compressed parameter data block class with objects: ' +
                'n_rb, coeff_rb, unpacked_bits')
        return strg


# *************************************************************************** #
#                                                                             #
#                   SSRZ Parameter Rice Block Definition (ZDB028)             #
#                                                                             #
# *************************************************************************** #
class zdb028:
    """ It includes the Parameter Rice Block definitions.
    """

    def __init__(self, msg, unpacked_bits, n_para):
        # parameter list bit mask
        [self.bit_mask, unpacked_bits] = fields.zdf047(msg, unpacked_bits,
                                                       n_para)
        # default bin size parameter
        [self.p0, unpacked_bits] = fields.zdf043(msg, unpacked_bits)
        self.unpacked_bits = unpacked_bits

    def __repr__(self):
        strg = ('Parameter Rice block class with objects: ' +
                'bit_mask, p0, unpacked_bits')
        return strg


# *************************************************************************** #
#                                                                             #
#                  SSRZ BE IOD Definition Block (ZDB030)                      #
#                                                                             #
# *************************************************************************** #
class zdb030:
    """The SSRZ BE IOD Definition Block includes information how to read and
       process the SSRZ BE IOD per GNSS
    """

    def __init__(self, msg, unpacked_bits, zdf020):
        if zdf020 == 1:
            # gnss id bit mask zdf012
            [self.gnss, unpacked_bits] = fields.zdf012(msg, unpacked_bits)

            # length of the SSRZ BE IOD field zdf199. It is the number of bits
            # nb_iod_gnss used to represent IOD values in ZDF033.
            # It repeats for each system in self.gnss
            self.nb_iod_gnss = []
            for ii in range(len(self.gnss)):
                [iod, unpacked_bits] = fields.zdf199(msg, unpacked_bits)
                self.nb_iod_gnss.append(iod)

            # ssrz iod tag zdf200
            [self.iod_tag, unpacked_bits] = fields.zdf200(msg, unpacked_bits)

        elif zdf020 == 2:
            self.nb_iod_gnss = []
            # ssrz be iod tag gps zdf201
            [self.be_iod_tag_g, unpacked_bits] = fields.zdf201(msg,
                                                               unpacked_bits)
            self.nb_iod_gnss.append(self.be_iod_tag_g)
            # ssrz be iod tag glonass zdf202
            [self.be_iod_tag_r, unpacked_bits] = fields.zdf202(msg,
                                                               unpacked_bits)
            self.nb_iod_gnss.append(self.be_iod_tag_r)
            # ssrz be iod tag galileo zdf203
            [self.be_iod_tag_e, unpacked_bits] = fields.zdf203(msg,
                                                               unpacked_bits)
            self.nb_iod_gnss.append(self.be_iod_tag_e)
            # ssrz be iod tag qzss zdf204
            [self.be_iod_tag_j, unpacked_bits] = fields.zdf204(msg,
                                                               unpacked_bits)
            self.nb_iod_gnss.append(self.be_iod_tag_j)
            # ssrz be iod tag sbas zdf205
            [self.be_iod_tag_s, unpacked_bits] = fields.zdf205(msg,
                                                               unpacked_bits)
            self.nb_iod_gnss.append(self.be_iod_tag_s)
            # ssrz be iod tag beidou zdf206
            [self.be_iod_tag_c, unpacked_bits] = fields.zdf206(msg,
                                                               unpacked_bits)
            self.nb_iod_gnss.append(self.be_iod_tag_c)
            # ssrz be iod tag irnss zdf207
            [self.be_iod_tag_i, unpacked_bits] = fields.zdf207(msg,
                                                               unpacked_bits)
            self.nb_iod_gnss.append(self.be_iod_tag_i)
        self.unpacked_bits = unpacked_bits

    def __repr__(self):
        strg = ('SSRZ BE IOD block class with main objects: ' +
                'gnss, nb_iod_gnss, iod_tag, unpacked_bits')
        return strg


# *************************************************************************** #
#                                                                             #
#                   SSRZ Phase Bias Signal Block (ZDB031)                     #
#                                                                             #
# *************************************************************************** #
class zdb031:
    """ SSRZ Phase signal block
    """

    def __init__(self, msg, unpacked_bits, n_pb_bits):
        # phase bias flag zdf310
        [self.pb_flag, unpacked_bits] = fields.zdf310(msg, unpacked_bits)
        # combined continuity overflow indicator
        if self.pb_flag == 1:
            # phase bias value indicator
            [self.pb_value, unpacked_bits] = fields.zdf311(msg, unpacked_bits,
                                                           n_pb_bits)
            [self.continuity, unpacked_bits] = fields.zdf312(msg,
                                                             unpacked_bits)
        else:
            self.pb_value = 0
            self.continuity = 0
        self.unpacked_bits = unpacked_bits

    def __repr__(self):
        strg = ('Phase signal block class with objects: ' +
                'pb_flag, pb_value, continuity, unpacked_bits')
        return strg


# *************************************************************************** #
#                                                                             #
#                         SSRZ Phase Bias Block (ZDB032)                      #
#                                                                             #
# *************************************************************************** #
class zdb032:
    """ It is a sequence of SSRZ phase bias signal blocks (zdb031) for all
        transmitted satellites and defined signals per satellite.
    """

    def __init__(self, msg, unpacked_bits, sat_array, res, n_bits, signals):
        self.pb = []
        self.conti = []
        for ii in range(len(sat_array)):
            self.pb.append([])
            self.conti.append([])

            sat_list = sat_array[ii]
            if len(sat_list) < 1:
                continue
            else:
                gnss = sat_list[0][0]
            for jj in range(len(sat_list)):
                self.pb[ii].append([])
                self.conti[ii].append([])
                try:
                    sig_list = signals[gnss]
                except:
                    KeyError
                    continue
                for kk in range(len(sig_list)):
                    phase_signal_block = zdb031(msg, unpacked_bits, n_bits)
                    unpacked_bits = phase_signal_block.unpacked_bits
                    self.pb[ii][jj].append(phase_signal_block.pb_value * res)
                    self.conti[ii][jj].append(phase_signal_block.continuity)
        # Sav bits
        self.unpacked_bits = unpacked_bits

    def __repr__(self):
        strg = ('Phase bias block class with objects: ' +
                'pb, unpacked_bits')
        return strg


# *************************************************************************** #
#                                                                             #
#                       SSRZ Update and Offset Block (ZDB040)                 #
#                                                                             #
# *************************************************************************** #
class zdb040:
    """ It includes the length and the offset of an SSR update interval and it
        is an improvement of the SSRZ Timing Block (ZDB018).
    """

    def __init__(self, msg, unpacked_bits):
        # Length update interval
        [self.t_upd, unpacked_bits] = fields.zdf056(msg, unpacked_bits)
        # Update interval offset
        [self.t_off, unpacked_bits] = fields.zdf057(msg, unpacked_bits)
        # Save bits
        self.unpacked_bits = unpacked_bits

    def __repr__(self):
        strg = ('Update and offset block class with objects: ' +
                't_upd, t_off, unpacked_bits')
        return strg


# *************************************************************************** #
#                                                                             #
#            SSRZ Grid-related Update and Offset Block (ZDB043)               #
#                                                                             #
# *************************************************************************** #
class zdb043:
    """ It indicates those grids that are associated with an
        Update and Offset Block (ZDB040).
    """

    def __init__(self, msg, unpacked_bits, max_gr_id):
        # Update and offset blcok
        self.upd_off_blk = zdb040(msg, unpacked_bits)
        unpacked_bits = self.upd_off_blk.unpacked_bits
        # SSSRZ region ID bit mask
        [self.gr_id_mask, unpacked_bits] = fields.zdf026(msg, unpacked_bits,
                                                         max_gr_id)
        # Save bits
        self.unpacked_bits = unpacked_bits

    def __repr__(self):
        strg = ('Region-related update and offset block class with objects: ' +
                'upd_off_blk, gr_id_mask, unpacked_bits')
        return strg


# *************************************************************************** #
#                                                                             #
#                   SSRZ Grid-related Timing Parameters (ZDB044)              #
#                                                                             #
# *************************************************************************** #
class zdb044:
    """ It indicates those SSRZ Grid IDs that are associated with specific
        Update and Offset Blocks.
    """

    def __init__(self, msg, unpacked_bits):
        # Maximum grid-related timing parameters M_grid_ID
        [self.max_gr_id, unpacked_bits] = fields.zdf025(msg, unpacked_bits)
        # Number of grid-related update and offset blocks
        [self.n_tmg, unpacked_bits] = fields.zdf058(msg, unpacked_bits)
        # Region-related update and offset blocks
        self.blk_list = []
        for ii in range(self.n_tmg):
            blk = zdb043(msg, unpacked_bits, self.max_gr_id)
            self.blk_list.append(blk)
            unpacked_bits = blk.unpacked_bits
        # Save bits
        self.unpacked_bits = unpacked_bits

    def __repr__(self):
        strg = ('Grid-related timing parameters class with objects: ' +
                'max_gr_id, n_tmg, blk_list, unpacked_bits')
        return strg


# *************************************************************************** #
#                                                                             #
#            SSRZ Region-related Update and Offset Block (ZDB045)             #
#                                                                             #
# *************************************************************************** #
class zdb045:
    """ It indicates those regions that are associated with an
        Update and Offset Block (ZDB040).
    """

    def __init__(self, msg, unpacked_bits, max_r_id):
        # Update and offset blcok
        self.upd_off_blk = zdb040(msg, unpacked_bits)
        unpacked_bits = self.upd_off_blk.unpacked_bits
        # SSSRZ region ID bit mask
        [self.r_id_mask, unpacked_bits] = fields.zdf028(msg, unpacked_bits,
                                                        max_r_id)
        # Save bits
        self.unpacked_bits = unpacked_bits

    def __repr__(self):
        strg = ('Region-related update and offset block class with objects: ' +
                'upd_off_blk, r_id_mask, unpacked_bits')
        return strg


# *************************************************************************** #
#                                                                             #
#                  SSRZ Region-related Timing Parameters (ZDB046)             #
#                                                                             #
# *************************************************************************** #
class zdb046:
    """ It indicates those SSRZ Region IDs that are associated with specific
        Update and Offset Blocks.
    """

    def __init__(self, msg, unpacked_bits):
        # Maximum region-related timing parameters M_regionID
        [self.max_r_id, unpacked_bits] = fields.zdf027(msg, unpacked_bits)
        # Number of region-related update and offset blocks
        [self.n_tmg, unpacked_bits] = fields.zdf058(msg, unpacked_bits)
        # Region-related update and offset blocks
        self.blk_list = []
        for ii in range(self.n_tmg):
            blk = zdb045(msg, unpacked_bits, self.max_r_id)
            self.blk_list.append(blk)
            unpacked_bits = blk.unpacked_bits
        # Save bits
        self.unpacked_bits = unpacked_bits

    def __repr__(self):
        strg = ('Region-related timing parameters class with objects: ' +
                'max__id, n_tmg, blk_list, unpacked_bits')
        return strg


# *************************************************************************** #
#                                                                             #
# SSRZ Basic Troposphere Component Coefficient Data Block Definition (ZDB050) #
#                                                                             #
# *************************************************************************** #
class zdb050:
    """ It includes the basic tropo component coefficient.
    """

    def __init__(self, msg, unpacked_bits, zdf107_bit0=False):
        # Max horizontal order of regional troposphere component c
        [self.m_hor, unpacked_bits] = fields.zdf166(msg, unpacked_bits)
        # Max vertical order of regional troposphere component c
        [self.m_hgt, unpacked_bits] = fields.zdf167(msg, unpacked_bits)
        # Resolution of regional tropo coefficients of component c
        [self.res, unpacked_bits] = fields.zdf085(msg, unpacked_bits)
        # Number of coefficients
        # The number of coefficients is given by the n coeff of triangle
        # configuration times the max order of height + 1
        # These coefficients can be spread among multiple rice blocks
        # depending on the following zdf108 flag.
        n_coeff_triangle = compute_ncoeff_from_order_triangle(self.m_hor)
        self.n_coeff = (self.m_hgt + 1) * n_coeff_triangle
        # Compressed coefficients block definition of regional tropo component
        self.blk = zdb023(msg, unpacked_bits, self.m_hor + 1)
        unpacked_bits = self.blk.unpacked_bits
        # Separated compressed coeff block per height
        if zdf107_bit0:
            # This flag indicates if different SSRZ compressed coeff blocks
            # (all identically defined by the subsequent zdb023) are used
            # for coefficients related to different height orders
            [self.sep_hgt, unpacked_bits] = fields.zdf108(msg, unpacked_bits)
        # Save bits
        self.unpacked_bits = unpacked_bits

    def __repr__(self):
        strg = ('Basic tropo component coefficient class with objects: ' +
                'm_hor, m_hgt, res, n_coeff, blk, sep_hgt, unpacked_bits')
        return strg


# *************************************************************************** #
#                                                                             #
# SSRZ Tropo Mapping Function Improvement Para Data Block Definition (ZDB051) #
#                                                                             #
# *************************************************************************** #
class zdb051:
    """ It includes mapping improvement parameter data block.
    """

    def __init__(self, msg, unpacked_bits):
        # Max horizontal order of mapping function improvements
        [self.m_hor, unpacked_bits] = fields.zdf166(msg, unpacked_bits)
        # Max vertical order of mapping function improvements
        [self.m_hgt, unpacked_bits] = fields.zdf167(msg, unpacked_bits)
        # Maximum elevation for mapping improvement
        [self.max_el, unpacked_bits] = fields.zdf171(msg, unpacked_bits)
        # Resolution of mapping function improvements
        [self.res, unpacked_bits] = fields.zdf086(msg, unpacked_bits)
        # Number of coefficients
        self.n_coeff = (self.m_hor + 1) * (self.m_hgt + 1)
        # Compressed coefficients block definition of regional tropo component
        self.blk = zdb027(msg, unpacked_bits, self.n_coeff)
        # Save bits
        self.unpacked_bits = self.blk.unpacked_bits

    def __repr__(self):
        strg = ('Basic tropo component coefficient class with objects: ' +
                'm_hor, m_hgt, res, max_el, n_coeff, blk, unpacked_bits')
        return strg


# *************************************************************************** #
#                                                                             #
#                  SSRZ Model Parameters Block (ZDB060)                       #
#                                                                             #
# *************************************************************************** #
class zdb060:
    """ It includes the model parameters of the atmospheric corrections
    """

    def __init__(self, msg, unpacked_bits):
        # model id zdf110
        [self.id, unpacked_bits] = fields.zdf110(msg, unpacked_bits)
        # model version zdf 111
        [self.v, unpacked_bits] = fields.zdf111(msg, unpacked_bits)
        # number of integer model parameters zdf112
        [self.n_int_mp, unpacked_bits] = fields.zdf112(msg, unpacked_bits)
        # integer parameters
        int_mp = []
        for ii in range(self.n_int_mp):
            # integer model parameter zdf114
            [value, unpacked_bits] = fields.zdf114(msg, unpacked_bits)
            int_mp.append(str(value))
        # number of float model parameters zdf113
        [self.n_flt_mp, unpacked_bits] = fields.zdf113(msg, unpacked_bits)
        # float parameters
        flt_mp = []
        for ii in range(self.n_flt_mp):
            # integer model parameter zdf115
            [value, unpacked_bits] = fields.zdf115(msg, unpacked_bits)
            flt_mp.append(str(value))
        self.int_mp = np.array(int_mp)
        self.flt_mp = np.array(flt_mp)
        self.unpacked_bits = unpacked_bits

    def __str__(self):
        strg = ('  SSRZ Model Parameters Block\n' +
                '      model    : ' + str(int(self.id)) + '\n' +
                '      version  : ' + str(int(self.v)) + '\n' +
                '      n int p  : ' + str(int(self.n_int_mp)) + '\n' +
                '      n float p: ' + str(int(self.n_flt_mp)) + '\n')
        if self.n_int_mp > 0:
            strg += '      Integer parameters: \n'
            for ii in range(self.n_int_mp):
                strg += '      Int param  : ' + self.int_mp[ii] + '\n'
        if self.n_flt_mp > 0:
            strg += '      Float   parameters: \n'
            for ii in range(self.n_flt_mp):
                strg += '     Float param: ' + self.flt_mp[ii] + '\n'
        return strg

    def __repr__(self):
        strg = ('Model block class with objects: ' +
                'id, v, n_int_mp, n_flt_mp, int_mp, flt_mp, unpacked_bits')
        return strg


# *************************************************************************** #
#                                                                             #
#                   SSRZ VTEC Ionosphere Layer Block (ZDB061)                 #
#                                                                             #
# *************************************************************************** #
class zdb061:
    """ Global VTEC iono layer block
    """

    def __init__(self, msg, unpacked_bits):
        # Height of ionospheric layer
        [self.hgt, unpacked_bits] = fields.zdf131(msg, unpacked_bits)
        # Spherical harmonics degree
        [self.sph_deg, unpacked_bits] = fields.zdf132(msg, unpacked_bits)
        # Spherical harmonics order
        [self.sph_ord, unpacked_bits] = fields.zdf133(msg, unpacked_bits)
        self.unpacked_bits = unpacked_bits

    def __str__(self):
        strg = "** VTEC Iono layer block\n"
        strg += f"{'':3}" + f"{'Iono layer height:':25}" + f"{self.hgt:8.3f}"
        strg += f"{'':3}" + f"{'Sph armonic deg:':25}" + f"{self.sph_deg:2d}"
        strg += f"{'':3}" + f"{'Sph armonic ord:':25}" + f"{self.sph_ord:2d}"
        return strg


# *************************************************************************** #
#                                                                             #
#             SSRZ Rice-encoded integer value for global VTEC                 #
#                                                                             #
# *************************************************************************** #
class zdb062:
    def __init__(self, msg, unpacked_bits, p):
        # rice quotient zdf 304
        [q, unpacked_bits] = fields.zdf304(msg, unpacked_bits)
        # sign bit zdf303
        [s, unpacked_bits] = fields.zdf303(msg, unpacked_bits)
        # rice remainder zdf305
        [r, unpacked_bits] = fields.zdf305(msg, unpacked_bits, p)
        # value m = s(2**p * q + r)
        if np.isnan(r):
            self.m = np.nan   # mark the number as not valid
        elif ((s == -1) & ((2.0 ** p * q + r) == 0)):
            self.m = np.nan
        else:
            self.m = s * (2.0 ** p * q + r)
        self.unpacked_bits = unpacked_bits


# =============================================================================
#                        METADATA MESSAGE BLOCKS (MB)
# =============================================================================
# *************************************************************************** #
#                                                                             #
#                 SSRZ High Rate Metadata Message Block (ZMB001)              #
#                                                                             #
# *************************************************************************** #
class zmb001:
    """SSRZ High Rate Metadata Message Block  ZMB001.
       It defines the metadata required for decoding
       the SSRZ High Rate message (ZM001).
    """

    def __init__(self, msg, unpacked_bits, n_g_hr, zdf020):
        # number Number of SSRZ High Rate Satellite Groups
        [self.n_g_hr, unpacked_bits] = fields.zdf011(msg, unpacked_bits,
                                                     zdf020)
        if zdf020 == 1:
            self.n_g_hr += 1
        # number of satellite-dependent timing blocks zdf055
        [self.n_timing, unpacked_bits] = fields.zdf055(msg, unpacked_bits)
        self.hr_timing_block = []
        for ii in range(self.n_timing):
            # ssrz high rate timing block zdb019
            self.hr_timing_block = np.append(self.hr_timing_block,
                                             zdb019(msg, unpacked_bits,
                                                    self.n_g_hr))
            # update the unpacked bits
            unpacked_bits = self.hr_timing_block[ii].unpacked_bits
        # high rate clock metadata ********************************************
        # high rate clock correction parameters zdf044
        [self.clk_p, unpacked_bits] = fields.zdf044(msg, unpacked_bits)
        # high rate clock resolution zdf060
        [self.clk_res, unpacked_bits] = fields.zdf060(msg, unpacked_bits)
        # number of high rate clock rice blocks zdf042
        [self.n_rb_clk, unpacked_bits] = fields.zdf042(msg, unpacked_bits)
        self.rb_clk = []
        for bb in range(self.n_rb_clk):
            # rice block definition zdb021
            self.rb_clk = np.append(self.rb_clk,
                                    zdb021(msg, unpacked_bits, self.clk_p))
            unpacked_bits = self.rb_clk[bb].unpacked_bits
        # SSRZ high rate orbit metadata ***************************************
        # number of orbit parameters zdf044
        [self.orb_p, unpacked_bits] = fields.zdf044(msg, unpacked_bits)
        if self.orb_p == 1:
            # ssrz high rate orbit default resolution
            [self.orb_res, unpacked_bits] = fields.zdf061(msg, unpacked_bits)
            # number of high rate orbit rice blocks zdf042
            [self.n_rb_orb, unpacked_bits] = fields.zdf042(msg, unpacked_bits)
            # satellite parameter rice block zdb021
            self.rb_orb = []
            for bb in range(self.n_rb_orb):
                self.rb_orb = np.append(self.rb_orb,
                                        zdb021(msg, unpacked_bits,
                                               self.orb_p))
                unpacked_bits = self.rb_orb[bb].unpacked_bits
        # save read bits
        self.unpacked_bits = unpacked_bits

    def __str__(self):
        strg = ' #** SSRZ High Rate Metadata Message Block' + '\n'
        strg += ('  Number of satellite-dependent timing blocks :' +
                 str(self.n_timing) + '\n')
        for tt in range(self.n_timing):
            strg += ('  #- SSRZ timing block ' + str(tt + 1) + '\n' +
                     self.hr_timing_block[tt].__str__())
        strg += ('  #* SSRZ High Rate Clock Metadata \n' +
                 '    Number of HR clock parameters :' + str(self.clk_p) +
                 '\n' +
                 '    HR clock resolution [m]: ' + str(self.clk_res) +
                 '\n' +
                 '    Number of HR clock rice blocks: ' +
                 str(self.n_rb_clk) + '\n')
        for rb in range(self.n_rb_clk):
            strg += '  ' + self.rb_clk[rb].__str__()
        strg += ('  #* SSRZ High Rate Orbit Metadata \n' +
                 '    Number of HR orbit parameters :' + str(self.orb_p) +
                 '\n')
        if self.orb_p == 1:
            strg += ('    HR orbit resolution [m]: ' + str(self.orb_res) +
                     '\n' +
                     '    Number of HR orbit rice blocks: ' +
                     str(self.n_rb_orb) + '\n')
            for rb in range(self.n_rb_orb):
                strg += '  ' + self.rb_orb[rb].__str__()
        return strg

    def __repr__(self):
        strg = ('Hight rate block class with main objects: ' +
                'orb_p, n_orb_p, n_rb_clk, n_timing, hr_timing_block, ' +
                'rb_clk, orb_res, rb_orb, n_rb_orb, clk_p, unpacked_bits')
        return strg


# *************************************************************************** #
#                                                                             #
#                 SSRZ Low Rate Metadata Message Block (ZMB002)               #
#                                                                             #
# *************************************************************************** #
class zmb002:
    def __init__(self, msg, unpacked_bits, n_g_lr, zdf020=None):
        self.zdf020 = zdf020
        # number Number of SSRZ Low Rate Satellite Groups
        [self.n_g_lr, unpacked_bits] = fields.zdf010(msg, unpacked_bits,
                                                     zdf020)
        if zdf020 == 1:
            self.n_g_lr += 1
        # number of satellite-dependent timing blocks zdf055
        [self.n_timing, unpacked_bits] = fields.zdf055(msg, unpacked_bits)
        self.lr_timing_block = []
        for ii in range(self.n_timing):
            # ssrz high rate timing block zdb019
            self.lr_timing_block = np.append(self.lr_timing_block,
                                             zdb019(msg, unpacked_bits,
                                                    n_g_lr))
            # update read bits
            unpacked_bits = self.lr_timing_block[ii].unpacked_bits
        # ssrz iod definition block zdb030
        self.iod_def_block = zdb030(msg, unpacked_bits, zdf020)
        unpacked_bits = self.iod_def_block.unpacked_bits
        # number of ssrz low rate clock metadata
        [self.n_lr_clk, unpacked_bits] = fields.zdf044(msg, unpacked_bits)
        if self.n_lr_clk == 1:
            # only constant c0 parameter is transmitted
            # low rate clock c0 resolution
            [self.c0_res, unpacked_bits] = fields.zdf062(msg, unpacked_bits)
        else:
            # constant term and linear term are transmitted
            # low rate clock c0 resolution
            [self.c0_res, unpacked_bits] = fields.zdf062(msg, unpacked_bits)
            # low rate clock c1 resolution
            [self.c1_res, unpacked_bits] = fields.zdf063(msg, unpacked_bits)
        # number of low rate clock rice blocks
        [self.n_rb_clk, unpacked_bits] = fields.zdf042(msg, unpacked_bits)
        self.rb_clk = []
        for bb in range(self.n_rb_clk):
            # rice block definition zdb021
            self.rb_clk = np.append(self.rb_clk,
                                    zdb021(msg, unpacked_bits, self.n_lr_clk))
            unpacked_bits = self.rb_clk[bb].unpacked_bits
        # =====================================================================
        #                    SSRZ low rate orbit metadata
        # =====================================================================
        # number of orbit parameters zdf044
        [self.orb_p, unpacked_bits] = fields.zdf044(msg, unpacked_bits)
        # the number of orbit components should be always 3 in this case
        if self.orb_p == 3:
            # ssrz low rate radial orbit default resolution
            [self.rad_res, unpacked_bits] = fields.zdf064(msg, unpacked_bits)
            # ssrz low rate along track orbit default resolution
            [self.atr_res, unpacked_bits] = fields.zdf065(msg, unpacked_bits)
            # ssrz low rate cross track orbit default resolution
            [self.ctr_res, unpacked_bits] = fields.zdf066(msg, unpacked_bits)
        else:
            print('Warning: something wrong in decoding metadata' +
                  ' low rate message')

        # number of low rate orbit rice blocks zdf042
        [self.n_rb_orb, unpacked_bits] = fields.zdf042(msg, unpacked_bits)
        # satellite parameter rice block zdb021
        self.rb_orb = []
        for bb in range(self.n_rb_orb):
            self.rb_orb = np.append(self.rb_orb,
                                    zdb021(msg, unpacked_bits,
                                           self.orb_p))
            unpacked_bits = self.rb_orb[bb].unpacked_bits
        # number of velocity components
        [self.n_vel, unpacked_bits] = fields.zdf044(msg, unpacked_bits)
        if self.n_vel == 3:
            # low rate radial velocity default resolution zdf067
            [self.rad_vel_res, unpacked_bits] = fields.zdf067(msg,
                                                              unpacked_bits)
            # low rate along-track velocity default resolution zdf068
            [self.atr_vel_res, unpacked_bits] = fields.zdf068(msg,
                                                              unpacked_bits)
            # low rate cross-track velocity default resolution zdf069
            [self.ctr_vel_res, unpacked_bits] = fields.zdf069(msg,
                                                              unpacked_bits)
        elif self.n_vel > 0:
            # number of low rate velocity rice blocks zdf042
            [self.n_rb_vel, unpacked_bits] = fields.zdf042(msg,
                                                           unpacked_bits)

            self.rb_vel = []
            # metadata satellite parameter rice block definition
            for bb in range(self.n_rb_vel):
                self.rb_vel = np.append(self.rb_vel,
                                        zdb021(msg, unpacked_bits,
                                               self.n_vel))
                unpacked_bits = self.rb_vel[bb].unpacked_bits
        # =====================================================================
        #                         Low Rate code bias
        # =====================================================================
        # code bias default resolution ZDF070
        [self.cb_res, unpacked_bits] = fields.zdf070(msg, unpacked_bits)
        # number of code bias rice blocks ZDF042
        [self.n_rb_cb, unpacked_bits] = fields.zdf045(msg, unpacked_bits)
        self.rb_cb = []
        for bb in range(self.n_rb_cb):
            rice_block = zdb022(msg, unpacked_bits)
            self.rb_cb = np.append(self.rb_cb, rice_block)
            unpacked_bits = rice_block.unpacked_bits
        # code bias reference signal bit mask block
        self.cb_ref_signals_blk = zdb020(msg, unpacked_bits)
        unpacked_bits = self.cb_ref_signals_blk.unpacked_bits
        # =====================================================================
        #                         Low Rate phase bias
        # =====================================================================
        # phase bias reference signal bit mask block
        self.pb_signals_blk = zdb020(msg, unpacked_bits)
        unpacked_bits = self.pb_signals_blk.unpacked_bits
        # ssrz low rate phase bias cycle range zdf071
        [self.pb_cr, unpacked_bits] = fields.zdf071(msg, unpacked_bits)
        # low rate phase bias bitfield length zdf072
        [self.pb_bl, unpacked_bits] = fields.zdf072(msg, unpacked_bits)
        # max number of continuity/overflow bits
        [self.pb_ofl, unpacked_bits] = fields.zdf073(msg, unpacked_bits)
        # phase bias resolution
        self.pb_res = self.pb_cr / (2 ** self.pb_bl)
        # =====================================================================
        #              Low Rate Satellite Dependent Global Iono
        # =====================================================================
        # satellite dependent global ionosphere correction flag zdf134
        [self.iono_flag, unpacked_bits] = fields.zdf134(msg, unpacked_bits)
        if self.iono_flag == 1:
            # model parameters block zdb060
            self.model = zdb060(msg, unpacked_bits)
            unpacked_bits = self.model.unpacked_bits
            # satellite-dependent global iono corr default resolution zdf074
            [self.iono_res, unpacked_bits] = fields.zdf074(msg, unpacked_bits)
            # metadata compressed coefficients date bock definition
            # for satellite-dependent global iono corrections
            # total number of satellite coefficients (rsi)
            self.rsi = int(self.model.int_mp[0])
            # number of different combined order m+l
            self.n_order = compute_order_list(self.rsi)
            # ssrz compressed coefficients block definition zdb023
            # if n_order is 0, then there is an offset only, and it should be
            # considered in zdb023, therefore one has to add 1 to n_order
            if self.n_order == 0:
                self.coeff_blk = zdb023(msg, unpacked_bits, self.n_order + 1)
            else:
                self.coeff_blk = zdb023(msg, unpacked_bits, self.n_order)
            unpacked_bits = self.coeff_blk.unpacked_bits
        # save read bits
        self.unpacked_bits = unpacked_bits

    def __str__(self):
        strg = ' #** SSRZ Low Rate Metadata Message Block' + '\n'
        strg += ('  Number of satellite-dependent timing blocks :' +
                 str(self.n_timing) + '\n')
        for tt in range(self.n_timing):
            strg += ('  #- SSRZ timing block ' + str(tt + 1) + '\n' +
                     self.lr_timing_block[tt].__str__())
        strg += ('  #* SSRZ Low Rate Clock Metadata \n' +
                 '    Number of LR clock parameters :' + str(self.n_lr_clk) +
                 '\n')
        if self.n_lr_clk < 2:
            strg += '    LR C0 resolution [m]: ' + str(self.c0_res) + '\n'
        else:
            strg += ('    LR C0 resolution [m]: ' + str(self.c0_res) + '\n' +
                     '    LR C1 resolution [m]: ' + str(self.c1_res) + '\n')
        strg += ('    Number of LR clock rice blocks: ' +
                 str(self.n_rb_clk) + '\n')
        for rb in range(self.n_rb_clk):
            strg += '  ' + self.rb_clk[rb].__str__()
        strg += ('  #* SSRZ Low Rate Orbit Metadata \n' +
                 '    Number of LR orbit parameters :' + str(self.orb_p) +
                 '\n')
        if self.orb_p == 3:
            strg += ('    LR rad orbit resolution [m]: ' + str(self.rad_res) +
                     '\n' +
                     '    LR atr orbit resolution [m]: ' + str(self.atr_res) +
                     '\n' +
                     '    LR ctr orbit resolution [m]: ' + str(self.ctr_res) +
                     '\n' +
                     '    Number of LR orbit rice blocks: ' +
                     str(self.n_rb_orb) + '\n')
            for rb in range(self.n_rb_orb):
                strg += '  ' + self.rb_orb[rb].__str__()
        strg += ('  #* SSRZ Low Rate Code Bias Metadata \n' +
                 '    LR code bias resolution [m]: ' + str(self.cb_res) + '\n'
                 '    Number of LR code bias rice blocks: ' +
                 str(self.n_rb_cb) + '\n')
        for rb in range(self.n_rb_cb):
            strg += '  ' + self.rb_cb[rb].__str__()
        strg += '    Code bias reference signals \n'
        strg += '  ' + self.cb_ref_signals_blk.__str__()
        strg += ('  #* SSRZ Low Rate Phase Bias Metadata \n')
        strg += '    Phase bias reference signals \n'
        strg += ('  ' + self.pb_signals_blk.__str__() +
                 '    LR phase bias cycle range [cy]: ' + str(self.pb_cr) +
                 '\n' +
                 '    LR phase bias bitfield length : ' + str(self.pb_bl) +
                 '\n' +
                 '    LR max num of overflow bits   : ' + str(self.pb_ofl) +
                 '\n' +
                 '    LR phase bias resolution [cy] : ' +
                 str(np.round(self.pb_res, 7)) + '\n')
        strg += ('  #* SSRZ Low Rate Satellite-dependent Global Ionosphere' +
                 ' correction Metadata \n')
        strg += ('    Ionosphere correction flag : ' + str(self.iono_flag) +
                 '\n')
        if self.iono_flag == 1:
            strg += '  ' + self.model.__str__()

        return strg

    def __repr__(self):
        strg = ('Low rate block class with main objects: ' +
                'n_timing, n_lr_clk, orb_p, c0_res, c1_res, rad_res, ' +
                'atr_res, ctr_res, rb_orb, cb_res, pb_cr,' +
                'cb_ref_signals_blk, ' +
                'pb_signals_blk, pb_cr, pb_bl, pb_ofl, pb_res, iono_flag, ' +
                'unpacked_bits')
        return strg


# *************************************************************************** #
#                                                                             #
#  SSRZ Gridded Ionosphere Correction Metadata Message Block Tag 1 (ZMB003-1) #
#                                                                             #
# *************************************************************************** #
class zmb003_1:
    """ It defines the metadata required for the decoding of the SSRZ
        Gridded Ionosphere Correction Message (ZM003). Here metadata tag 1
        is considered
    """

    def __init__(self, msg, unpacked_bits):
        # number of SSRZ LR satellite groups
        [self.n_lr_groups, unpacked_bits] = fields.zdf010(msg,
                                                          unpacked_bits, 1)
        self.n_lr_groups += 1
        # number of satellite-dependent timing blocks
        [self.n_timing, unpacked_bits] = fields.zdf055(msg, unpacked_bits)
        # SSRZ GRI timing block
        gri_time_blk = []
        for ii in range(self.n_timing):
            time_blk = zdb019(msg, unpacked_bits, self.n_lr_groups)
            unpacked_bits = time_blk.unpacked_bits
            gri_time_blk.append(time_blk)
        self.gri_time_blk = np.array(gri_time_blk)
        # ssrz default resolution of the gridded ionospher
        [self.gr_iono_res, unpacked_bits] = fields.zdf075(msg, unpacked_bits)
        # default bin size parameter zdf043
        [self.bin_size, unpacked_bits] = fields.zdf043(msg, unpacked_bits)
        self.unpacked_bits = unpacked_bits

    def __str__(self):
        strg = ' #** SSRZ Gridded Ionosphere Metadata Message Block' + '\n'
        strg += ('  Number of SSRZ LR satellite groups          :' +
                 str(self.n_lr_groups) + '\n')
        strg += ('  Number of satellite-dependent timing blocks :' +
                 str(self.n_timing) + '\n')
        for tt in range(self.n_timing):
            strg += ('  #- SSRZ timing block ' + str(tt + 1) + '\n' +
                     self.gri_time_blk[tt].__str__())
        strg += ('  Ionosphere resolution [TECU]:' +
                 str(self.gr_iono_res) + '\n')
        strg += ('  Default bin size parameter (p0): ' +
                 str(self.bin_size) + '\n')
        return strg

    def __repr__(self):
        strg = ('Gridded ionosphere block class with objects: ' +
                'n_lf_groups, n_timing, gri_time_blk, gr_iono_res,' +
                'bin_size, unpacked_bits')
        return strg


# *************************************************************************** #
#                                                                             #
#  SSRZ Gridded Ionosphere Correction Metadata Message Block Tag 2 (ZMB003-2) #
#                                                                             #
# *************************************************************************** #
class zmb003_2:
    """ It defines the metadata required for the decoding of the SSRZ
        Gridded Ionosphere Correction Message (ZM003). Here metadata tag 2
        is considered
    """

    def __init__(self, msg, unpacked_bits):
        # SSRZ grid-related timing parameters
        self.time_para = zdb044(msg, unpacked_bits)
        unpacked_bits = self.time_para.unpacked_bits
        # ssrz default resolution of the gridded ionosphere
        [self.gr_iono_res, unpacked_bits] = fields.zdf075(msg, unpacked_bits)
        # default bin size parameter zdf043
        [self.bin_size, unpacked_bits] = fields.zdf043(msg, unpacked_bits)
        self.unpacked_bits = unpacked_bits

    def __str__(self):
        strg = ' #** SSRZ Gridded Ionosphere Metadata Message Block' + '\n'
        strg += ('  Grid-related Timing parameters\n' +
                 self.time_para.__str__())
        strg += ('  Ionosphere resolution [TECU]:' +
                 str(self.gr_iono_res) + '\n')
        strg += ('  Default bin size parameter (p0): ' +
                 str(self.bin_size) + '\n')
        return strg

    def __repr__(self):
        strg = ('Gridded ionosphere block class with objects: ' +
                'time_para, gr_iono_res,' +
                'bin_size, unpacked_bits')
        return strg


# *************************************************************************** #
#                                                                             #
#  SSRZ Gridded Tropo Correction Metadata Message Block with tag 2(ZMB004-2)  #
#                                                                             #
# *************************************************************************** #
class zmb004_2:
    """ It defines the metadata required for the decoding of the SSRZ
        Gridded Troposphere Correction Message (ZM004).
        Here tag 2 is considered.
    """

    def __init__(self, msg, unpacked_bits):
        # ssrz grt timing block
        self.timing_blk = zdb018(msg, unpacked_bits)
        unpacked_bits = self.timing_blk.unpacked_bits
        # ssrz grt model parameters block
        self.model_blk = zdb060(msg, unpacked_bits)
        unpacked_bits = self.model_blk.unpacked_bits
        # troposphere component bit mask zdf107
        [self.mask, unpacked_bits] = fields.zdf107(msg, unpacked_bits)
        self.components = []
        self.res = []
        self.p0 = []
        if self.mask[0] == 1:
            # dry component
            self.components.append('d')
            # ssrz gridded troposphere scale factor resolution of dry component
            [res, unpacked_bits] = fields.zdf076(msg, unpacked_bits)
            self.res.append(res)
            # bin size parameter zdf043
            [bin_size, unpacked_bits] = fields.zdf043(msg, unpacked_bits)
            self.p0.append(bin_size)
        if self.mask[1] == 1:
            # wet component
            self.components.append('w')
            # ssrz gridded troposphere scale factor resolution of dry component
            [res, unpacked_bits] = fields.zdf076(msg, unpacked_bits)
            self.res.append(res)
            # bin size parameter zdf043
            [bin_size, unpacked_bits] = fields.zdf043(msg, unpacked_bits)
            self.p0.append(bin_size)
        if self.mask[2] == 1:
            # total component:
            self.components.append('t')
            # ssrz gridded troposphere scale factor resolution of dry component
            [res, unpacked_bits] = fields.zdf076(msg, unpacked_bits)
            self.res.append(res)
            # bin size parameter zdf043
            [bin_size, unpacked_bits] = fields.zdf043(msg, unpacked_bits)
            self.p0.append(bin_size)
        self.unpacked_bits = unpacked_bits

    def __str__(self):
        strg = ' #** SSRZ Gridded Troposphere Metadata Message Block' + '\n'
        strg += self.timing_blk.__str__()
        strg += self.model_blk.__str__()
        strg += '     Component    Resolution [m]    Bin size (p0) ' + '\n'
        for ii in range(len(self.components)):
            strg += ('        ' + self.components[ii] +
                     '            ' + str(self.res[ii]) +
                     '               ' + str(self.p0[ii]) + '\n')
        return strg

    def __repr__(self):
        strg = ('Gridded troposphere block with main objects: ' +
                'timing_blk, model_blk, components, res, p0')
        return strg


# *************************************************************************** #
#                                                                             #
#  SSRZ Gridded Tropo Correction Metadata Message Block with tag 3 (ZMB004-3) #
#                                                                             #
# *************************************************************************** #
class zmb004_3:
    """ It defines the metadata required for the decoding of the SSRZ
        Gridded Troposphere Correction Message (ZM004).
        Here tag 3 is considered.
    """

    def __init__(self, msg, unpacked_bits):
        # ssrz grid-related timing parameters
        self.time_para = zdb044(msg, unpacked_bits)
        unpacked_bits = self.time_para.unpacked_bits
        # ssrz grt model parameters block
        self.model_blk = zdb060(msg, unpacked_bits)
        unpacked_bits = self.model_blk.unpacked_bits
        # troposphere component bit mask zdf107
        [self.mask, unpacked_bits] = fields.zdf107(msg, unpacked_bits)
        self.components = []
        self.res = []
        self.p0 = []
        if self.mask[0] == 1:
            # dry component
            self.components.append('d')
            # ssrz gridded troposphere scale factor resolution of dry component
            [res, unpacked_bits] = fields.zdf076(msg, unpacked_bits)
            self.res.append(res)
            # bin size parameter zdf043
            [bin_size, unpacked_bits] = fields.zdf043(msg, unpacked_bits)
            self.p0.append(bin_size)
        if self.mask[1] == 1:
            # wet component
            self.components.append('w')
            # ssrz gridded troposphere scale factor resolution of dry component
            [res, unpacked_bits] = fields.zdf076(msg, unpacked_bits)
            self.res.append(res)
            # bin size parameter zdf043
            [bin_size, unpacked_bits] = fields.zdf043(msg, unpacked_bits)
            self.p0.append(bin_size)
        if self.mask[2] == 1:
            # total component:
            self.components.append('t')
            # ssrz gridded troposphere scale factor resolution of dry component
            [res, unpacked_bits] = fields.zdf076(msg, unpacked_bits)
            self.res.append(res)
            # bin size parameter zdf043
            [bin_size, unpacked_bits] = fields.zdf043(msg, unpacked_bits)
            self.p0.append(bin_size)
        self.unpacked_bits = unpacked_bits

    def __str__(self):
        strg = ' #** SSRZ Gridded Troposphere Metadata Message Block' + '\n'
        strg += self.time_para.__str__()
        strg += self.model_blk.__str__()
        strg += '     Component    Resolution [m]    Bin size (p0) ' + '\n'
        for ii in range(len(self.components)):
            strg += ('        ' + self.components[ii] +
                     '            ' + str(self.res[ii]) +
                     '               ' + str(self.p0[ii]) + '\n')
        return strg

    def __repr__(self):
        strg = ('Gridded troposphere block with main objects: ' +
                'time_para, model_blk, components, res, p0')
        return strg


# *************************************************************************** #
#                                                                             #
#           SSRZ Satellite dependent Regional Ionosphere Correction           #
#                      Metadata Message Block (ZMB005)                        #
#                                                                             #
# *************************************************************************** #
class zmb005:
    """ It defines the metadata required for the decoding of the SSRZ
        Satellite dependent Regional Ionosphere Correction Message (ZM005).
    """

    def __init__(self, msg, unpacked_bits, zdf020):
        # number of low-rate satellite groups zdf010. It has to be the same
        # reported in zmb011
        self.md_tag = zdf020
        [self.n_g_lr, unpacked_bits] = fields.zdf010(msg, unpacked_bits,
                                                     zdf020)
        self.n_g_lr += 1
        # number of satellite-dependent timing blocks zdf055
        [self.n_t_blk, unpacked_bits] = fields.zdf055(msg, unpacked_bits)
        # timing block of satellite dependent ionosphere corrections
        self.time_sat_iono_blk = []
        for tt in range(self.n_t_blk):
            block = zdb019(msg, unpacked_bits, self.n_g_lr)
            self.time_sat_iono_blk = np.append(self.time_sat_iono_blk, block)
            unpacked_bits = block.unpacked_bits
        # ssrz model parameters block zdb060
        self.model_blk = zdb060(msg, unpacked_bits)
        unpacked_bits = self.model_blk.unpacked_bits
        # total number of satellite coefficients (rsi)
        self.rsi = int(self.model_blk.int_mp[0])
        # number of different combined order m+l
        self.n_order = compute_order_list(self.rsi)
        # ssrz default resolution of the satellite-dependent regional
        # ionosphere corrections zdf080
        [self.sat_reg_res, unpacked_bits] = fields.zdf080(msg, unpacked_bits)
        # ssrz compressed coefficients block definition zdb023
        # if n_order is 0, it means there is an offset only, and it should be
        # considered in zdb023, therefore one has to add 1 to n_order
        if self.n_order == 0:
            self.coeff_blk = zdb023(msg, unpacked_bits, self.n_order + 1)
        else:
            self.coeff_blk = zdb023(msg, unpacked_bits, self.n_order)
        self.unpacked_bits = self.coeff_blk.unpacked_bits

    def __str__(self):
        strg = (' #** SSRZ Satellite dependent Regional Ionosphere Metadata ' +
                'Message Block' + '\n')
        for tb in range(self.n_t_blk):
            strg += self.time_sat_iono_blk[tb].__str__()
        strg += '  ' + self.model_blk.__str__()
        strg += '    Resolution [TECU]:' + str(self.sat_reg_res) + '\n'
        strg += '    Rice block parameters: \n'
        strg += ('    Number of different combined order m+l: ' +
                 str(self.n_order) + '\n')
        for ii in range(len(self.coeff_blk.coeff_rb)):
            strg += ('      Default bin size p0: ' +
                     str(self.coeff_blk.coeff_rb[ii].p0) + '\n')
        return strg

    def __repr__(self):
        strg = ('Satellite dependent regional ionosphere block ' +
                'with main objects: ' +
                'time_sat_iono_blk, model_blk, n_order, coeff_blk.')
        return strg


# *************************************************************************** #
#                                                                             #
#                   SSRZ Global VTEC Ionosphere Correction                    #
#                      Metadata Message Block (ZMB006)                        #
#                                                                             #
# *************************************************************************** #
class zmb006:
    """ It defines the metadata required for the decoding of the SSRZ
        Global VTEC Ionosphere Correction Message (ZM006).
    """

    def __init__(self, msg, unpacked_bits, zdf020):
        self.zdf020 = zdf020
        if ((zdf020 == 2) | (zdf020 == 3)):
            # global vtec timing block zdb018
            self.t_blk = zdb018(msg, unpacked_bits)
            unpacked_bits = self.t_blk.unpacked_bits
            # ssrz model parameter block zdb060
            self.model_blk = zdb060(msg, unpacked_bits)
            unpacked_bits = self.model_blk.unpacked_bits
            # ssrz encoder type zdf129
            [self.enc_type, unpacked_bits] = fields.zdf129(msg, unpacked_bits)
        elif zdf020 == 4:
            # SSRZ update and offset block
            self.upoff_blk = zdb040(msg, unpacked_bits)
            unpacked_bits = self.upoff_blk.unpacked_bits
        else:
            pass
        # Global VTEC resolution
        if zdf020 == 2:
            [zdf081, unpacked_bits] = fields.zdf081(msg, unpacked_bits)
            self.res = zdf081 * 1e-3  # [TECU]
        elif ((zdf020 == 3) | ((zdf020 == 4))):
            [zdf082, unpacked_bits] = fields.zdf082(msg, unpacked_bits)
            self.res = zdf082 * 1e-5  # [TECU]
            if zdf020 == 4:
                # Number of ionospheric layers
                [n_layers, unpacked_bits] = fields.zdf130(msg, unpacked_bits)
                self.layer_blk = []
                self.coeff_blk = []
                for ii in range(n_layers):
                    layer_blk = zdb061(msg, unpacked_bits)
                    unpacked_bits = layer_blk.unpacked_bits
                    self.layer_blk.append(layer_blk)
                    # Coeff block
                    n_para = layer_blk.sph_ord + 1
                    coeff_blk = zdb023(msg, unpacked_bits, n_para)
                    unpacked_bits = coeff_blk.unpacked_bits
                    self.coeff_blk.append(coeff_blk)

        self.unpacked_bits = unpacked_bits

    def __str__(self):
        strg = (' #** SSRZ Global VTEC Ionosphere Correction Metadata '
                'Message Block ZMB006' + '\n')
        strg += self.t_blk.__str__()
        if self.zdf020 == 3:
            strg += ' ' + self.model_blk.__str__()
        strg += '  Resolution [TECU]:' + str(self.res) + '\n'
        return strg

    def __repr__(self):
        strg = ('Global VTEC ionosphere block with main objects: ' +
                't_blk, model_blk, res')
        return strg


# *************************************************************************** #
#                                                                             #
#                   SSRZ Regional Troposphere Correction                      #
#                    Metadata Message Block tag 2 (ZMB007_2)                  #
#                                                                             #
# *************************************************************************** #
class zmb007_2:
    """ ZMB007-2_ It defines the metadata required for the decoding of
        the SSRZ Regional Troposphere Correction Message (ZM007) with
        metadata tag 2
    """

    def __init__(self, msg, unpacked_bits):
        # regional troposphere timing block ZDB018
        self.t_blk = zdb018(msg, unpacked_bits)
        unpacked_bits = self.t_blk.unpacked_bits

        # troposphere basic component bit mask zdf107
        [self.components, unpacked_bits] = fields.zdf107(msg, unpacked_bits)

        # ssrz model parameter block ZDB060
        self.model_blk = zdb060(msg, unpacked_bits)
        unpacked_bits = self.model_blk.unpacked_bits

        # derive parameters from ZDB060 (see Table 15.5)
        # number of different orders in latitude
        n_lat = int(self.model_blk.int_mp[1])
        # number of different orders in longitude
        n_lon = int(self.model_blk.int_mp[1])
        self.n_order = (n_lat - 1) + (n_lon - 1) + 1

        self.cff = []
        self.res = []

        # identify components and decode blocks
        self.trans_comp = []

        # check if parameters for total component are available
        if self.components[0] == 1:
            self.trans_comp.append('t')
            # ssrz default resolution of the regional tropo dry part zdf 085
            [self.res_t, unpacked_bits] = fields.zdf085(msg, unpacked_bits)
            # ssrz compressed coeff of the regional tropo dry part zdb023
            self.cff_t = zdb023(msg, unpacked_bits, self.n_order)
            unpacked_bits = self.cff_t.unpacked_bits
            # separated by height flag
            [self.sep_hgt_t, unpacked_bits] = fields.zdf108(msg, unpacked_bits)
            # Save coeff and resolution
            self.cff.append(self.cff_t)
            self.res.append(self.res_t)

        # check if parameters for wet component are available
        if self.components[1] == 1:
            self.trans_comp.append('w')
            # ssrz default resolution of the regional tropo dry part zdf 085
            [self.res_w, unpacked_bits] = fields.zdf085(msg, unpacked_bits)
            # ssrz compressed coeff of the regional tropo dry part zdb023
            self.cff_w = zdb023(msg, unpacked_bits, self.n_order)
            unpacked_bits = self.cff_w.unpacked_bits
            # separated by height flag
            [self.sep_hgt_w, unpacked_bits] = fields.zdf108(msg, unpacked_bits)
            # Save coeff and res
            self.cff.append(self.cff_w)
            self.res.append(self.res_w)
        # check if parameters for dry component are available
        if self.components[2] == 1:
            self.trans_comp.append('d')
            # ssrz default resolution of the regional tropo dry part zdf 085
            [self.res_d, unpacked_bits] = fields.zdf085(msg, unpacked_bits)
            # ssrz compressed coeff of the regional tropo dry part zdb023
            self.cff_d = zdb023(msg, unpacked_bits, self.n_order)
            unpacked_bits = self.cff_d.unpacked_bits
            # separated by height flag
            [self.sep_hgt_d, unpacked_bits] = fields.zdf108(msg, unpacked_bits)
            # Save coeff and res
            self.cff.append(self.cff_d)
            self.res.append(self.res_d)

        if self.components[3] == 1:
            self.trans_comp.append('m')
            # resolution of the mapping improvement zdb086
            [self.res_m, unpacked_bits] = fields.zdf086(msg, unpacked_bits)
            self.res.append(self.res_m)
            # ssrz compressed mapping improvement data block definition zdb027
            self.cff_m = zdb027(msg, unpacked_bits, n_para=self.n_order)
            self.cff.append(self.cff_m)

    def __str__(self):
        strg = (' #** SSRZ Regional Troposphere Correction ' +
                'Metadata Message Block (ZMB007-2) ' + '\n')
        strg += self.t_blk.__str__()
        strg += self.model_blk.__str__()
        strg += '     Component    Resolution [-]    Bin size (p0) ' + '\n'
        for ii in range(len(self.trans_comp)):
            strg += ('        ' + str(self.trans_comp[ii]) +
                     '            ' + str(self.res[ii]) + '\n')
        return strg


# *************************************************************************** #
#                                                                             #
#                   SSRZ Regional Troposphere Correction                      #
#                    Metadata Message Block tag 3 (ZMB007_3)                  #
#                                                                             #
# *************************************************************************** #
class zmb007_3:
    """ ZMB007-3_ It defines the metadata required for the decoding of the SSRZ
        Regional Troposphere Correction Message (ZM007) with metadata
        tag 3
    """

    def __init__(self, msg, unpacked_bits):
        [self.region_id, unpacked_bits] = fields.zdf027(msg, unpacked_bits)
        # latitude of ground point origin zdf168
        [self.lat_gpo, unpacked_bits] = fields.zdf168(msg, unpacked_bits)

        # longitude of ground point origin zdf169
        [self.lon_gpo, unpacked_bits] = fields.zdf169(msg, unpacked_bits)

        # height of ground point origin zdf170
        [self.hgt_gpo, unpacked_bits] = fields.zdf170(msg, unpacked_bits)

        # Horizontal scale factor zdf172
        [self.d_rt, unpacked_bits] = fields.zdf172(msg, unpacked_bits)

        # Vertical scale factor zdf173
        [self.h_rt, unpacked_bits] = fields.zdf173(msg, unpacked_bits)

        # troposphere basic component bit mask zdf107
        [self.components, unpacked_bits] = fields.zdf107(msg, unpacked_bits)
        # Append quantities per component based on the bit mask
        self.cff = []
        self.res = []
        self.trans_comp = []
        self.max_hor = []
        self.max_hgt = []
        if self.components[0] == 1:
            self.trans_comp.append('d')
            # basic tropo component dry part
            self.cff_d = zdb050(msg, unpacked_bits, zdf107_bit0=True)
            unpacked_bits = self.cff_d.unpacked_bits
            self.cff.append(self.cff_d)
            self.res.append(self.cff_d.res)
            self.max_hor.append(self.cff_d.m_hor)
            self.max_hgt.append(self.cff_d.m_hgt)
        if self.components[1] == 1:
            self.trans_comp.append('w')
            # basic tropo component wet part
            self.cff_w = zdb050(msg, unpacked_bits)
            unpacked_bits = self.cff_w.unpacked_bits
            self.cff.append(self.cff_w)
            self.res.append(self.cff_w.res)
            self.max_hor.append(self.cff_w.m_hor)
            self.max_hgt.append(self.cff_w.m_hgt)
        if self.components[2] == 1:
            self.trans_comp.append('t')
            # basic tropo component total part
            self.cff_t = zdb050(msg, unpacked_bits)
            unpacked_bits = self.cff_t.unpacked_bits
            self.cff.append(self.cff_t)
            self.res.append(self.cff_t.res)
            self.max_hor.append(self.cff_t.m_hor)
            self.max_hgt.append(self.cff_t.m_hgt)
        if self.components[3] == 1:
            self.trans_comp.append('m')
            # tropo mapping function improvement parameter data block
            self.cff_m = zdb051(msg, unpacked_bits)
            unpacked_bits = self.cff_m.unpacked_bits
            self.cff.append(self.cff_m)
            self.res.append(self.cff_m.res)
            self.max_hor.append(self.cff_m.m_hor)
            self.max_hgt.append(self.cff_m.m_hgt)
        # Multi-region flag
        [self.mreg_flag, unpacked_bits] = fields.zdf030(msg, unpacked_bits)
        # Save bits
        self.unpacked_bits = unpacked_bits

    def __str__(self):
        strg = (' #** SSRZ Regional Troposphere Correction ' +
                'Metadata Message Block (ZMB007-3) ' + '\n')
        strg += ('     Ground point origin:\n' +
                 '      lat[deg]      lon [deg]   ' +
                 ' hgt [m]\n')
        strg += ("      " + str(self.lat_gpo) + '        ' +
                 str(self.lon_gpo) +
                 '      ' +
                 str(self.hgt_gpo) + "\n")
        strg += '     Component    Resolution [-]' + '\n'
        for ii in range(len(self.trans_comp)):
            strg += ('        ' + str(self.trans_comp[ii]) +
                     '            ' + str(self.res[ii]) + '\n')
        return strg


# *************************************************************************** #
#                                                                             #
#                   SSRZ Regional Troposphere Correction                      #
#                      Metadata Message Block (ZMB007)                        #
#                                                                             #
# *************************************************************************** #
class zmb007:
    """ It defines the metadata required for the decoding of the SSRZ
        Regional Troposphere Correction Message (ZM007).
    """

    def __init__(self, msg, unpacked_bits, zdf020):
        self.zdf020 = zdf020
        if zdf020 == 2:
            # ssrz block for metadata tag 2
            self.blk = zmb007_2(msg, unpacked_bits)
        elif zdf020 == 3:
            # Region-related timing parameters
            self.tmg_blk = zdb046(msg, unpacked_bits)
            unpacked_bits = self.tmg_blk.unpacked_bits
            # ssrz block for metadata tag 3
            blk0 = zmb007_3(msg, unpacked_bits)
            self.blk = [blk0]
            unpacked_bits = blk0.unpacked_bits
            ii = 0
            while (self.blk[ii].mreg_flag != 0):
                blk = zmb007_3(msg, unpacked_bits)
                self.blk.append(blk)
                unpacked_bits = blk.unpacked_bits
                ii += 1
        # Save unpacked bits
        self.unpacked_bits = unpacked_bits

    def __str__(self):
        strg = (' #** SSRZ Regional Troposphere Correction ' +
                'Metadata Message Block (ZMB007) ' + '\n')
        if self.zdf020 == 2:
            strg += "   " + self.blk.__str__()
        else:
            for ii in range(len(self.blk)):
                blk = self.blk[ii]
                strg += "   " + "Region " + f"{ii}" + "\n"
                strg += "   " + blk.__str__()
        return strg

    def __repr__(self):
        strg = ('Regional troposphere correction block with main objects: ' +
                'blk and zdf020.')
        return strg


# *************************************************************************** #
#                                                                             #
#                             SSRZ QIX Bias                                   #
#                      Metadata Message Block (ZMB008)                        #
#                                                                             #
# *************************************************************************** #
class zmb008:
    """  This block can be part of the SSRZ Metadata Message (ZM012) or of
        the SSRZ QIX Bias Message ZM008 itself.
    """

    def __init__(self, msg, unpacked_bits, zdf020):
        self.zdf020 = zdf020
        # ssrz qix code bias flag zdf 190
        [self.qix_cb_flag, unpacked_bits] = fields.zdf190(msg, unpacked_bits)
        # ssrz qix phase bias flag zdf 191
        [self.qix_pb_flag, unpacked_bits] = fields.zdf191(msg, unpacked_bits)
        if ((zdf020 == 1) | ((zdf020 == 2) & (self.qix_cb_flag == 1))):
            # default resolution
            [self.cb_res, unpacked_bits] = fields.zdf089(msg, unpacked_bits)
            # number of qix bias rb
            [self.n_cb_rb, unpacked_bits] = fields.zdf045(msg, unpacked_bits)
            # ssrz qix code bias signal rice block definition
            self.cb_rb_list = []
            for ii in range(self.n_cb_rb):
                rb = zdb022(msg, unpacked_bits)
                self.cb_rb_list = np.append(self.cb_rb_list, rb)
                unpacked_bits = rb.unpacked_bits
        if ((zdf020 == 1) | ((zdf020 == 2) & (self.qix_pb_flag == 1))):
            # default resolution
            [self.pb_res, unpacked_bits] = fields.zdf089(msg, unpacked_bits)
            # number of qix bias rb
            [self.n_pb_rb, unpacked_bits] = fields.zdf045(msg, unpacked_bits)
            # ssrz qix code bias signal rice block definition
            self.pb_rb_list = []
            for ii in range(self.n_pb_rb):
                rb = zdb022(msg, unpacked_bits)
                self.pb_rb_list = np.append(self.pb_rb_list, rb)
                unpacked_bits = rb.unpacked_bits
        self.unpacked_bits = unpacked_bits

    def __repr__(self):
        strg = ('Qix Bias block class with objects: qix_cb_flag, qix_pb_flag' +
                'cb_res, n_cb_rb, cb_rb_list (when cb are available), and ' +
                'pb_res, n_pb_rb, cb_pb_list (when pb are available)')
        return strg

    def __str__(self):
        strg = ' #** SSRZ QIX Bias Metadata Message Block (ZMB008) ' + '\n'
        strg += ('Code bias flag : ' + str(self.qix_cb_flag) +
                 'Phase bias flag: ' + str(self.qix_pb_flag))
        if ((self.zdf020 == 1) | ((self.zdf020 == 2) &
                                  (self.qix_cb_flag == 1))):
            strg += ('QIX code bias default resolution [mm]:' +
                     str(self.cb_res) + '\n')
            strg += ('Number of QIX code bias rice blocks: ' +
                     str(self.n_cb_rb) + '\n')
            for ii in range(self.n_cb_rb):
                strg += self.cb_rb_list[ii].__str__()
        if ((self.zdf020 == 1) | ((self.zdf020 == 2) &
                                  (self.qix_pb_flag == 1))):
            strg += ('QIX phase bias default resolution [mm]:' +
                     str(self.pb_res) + '\n')
            strg += ('Number of QIX phase bias rice blocks: ' +
                     str(self.n_pb_rb) + '\n')
            for ii in range(self.n_pb_rb):
                strg += self.pb_rb_list[ii].__str__()
        return strg


# *************************************************************************** #
#                                                                             #
#        SSRZ Satellite Group Definition Metadata Message Block (ZMB011)      #
#                                                                             #
# *************************************************************************** #
class zmb011:
    """ Satellite Group Definition Metadata Message Block (ZMB011).
         It defines the metadata required for decoding
         the SSRZ Low Rate message (ZM002).
    """

    def __init__(self, msg, unpacked_bits, zdf020):
        # zdf010 number of ssrz low rate satellite groups
        self.n_g_lr = fields.zdf010(msg, unpacked_bits, zdf020)[0]
        unpacked_bits = fields.zdf010(msg, unpacked_bits, zdf020)[1]
        # low rate satellite group definition block
        self.lr_block = []
        for s in range(self.n_g_lr):
            self.lr_block.append([])
            self.lr_block[s] = zdb017(msg, unpacked_bits)
            # update the unpacked_bits
            unpacked_bits = self.lr_block[s].unpacked_bits

        # zdf010 number of ssrz high rate satellite groups
        self.n_g_hr = fields.zdf010(msg, unpacked_bits, zdf020)[0]
        unpacked_bits = fields.zdf010(msg, unpacked_bits, zdf020)[1]
        # high rate satellite group definition block
        self.hr_block = []
        for s in range(self.n_g_hr):
            self.hr_block.append([])
            self.hr_block[s] = zdb017(msg, unpacked_bits)
            # update the unpacked_bits
            unpacked_bits = self.hr_block[s].unpacked_bits
        self.unpacked_bits = unpacked_bits

    def __str__(self):
        strg = ('Number of low-rate satellite groups: ' +
                str(self.n_g_lr) + '\n')
        # loop over the satellite low rate groups
        for g in range(self.n_g_lr):
            strg += (' Satellite low-rate group block ' + str(g + 1) + '\n' +
                     '   GNSS                : ')
            for syst in self.lr_block[g].gnss:
                if syst == self.lr_block[g].gnss[-1]:
                    strg += str(syst)
                else:
                    strg += str(syst) + ','
            strg += '\n' + '   Max satellite ID/PRN: '
            for ii in range(len(self.lr_block[g].max_sat_id)):
                max_id = self.lr_block[g].max_sat_id[ii]
                syst = self.lr_block[g].gnss[ii]
                if syst == self.lr_block[g].gnss[-1]:
                    strg += str(max_id)
                else:
                    strg += str(max_id) + ','
            strg += ('\n' + '   Satellite group mode: ' +
                     str(self.lr_block[g].sat_group_mode) + '\n')
            # print all the satellite only when the group mode is 0
            # to check which satellite is included in the group or not
            if self.lr_block[g].sat_group_mode == 0:
                for syst in self.lr_block[g].gnss:
                    strg += '\n' + '   '
                    for sat in self.lr_block[g].sat_group[syst]:
                        if sat == self.lr_block[g].sat_group[syst][-1]:
                            strg += str(sat)
                        else:
                            strg += str(sat) + ','

        strg += ('\n' + 'Number of high-rate satellite groups: ' +
                 str(self.n_g_hr) + '\n')
        # loop over the satellite high rate groups
        for g in range(self.n_g_hr):
            strg += (' Satellite high-rate group block ' + str(g + 1) + '\n' +
                     '   GNSS                : ')
            for syst in self.hr_block[g].gnss:
                if syst == self.hr_block[g].gnss[-1]:
                    strg += str(syst)
                else:
                    strg += str(syst) + ','
            strg += '\n' + '   Max satellite ID/PRN: '
            for ii in range(len(self.hr_block[g].max_sat_id)):
                max_id = self.hr_block[g].max_sat_id[ii]
                syst = self.hr_block[g].gnss[ii]
                if syst == self.hr_block[g].gnss[-1]:
                    strg += str(max_id)
                else:
                    strg += str(max_id) + ','
            strg += ('\n' + '   Satellite group mode: ' +
                     str(self.hr_block[g].sat_group_mode))
            # print all the satellite only when the group mode is 0
            # to check which satellite is included in the group or not
            if self.hr_block[g].sat_group_mode == 0:
                for syst in self.hr_block[g].gnss:
                    strg += '\n' + '  '
                    for sat in self.hr_block[g].sat_group[syst]:
                        if sat == self.hr_block[g].sat_group[syst][-1]:
                            strg += str(sat)
                        else:
                            strg += str(sat) + ','
        return strg

    def __repr__(self):
        strg = ('Satellite metadata block with main objects: hr_block, ' +
                'lr_block')
        return strg


# *************************************************************************** #
#                                                                             #
#                      SSRZ Metadata Message Block (ZMB012)                   #
#                                                                             #
# *************************************************************************** #
class zmb012:
    def __init__(self, msg, unpacked_bits, md):
        # zdf020
        [zdf020, unpacked_bits] = fields.zdf020(msg, unpacked_bits)
        # Variable initialization
        n_g_hr, n_g_lr = None, None
        self.hr_md_block, self.lr_md_block = None, None
        # loop till the end of message blocks
        while zdf020 != 0:
            # ssrz metadata type number
            [self.md_type, unpacked_bits] = fields.zdf003(msg, unpacked_bits)
            # For debug: print(zdf020, self.md_type)
            # number of bits of the metadata block zdf021
            [n_bits, unpacked_bits] = fields.zdf021(msg, unpacked_bits)
            # make a string of bits
            bits_md_block = ''
            for bb in range(n_bits):
                bits_md_block += 'u1'
            # metadata block
            if ((zdf020 == 1) & (self.md_type == 1)):
                # high rate metadata message block
                if n_g_hr is None:
                    # check if the satellite group block has been already
                    # decoded
                    if md is None:
                        # metadata not yet available, keep in mind num of bits
                        # to possibly continue later
                        bits2unpack_hr_md_block = unpacked_bits
                        unpacked_bits += bits_md_block
                        continue
                    else:
                        n_g_hr = md.sat_gr.sat_md_blk.n_g_hr
                        self.hr_md_block = zmb001(msg, unpacked_bits, n_g_hr,
                                                  zdf020)
                        # update read bits
                        unpacked_bits = self.hr_md_block.unpacked_bits
                else:
                    self.hr_md_block = zmb001(msg, unpacked_bits, n_g_hr,
                                              zdf020)
                    # update read bits
                    unpacked_bits = self.hr_md_block.unpacked_bits
            elif ((zdf020 == 1) & (self.md_type == 2)):
                # low rate metadata message block
                if n_g_lr is None:
                    if md is None:
                        bits2unpack_lr_md_block = unpacked_bits
                        unpacked_bits += bits_md_block
                        continue
                    else:
                        n_g_lr = md.sat_gr.sat_md_blk.n_g_lr
                        self.lr_md_block = zmb002(msg, unpacked_bits, n_g_lr,
                                                  zdf020=1)
                        # update read bits
                        unpacked_bits = self.lr_md_block.unpacked_bits
                else:
                    self.lr_md_block = zmb002(msg, unpacked_bits, n_g_lr,
                                              zdf020=1)
                    # update read bits
                    unpacked_bits = self.lr_md_block.unpacked_bits
            elif ((zdf020 == 1) & (self.md_type == 3)):
                # SSRZ Gridded Iono Correction Metadata Message Block zmb003-1
                self.grid_iono_block = zmb003_1(msg, unpacked_bits)
                unpacked_bits = self.grid_iono_block.unpacked_bits
            elif ((zdf020 == 2) & (self.md_type == 3)):
                # SSRZ Gridded Iono Correction Metadata Message Block zmb003-2
                self.grid_iono_block = zmb003_2(msg, unpacked_bits)
                unpacked_bits = self.grid_iono_block.unpacked_bits
            elif ((zdf020 == 2) & (self.md_type == 4)):
                # SSRZ Gridded Tropo Correction Metadata Message Block zmb004
                self.grid_tropo_block = zmb004_2(msg, unpacked_bits)
                unpacked_bits = self.grid_tropo_block.unpacked_bits
            elif ((zdf020 == 3) & (self.md_type == 4)):
                # SSRZ Gridded Tropo Correction Metadata Message Block zmb004
                self.grid_tropo_block = zmb004_3(msg, unpacked_bits)
                unpacked_bits = self.grid_tropo_block.unpacked_bits
            elif ((zdf020 == 1) & (self.md_type == 5)):
                # SSRZ Satellite-depending Regional Ionosphere correction
                # Metadata Message Block zmb005
                self.sat_reg_iono_block = zmb005(msg, unpacked_bits, zdf020)
                unpacked_bits = self.sat_reg_iono_block.unpacked_bits
            elif (((zdf020 == 3) | (zdf020 == 4)) & (self.md_type == 6)):
                # SSRZ Global VTEC Ionosphere Correction Metadata
                # Message Block zmb006
                self.glo_vtec = zmb006(msg, unpacked_bits, zdf020)
                unpacked_bits = self.glo_vtec.unpacked_bits
            elif (((zdf020 == 2) | (zdf020 == 3)) & (self.md_type == 7)):
                # SSRZ Regional Troposphere Correction Metadata
                # Message Block zmb007
                self.reg_tropo = zmb007(msg, unpacked_bits, zdf020)
                unpacked_bits = self.reg_tropo.unpacked_bits
                # unpacked_bits += bits_md_block
            elif (((zdf020 == 1) | (zdf020 == 2)) & (self.md_type == 8)):
                # QIX biases
                self.qix_bias = zmb008(msg, unpacked_bits, zdf020)
                unpacked_bits = self.qix_bias.unpacked_bits
            elif (((zdf020 == 1) | (zdf020 == 2)) & (self.md_type == 11)):
                # satellite group definition metadata message block zmb011
                self.sat_md_block = zmb011(msg, unpacked_bits, zdf020)
                unpacked_bits = self.sat_md_block.unpacked_bits
                n_g_lr = self.sat_md_block.n_g_lr
                n_g_hr = self.sat_md_block.n_g_hr
            elif (((zdf020 == 3) | (zdf020 == 4)) & (self.md_type == 13)):
                # grid definition metadata block
                self.grid_block = zmb013(msg, unpacked_bits, zdf020)
                unpacked_bits = self.grid_block.unpacked_bits
            # compute again zdf020 and update unpacked bits
            [zdf020, unpacked_bits] = fields.zdf020(msg, unpacked_bits)

        # Reconstruct block that was not directly possible
        if ((self.hr_md_block is None) & (n_g_hr is not None)):
            self.hr_md_block = zmb001(msg, bits2unpack_hr_md_block, n_g_hr,
                                      zdf020)

        if ((self.lr_md_block is None) & (n_g_lr is not None)):
            self.lr_md_block = zmb002(msg, bits2unpack_lr_md_block, n_g_lr)

    def __str__(self):
        strg = 'SSRZ Metadata Message Block \n'
        if self.hr_md_block is not None:
            strg += ' ' + self.hr_md_block.__str__()
        if self.lr_md_block is not None:
            strg += ' ' + self.lr_md_block.__str__()
        if self.grid_iono_block is not None:
            strg += ' ' + self.grid_iono_block.__str__()
        return strg

    def __repr__(self):
        strg = ('Metadata block class with main objects: ' +
                'hr_md_block, lr_md_block, sat_md_block, grid_block, ' +
                'grid_iono_block, grid_tropo_block, sat_reg_iono_block, ' +
                'reg_tropo')
        return strg


# *************************************************************************** #
#                                                                             #
#         SSRZ Grid Definition Metadata Message Block (ZMB013)                #
#                                                                             #
# *************************************************************************** #
class zmb013:
    def __init__(self, msg, unpacked_bits, zdf020):
        self.zdf020 = zdf020
        # number of grids zdf091
        [zdf091, unpacked_bits] = fields.zdf091(msg, unpacked_bits)
        self.n_grids = zdf091
        if self.zdf020 == 4:
            # integer part of the grid point coordinate resolution
            [self.n, unpacked_bits] = fields.zdf092(msg, unpacked_bits)
            # order part of the frid point coordinate resolution zdf093
            [self.m, unpacked_bits] = fields.zdf093(msg, unpacked_bits)
            # final resolution in km is:
            self.dx = self.n * 10 ** (-1 * (self.m + 2))
            [self.hgt_res, unpacked_bits] = fields.zdf102(msg, unpacked_bits)
        else:
            self.dx = None
            self.hgt_res = None
        # ssrz grid definition block
        self.grid_blk_list = []
        for nn in range(self.n_grids):
            grid_blk = zdb012(msg, unpacked_bits, zdf020, self.dx,
                              self.hgt_res)
            unpacked_bits = grid_blk.unpacked_bits
            self.grid_blk_list.append(grid_blk)

        self.unpacked_bits = unpacked_bits

    def __str__(self):
        strg = ('SSRZ Grid Definition Block \n' +
                'Number of grids: ' + str(self.n_grids) + '\n')
        if self.zdf020 == 4:
            strg += ('Order part of the grid point coord res: ' + str(self.m) +
                     '\n' +
                     'Integer part of the grid point coord res: ' +
                     str(self.n) + '\n' +
                     'Grid resolution [km]: ' + str(self.dx) + '\n' +
                     'Height resolution: ' + str(self.hgt_res))
        for ii in range(self.n_grids):
            strg += ' ## Grid n.°' + str(ii + 1) + ': \n'
            strg += self.grid_blk_list[ii].__str__()
        return strg

    def __repr__(self):
        strg = ('Grid definition class with objects; zdf020, n_grids, dx, ' +
                'hgt_res, grid_blk_list')
        return strg
