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
    SSRZ data-field multi-method file
    Description:
    the module contains a collection of methods to decode the SSRZ data fields
    (ZDF). Each ZDF is a method. The constructor of the methods expects
    specific parts of the SSRZ message content as a byte object.
    Each ZDF method receives in input the message and the list of already
    unpacked bits. In output, each method provides the decoded content of the
    field and the unpacked bits after decoding the field.
    ***************************************************************************

    References:
       - Geo++ SSRZ documentation v1.1.2
"""
from atexit import register
import cbitstruct as bitstruct
import numpy as np


# =============================================================================
#                           Complementary functions
# =============================================================================
def prefix_coding(msg, n0, db, n_max=0, unpacked_bits=None):
    """
        Function to compute the decode the prefix codes used in the msgs.
        Input:
            - msg to decode
            - n0 number of bits of the first block
            - db delta bit of increasing bits for the next block
            - n_max of bits that can be considering in the additive block
        Output
    """
    if unpacked_bits is None:
        unpacked_bits = ''
        # define the indices to skip in the test. The number is
        # related to the already unpacked bits
        ii2skip = 0
    else:
        unpacked_bits = unpacked_bits
        # define the indices to skip in the test. The number is
        # related to the already unpacked bits
        ii2skip = len(bitstruct.unpack(unpacked_bits, msg))

    # initialization of the variables
    delta = n0
    bits2add = delta
    fail = ''
    for kk in range(delta):
        fail += '1'
    test2check = ''
    bits2unpack = unpacked_bits
    for ii in range(bits2add):
        bits2unpack += 'u1'
    test = bitstruct.unpack(bits2unpack, msg)
    for ii in range(bits2add):
        test2check += str(test[ii2skip + ii])
    int_val = int(test2check[-delta:], 2)

    # the loop continue until the last n0 bits are equal to n0 times '1'
    # or until the number of delta bits to add is lower than n_max
    while test2check[-delta:] == fail:
        if n_max == 0:
            delta += db
        else:
            if delta < n_max:
                delta += db
        fail = ''
        for kk in range(delta):
            fail += '1'
        bits2add += delta
        test2check = ''
        bits2unpack = unpacked_bits
        for ii in range(bits2add):
            bits2unpack += 'u1'
        test = bitstruct.unpack(bits2unpack, msg)

        for ii in range(bits2add):
            test2check += str(test[ii2skip + ii])
        int_val += int(test2check[-delta:], 2)

    return [test2check, int_val, bits2unpack]


# =============================================================================
#                               SSRZ data fields
# =============================================================================
def zdf002(msg):
    """SSRZ Message Type number Indicator
    """
    [prefix_code,
     zdf002_integer,
     unpacked_bits] = prefix_coding(msg, 2, 1, 5)
    return [zdf002_integer, unpacked_bits]


def zdf003(msg, unpacked_bits):
    """SSRZ Metadata Type number
    """
    bits2unpack = unpacked_bits + 'u8'
    type_n = bitstruct.unpack(bits2unpack, msg)[-1]
    return [type_n, bits2unpack]


def zdf005(msg, unpacked_bits):
    """ SSRZ issue of metadata
    """
    # ZDF005 issue of metadata, necessary for the correct association between
    # updata SSR and metadata
    bits2unpack = unpacked_bits + 'u2'
    iodm = bitstruct.unpack(bits2unpack, msg)[-1]
    return [iodm, bits2unpack]


def zdf006(msg, unpacked_bits):
    """ SSRZ metadata announcement bit. It announce the availability of new
        metadata.
    """
    bits2unpack = unpacked_bits + 'u1'
    md_ann_bit = bitstruct.unpack(bits2unpack, msg)[-1]
    return [md_ann_bit, bits2unpack]


def zdf010(msg, unpacked_bits, zdf020):
    """ Number of SSRZ Low Rate Satellite Groups
    """
    bits2unpack = unpacked_bits + 'u5'
    if int(zdf020) == 1:
        n_g_lr = bitstruct.unpack(bits2unpack, msg)[-1]
    elif int(zdf020) == 2:
        n_g_lr = bitstruct.unpack(bits2unpack, msg)[-1] + 1

    return [n_g_lr, bits2unpack]


def zdf011(msg, unpacked_bits, zdf020):
    """ Number of SSRZ Satellite Groups
    """
    bits2unpack = unpacked_bits + 'u5'
    if int(zdf020) == 1:
        n_g = bitstruct.unpack(bits2unpack, msg)[-1]
    elif int(zdf020) == 2:
        n_g = bitstruct.unpack(bits2unpack, msg)[-1] + 1

    return [n_g, bits2unpack]


def zdf012(msg, unpacked_bits):
    """ SSRZ GNSS ID Bit Mask
    """
    bits2unpack = unpacked_bits + 'u1' * 16
    gnss_id_bit_mask = bitstruct.unpack(bits2unpack, msg)
    gnss_list = ['G', 'R', 'E', 'J', 'S', 'C', 'I']
    gnss = ''
    for ii in range(len(gnss_list)):
        if gnss_id_bit_mask[-16 + ii] == 1:
            gnss += gnss_list[ii]
    if gnss == '':
        gnss = ['n/a']
    return [gnss, bits2unpack]


def zdf013(msg, unpacked_bits):
    """SSRZ Maximum Satellite ID per GNSS and Group
    """
    bits2unpack = unpacked_bits + 'u7'
    max_sat_id = bitstruct.unpack(bits2unpack, msg)[-1]
    return [max_sat_id, bits2unpack]


def zdf014(msg, unpacked_bits):
    """Satellite Group Definition Mode
    """
    bits2unpack = unpacked_bits + 'u4'
    sat_group_mode = bitstruct.unpack(bits2unpack, msg)[-1]
    return [sat_group_mode, bits2unpack]


def zdf015(msg, unpacked_bits):
    """Satellite Group Bit Mask per GNSS
    """
    bits2unpack = unpacked_bits + 'u1'
    bit_mask = bitstruct.unpack(bits2unpack, msg)[-1]
    return [bit_mask, bits2unpack]


def zdf016(msg, unpacked_bits, n_groups):
    """ SSRZ Satellite Group List Bit Mask
    """
    bit_mask = []
    bits2unpack = unpacked_bits
    for ii in range(n_groups):
        bits2unpack += 'u1'
        bit_mask = np.append(bit_mask,
                             int(bitstruct.unpack(bits2unpack, msg)[-1]))
    return [bit_mask, bits2unpack]


def zdf017(msg, unpacked_bits, gnss, max_id):
    """ Satellite bit mask per GNSS
    """
    bits2unpack = unpacked_bits
    prn_list = []
    for ii in range(max_id):
        bits2unpack += 'u1'
        bit_mask = bitstruct.unpack(bits2unpack, msg)[-1]
        if bit_mask == 1:
            if ii + 1 < 10:
                prn_list.append(gnss + '0' + str(ii + 1))
            else:
                prn_list.append(gnss + str(ii + 1))
    return [prn_list, bits2unpack]


def zdf018(msg, unpacked_bits, gnss_list, max_id_list):
    """ Satellite bit Mask. It is the sequence of the satellite bit masks
        per gnss zdf017.
    """
    sat_list = []
    for ii in range(len(gnss_list)):
        gnss = gnss_list[ii]
        max_id = max_id_list[ii]
        sat_list.append([])
        [prn_list, unpacked_bits] = zdf017(msg, unpacked_bits, gnss, max_id)
        sat_list[ii] = prn_list
    return [sat_list, unpacked_bits]


def zdf019(msg, unpacked_bits, gnss):
    """ SSRZ Signal Bit Mask per GNSS
    """
    n_bits = 32
    bits2unpack = unpacked_bits
    bit_mask = []
    for ii in range(n_bits):
        bits2unpack += 'u1'
        bit_mask = np.append(bit_mask, bitstruct.unpack(bits2unpack, msg)[-1])
    # rinex 3 code
    r3code = {}
    r3code['G'] = ['1C', '1P', '1W', '', '', '2C', '2D', '2S', '2L', '2X',
                   '2P', '2W', '', '', '5I', '5Q', '5X', '1S', '1L', '1X', '',
                   '', '', '', '', '', '', '', '', '', '']
    r3code['R'] = ['1C', '1P', '2C', '2P', '4A', '4B', '4X', '6A', '6B', '6X',
                   '3I', '3Q', '3X', '', '', '', '', '', '', '', '',
                   '', '', '', '', '', '', '', '', '', '']
    r3code['E'] = ['1A', '1B', '1C', '1X', '1Z', '5I', '5Q', '5X', '7I', '7Q',
                   '7X', '8I', '8Q', '8X', '6A', '6B', '6C', '6X', '6Z', '',
                   '', '', '', '', '', '', '', '', '', '']
    r3code['J'] = ['1C', '1S', '1L', '2S', '2L', '2X', '5I', '5Q', '5X', '6S',
                   '6L', '6X', '1X', '', '', '', '', '', '', '', '',
                   '', '', '', '', '', '', '', '', '', '', '']
    r3code['S'] = ['1C', '5I', '5Q', '5X', '', '', '', '', '', '',
                   '', '', '', '', '', '', '', '', '', '', '',
                   '', '', '', '', '', '', '', '', '', '', '']
    r3code['C'] = ['2I', '2Q', '2X', '6I', '6Q', '6X', '7I', '7Q', '7X', '1D',
                   '1P', '1X', '1A', '1N', '5D', '5P', '5X', '7D', '7P', '7Z',
                   '8D', '8P', '8X', '', '', '', '', '', '', '', '']
    indices = np.where(bit_mask == 1)[0]
    sig_list = np.array(r3code[gnss])[indices]
    return [sig_list, bits2unpack]


def zdf020(msg, unpacked_bits):
    """SSRZ Metadata Tag
    """
    [prefix_code, md_tag,
     zdf020_bits] = prefix_coding(msg, 8, 0, unpacked_bits=unpacked_bits)
    return [md_tag, zdf020_bits]


def zdf021(msg, unpacked_bits):
    """ Size of specific SSRZ Metadata Message Block
    """
    bits2unpack = unpacked_bits + 'u13'
    n_bits = bitstruct.unpack(bits2unpack, msg)[-1]
    return [n_bits, bits2unpack]


def zdf024(msg, unpacked_bits):
    """ SSRZ number of satellite groups
    """
    [pr_code, ngroups,
     bits] = prefix_coding(msg, 5, 0, unpacked_bits=unpacked_bits)
    return [ngroups, bits]


def zdf025(msg, unpacked_bits):
    """ SSRZ Grid ID
    """
    [pr_code, id,
     bits] = prefix_coding(msg, 8, 0, unpacked_bits=unpacked_bits)
    return [id + 1, bits]


def zdf026(msg, unpacked_bits, m_grid_id):
    """ The SSRZ Grid ID Bit Mask indicates in ascending order if a grid
        with SSRZ grid ID is present.
        0: not present
        1: present
    """
    bit_mask = []
    bits2unpack = unpacked_bits
    for ii in range(m_grid_id):
        bits2unpack += 'u1'
        bit_mask = np.append(bit_mask,
                             int(bitstruct.unpack(bits2unpack, msg)[-1]))
    return [bit_mask, bits2unpack]


def zdf027(msg, unpacked_bits):
    """ SSRZ Region ID 1-X
    """
    [pr_code, region_id,
     zdf027_bits] = prefix_coding(msg, 8, 0, unpacked_bits=unpacked_bits)
    region_id += 1
    return [region_id, zdf027_bits]


def zdf028(msg, unpacked_bits, m_reg_id):
    """ The SSRZ Region ID Bit Mask indicates in ascending order if a region
        with SSRZ Region ID is present.
        0: not present
        1: present
    """
    bit_mask = []
    bits2unpack = unpacked_bits
    for ii in range(m_reg_id):
        bits2unpack += 'u1'
        bit_mask = np.append(bit_mask,
                             int(bitstruct.unpack(bits2unpack, msg)[-1]))
    return [bit_mask, bits2unpack]


def zdf030(msg, unpacked_bits):
    """ SSRZ Multi-Region flag
    """
    bits2unpack = unpacked_bits + 'u1'
    flag = bitstruct.unpack(bits2unpack, msg)[-1]
    return [flag, bits2unpack]


def zdf042(msg, unpacked_bits):
    """ Number of SSRZ Rice Blocks
    """
    [pr_code, n_rice_blk,
     zdf042_bits] = prefix_coding(msg, 2, 1, 5, unpacked_bits=unpacked_bits)
    return [n_rice_blk, zdf042_bits]


def zdf043(msg, unpacked_bits):
    """ Default Bin Size Parameter
    """
    [pr_code, p0, zdf043_bits] = prefix_coding(msg, 2, 1, n_max=5,
                                               unpacked_bits=unpacked_bits)
    return [p0, zdf043_bits]


def zdf044(msg, unpacked_bits):
    """ Number of components per SSR parameter type
    """
    bits2unpack = unpacked_bits + 'u2'
    n_components = bitstruct.unpack(bits2unpack, msg)[-1]
    return [n_components, bits2unpack]


def zdf045(msg, unpacked_bits):
    """ Number of SSRZ Signal Rice Blocks
    """
    [pr_code, n_rice_blk,
     zdf045_bits] = prefix_coding(msg, 2, 0, unpacked_bits=unpacked_bits)
    return [n_rice_blk, zdf045_bits]


def zdf046(msg, unpacked_bits):
    """ Default Bin Size Parameter of Signal Bias
    """
    [pr_code, p0, zdf046_bits] = prefix_coding(msg, 2, 0,
                                               unpacked_bits=unpacked_bits)
    return [p0, zdf046_bits]


def zdf047(msg, unpacked_bits, n_para):
    """ This general bit mask indicates the presence of general parameters
        in a list. The sequence of parameters and length of this bit mask
        n_para is context-sensitive.
    """
    bit_mask = []
    bits2unpack = unpacked_bits
    for ii in range(n_para):
        bits2unpack += 'u1'
        bit_mask = np.append(bit_mask,
                             int(bitstruct.unpack(bits2unpack, msg)[-1]))
    return [bit_mask, bits2unpack]


def zdf050(msg, unpacked_bits):
    """ Full seconds since the beginnning of a 15 min interval.
    """
    bits2unpack = unpacked_bits + 'u10'
    time_tag = bitstruct.unpack(bits2unpack, msg)[-1]  # [s]
    return [time_tag, bits2unpack]


def zdf051(msg, unpacked_bits):
    """ GPS week number. [0-4095]. Starting at midnight of January 5, 1980
    """
    bits2unpack = unpacked_bits + 'u12'
    gps_week = bitstruct.unpack(bits2unpack, msg)[-1]  # [week]
    return [gps_week, bits2unpack]


def zdf052(msg, unpacked_bits):
    """ GPS epoch time. [0-604799]. Full seconds since the beginning of the
        GPS week.
    """
    bits2unpack = unpacked_bits + 'u20'
    gps_tow = bitstruct.unpack(bits2unpack, msg)[-1]  # [week]
    return [gps_tow, bits2unpack]


def zdf053(msg, unpacked_bits):
    """ Length of SSR Update interval
    """
    bits2unpack = unpacked_bits + 'u6'
    ui = bitstruct.unpack(bits2unpack, msg)[-1]  # [s]
    return [ui, bits2unpack]


def zdf054(msg, unpacked_bits):
    """ SSR offset interval
    """
    bits2unpack = unpacked_bits + 'u6'
    offset = bitstruct.unpack(bits2unpack, msg)[-1]  # [s]
    return [offset, bits2unpack]


def zdf055(msg, unpacked_bits):
    """Number of satellite dependent timing blocks
    """
    bits2unpack = unpacked_bits + 'u5'
    n_timing = bitstruct.unpack(bits2unpack, msg)[-1] + 1
    return [n_timing, bits2unpack]


def zdf056(msg, unpacked_bits):
    """ Length of an SSR update interval T_update that is identical to its
        nominal validity time of an SSR parameter.
        T_update = ZDF056 + 1
    """
    [pr_code, num, bits_zdf056] = prefix_coding(msg, 6, 0,
                                                unpacked_bits=unpacked_bits)
    t_update = num + 1
    return [t_update, bits_zdf056]


def zdf057(msg, unpacked_bits):
    """ SSR update interval offset of an SSR update interval.
        T_offset = ZDF057
    """
    [pr_code, num, bits_zdf057] = prefix_coding(msg, 6, 0,
                                                unpacked_bits=unpacked_bits)
    t_offset = num
    return [t_offset, bits_zdf057]


def zdf058(msg, unpacked_bits):
    """ Number of update and offset blocks (ZDB040) within an
        SSRZ Timing Parameter Block (ZDB041).
        N_tb = ZDF058 + 1
    """
    [pr_code, num, bits_zdf058] = prefix_coding(msg, 3, 0,
                                                unpacked_bits=unpacked_bits)
    n_tb = num + 1
    return [n_tb, bits_zdf058]


def zdf060(msg, unpacked_bits):
    """ SSRZ High Rate Clock Default Resolution
    """
    [prefix_code, hr_res,
     bits_zdf060] = prefix_coding(msg, 10, 0, unpacked_bits=unpacked_bits)
    # apply the resolution to the integer
    hr_res_resol = hr_res * 1e-4  # [m]
    return [hr_res_resol, bits_zdf060]


def zdf061(msg, unpacked_bits):
    """ SSRZ High Rate Radial orbit Default Resolution
    """
    [pr_code, hr_res, bits_zdf061] = prefix_coding(msg, 10, 0,
                                                   unpacked_bits=unpacked_bits)
    # apply the resolution to the integer
    hr_res_resol = hr_res * 1e-4  # [m]
    return [hr_res_resol, bits_zdf061]


def zdf062(msg, unpacked_bits):
    """ SSRZ Low Rate Clock C0 Default Resolution
    """
    [pr_code, c0, bits_zdf062] = prefix_coding(msg, 10, 0,
                                               unpacked_bits=unpacked_bits)
    # apply the resolution to the integer
    c0_res = c0 * 1e-4  # [m]
    return [c0_res, bits_zdf062]


def zdf063(msg, unpacked_bits):
    """ SSRZ Low Rate Clock C1 Default Resolution
    """
    [pr_code, c1, bits_zdf063] = prefix_coding(msg, 10, 0,
                                               unpacked_bits=unpacked_bits)
    # apply the resolution to the integer
    c1_res = c1 * 1e-2  # [mm/s]
    return [c1_res, bits_zdf063]


def zdf064(msg, unpacked_bits):
    """ SSRZ Low Rate Radial orbit Default Resolution
    """
    [pr_code, rad, bits_zdf064] = prefix_coding(msg, 10, 0,
                                                unpacked_bits=unpacked_bits)
    # apply the resolution to the integer
    rad_res = rad * 1e-4  # [m]
    return [rad_res, bits_zdf064]


def zdf065(msg, unpacked_bits):
    """ SSRZ Low Rate Along-Track Orbit Default Resolution
    """
    [pr_code, atr, bits_zdf065] = prefix_coding(msg, 8, 0,
                                                unpacked_bits=unpacked_bits)
    # apply the resolution to the integer
    atr_res = atr * 1e-4  # [m]
    return [atr_res, bits_zdf065]


def zdf066(msg, unpacked_bits):
    """ SSRZ Low Rate Cross-Track Orbit Default Resolution
    """
    [pr_code, ctr, bits_zdf066] = prefix_coding(msg, 8, 0,
                                                unpacked_bits=unpacked_bits)
    # apply the resolution to the integer
    ctr_res = ctr * 1e-4  # [m]
    return [ctr_res, bits_zdf066]


def zdf067(msg, unpacked_bits):
    """ SSRZ Low Rate Radial velocity Default Resolution
    """
    [pr_code, rad_vel,
     bits_zdf067] = prefix_coding(msg, 10, 0, unpacked_bits=unpacked_bits)
    # apply the resolution to the integer
    rad_vel_res = rad_vel * 1e-3  # [mm/s]
    return [rad_vel_res, bits_zdf067]


def zdf068(msg, unpacked_bits):
    """ SSRZ Low Rate Along-Track velocity Default Resolution
    """
    [pr_code, atr_vel,
     bits_zdf068] = prefix_coding(msg, 10, 0, unpacked_bits=unpacked_bits)
    # apply the resolution to the integer
    atr_vel_res = atr_vel * 1e-3  # [m]
    return [atr_vel_res, bits_zdf068]


def zdf069(msg, unpacked_bits):
    """ SSRZ Low Rate Cross-Track velocity Default Resolution
    """
    [pr_code, ctr_vel,
     bits_zdf069] = prefix_coding(msg, 10, 0, unpacked_bits=unpacked_bits)
    # apply the resolution to the integer
    ctr_vel_res = ctr_vel * 1e-3  # [m]
    return [ctr_vel_res, bits_zdf069]


def zdf070(msg, unpacked_bits):
    """ SSRZ Low Rate Code bias Default Resolution
    """
    [pr_code, cb,
     bits_zdf070] = prefix_coding(msg, 11, 0, unpacked_bits=unpacked_bits)
    # apply the resolution to the integer
    cb_res = cb * 1e-4  # [m]
    return [cb_res, bits_zdf070]


def zdf071(msg, unpacked_bits):
    """ SSRZ low rate phase bias cycle range
    """
    [pr_code, pb_cr, bits_zdf071] = prefix_coding(msg, 5, 0,
                                                  unpacked_bits=unpacked_bits)
    return [pb_cr, bits_zdf071]


def zdf072(msg, unpacked_bits):
    """ SSRZ low rate phase bias bitfield length
    """
    [pr_code, pb_bl, bits_zdf072] = prefix_coding(msg, 5, 0,
                                                  unpacked_bits=unpacked_bits)
    return [pb_bl, bits_zdf072]


def zdf073(msg, unpacked_bits):
    """ SSRZ low rate max number of continuity/overflow bits
    """
    [pr_code, ofl, bits_zdf073] = prefix_coding(msg, 5, 0,
                                                unpacked_bits=unpacked_bits)
    return [ofl, bits_zdf073]


def zdf074(msg, unpacked_bits):
    """ SSRZ global iono resolution
    """
    [pr_code, gi_res, bits_zdf074] = prefix_coding(msg, 4, 0,
                                                   unpacked_bits=unpacked_bits)
    resolution = 1e-3  # [TECU]
    return [gi_res*resolution, bits_zdf074]


def zdf075(msg, unpacked_bits):
    """ SSRZ global iono resolution
    """
    [pr_code, gr_iono_res,
     bits_zdf075] = prefix_coding(msg, 4, 0, unpacked_bits=unpacked_bits)
    resolution = 1e-3  # [TECU]
    return [gr_iono_res*resolution, bits_zdf075]


def zdf076(msg, unpacked_bits):
    """ SSRZ gridded troposphere scale factor default resolution
    """
    [pr_code, gr_tropo_res,
     bits_zdf076] = prefix_coding(msg, 8, 0, unpacked_bits=unpacked_bits)
    resolution = 1e-5  # [m]
    return [gr_tropo_res*resolution, bits_zdf076]


def zdf080(msg, unpacked_bits):
    """ SSRZ default resolution of the satellite-dependent
        regional ionosphere coefficients
    """
    [pr_code, value,
     bits_zdf080] = prefix_coding(msg, 4, 0, unpacked_bits=unpacked_bits)
    resolution = 1e-3  # [TECU]
    sat_reg_res = value * resolution
    return [sat_reg_res, bits_zdf080]


def zdf081(msg, unpacked_bits):
    """ SSRZ default resolution of Global VTEC Ionosphere coefficients
    """
    [pr_code, value, bits_zdf081] = prefix_coding(msg, 4, 0,
                                                  unpacked_bits=unpacked_bits)
    return [value, bits_zdf081]


def zdf082(msg, unpacked_bits):
    """ SSRZ default resolution of Global VTEC Ionosphere coefficients
    """
    [pr_code, value, bits_zdf082] = prefix_coding(msg, 8, 0,
                                                  unpacked_bits=unpacked_bits)
    return [value, bits_zdf082]


def zdf085(msg, unpacked_bits):
    """ SSRZ default resolution of the regional tropo scale factor
    """
    [pr_code, value, bits] = prefix_coding(msg, 8, 0,
                                           unpacked_bits=unpacked_bits)
    res = value * 1e-5
    return [res, bits]


def zdf086(msg, unpacked_bits):
    """ SSRZ default resolution of the regional tropo mapping improvements
    """
    [pr_code, value, bits] = prefix_coding(msg, 8, 0,
                                           unpacked_bits=unpacked_bits)
    res = value * 1e-6
    return [res, bits]


def zdf089(msg, unpacked_bits):
    """ SSRZ QIX Code Bias Default Resolution
    """
    [pr_code, value, bits] = prefix_coding(msg, 7, 0,
                                           unpacked_bits=unpacked_bits)
    qix_code_bias = value * 0.01  # [mm]
    return [qix_code_bias, bits]


def zdf090(msg, unpacked_bits):
    """ SSRZ QIX Phase Bias Default Resolution
    """
    [pr_code, value, bits] = prefix_coding(msg, 7, 0,
                                           unpacked_bits=unpacked_bits)
    qix_phase_bias = value * 0.01  # [mm]
    return [qix_phase_bias, bits]


def zdf091(msg, unpacked_bits):
    """ Number of grids
    """
    [code, value, bits] = prefix_coding(msg, 4, 0,
                                        unpacked_bits=unpacked_bits)
    return [value, bits]


def zdf092(msg, unpacked_bits):
    """ Order indicator of the grid point coordinate resolution
    """
    [code, value, bits] = prefix_coding(msg, 2, 0, unpacked_bits=unpacked_bits)
    return [value, bits]


def zdf093(msg, unpacked_bits):
    """ Integer part of the coordinate resolution (0-255)
    """
    bits2unpack = unpacked_bits + 'u8'
    int_res = bitstruct.unpack(bits2unpack, msg)[-1]
    return [int_res, bits2unpack]


def zdf094(msg, unpacked_bits):
    """ Number of chains
    Be aware, the number of chain is zdf094 + 1!
    """
    [code, value, bits] = prefix_coding(msg, 4, 0, unpacked_bits=unpacked_bits)
    return [value, bits]


def zdf095(msg, unpacked_bits):
    """ Number of grid points per chain
    Be aware, the number of points per chain is zdf095 + 1!
    """
    [code, value, bits] = prefix_coding(msg, 5, 0, unpacked_bits=unpacked_bits)
    return [value, bits]


def zdf096(msg, unpacked_bits):
    """ Grid Bin Size parameter
    """
    [code, value, bits] = prefix_coding(msg, 4, 0, unpacked_bits=unpacked_bits)
    return [value, bits]


def zdf097(msg, unpacked_bits):
    """ Use baseline flag. If 0-->baseline of the previous triangle,
                           if 1-->baseline of the current triangle
    """
    bits2unpack = unpacked_bits + 'u1'
    use_flag = bitstruct.unpack(bits2unpack, msg)[-1]
    return [use_flag, bits2unpack]


def zdf098(msg, unpacked_bits):
    """ Point Position Flag. The parameter defines if the third point
        of the triangle is located left(1) or right(0) with respect to
        the baseline pointing from base point 1 to base point 2
    """
    bits2unpack = unpacked_bits + 'u1'
    point_flag = bitstruct.unpack(bits2unpack, msg)[-1]
    return [point_flag, bits2unpack]


def zdf099(msg, unpacked_bits):
    """ Add baseline left flag. If 1 new baseline from left to third point.
        Be aware: the left depends on position of the third point. If the
        point is on the left w.r.t. the current baseline (r2-r1), then the
        new baseline is r1 - r3. Otherwise, if the point is on the right,
        the new baseline is r2 - r3. I.e. the direction depends on
        the point position flag.
    """
    bits2unpack = unpacked_bits + 'u1'
    left_flag = bitstruct.unpack(bits2unpack, msg)[-1]
    return [left_flag, bits2unpack]


def zdf100(msg, unpacked_bits):
    """ Add baseline right flag. If 1 new baseline from right to third point.
        Be aware: the right depends on position of the third point. If the
        point is on the left w.r.t. the current baseline (r2-r1), then the
        new baseline is r3 - r2. Otherwise, if the point is on the right,
        the new baseline is r3 - r1. I.e. the direction depends on
        the point position flag.
    """
    bits2unpack = unpacked_bits + 'u1'
    right_flag = bitstruct.unpack(bits2unpack, msg)[-1]
    return [right_flag, bits2unpack]


def zdf101(msg, unpacked_bits):
    """ Height flag. It indicates if grid point height data are provided (1)
        or not.
    """
    bits2unpack = unpacked_bits + 'u1'
    right_flag = bitstruct.unpack(bits2unpack, msg)[-1]
    return [right_flag, bits2unpack]


def zdf102(msg, unpacked_bits):
    """ Grid point height resolution if 0: no height info, if >0 height info
    is available with resolution zdf102 in meters
    """
    [code, value, bits] = prefix_coding(msg, 5, 0, unpacked_bits=unpacked_bits)
    return [value, bits]


def zdf103(msg, unpacked_bits):
    """ Gridded data predictor points. It indicates if the grid point indices
        for the gridded data predictor are provided (1) or nor (0)
    """
    bits2unpack = unpacked_bits + 'u1'
    grid_pred_p = bitstruct.unpack(bits2unpack, msg)[-1]
    return [grid_pred_p, bits2unpack]


def zdf104(msg, unpacked_bits):
    """ Predictor poiunt indicator. It enables the computation of a point index
        required for the gridded-data predictor
    """
    [code, value, bits] = prefix_coding(msg, 2, 0, unpacked_bits=unpacked_bits)
    return [value, bits]


def zdf105(msg, unpacked_bits):
    """ Grid ID
    """
    [grid_code, grid_id,
     bits_zdf105] = prefix_coding(msg, 4, 0, unpacked_bits=unpacked_bits)
    grid_id += 1  # According to SSRZ doc v1.1.2
    return [grid_id, bits_zdf105]


def zdf106(msg, unpacked_bits):
    """ Grid IOD. Issue of data of a grid with a certain grid id given by
        zdf105. zdf105 + zdf106 identify a valid grid.
    """
    bits2unpack = unpacked_bits + 'u3'
    iod = bitstruct.unpack(bits2unpack, msg)[-1]
    return [iod, bits2unpack]


def zdf107(msg, unpacked_bits):
    """ Troposphere Basic Component Bit Mask
    """
    n_bits = 16
    mask = []
    bits2unpack = unpacked_bits
    for ii in range(n_bits):
        bits2unpack += 'u1'
        mask.append(bitstruct.unpack(bits2unpack, msg)[-1])
    return [mask, bits2unpack]


def zdf108(msg, unpacked_bits):
    """ Separated Compressed Coefficient Blocks Flag
    """
    bits2unpack = unpacked_bits + 'u1'
    separated_by_height = bitstruct.unpack(bits2unpack, msg)[-1]
    return [separated_by_height, bits2unpack]


def zdf110(msg, unpacked_bits):
    """ Model ID
    """
    bits2unpack = unpacked_bits + 'u8'
    m_id = bitstruct.unpack(bits2unpack, msg)[-1]
    return [m_id, bits2unpack]


def zdf111(msg, unpacked_bits):
    """ Model version
    """
    bits2unpack = unpacked_bits + 'u8'
    m_vers = bitstruct.unpack(bits2unpack, msg)[-1]
    return [m_vers, bits2unpack]


def zdf112(msg, unpacked_bits):
    """ Number of integers model parameters
    """
    bits2unpack = unpacked_bits + 'u8'
    n_int_mp = bitstruct.unpack(bits2unpack, msg)[-1]
    return [n_int_mp, bits2unpack]


def zdf113(msg, unpacked_bits):
    """ Number of float model parameters
    """
    bits2unpack = unpacked_bits + 'u8'
    n_float_mp = bitstruct.unpack(bits2unpack, msg)[-1]
    return [n_float_mp, bits2unpack]


def zdf114(msg, unpacked_bits):
    """ Integer model parameter
    """
    bits2unpack = unpacked_bits + 's32'
    int_mp = bitstruct.unpack(bits2unpack, msg)[-1]
    return [int_mp, bits2unpack]


def zdf115(msg, unpacked_bits):
    """ Float model parameter
    """
    bits2unpack = unpacked_bits + 'f32'
    float_mp = bitstruct.unpack(bits2unpack, msg)[-1]
    return [float_mp, bits2unpack]


def zdf116(msg, unpacked_bits, num_para):
    """ Coefficient order bit mask
    """
    bits2unpack = unpacked_bits
    bit_mask = []
    for ii in range(num_para):
        bits2unpack += 'u1'
        bit_mask.append(bitstruct.unpack(bits2unpack, msg)[-1])
    return [bit_mask, bits2unpack]


def zdf129(msg, unpacked_bits):
    """ SSRZ encoder type
    """
    [prefix_code, enc_type,
     zdf129_bits] = prefix_coding(msg, 2, 0, unpacked_bits=unpacked_bits)
    return [enc_type, zdf129_bits]


def zdf130(msg, unpacked_bits):
    """ Number of ionospheric layers.
    """
    [prefix_code, n,
     zdf130_bits] = prefix_coding(msg, 2, 0, unpacked_bits=unpacked_bits)
    n_iono_layers = n + 1
    return [n_iono_layers, zdf130_bits]


def zdf131(msg, unpacked_bits):
    """ Height of iono layer in km.
    """
    [prefix_code, hgt,
     zdf131_bits] = prefix_coding(msg, 10, 0, unpacked_bits=unpacked_bits)

    return [hgt, zdf131_bits]


def zdf132(msg, unpacked_bits):
    """ Spherical harmonic degree N_l.
    """
    [prefix_code, degree,
     zdf132_bits] = prefix_coding(msg, 6, 0, unpacked_bits=unpacked_bits)

    return [degree, zdf132_bits]


def zdf133(msg, unpacked_bits):
    """ Spherical harmonic order M_l.
    """
    [prefix_code, order,
     zdf133_bits] = prefix_coding(msg, 6, 0, unpacked_bits=unpacked_bits)

    return [order, zdf133_bits]


def zdf134(msg, unpacked_bits):
    """ Satellite-dependent global ionosphere correction flag
        It defines if the corrections are part of the low-rate message.
    """
    bits2unpack = unpacked_bits + 'u1'
    lr_iono = bitstruct.unpack(bits2unpack, msg)[-1]
    return [lr_iono, bits2unpack]


def zdf143(msg, unpacked_bits):
    """ Number of regions
    """
    [prefix_code, n, bits] = prefix_coding(msg, 4, 0,
                                           unpacked_bits=unpacked_bits)
    n += 1
    return [n, bits]


def zdf144(msg, unpacked_bits):
    """ Region ID
    """
    [prefix_code, reg_id, bits] = prefix_coding(msg, 4, 0,
                                                unpacked_bits=unpacked_bits)
    reg_id += 1
    return [reg_id, bits]


def zdf166(msg, unpacked_bits):
    """ Maximum horizontal order M_hor
    """
    [prefix_code, order, bits] = prefix_coding(msg, 2, 0,
                                               unpacked_bits=unpacked_bits)
    return [order, bits]


def zdf167(msg, unpacked_bits):
    """ Maximum vertical order M_hgt
    """
    [prefix_code, order, bits] = prefix_coding(msg, 2, 0,
                                               unpacked_bits=unpacked_bits)
    return [order, bits]


def zdf168(msg, unpacked_bits):
    """ Latitude of the regional tropo ground point origin (+- 90°)
    """
    bits2unpack = unpacked_bits + 's11'
    lat = bitstruct.unpack(bits2unpack, msg)[-1] * 0.1  # [deg]
    return [lat, bits2unpack]


def zdf169(msg, unpacked_bits):
    """ Longitude of the regional tropo ground point origin (+- 180°)
    """
    bits2unpack = unpacked_bits + 's12'
    lon = bitstruct.unpack(bits2unpack, msg)[-1] * 0.1  # [deg]
    return [lon, bits2unpack]


def zdf170(msg, unpacked_bits):
    """ Height of the regional tropo ground point origin (+- 8191)
    """
    bits2unpack = unpacked_bits + 's14'
    lat = bitstruct.unpack(bits2unpack, msg)[-1] * 1.0  # [m]
    return [lat, bits2unpack]


def zdf171(msg, unpacked_bits):
    """Maximum elevation for mapping improvement. Max elevation at which
    the mapping improvements should be applied.
    """
    bits2unpack = unpacked_bits + 'u6'
    max_el = bitstruct.unpack(bits2unpack, msg)[-1] * 1.0  # [deg]
    return [max_el, bits2unpack]


def zdf172(msg, unpacked_bits):
    """ Coverage dependent factor. It is used to normalize the arguments
    of the Chebyshev polynomials.
    """
    [prefix_code, out, bits] = prefix_coding(msg, 8, 0,
                                             unpacked_bits=unpacked_bits)
    res = 10.0e3  # [m]
    bias = 100.0e3  # [m]
    d_rt = bias + out * res  # [m]
    return [d_rt, bits]


def zdf173(msg, unpacked_bits):
    """ Vertical scale factor
    """
    [prefix_code, out, bits] = prefix_coding(msg, 8, 0,
                                             unpacked_bits=unpacked_bits)
    res = 1.0e1  # [m]
    bias = 1.0e2  # [m]
    h_rt = bias + out * res  # [m]
    return [h_rt, bits]


def zdf190(msg, unpacked_bits):
    """ It indicated the presence of the ssrz qix code bias
        rice block in ZM008.
    """
    bits2unpack = unpacked_bits + 'u1'
    qix_cb_flag = bitstruct.unpack(bits2unpack, msg)[-1]  # [deg]
    return [qix_cb_flag, bits2unpack]


def zdf191(msg, unpacked_bits):
    """ It indicated the presence of the ssrz qix phase bias
        rice block in ZM008.
    """
    bits2unpack = unpacked_bits + 'u1'
    qix_pb_flag = bitstruct.unpack(bits2unpack, msg)[-1]  # [deg]
    return [qix_pb_flag, bits2unpack]


def zdf199(msg, unpacked_bits):
    """ Length of the SSRZ BE IOD bit field
    """
    # length of the SSRZ BE IOD field zdf199. It is the number of bits
    # nb_iod_gnss used to represent IOD values in ZDF033
    bits2unpack = unpacked_bits + 'u4'
    nb_iod_gnss = bitstruct.unpack(bits2unpack, msg)[-1]
    return [nb_iod_gnss, bits2unpack]


def zdf200(msg, unpacked_bits):
    """ SSRZ IOD Tag
    """
    [prefix_code, iod_tag,
     zdf200_bits] = prefix_coding(msg, 2, 0, unpacked_bits=unpacked_bits)
    return [iod_tag, zdf200_bits]


def zdf201(msg, unpacked_bits):
    """ SSRZ BE IOD Tag GPS
    """
    [prefix_code, iod_tag,
     zdf201_bits] = prefix_coding(msg, 2, 0, unpacked_bits=unpacked_bits)

    return [iod_tag, zdf201_bits]


def zdf202(msg, unpacked_bits):
    """ SSRZ BE IOD Tag GLONASS
    """
    [prefix_code, iod_tag,
     zdf202_bits] = prefix_coding(msg, 2, 0, unpacked_bits=unpacked_bits)
    return [iod_tag, zdf202_bits]


def zdf203(msg, unpacked_bits):
    """ SSRZ BE IOD Tag Galileo
    """
    [prefix_code, iod_tag,
     zdf203_bits] = prefix_coding(msg, 2, 0, unpacked_bits=unpacked_bits)
    return [iod_tag, zdf203_bits]


def zdf204(msg, unpacked_bits):
    """ SSRZ BE IOD Tag QZSS
    """
    [prefix_code, iod_tag,
     zdf204_bits] = prefix_coding(msg, 2, 0, unpacked_bits=unpacked_bits)
    return [iod_tag, zdf204_bits]


def zdf205(msg, unpacked_bits):
    """ SSRZ BE IOD Tag SBAS
    """
    [prefix_code, iod_tag,
     zdf205_bits] = prefix_coding(msg, 2, 0, unpacked_bits=unpacked_bits)
    return [iod_tag, zdf205_bits]


def zdf206(msg, unpacked_bits):
    """ SSRZ BE IOD Tag BDS
    """
    [prefix_code, iod_tag,
     zdf206_bits] = prefix_coding(msg, 2, 0, unpacked_bits=unpacked_bits)
    return [iod_tag, zdf206_bits]


def zdf207(msg, unpacked_bits):
    """ SSRZ BE IOD Tag IRNSS
    """
    [prefix_code, iod_tag,
     zdf207_bits] = prefix_coding(msg, 2, 0, unpacked_bits=unpacked_bits)

    return [iod_tag, zdf207_bits]


def zdf230(msg, unpacked_bits):
    """ SSRZ Time tag definition
    """
    [prefix_code, time_tag,
     zdf230_bits] = prefix_coding(msg, 2, 1, unpacked_bits=unpacked_bits)
    return [time_tag, zdf230_bits]


def zdf231(msg, unpacked_bits):
    """ SSRZ 1h-30 seconds time tag
    """
    bits2unpack = unpacked_bits + 'u7'
    time_tag = bitstruct.unpack(bits2unpack, msg)[-1] * 30  # [s]
    return [time_tag, unpacked_bits]


def zdf232(msg, unpacked_bits):
    """ SSRZ 1h-5 seconds time tag
    """
    bits2unpack = unpacked_bits + 'u10'
    time_tag = bitstruct.unpack(bits2unpack, msg)[-1] * 5  # [s]
    return [time_tag, unpacked_bits]


def zdf233(msg, unpacked_bits):
    """ SSRZ 1h-1 second time tag
    """
    bits2unpack = unpacked_bits + 'u12'
    time_tag = bitstruct.unpack(bits2unpack, msg)[-1] * 1  # [s]
    return [time_tag, unpacked_bits]


def zdf234(msg, unpacked_bits):
    """ SSRZ 1day-1 second time tag
    """
    bits2unpack = unpacked_bits + 'u17'
    time_tag = bitstruct.unpack(bits2unpack, msg)[-1] * 1  # [s]
    return [time_tag, unpacked_bits]


def zdf301(msg, unpacked_bits):
    """ Scale factor indicator
    """
    [scale_factor_code, scale_factor,
     zdf301_bits] = prefix_coding(msg, 2, 0, unpacked_bits=unpacked_bits)
    return [scale_factor, zdf301_bits]


def zdf302(msg, unpacked_bits):
    """ Bin Size indicator
    """
    [bin_size_code, bin_size,
     zdf302_bits] = prefix_coding(msg, 2, 0, unpacked_bits=unpacked_bits)
    return [bin_size, zdf302_bits]


def zdf303(msg, unpacked_bits):
    """ Sign bit
    """
    bits2unpack = unpacked_bits + 'u1'
    test = bitstruct.unpack(bits2unpack, msg)[-1]
    if test == 0:
        s = 1
    elif test == 1:
        s = -1
    return [s, bits2unpack]


def zdf304(msg, unpacked_bits):
    """ Rice quotient
    """
    [rice_q_code, rice_q,
     zdf304_bits] = prefix_coding(msg, 1, 0, unpacked_bits=unpacked_bits)

    return [rice_q, zdf304_bits]


def zdf305(msg, unpacked_bits, bin_size):
    """ Rice remainder
    """
    if bin_size != 0:
        bits2unpack = unpacked_bits + 'u' + str(int(bin_size))
        rice_rem = bitstruct.unpack(bits2unpack, msg)[-1]
    else:
        bits2unpack = unpacked_bits
        rice_rem = 0
    return [rice_rem, bits2unpack]


def zdf309(msg, unpacked_bits, sv_bit_mask, gnss_be_iod_length):
    """ Issue of data of the broadcasted ephemeris. It takes in input a gnss
        and its sat_list based on zdf018 and provides the iod considering
        zdf201-zdf216
    """
    bits2unpack = unpacked_bits
    iod_list = []
    for ii in range(len(sv_bit_mask)):
        sat_list = sv_bit_mask[ii]
        bits_length = gnss_be_iod_length[ii]
        if len(sat_list) > 0:
            iod_list.append([])
        else:
            iod_list.append([])
            continue
        for sat in sat_list:
            bits2unpack += 'u' + str(bits_length)
            iod = int(bitstruct.unpack(bits2unpack, msg)[-1])
            iod_list[ii] = np.append(iod_list[ii], iod)

    return [iod_list, bits2unpack]


def zdf310(msg, unpacked_bits):
    """ This flag indicates if the phase bias value is valid.
    """
    bits2unpack = unpacked_bits + 'u1'
    pb_flag = bitstruct.unpack(bits2unpack, msg)[-1]
    return [pb_flag, bits2unpack]


def zdf311(msg, unpacked_bits, n_pb_bits):
    """ Phase bias indicator.
    """
    bits2unpack = unpacked_bits + 'u%d' % n_pb_bits
    pb_bias_indicator = bitstruct.unpack(bits2unpack, msg)[-1]
    return [pb_bias_indicator, bits2unpack]


def zdf312(msg, unpacked_bits):
    """ Overflow/Discontinuity Indicator
    """
    [code, value, bits] = prefix_coding(msg, 1, 0, unpacked_bits=unpacked_bits)
    return [value, bits]


def zdf315(msg, unpacked_bits):
    """ SSRZ predictor flag: it it indicates if the gridded data are encoded
        with the gridded-data predictor and how the resulting SSRZ Rice Blocks
        have to read
    """
    bits2unpack = unpacked_bits + 'u1'
    pred_flag = bitstruct.unpack(bits2unpack, msg)[-1]
    return [pred_flag, bits2unpack]


def zdf320(msg, unpacked_bits):
    """ SSRZ QIX Bias metadata flag. It indicates the presence of QIX metadata.
    """
    bits2unpack = unpacked_bits + 'u1'
    qix_md_flag = bitstruct.unpack(bits2unpack, msg)[-1]
    return [qix_md_flag, bits2unpack]


def zdf330(msg, unpacked_bits):
    """ VTEC flag. It indicates if the message contains valid VTEC.
    """
    bits2unpack = unpacked_bits + 'u1'
    vtec_flag = bitstruct.unpack(bits2unpack, msg)[-1]
    return [vtec_flag, bits2unpack]


def zdf331(msg, unpacked_bits):
    """ SSRZ Global VTEC bin size indicator
    """
    [code, value, bits] = prefix_coding(msg, 3, 1, 0,
                                        unpacked_bits=unpacked_bits)
    return [value, bits]
