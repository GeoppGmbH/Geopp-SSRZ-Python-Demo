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
    the module contains classes to temporary store SSRZ metadata and
    corrections.
    The 'Metadata' and 'Corrections' classes initialize their objects as either
    None or empty arrays (depending on the content). Each class has an 'update'
    method to update the desired object for the current epoch.
    ***************************************************************************
"""


class Metadata:
    """ SSRZ Metadata class
    """

    def __init__(self):
        # satellite goup definition
        self.sat_gr = None
        # metadata group
        self.md_gr = None
        # grid group
        self.grid_gr = []

    def __repr__(self):
        return ('SSRZ Metadata objects: sat_gr, md_gr, grid_gr')

    def update(self, sat_gr=None, md_gr=None, grid_gr=None):
        if sat_gr is not None:
            self.sat_gr = sat_gr
        if md_gr is not None:
            self.md_gr = md_gr
        if grid_gr is not None:
            if len(self.grid_gr) > 0:
                if ((grid_gr.grid_block.grid_blk_list[0].id == self.grid_gr[-1].grid_block.grid_blk_list[0].id) &
                        (grid_gr.grid_block.grid_blk_list[0].iod == self.grid_gr[-1].grid_block.grid_blk_list[0].iod)):
                    pass
                else:
                    self.grid_gr.append(grid_gr)
            else:
                self.grid_gr.append(grid_gr)


class Corrections:
    def __init__(self, md=None):
        if md is None:
            self.lr = None
            self.hr = None
            # regional satellite iono
            self.rsi = None
            # gridded iono
            self.gri = None
        else:
            # low rate
            # create structure with timing blocks and
            # satellite grups based on md
            self.lr = []
            self.rsi = []
            self.gri = []
            n_timing = md.md_gr.md_block.lr_md_block.n_timing
            for ii in range(n_timing):
                self.lr.append([])
                # regional satellite iono
                self.rsi.append([])
                # gridded iono
                self.gri.append([])
                # satellite groups
                n_g_lr = md.md_gr.md_block.lr_md_block.n_g_lr
                for jj in range(n_g_lr):
                    self.lr[ii].append([])
                    self.rsi[ii].append([])
                    self.gri[ii].append([])
            # high rate
            # create structure with timing blocks and
            # satellite grups based on md
            self.hr = []
            n_timing = md.md_gr.md_block.hr_md_block.n_timing
            for ii in range(n_timing):
                self.hr.append([])
                # satellite groups
                n_g_hr = md.md_gr.md_block.hr_md_block.n_g_hr
                for jj in range(n_g_hr):
                    self.hr[ii].append([])
        # global vtec iono
        self.gvi = None

        # gridded tropo
        self.grt = []
        # regional tropo
        self.rt = None
        # qix bias
        self.qix = None
        # time tag
        self.tt = None

    def update(self, lr=None, hr=None, gvi=None, rsi=None, gri=None, grt=None,
               rt=None, qix=None, tt=None, t_blk=None, sat_gr=None,
               grid_number=None):
        if lr is not None:
            self.lr[t_blk][sat_gr] = lr
        if hr is not None:
            self.hr[t_blk][sat_gr] = hr
        if gvi is not None:
            self.gvi = gvi
        if rsi is not None:
            self.rsi[t_blk][sat_gr] = rsi
        if gri is not None:
            if grid_number is None:
                self.gri[t_blk][sat_gr] = gri
            else:
                try:
                    self.gri[t_blk][sat_gr][grid_number] = gri
                except IndexError:
                    print("Grid index error")
        if grt is not None:
            if grid_number is None:
                self.grt = grt
            else:
                self.grt[grid_number] = grt
        if rt is not None:
            self.rt = rt
        if qix is not None:
            self.qix = qix
        if tt is not None:
            self.tt = tt

    def __repr__(self):
        return "".join(['SSRZ Corrections objects: lr, hr, gvi, rsi, grt, gri, rt, ',
                'qix, tt.',
                'lr and hr are organized in timing and satellite ',
                'groups blocks,',
                'e.g. for timing_block = 0 and sat_group_block = 1, the ',
                'low rate msg can be retrieved as lr[0][1].'])
