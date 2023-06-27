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
   ****************************************************************************
   Group of classes to create SSRZ-SSR objects.
   Input : decoded ssrz-ssr message
   Output: object oriented ssr message
   ****************************************************************************
   Description:
   the SSRObjects class is initialized with objects: epochs, metadata and
   ssr corrections. They are organized for epochs. The corrections have
   as objects the different messages as defined in ssrz_messages_temp.py
   (lr, hr, gvi, gsi, rsi, gri, gt, rt, grt).
   This SSRObjects class is updated for each epoch there is a new
   valid high-rate message adding a new epoch to the objects.
   The purpose of this class is mainly related to the possibility to
   post-process the data.
   ****************************************************************************
"""
import numpy as np


class SSRObjects:
    def __init__(self, epochs=None, metadata=None, corrections=None):
        if epochs is None:
            self.epochs = []
        else:
            self.epochs = epochs

        if corrections is None:
            self.corrections = []
        else:
            self.corrections = corrections

        if metadata is None:
            self.metadata = []
        else:
            self.metadata = metadata

    def __repr__(self):
        return "".join(['SSR objects: epochs, metadata, corrections. ',
                'Corrections are organized per epoch and ',
                'has the following objects: ',
                'lr, hr, gvi, gsi, rsi, gri, grt, rt. ',
                'The objects are organized per GNSS, per satellite, ',
                'per signal when applicable.'])

    def add_epoch(self, epoch, dec_md, dec_corr):
        if epoch not in self.epochs:
            self.epochs = np.append(self.epochs, epoch)
            self.metadata = np.append(self.metadata, dec_md)
            self.corrections = np.append(self.corrections, dec_corr)

    def get_closest_ssr(self, ssr, epoch):
        # find the index among the iono epochs
        ssr_index = np.where(np.abs(ssr.epochs - epoch) ==
                             np.nanmin(np.abs(ssr.epochs - epoch)))[0]
        # consider the index in the common list of epochs
        if len(ssr_index) == 1:
            index = ssr_index[0]
        else:
            index = ssr_index[0][0]
        return [ssr.metadata, ssr.corrections[index[0]]]
