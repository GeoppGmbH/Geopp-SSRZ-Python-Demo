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
"""
import numpy as np


def do_bil_int(N, x1, x2, q1, q2):
    """ Bilinear Interpolation
    Input:
        - N: vector of four grid points,
             rectangle which contains the query point (known values of the fun)
        - x1: vector of two components, it defines the interval in the first
              direction
        - x2: vector of two components, it defines the interval in the second
              direction
        - q1,q2: desidered point in which calculate the value of the function
    Output:
        - f: interpolated vaule of the function required
    Remark:
        N is considered as follow:
            N[0] = f(x1[0],x2[0])
            N[1] = f(x1[0],x2[1])
            N[2] = f(x1[1],x2[1])
            N[3] = f(x1[1],x2[0])
    Reference:
        DEPARTMENT OF DEFENSE WORLD GEODETIC SYSTEM,1984
        ITS DEFINITION AND RELATIONSHIPS WITHLOCAL GEODETIC SYSTEMS
    Example:
    >>> '%.2f' % bil_int([1, 2, 3, 4], [5, 6], [7, 8], 5.5, 7.5)
    '2.50'
"""
    # Coefficient for the bilinear interpolation
    a0 = N[0]
    a1 = N[1] - N[0]
    a2 = N[3] - N[0]
    a3 = N[0] + N[2] - N[1] - N[3]
    X = (q2 - x2[0]) / (x2[1] - x2[0])
    Y = (q1 - x1[0]) / (x1[1] - x1[0])
    # Interpolation
    f = a0 + a1 * X + a2 * Y + a3 * X * Y
    return f


class Interpolator2D:
    def __init__(self, grid_data, grid_values, use_geodetic_coordinates=False):
        """ Class to do 2d interpolation.
        Various interpolating methods possible.
        Nomenclature in this class:
        n        Number of grid points
        m        Number of interpolation points (where the grid is evaluated)
        l        Number of different values to be interpolated l,
                 e.g. l=2 for zhd,zwd
        Input arguments:
        grid_data                 n times 2 array with 2D-coordinates
        grid_values               n times l array with
        use_geodetic_coordinates  if true, the longitude will be scaled
                                  accoriding to the latitude of the
                                  respective coordinate when
                                  calculating distances.
        """
        self.grid = grid_data
        self.values = grid_values
        self.n = self.grid.shape[0]
        self.ll = self.values.shape[1]
        self.mm = 0  # initialized, once an iterating function is called
        self.distances = np.array([])  # init by calling compute_distances
        self.coords = np.array([])
        self.results = np.array([])
        self.use_geodetic_coordinates = use_geodetic_coordinates

    def compute_distances(self):
        """
        Helper function to nompute the distances between the grid points.
        These are used by some of the interpolators as weights.

        >>> test = Interpolator2D(np.array([[1,0],[0,7]]),np.array([[1],[1]]))
        >>> test.coords = np.array([[0,0],[4,4]]) # this should normally be set
            by the interpolation methods.
        >>> test.compute_distances()
        >>> test.distances
        array([[ 1.,  7.],
               [ 5.,  5.]])

        Input arguments:
            coords: m times 2 array of coordinates where interpolation
                    should be performed
        """
        if self.use_geodetic_coordinates:
            e_x = [1.0, 0.0]
            self.distances = []
            for ii in range(len(self.coords)):
                c = self.coords[ii]
                self.distances.append([])
                for jj in range(len(self.grid)):
                    self.distances[ii].append([])
                    g = self.grid[jj]
                    corr = [0.0, np.cos(np.pi / 180. * c[0])]
                    proj = np.dot([e_x, corr], g - c)
                    square = np.dot(proj, proj)
                    self.distances[ii][jj] = np.array([np.sqrt(square)])
        else:
            self.distances = np.array([[np.sqrt(np.dot(g - c, g - c))
                                        for g in self.grid]
                                       for c in self.coords])

    def IDW(self, coords):
        self.coords = coords
        self.compute_distances()
        weights = 1.0 / np.power(self.distances, 2)
        # check if closest grid point has valid data
        maxWeightIndex = np.argmax(weights)
        if np.isinf(weights).any():
            # The interpolated value is one of the grid points
            # Here, we assume that the point to interpolate is one
            self.results = self.values[maxWeightIndex][0]
        else:
            self.results = (np.nansum(np.transpose(weights) * self.values) /
                            np.nansum(weights))
