"""Nodes

This script implements a class for each communication entity: BS, RIS, and UE.

Authors: @victorcroisfelt, @lostinafro
Date: 28/07/2022
"""

import numpy as np
from scipy.constants import speed_of_light

class Node:
    """Creates a communication entity.

    Arguments
    ---------
        n : int
            Number of nodes.

        pos : ndarray of shape (n, 3) or (3,) if n = 1
          Position of the node in rectangular coordinates.

        gain : float
            Antenna gain of the node.
    """

    def __init__(
            self,
            n: int,
            pos: np.ndarray,
            gain: float or np.ndarray = None
    ):

        # Control on INPUT
        if pos.shape != (n, 3) and pos.shape != (3, ):
            raise ValueError(f'Illegal positioning: for Node, pos.shape must be ({n}, 3), instead it is {pos.shape}')

        # Set attributes
        self.n = n
        self.pos = pos
        self.gain = gain


class BS(Node):
    """Base station.

    Arguments
    ---------
        pos : ndarray of shape (3,)
            Position of the BS in rectangular coordinates.

        gain : float
            BS antenna gain. Default is 5.00 dB.
    """

    def __init__(
            self,
            pos: np.ndarray = None,
            gain: float = None
    ):

        if gain is None:
            gain = 10**(5/10)

        super().__init__(1, pos, gain)

        self.distance = np.linalg.norm(self.pos)
        self.angle = np.abs(np.arctan2(self.pos[0], self.pos[1]))

    def __repr__(self):
        return f'BS-{self.n}'


class RIS(Node):
    """Reflective intelligent surface.

    Arguments
    ---------
        pos : ndarray of shape (3,)
            Position of the RIS in rectangular coordinates.

        num_els_ver : int
            Number of elements along z-axis.

        num_els_hor : int
            Number of elements along x-axis.

        wavelength : float
            Wavelength in meters.

        size_el : float
            Size of each element. Default: wavelength/2

        num_configs : int
            Number of configurations.
    """

    def __init__(
            self,
            pos: np.ndarray = None,
            num_els_ver: int = None,
            num_els_hor: int = None,
            wavelength: float = None,
            size_el: float = None,
    ):

        # Default values
        if pos is None:
            pos = np.array([0, 0, 0])
        if num_els_ver is None:
            num_els_ver = 10
        if num_els_hor is None:
            num_els_hor = 10
        if size_el is None:
            size_el = wavelength/2

        super().__init__(1, pos, 0.0)

        self.num_els_ver = num_els_ver  # vertical number of elements
        self.num_els_hor = num_els_hor  # horizontal number of elements
        self.num_els = num_els_ver * num_els_hor  # total number of elements
        self.size_el = size_el  # size of each element

        # Compute RIS sizes
        self.size_z = num_els_ver * self.size_el  # vertical size [m]
        self.size_x = num_els_hor * self.size_el  # horizontal size [m]
        self.area = self.size_z * self.size_x   # area [m^2]

    def __repr__(self):
        return f'RIS-{self.n}'


class UE(Node):
    """User.

    Arguments
    ---------
        n : int
            Number of UEs.

        pos : ndarray of shape (n, 3)
            Position of the UEs in rectangular coordinates.

        gain : float
            BS antenna gain. Default is 5.00 dB.
    """

    def __init__(
            self,
            n: int,
            pos: np.ndarray,
            gain: float = None
    ):

        if gain is None:
            gain = 10**(5/10)

        # Init parent class
        super().__init__(n, pos, gain)

        self.distances = np.linalg.norm(self.pos, axis=-1)
        self.angles = np.abs(np.arctan2(self.pos[:, 0], self.pos[:, 1]))

    def __repr__(self):
        return f'UE-{self.n}'


