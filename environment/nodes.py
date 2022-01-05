#!/usr/bin/env python3
# filename "nodes.py"

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light


class Node:
    """Node definition. This class is called by BS, UE, and RIS classes to defining common parameters."""

    def __init__(self,
                 n: int,
                 pos: np.ndarray,
                 ant: int or np.ndarray = None,
                 gain: float or np.ndarray = None,
                 max_pow: float or np.ndarray = None,
                 noise_power: float or np.ndarray = None):
        """Constructor of the Node class.

        Parameters
        ----------
        :param n: number of nodes.
        :param pos: np.ndarray n x 3 representing the x,y,z cartesian coordinates of each node.
        :param ant: np.ndarray n x 1 of int > 0, number of antennas of the node.
        :param gain: float representing the antenna gain of the node.
        :param max_pow: float representing the max power available on transmission.
        """
        # Control on INPUT
        if pos.shape != (n, 3):
            raise ValueError(f'Illegal positioning: for Node, pos.shape must be ({n},3), instead it is {pos.shape}')
        # Input reformat
        if not isinstance(ant, np.ndarray):
            ant = np.array([ant] * n)
        if not isinstance(gain,  np.ndarray):
            gain = np.array([gain] * n)
        if not isinstance(max_pow,  np.ndarray):
            max_pow = np.array([max_pow] * n)
        if not isinstance(noise_power,  np.ndarray):
            noise_power = np.array([noise_power] * n)       
        # Set attributes
        self.n = n
        self.pos = pos
        self.ant = ant
        self.gain = gain
        self.max_pow = max_pow       
        # Noise definition
        self.noise = RxNoise(dBm=noise_power)    


class BS(Node):
    """BS definition."""

    def __init__(self, n: int, pos: np.ndarray, ant: int, gain: float, max_pow: float, noise_power: float):
        """BS class for simulation purpose.

        :param n: int, number of BS
        :param pos: ndarray n x 3, the x,y cartesian coordinates of the node.
        :param ant: int > 0, number of antennas of the node. Default value is 1.
        :param gain : float [dBi] representing antenna gain of the node. Default is 5.00 dB
        :param max_pow: float [dBm], max power available on transmission. Default is 43 dBm.
        :param noise_power: float [dBm], noise power at RF chain
        """
        # Default values
        if ant is None:
            ant = 1
        if gain is None:
            gain = 10**(5/10)
        if max_pow is None:
            max_pow = 43
        if noise_power is None:
            noise_power = -92.5  # [dBm] TODO: update to a reasonable value

        # Init parent class
        super(BS, self).__init__(n, pos, ant, gain, max_pow, noise_power)

        # Compute incidence angle (always positive)
        self.incidence_angle = np.abs(np.arctan(self.pos[0][0] / self.pos[0][1]))

    def __repr__(self):
        return f'BS-{self.n}'


class UE(Node):
    """User definition."""

    def __init__(self, 
                 n: int, 
                 pos: np.ndarray,
                 ant: int or np.ndarray = None, 
                 gain: float or np.ndarray = None,
                 max_pow: float or np.ndarray = None, 
                 noise_power: float or np.ndarray = None):
        """User class for simulation purpose.

        :param pos: ndarray n x 3, the x,y,z cartesian coordinates of the node.
        :param ant: int > 0, number of antennas of the node. Default value is 1.
        :param gain: float [dBi], antenna gains of the node. Default value is 5.00 dB.
        :param max_pow: float [dBm], max power available on transmission in dBm.
                Default value is 24 dBm.
        """
        # Default values
        if ant is None:
            ant = 1
        if gain is None:
            gain = 10**(5/10)
        if max_pow is None:
            max_pow = 24
        if noise_power is None:
            noise_power = -92.5  # [dBm] TODO: update to a reasonable value

        # Init parent class
        super(UE, self).__init__(n, pos, ant, gain, max_pow, noise_power)

        # Compute incidence angle (always positive)
        self.incidence_angle = np.abs(np.arctan(self.pos[:, 0] / self.pos[:, 1]))

    def __repr__(self):
        return f'UE-{self.n}'


class RIS(Node):
    """RIS definition. A class that defines a reflective intelligent surface (RIS)."""

    def __init__(self,
                 n: int,
                 pos: np.ndarray,
                 Na: int = None,
                 Nb: int = None,
                 S: int = None,
                 wavelength: float = None):
        """ Constructor.

        :param n: int > 0, number of RISs
        :param pos: ndarray n x 3, the x,y,z cartesian coordinates of each node.
        :param Na: int > 0, number of elements along vertical dimension
        :param Nb: int > 0, number of elements along horizontal dimension
        :param S: int > 0, number of configurations
        """
        # Default values
        if Na is None:
            Na = 4
        if Nb is None:
            Nb = 4
        if S is None:
            S = 4
        if wavelength is None:
            wavelength = 0.1
        # Instance variables
        self.num_els_v = Na  # vertical number of elements
        self.num_els_h = Nb  # horizontal number of elements
        self.num_els = Na * Nb  # total number of elements
        self.num_configs = S  # number of configurations

        # Initialize the parent, considering that the antenna gain of the ris is 0.0,
        # max_pow and noise_power are -np.inf,
        # the number of antenna is the number or RIS elements
        super(RIS, self).__init__(n, pos, self.num_els, 0.0, -np.inf, -np.inf)
        # In this way every ris instantiated is equal to the others

        # Store index of elements considering total number
        self.enumeration_els = np.arange(self.num_els)

        # Each antenna element is a square of size wavelength/4
        self.size_el = wavelength / 4

        # Compute RIS sizes
        self.size_h = Nb * self.size_el  # horizontal size [m]
        self.size_v = Na * self.size_el  # vertical size [m]
        self.area = self.size_h * self.size_v   # area [m^2]

        # Organizing elements over the RIS
        self.id_els = self.indexing_els()
        self.pos_els = self.positioning_els()

        # Configure RIS
        self.angular_resolution = self.get_angular_resolution()
        self.set_configs = self.configure()

    def indexing_els(self):
        """
        Define an array of tuples where each represents the id of an element.

        Returns
        -------
        id_els : ndarray of tuples and shape (self.num_els)
            Each ndarray entry has a tuple (id_v, id_h), which indexes the elements arranged in a planar array. Vertical
            index is given as id_v, while horizontal index is id_h.

        Example
        -------
        For a Na = 3 x Nb = 3 RIS, the elements are indexed as follows:

                                                (2,0) -- (2,1) -- (2,2)
                                                (1,0) -- (1,1) -- (1,2)
                                                (0,0) -- (0,1) -- (0,2),

        the corresponding id_els should contain:

                        id_els = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)].

        While the respective enumeration of the elements is:

                                                    6 -- 7 -- 8
                                                    3 -- 4 -- 5
                                                    0 -- 1 -- 2,

        the enumeration is stored at:

                                            self.els = range(num_els).

        Therefore, id_els and self.els are two different index methods for the elements. The former is used to
        characterize the geometry features of each element, while the latter is used for storage purposes.
        """
        # Get vertical ids
        id_v = self.enumeration_els // self.num_els_v

        # Get horizontal ids
        id_h = np.mod(self.enumeration_els, self.num_els_h)

        # Get array of tuples with complete id
        id_els = [(id_v[el], id_h[el]) for el in self.enumeration_els]

        return id_els

    def positioning_els(self):
        """
        Compute position of each element in the planar array.

        Returns
        -------

        """
        # Compute offsets
        offset_x = (self.num_els_h - 1) * self.size_el / 2
        offset_z = (self.num_els_v - 1) * self.size_el / 2

        # Prepare to store the 3D position vector of each element
        pos_els = np.zeros((self.num_els, 3))

        # Go through all elements
        for el in self.enumeration_els:
            pos_els[el, 0] = (self.id_els[el][1] * self.size_el) - offset_x
            pos_els[el, 2] = (self.id_els[el][0] * self.size_el) - offset_z

        return pos_els

    def plot(self):
        """
        Plot RIS along with the index of each element.

        Returns
        -------
        None.

        """
        fig, ax = plt.subplots()

        # Go through all elements
        for el in self.enumeration_els:
            ax.plot(self.pos_els[el, 0], self.pos_els[el, 2], 'x', color='black')
            ax.text(self.pos_els[el, 0] - 0.003, self.pos_els[el, 2] - 0.0075, str(self.id_els[el]))

        # Plot origin
        ax.plot(0, 0, '.', color='black')

        ax.set_xlim([-0.05, 0.05])
        ax.set_ylim([-0.05, 0.05])

        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")

        plt.show()

    def get_angular_resolution(self):
        """
        Get RIS angular resolution. Observation space is ever considered to be pi/2 (half-plane) given our system setup.

        Returns
        -------
        angular_resolution : float
            RIS angular resolution in radians given the number of configurations and uniform division of the observation
            space.

        Example
        -------
        For S = 4, angular_resolution evaluates to pi/8.

        """
        angular_resolution = (np.pi / 2 - 0) / self.num_configs

        return angular_resolution

    def configure(self):
        """
        Define set of S configurations offered by the RIS.

        Returns
        -------
        set_configs : ndarray of shape (self.num_configs-1,)
            Discrete set of configurations containing all possible angles (theta_s) in radians in which the RIS can
            steer the incoming signal.

        Example
        -------
        For S = 4, angular resolution is pi/8. The set of configurations evaluates to:

                                 set_configs = [pi/8, pi/4, 3pi/8]

        0 and pi/2 are not included. Note that the observation space is divided into 4 zones.
        """
        set_configs = np.arange(0, np.pi / 2, self.angular_resolution)

        return set_configs

    def __repr__(self):
        return f'RIS-{self.n}'



class RxNoise:
    """Represent the noise value at the physical receiver
    # TODO: match with ambient noise and noise figure
    """

    def __init__(self, linear=None, dB=None, dBm: np.ndarray = np.array([-92.5])):
        if (linear is None) and (dB is None):
            self.dBm = dBm
            self.dB = dBm - 30
            self.linear = 10 ** (self.dB / 10)
        elif (linear is not None) and (dB is None):
            self.linear = linear
            if self.linear != 0:
                self.dB = 10 * np.log10(self.linear)
                self.dBm = 10 * np.log10(self.linear * 1e3)
            else:
                self.dB = -np.inf
                self.dBm = -np.inf
        else:
            self.dB = dB
            self.dBm = dB + 30
            self.linear = 10 ** (self.dB / 10)

    def __repr__(self):
        return (f'noise({self.linear:.3e}, '
                f'dB={self.dB:.1f}, dBm={self.dBm:.1f})')
