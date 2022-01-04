#!/usr/bin/env python3
# filename "nodes.py"

import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import speed_of_light

class Node:
    """Node definition. This class is called by BS, UE, and RIS classes to defining common parameters."""

    def __init__(self, label: str, pos: np.ndarray, ant: int, gain: float, max_pow: float, noise_power: float):
        """Constructor of the Node class.

        Parameters
        ----------
        :param label: str representing a label for representation purpose.
        :param pos: np.ndarray 1 x 2 representing the x,y cartesian coordinates of the node.
        :param ant: int > 0, number of antennas of the node.
        :param gain: float representing the antenna gain of the node.
        :param max_pow: float representing the max power available on transmission.
        """
        # Control on INPUT
        if pos.shape != (2,):
            raise ValueError(f'Illegal positioning: for Node, '
                             f'coord.shape must be (2,), '
                             f'instead it is {pos.shape}')
        elif ant < 1 or (not isinstance(ant, int)):
            raise ValueError('ant must be a single integer >= 1')
        elif not isinstance(gain, (float, int)):
            raise ValueError('gain [dB] must be a single number')
        elif not isinstance(max_pow, (float, int)):
            raise ValueError('max_pow [dBm] must be a float or integer')

        # Set attributes
        self.label = label
        self.pos = pos
        self.ant = ant
        self.gain = gain
        self.max_pow = max_pow
        self.id = None  # id of the user, initialized by the box.place_node method
        self.ord = None  # id of the user in multi-cell, initialized by the box.place_node method. Not used

        # Noise definition
        self.noise = RxNoise(dBm=noise_power)

    # Operator definition
    def __lt__(self, other):
        if isinstance(other, (self.__class__, self.__class__.__bases__)):
            return True if self.id < other.id else False
        else:
            raise TypeError('< applied for element of different classes')

    def __repr__(self):
        return f'{self.id}-{self.label:3}'


class BS(Node):
    """BS definition."""

    def __init__(self, pos: np.ndarray, ant: int, gain: float, max_pow: float, noise_power: float):
        """BS class for simulation purpose.

        :param pos: ndarray 1 x 2,  representing the x,y cartesian coordinates of the node.
        :param ant: int > 0, number of antennas of the node. Default value is 1.
        :param gain : float [dBi] representing antenna gain of the node. Default is 5.85 dB
        :param max_pow: float [dBm], max power available on transmission. Default is 43 dBm.
        """
        # Default values
        if ant is None:
            ant = 1
        if gain is None:
            gain = 5.85
        if max_pow is None:
            max_pow = 43
        if noise_power is None:
            noise_power = -92.5  # [dBm] TODO: update to a reasonable value

        # Init parent class
        super(BS, self).__init__('BS', pos, ant, gain, max_pow, noise_power)


class UE(Node):
    """User definition."""

    def __init__(self, pos: np.ndarray, ant: int = None, gain: float = None,
                 max_pow: float = None, noise_power: float = None):
        """User class for simulation purpose.

        :param pos: ndarray 1 x 2,
                it represents the x,y cartesian coordinates of the node.
        :param ant: int > 0
                number of antennas of the node. Default value is 1.
        :param gain: float, [dBi]
                It represents the antenna gain of the node. Default value 2.15.
        :param max_pow: float [dBm], max power available on transmission in dBm.
                Default value is 24 dBm.
        """
        # Default values
        if ant is None:
            ant = 1
        if gain is None:
            gain = 2.15
        if max_pow is None:
            max_pow = 24
        if noise_power is None:
            noise_power = -92.5  # [dBm] TODO: update to a reasonable value

        # Init parent class
        super(UE, self).__init__('UE', pos, ant, gain, max_pow, noise_power)


class RxNoise:
    """Represent the noise value at the physical receiver
    # TODO: match with ambient noise and noise figure
    """

    def __init__(self, linear=None, dB=None, dBm=-92.5):
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


class RIS:
    """RIS definition. A class that defines a reflective intelligent surface (RIS)."""

    # Class variables
    carrier_frequency = 3e9
    wavelength = speed_of_light / carrier_frequency
    wavenumber = 2 * np.pi / wavelength
    pos = np.array([0, 0, 0])
    incidence_angle = np.deg2rad(30)

    def __init__(self, Na=4, Nb=4, S=4):

        # Instance variables
        self.num_els = Na * Nb  # total number of elements
        self.num_els_v = Na  # vertical number of elements
        self.num_els_h = Nb  # horizontal number of elements
        self.num_configs = S  # number of configurations

        # Store index of elements considering total number
        self.els = np.arange(self.num_els)

        # Each antenna element is a square of size lambda/4
        self.size_el = (self.wavelength / 4)

        # Compute sizes
        self.size_h = Nb * self.size_el  # horizontal size [m]
        self.size_v = Na * self.size_el  # vertical size [m]

        # Organizing elements over the RIS
        self.id_els = self.indexing_els()
        self.pos_els = self.positioning_els()

        # Configure RIS
        self.set_configs = self.configuration()

        self.Phi_dl = self.dl_reflection_coefficients()

    def indexing_els(self):
        """
        Define a array of tuples where each represents the id of an element.

        Returns
        -------


        """
        # Get vertical ids
        id_v = self.els // self.num_els_v

        # Get horizontal ids
        id_h = np.mod(self.els, self.num_els_h)

        # Get array of tuples with complete id
        id_els = [(id_v[el], id_h[el]) for el in self.els]

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
        for el in self.els:
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
        for el in self.els:
            ax.plot(self.pos_els[el, 0], self.pos_els[el, 2], 'x', color='black')
            ax.text(self.pos_els[el, 0] - 0.003, self.pos_els[el, 2] - 0.0075, str(self.id_els[el]))

        # Plot origin
        ax.plot(0, 0, '.', color='black')

        ax.set_xlim([-0.05, 0.05])
        ax.set_ylim([-0.05, 0.05])

        ax.set_xlabel("x [m]")
        ax.set_ylabel("z [m]")

        plt.show()

    def configuration(self):
        """
        Define set of S configurations offered by the RIS.

        Returns
        -------
        None.

        """
        step_size = (np.pi / 2 - 0) / self.num_configs
        set_configs = np.arange(0, np.pi / 2, step_size)

        return set_configs

    def dl_reflection_coefficients(self):
        """
        Define vectors of DL reflection coefficients that steer the signal towards each configuration, having the
        azimuth angle of the BS as the incidence angle.

        Returns
        -------
        None.

        """
        # Go along x dimension
        x_range = np.arange(-self.size_h / 2, +self.size_h / 2, self.size_el)  # [m]

        # Check if the size of x_range meets the number of horizonal els
        if len(x_range) != self.num_els_h:
            raise Exception("Range over x-axis does not meet number of horizontal Nb elements.")

        # Prepare to save the reflection coefficients for each configuration
        local_surface_phase = np.zeros((self.num_configs, self.num_els_h))

        # Go through all configurations
        for config, theta_s in enumerate(self.set_configs):
            local_surface_phase[config, :] = (2 * np.pi * np.mod(
                self.wavenumber * (-np.sin(theta_s) + np.sin(self.incidence_angle)) * x_range, 1))

        # Store the reflection coefficient of each element for each by repeating
        # the same local surface phase along x-axis
        Phi_dl = np.tile(local_surface_phase, rep=self.num_els_h)

        return Phi_dl