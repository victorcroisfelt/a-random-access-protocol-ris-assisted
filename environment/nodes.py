#!/usr/bin/env python3
# filename "nodes.py"

import numpy as np


class Node:
    """Node definition. This class is called by BS, UE and RIS classes to defining common parameters."""

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
        self.id = None      # id of the user, initialized by the box.place_node method
        self.ord = None     # id of the user in multi-cell, initialized by the box.place_node method. Not used
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
    """Node definition"""

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
    """User definition"""

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
