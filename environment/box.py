#!/usr/bin/env python3
# filename "cells.py"

import numpy as np
import environment.common as common
from environment.nodes import UE, BS, RIS
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib import rc
from scipy.constants import speed_of_light


def quant(x: np.ndarray, bits: int):
    """
    Quantize a signal x considering the given number of bits.

    Parameters
    ----------

    x : array of floats
        input signal

    bits : integer
        number of bits that defines the step size (resolution) of the
        quantization process.

    Returns
    -------

    yk : array of floats
        quantized version of the input signal
    """

    # Obtain step size
    Delta = (1 / 2) ** (bits - 1)

    # Re-scale the signal
    x_zeros = x * (1 - 1e-12)
    x_scaled = x_zeros - Delta / 2

    # Quantization stage
    k = np.round(x / Delta) * Delta

    # Reconstruction stage
    yk = k + Delta / 2

    return yk


class Box:
    """Creates an environment defined by a square box of UEs.

    Arguments:
        square_length : float
            Length of the box where the UEs are.

        min_square_length : float
            Distance between the x-axis and the lowest y-coordinate of the box.

        carrier_frequency : float 
            Central frequency in Hertz.

        bandwidth : float 
            Bandwidth in Hertz.

    """

    def __init__(
            self,
            square_length: float,
            min_square_length: float,
            carrier_frequency: float = 3e9,
            bandwidth: float = 180e3,
            rng: np.random.RandomState = None
    ):

        if (square_length < min_square_length) or (min_square_length <= 0) or (square_length <= 0):
            raise ValueError('Invalid definition of the box concerning its length.')

        # Physical attributes of the box
        # self.pos = np.sqrt(2) * (ell0 + ell / 2) * np.array([1, 1])  # center coordinates (I don't know if useful)
        self.square_length = square_length
        self.min_square_length = min_square_length

        # # Propagation environment
        # self.pl_exp = pl_exp
        # self.sh_std = sh_std

        # Bandwidth available
        self.fc = carrier_frequency
        self.wavelength = speed_of_light / carrier_frequency
        self.wavenumber = 2 * np.pi / self.wavelength
        self.bw = bandwidth

        # Random State generator
        self.rng = np.random.RandomState() if rng is None else rng

        # Nodes
        self.bs = None
        self.ue = None
        self.ris = None

    def place_bs(
            self,
            distance: float = None,
            zenith_angle_deg: float = None,
            gain: float = None,
            max_pow: float = None
    ):
        """Place a single BS in the environment. Following the paper, the BS is always located at the second quadrant of
         the coordinate system. If a new BS is set the old one is canceled.

        Parameters
        ----------
        distance : float
            Distance to the origin "0" of the coordinate system. Default: 25 meters.

        zenith_angle_deg : float
            Zenith angle in degrees. Default: 45 degrees

        gain : float
            BS antenna gain G_b.

        max_pow : float
           Maximum power available at the BS.
        """
        if distance is None:
            distance = 25
        if zenith_angle_deg is None:
            zenith_angle_deg = 45

        # Compute rectangular coordinates
        pos = distance * np.array([-np.sin(zenith_angle_deg), np.cos(zenith_angle_deg), 0])

        # Append BS
        self.bs = BS(1, pos, gain, max_pow)

    def place_ue(
            self,
            n: int,
            gain: float = None,
            max_pow: float = None
    ):
        """Place a predefined number n of UEs in the box. If a new set of UE is set the old one is canceled.

        Parameters
        ----------

        n : int
            Number of UEs to be placed.

        gain : float
            UE antenna gain G_k.

        max_pow : float
           Maximum power available at each UE.
        """
        # Control on input
        if not isinstance(n, int) or (n <= 0):  # Cannot add a negative number od nodes
            raise ValueError('n must be int >= 0.')

        # Drop UEs, pos is a ndarray of shape (K, 3)
        pos = np.zeros((n, 3))
        pos[:, :-1] = self.square_length * self.rng.rand(n, 2) + self.min_square_length

        # Append UEs
        self.ue = UE(n, pos, gain, max_pow)

    def place_ris(self,
                  pos: np.ndarray = None,
                  num_els_v: int = None,
                  num_els_h: int = None,
                  size_el: float = None,
                  num_configs: int = None
                  ):
        """Place a single RIS in the environment. If a new RIS is set the old one is canceled.

        Parameters
        ----------

        pos : ndarray of shape (3,)
            Position of the RIS in rectangular coordinates.

        num_els_v : int
            Number of elements in the vertical dimension.

        num_els_h : int
            Number of elements in the horizontal dimension.

        size_el : float
            Size of each element. Default: wavelength/4

        num_configs : int
            Number of configurations.
        """

        # Append RIS
        self.ris = RIS(pos=pos, num_els_v=num_els_v,  wavelength=self.wavelength, size_el=size_el,  num_els_h=num_els_h,
                       num_configs=num_configs)

    def get_channel_model(self):
        """Get Downlink (DL) and Uplink (UL) channel gain.

        Returns
        -------
        channel_gains_dl : ndarray of shape (num_ues, )
            Downlink channel gain between the BS and each UE for each RIS element and K UEs.

        channel_gains_ul : ndarray of shape (num_ues, )
            Uplink channel gain between the BS and each UE for each RIS element and K UEs.

        """
        # Compute DL pathloss component of shape (num_ues, )
        num = self.bs.gain * self.ue.gain * self.ris.area**2
        den = (4 * np.pi * self.ris.num_els * self.bs.distance * self.ue.distances)**2

        pathloss_dl = num / den * np.cos(self.bs.angle)**2

        # Compute UL pathloss component of shape (num_ues, )
        pathloss_ul = num / den * np.cos(self.ue.angles)**2

        # Compute constant phase component of shape (num_ues, )
        phi = - self.wavenumber * (self.bs.distance + self.ue.distances) - (np.sin(self.bs.angle) - np.sin(self.ue.angles)) * (self.ris.num_els_h + 1) / 2 * self.ris.size_el

        # Compute array factor of shape (num_configs, num_ues)
        enumeration_num_els_h = np.arange(1, self.ris.num_els_h + 1)
        argument = self.wavenumber * (np.sin(self.ue.angles[np.newaxis, :, np.newaxis]) - np.sin(self.ris.configs[:, np.newaxis, np.newaxis])) * enumeration_num_els_h[np.newaxis, np.newaxis, :] *  self.ris.size_el

        array_factor_dl = self.ris.num_els_v * np.sum(np.exp(+1j * argument), axis=-1)
        array_factor_ul = array_factor_dl.conj()

        # Compute channel gains of shape (num_configs, num_ues)
        channel_gains_dl = np.sqrt(pathloss_dl[np.newaxis, :]) * np.exp(+1j * phi[np.newaxis, :]) * array_factor_dl
        channel_gains_ul = np.sqrt(pathloss_ul[np.newaxis, :]) * np.exp(-1j * phi[np.newaxis, :]) * array_factor_ul

        return channel_gains_dl, channel_gains_ul

    def get_channel_model_slotted_aloha(self):
        """Get Downlink (DL) and Uplink (UL) channel gain.

        Returns
        -------
        channel_gains_dl : ndarray of shape (num_ues, )
            Downlink channel gain between the BS and each UE for each RIS element and K UEs.

        channel_gains_ul : ndarray of shape (num_ues, )
            Uplink channel gain between the BS and each UE for each RIS element and K UEs.

        """
        # Compute DL pathloss component of shape (num_ues, )
        distance_bs_ue = np.linalg.norm(self.ue.pos - self.bs.pos, axis=-1)

        num = self.bs.gain * self.ue.gain
        den = (4 * np.pi * distance_bs_ue)**2

        pathloss_dl = num / den

        # Compute UL pathloss component of shape (num_ues, )
        pathloss_ul = pathloss_dl

        # Compute constant phase component of shape (num_ues, )
        phi = - self.wavenumber * distance_bs_ue

        # Compute channel gains of shape (num_configs, num_ues)
        channel_gains_dl = np.sqrt(pathloss_dl[np.newaxis, :]) * np.exp(+1j * phi[np.newaxis, :])
        channel_gains_ul = np.sqrt(pathloss_ul[np.newaxis, :]) * np.exp(-1j * phi[np.newaxis, :])

        return channel_gains_dl, channel_gains_ul

    def plot_scenario(self):
        """This method will plot the scenario of communication
        """
        # LaTeX type definitions
        rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
        rc('text', usetex=True)
        # Open axes
        fig, ax = plt.subplots()

        # Box positioning
        box = plt.Rectangle((self.min_square_length, self.min_square_length), self.square_length, self.square_length, ec="black", ls="--", lw=1, fc='#45EF0605')
        ax.add_patch(box)
        # User positioning
        delta = self.min_square_length / 100
        # BS
        plt.scatter(self.bs.pos[0], self.bs.pos[1], c=common.node_color['BS'], marker=common.node_mark['BS'],
                    label='BS')
        # plt.text(self.bs.pos[:, 0], self.bs.pos[:, 1] + delta, s='BS', fontsize=10)
        # UE
        plt.scatter(self.ue.pos[:, 0], self.ue.pos[:, 1], c=common.node_color['UE'], marker=common.node_mark['UE'],
                    label='UE')
        for k in np.arange(self.ue.n):
            plt.text(self.ue.pos[k, 0], self.ue.pos[k, 1] + delta, s=f'{k}', fontsize=10)
        # RIS
        plt.scatter(self.ris.pos[0], self.ris.pos[1], c=common.node_color['RIS'], marker=common.node_mark['RIS'],
                    label='RIS')
        # plt.text(self.ris.pos[:, 0], self.ris.pos[:, 1] + delta, s='RIS', fontsize=10)
        # Set axis
        # ax.axis('equal')
        ax.set_xlabel('$x$ [m]')
        ax.set_ylabel('$y$ [m]')
        # limits
        ax.set_ylim(ymin=-self.min_square_length / 2)
        # Legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        # Finally
        plt.grid(color='#E9E9E9', linestyle='--', linewidth=0.5)
        plt.show(block=False)
