"""Box

This script creates the RIS-assisted environment, where we have a BS, a RIS, and UEs.
The environment is as specified in Fig. 1 of the paper:

V. Croisfelt, F. Saggese, I. Leyva-Mayorga, R. Kotaba, G. Gradoni and P. Popovski, "A Random Access Protocol for RIS-
Aided Wireless Communications," 2022 IEEE 23rd International Workshop on Signal Processing Advances in Wireless
Communication (SPAWC), 2022, pp. 1-5, doi: 10.1109/SPAWC51304.2022.9833984.

Authors: @victorcroisfelt, @lostinafro
Date: 28/07/2022

Specific dependencies:
    - src/nodes.py
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc

from scipy.constants import speed_of_light

from src.nodes import UE, BS, RIS

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
            maximum_distance: float,
            minimum_distance: float,
            carrier_frequency: float = 3e9,
            rng: np.random.RandomState = None
    ):

        # Physical dimensions
        self.maximum_distance = maximum_distance
        self.minimum_distance = minimum_distance

        # Signal parameters
        self.fc = carrier_frequency
        self.wavelength = speed_of_light / carrier_frequency
        self.wavenumber = 2 * np.pi / self.wavelength

        # Random state generator
        self.rng = np.random.RandomState() if rng is None else rng

        # Nodes
        self.bs = None
        self.ue = None
        self.ris = None

        # Downlink
        self.incoming_angle_dl = None   # zenith angle

        self.incoming_direction_dl = None   # unit vector containing direction of incoming wave propagation
        self.scattered_direction_dl = None  # unit vector containing direction of scattered wave propagation

    def place_bs(
            self,
            distance: float = None,
            zenith_angle_deg: float = None,
            gain: float = None
    ):
        """Place a single BS in the environment.

        Parameters
        ----------
        distance : float
            Distance to the origin "0" of the coordinate system. Default: 25 meters.

        zenith_angle_deg : float
            Zenith angle in degrees. Default: 45 degrees

        gain : float
            BS antenna gain G_b.
        """
        if distance is None:
            distance = 25
        if zenith_angle_deg is None:
            zenith_angle_deg = 45

        # Compute rectangular coordinates
        pos = distance * np.array([-np.sin(zenith_angle_deg), np.cos(zenith_angle_deg), 0])

        # Append BS
        self.bs = BS(pos, gain)

    def place_ris(self,
                  pos: np.ndarray = None,
                  num_els_ver: int = None,
                  num_els_hor: int = None,
                  size_el: float = None,
                  # num_ce_configs: int = None,
                  # num_access_configs: int = None
                  ):
        """Place a single RIS in the environment. If a new RIS is set the old one is canceled.

        Parameters
        ----------

        pos : ndarray of shape (3,)
            Position of the RIS in rectangular coordinates.

        num_els_ver : int
            Number of elements along z-axis.

        num_els_hor : int
            Number of elements along x-axis.

        size_el : float
            Size of each element. Default: wavelength

        num_configs : int
            Number of configurations.
        """

        # Append RIS
        self.ris = RIS(
            pos=pos,
            num_els_ver=num_els_ver,
            num_els_hor=num_els_hor,
            wavelength=self.wavelength,
            size_el=size_el
        )

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

        # Compute distances
        distances = np.sqrt(self.rng.rand(n, 1) * (self.maximum_distance**2 - self.minimum_distance**2) + self.minimum_distance**2)
        distances = np.squeeze(distances)

        # Compute angles
        angles = np.pi/2 * self.rng.rand(n, 1)
        angles = np.squeeze(angles)

        # Compute positions
        pos = np.zeros((n, 3))
        pos[:, 0] = distances * np.sin(angles)
        pos[:, 1] = distances * np.cos(angles)

        # Append UEs
        self.ue = UE(n, pos, gain)

    def get_channels(self, tx_power, noise_power, codebook, direction=None, mask=None):
        """Calculate channel gains for the configuration estimation phase.

        Returns
        -------
        channel_gains_ce: ndarray of shape (num_ce_configs, num_ues)
            Downlink channel gains between the BS and UEs given each RIS configuration.
        """

        if mask is not None:
            ue_distances = self.ue.distances[mask]
            ue_angles = self.ue.angles[mask]
        else:
            ue_distances = self.ue.distances
            ue_angles = self.ue.angles

        if isinstance(ue_distances, float):
            ue_distances = np.array([ue_distances, ])
            ue_angles = np.array([ue_angles, ])

        if isinstance(codebook, float):
            codebook = np.array([codebook,])

        # Compute constant term
        num = self.bs.gain * self.ue.gain * (self.ris.size_el * self.ris.size_el)**2
        den = (4 * np.pi * self.bs.distance * ue_distances)**2

        # Compute pathloss component of shape (num_ues, )
        if direction == 'dl':
            pathloss = num / den * np.cos(self.bs.angle)**2
        elif direction == 'ul':
            pathloss = num / den * np.cos(ue_angles)**2

        # Compute propagation phase-shift
        propagation_angle = - (self.bs.distance + ue_distances -
        ((np.sin(self.bs.angle) - np.sin(ue_angles)) * ((self.ris.num_els_hor + 1) / 2) * self.ris.size_el))

        propagation_phase_shift = np.exp(1j * self.wavenumber * propagation_angle)

        # Define enumeration of the number of horizontal elements
        enumeration_num_els_hor = np.arange(1, self.ris.num_els_hor + 1)

        # Compute phase-shift contribution
        contribution = enumeration_num_els_hor[:, None, None] * (np.sin(ue_angles)[:, None] - np.sin(codebook[None, :]))

        # Compute array factors
        array_factor = np.exp(1j * self.wavenumber * self.ris.size_el * contribution)
        array_factor = array_factor.sum(axis=0)

        # Compute channels
        channels = np.sqrt(pathloss[:, None]) * propagation_phase_shift[:, None] * array_factor

        if direction == 'ul':
            channels = channels.conj()

        # Normalize channels
        channels *= np.sqrt(tx_power / noise_power)

        return channels