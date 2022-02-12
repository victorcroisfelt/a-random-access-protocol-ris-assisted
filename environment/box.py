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

        # Downlink
        self.incoming_angle_dl = None   # BS zenith angle (theta_b); float

        self.incoming_direction_dl = None   # unit vector of shape (3, ) containing direction of DL incoming wave propagation
        self.scattered_direction_dl = None  # unit vector of shape (num_configs, 3) containing direction of DL scattered wave propagation

        # Uplink
        self.incoming_angle_ul = None   # UE's zenith angles (theta_k); ndarray of shape (num_ues, )

        self.incoming_direction_ul = None   # unit vector of shape (num_ues, 3) containing direction of DL incoming wave propagation
        self.scattered_direction_ul = None  # unit vector of shape (num_configs, 3) containing direction of DL scattered wave propagation

    def set_incoming_direction_dl(self, bs_zenith_angle):
        """Set Downlink incoming direction as a unit vector of shape (3, ).

        Parameters
        ----------
        bs_zenith_angle : float
            BS zenith angles in radians.
        """
        self.incoming_direction_dl = np.zeros(3)

        self.incoming_direction_dl[0] = +np.sin(bs_zenith_angle)
        self.incoming_direction_dl[1] = -np.cos(bs_zenith_angle)

    def set_scattered_direction_dl(self, configs):
        """Set Downlink scattered direction as a unit vector of shape (num_configs, 3).

        Parameters
        ----------
        configs : ndarray of shape (num_configs, )
            Set of configuration angles.
        """
        self.scattered_direction_dl = np.zeros((configs.size, 3))

        self.scattered_direction_dl[:, 0] = +np.sin(configs)
        self.scattered_direction_dl[:, 1] = +np.cos(configs)

    def set_incoming_direction_ul(self, ue_zenith_angles):
        """Set Uplink incoming directions as unit vectors of shape (num_ues, 3).

        Parameters
        ----------
        ue_zenith_angles : ndarray of shape (num_ues, )
             UE's zenith angles.
        """
        self.incoming_direction_ul = np.zeros((ue_zenith_angles.size, 3))

        self.incoming_direction_ul[:, 0] = -np.sin(ue_zenith_angles)
        self.incoming_direction_ul[:, 1] = -np.cos(ue_zenith_angles)

    def set_scattered_direction_ul(self):
        """Set Uplink scattered direction as a unit vector of shape (3, ).

        """
        self.scattered_direction_ul = np.zeros(3)

        self.scattered_direction_ul[0] = -np.sin(self.incoming_angle_dl)
        self.scattered_direction_ul[1] = +np.cos(self.incoming_angle_dl)

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

        # Store DL incoming angle
        self.incoming_angle_dl = np.deg2rad(zenith_angle_deg)

        # Set incoming direction in the DL
        self.set_incoming_direction_dl(self.incoming_angle_dl)

        # Compute rectangular coordinates
        pos = distance * -self.incoming_direction_dl

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

        # Store UL incoming angles
        self.incoming_angle_ul = np.arctan2(pos[:, 0], pos[:, 1])

        # Set incoming directions in the UL
        self.set_incoming_direction_ul(self.incoming_angle_ul)

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

        # Set scattered direction in the DL
        self.set_scattered_direction_dl(self.ris.configs)

        # Set scattered direction in the UL
        self.set_scattered_direction_ul()

    @property
    def get_reflection_coefficients_dl(self):
        """Compute DL reflection coefficients for each configuration.

        Returns
        -------
        reflection_coefficients_dl : ndarray of shape (num_configs, num_els)
            Matrix containing the DL reflection coefficients for each configuration and each element.
        """

        # Generalized Snell's Law
        constants = self.wavenumber * (np.sin(self.incoming_angle_dl) - np.sin(self.ris.configs))
        reflection_coefficients_dl = constants[:, np.newaxis] * self.ris.pos_els[:, 0]

        return reflection_coefficients_dl

    @property
    def get_reflection_coefficients_ul(self):
        """Compute UL reflection coefficients for each configuration.

        Returns
        -------
        reflection_coefficients_ul : ndarray of shape (num_configs, num_els)
            Matrix containing the UL reflection coefficients for each configuration and each element.
        """

        return -self.get_reflection_coefficients_dl

    def get_channel_model_dl(self):
        """Get Downlink channel gains and phase shifts.

        Returns
        -------
        channel_gains_dl : ndarray of shape (num_ues, )
            Downlink channel gain between the BS and each UE for each RIS element and K UEs.

        phase_shifts_dl : ndarray of shape (num_configs, num_ues, num_els)
            Total Downlink phase shifts between the BS and each RIS element for N elements.

        """
        # Compute distance BS-RIS
        dist_bs = np.linalg.norm(self.bs.pos - self.ris.pos)

        # Compute distance RIS-UE of shape (num_ues, )
        dist_ue = np.linalg.norm(self.ue.pos - self.ris.pos, axis=-1)

        # Compute pathloss constant of shape (num_ues, )
        pathloss_constant = self.bs.gain * self.ue.gain * self.ris.area ** 2 / (4 * np.pi * dist_bs * self.ris.num_els) ** 2

        # Common factor of shape (num_ues, )
        common_factor = pathloss_constant / (dist_ue ** 2)

        # DL channel gains of shape (num_ues, )
        channel_gains_dl = common_factor * np.cos(self.incoming_angle_dl) ** 2

        # BS phase shifts of shape (num_els, )
        dist_b_onto_t = (self.bs.pos * self.incoming_direction_dl).sum()    # float
        dist_n_onto_t = (self.ris.pos_els * self.incoming_direction_dl[np.newaxis, :]).sum(axis=-1)     # shape (num_els, )

        phase_shifts_b_to_n = self.wavenumber * (dist_b_onto_t - dist_n_onto_t)

        # Get reflection coefficients DL (num_configs, num_els)
        reflection_coefficients_dl = self.get_reflection_coefficients_dl

        # UE's phase shifts of shape (num_configs, num_ues, num_els)
        dist_n_onto_r = (self.ris.pos_els[np.newaxis, :, :] * self.scattered_direction_dl[:, np.newaxis, :]).sum(axis=-1)   # shape (num_configs, num_els)
        dist_k_onto_r = (self.ue.pos[np.newaxis, :, :] * self.scattered_direction_dl[:, np.newaxis, :]).sum(axis=-1)    # shape (num_configs, num_ues)

        phase_shifts_n_to_k = self.wavenumber * (dist_n_onto_r[:, np.newaxis, :] - dist_k_onto_r[:, :, np.newaxis])

        # Final phase shift of shape (num_configs, num_ues, num_els)
        phase_shifts_dl = phase_shifts_b_to_n[np.newaxis, np.newaxis, :] +\
                          reflection_coefficients_dl[:, np.newaxis, :] +\
                          phase_shifts_n_to_k

        return channel_gains_dl, phase_shifts_dl

    def get_channel_model_ul(self):
        """Get Uplink channel gains and phase shifts.

        Returns
        -------
        channel_gains_ul : ndarray of shape (K, )
            Uplink channel gain between the BS and each UE for each RIS element and K UEs.

        phase_shifts_ul : ndarray of shape (num_configs, K, N)
            Total uplink phase shift between the BS and each RIS element for N elements.
        """
        # Compute distance BS-RIS
        dist_bs = np.linalg.norm(self.bs.pos - self.ris.pos)

        # Compute distance RIS-UE of shape (K,)
        dist_ue = np.linalg.norm(self.ue.pos - self.ris.pos, axis=-1)

        # Compute pathloss constant
        pathloss_constant = self.bs.gain * self.ue.gain * self.ris.area ** 2 / (4 * np.pi * dist_bs * self.ris.num_els)**2

        # Common factor
        common_factor = pathloss_constant / (dist_ue ** 2)

        # UL channel gains
        channel_gains_ul = common_factor * np.cos(self.incoming_angle_ul) ** 2

        # UE's phase shifts
        dist_ue_onto_t = (self.ue.pos * self.incoming_direction_ul).sum(axis=-1) # shape (num_ues, )
        dist_n_onto_t = (self.ris.pos_els[np.newaxis, :] * self.incoming_direction_ul[:, np.newaxis, :]).sum(axis=-1) # shape (num_ues, num_els)

        phase_shifts_ue_to_n = self.wavenumber * (dist_ue_onto_t[:, np.newaxis] - dist_n_onto_t) # shape (num_ues, num_els)

        # Get reflection coefficients UL (num_configs, num_els)
        reflection_coefficients_ul = self.get_reflection_coefficients_ul

        # BS phase shifts of shape (num_els, )
        dist_n_onto_r = (self.ris.pos_els * self.scattered_direction_ul).sum(axis=-1)  # shape (num_els, )
        dist_b_onto_r = (self.bs.pos * self.scattered_direction_ul).sum(axis=-1)  # float

        phase_shifts_n_to_bs = self.wavenumber * (dist_n_onto_r - dist_b_onto_r)  # shape (num_els, )

        # Final phase shift of shape (num_configs, num_ues, num_els)
        phase_shifts_ul = phase_shifts_ue_to_n[np.newaxis, :, :] +\
                          reflection_coefficients_ul[:, np.newaxis, :] +\
                          phase_shifts_n_to_bs[np.newaxis, np.newaxis, :]

        return channel_gains_ul, phase_shifts_ul

    # Visualization methods
    def plot_reflection_coefficients(self, reflection_coefficients):
        """This method plots how each element of the RIS is configured in each different configuration.
        """

        # Decompose number of configurations
        decompose = int(np.sqrt(self.ris.num_configs))

        # Define min and max values of the reflection coefficient
        min_val, max_val = 0, 2 * np.pi

        # Flipped version of input vector: by convention, the first element of reflection coefficients stores the bottom
        # leftmost element, that is,
        #
        # x   x   x
        # x   x   x
        # o   x   x
        #
        reflection_coefficients_flipped = np.flip(reflection_coefficients, axis=-1)

        # Reshape the reflection coefficient tensor from (S,N) -> (S,Na,Nb)
        reflection_coefficients_matrix = reflection_coefficients_flipped.reshape(self.ris.num_configs,
                                                                                 self.ris.num_els_v, self.ris.num_els_h)
        # LaTeX type definitions
        rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
        rc('text', usetex=True)

        # Open axes
        fig, ax = plt.subplots(nrows=decompose, ncols=decompose)

        # Go through all configurations
        for config, theta_s in enumerate(self.ris.set_configs):

            id_r = int(np.mod(config, decompose))
            id_c = config // decompose

            ax[id_r][id_c].matshow(reflection_coefficients_matrix[config][:, :], cmap=plt.cm.Blues)

            ax[id_r][id_c].set_title(("desired direction is: " + str(np.round(np.rad2deg(theta_s), 2))))

            # Go through RIS in the reverse direction
            for i in range(self.ris.num_els_v):
                for j in range(self.ris.num_els_h):
                    # Get value in degrees
                    value_deg = np.round(np.rad2deg(reflection_coefficients_matrix[config][i, j]))

                    # Print value, note that j indexes the x-dimension (horizontal plot dimension)
                    ax[id_r][id_c].text(j, i, str(value_deg), va='center', ha='center', color='black',
                                        fontsize='x-small', fontweight='bold')

                    ax[id_r][id_c].set_xlabel('$x$ [m]')
                    ax[id_r][id_c].set_ylabel('$z$ [m]')

        plt.show()

        fig, ax = plt.subplots(figsize=(3.15, 3))

        ax.matshow(reflection_coefficients_matrix[0][:, :], cmap=plt.cm.Reds)

        # Go through RIS in the reverse direction
        for i in range(self.ris.num_els_v):
            for j in range(self.ris.num_els_h):
                # Get value in degrees
                value_deg = int(np.round(np.rad2deg(reflection_coefficients_matrix[0][i, j])))

                # Print value, note that j indexes the x-dimension (horizontal plot dimension)
                ax.text(j, i, str(value_deg), va='center', ha='center', color='black',
                        fontsize='xx-small', fontweight='bold')

                ax.set_xlabel('$y$ [m]')
                ax.set_ylabel('$x$ [m]')

        plt.show()

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
