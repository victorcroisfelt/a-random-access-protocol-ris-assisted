#!/usr/bin/env python3
# filename "cells.py"

import numpy as np
import environment.common as common
from environment.nodes import UE, BS, RIS
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib import rc
from scipy.constants import speed_of_light


class Box:
    """Class Box creates an environment square box with specific parameters and nodes."""
    def __init__(self,
                 ell: float, ell0: float,
                 carrier_frequency: float = 3e9, bandwidth: float = 180e3,
                 pl_exp: float = 2, sh_std: float = -np.inf,
                 rng: np.random.RandomState = None
                 ):
        """Constructor of the cell.

        :param ell: float, side length of the users (nodes can be placed between outer and inner).
        :param ell0: float, distance between the x-axis and the lowest y-coordinate of the box.
        :param carrier_frequency: float [Hz], central frequency of the environment.
        :param bandwidth: float [Hz], bandwidth of the signal,
        :param pl_exp: float, path loss exponent in the cell (default is 2 as free space).
        :param sh_std: float, standard deviation of the shadowing phenomena (default is 0).
        """
        if (ell < ell0) or (ell0 <= 0) or (ell <= 0):
            raise ValueError('ell and ell0 must be >= 0 and ell > ell0')
        elif pl_exp <= 0:
            raise ValueError('pl_exp must be >= 0')

        # Physical attributes of the box
        self.pos = np.sqrt(2) * (ell0 + ell / 2) * np.array([1, 1])  # center coordinates (I don't know if useful)
        self.ell = ell
        self.ell0 = ell0

        # Propagation environment
        self.pl_exp = pl_exp
        self.sh_std = sh_std

        # Bandwidth available
        self.fc = carrier_frequency
        self.wavelength = speed_of_light / carrier_frequency
        self.wavenumber = 2 * np.pi / self.wavelength
        self.bw = bandwidth

        # Random State generator
        self.rng = np.random.RandomState() if rng is None else rng

        # Channel gain
        self.chan_gain = None

        # Nodes
        self.bs_height = 10     # [m]
        self.bs = None
        self.ue = None
        self.ris = None

    def place_bs(self, pos_polar: np.ndarray = None, ant: int = None, gain: float = None,
                 max_pow: float = None, noise_power: float = None):
        """Place a single BS in the environment. Following the paper environment, the BS is always located at the second
        quadrant of the coordinate system. If a new BS is set the old one is canceled.

        :param pos_polar: ndarray of shape (2, 1)
            Radius and positional angle of the BS with respect to the positive horizontal axis. If None, the position is
            randomly selected.

        :param ant: int > 0
            Number of BS antennas.

        :param gain: float
            BS antenna gain G_b.

        :param max_pow: float
           Maximum power available at the BS.

        :param noise_power: float
            Represent the noise power in dBm of the BS RF chain.
        """
        # Compute the position
        if pos_polar is None:
            # if the position is not given, a random position inside a specular box in second quadrant computed
            pos = self.rng.uniform([-self.ell0, self.ell0], [-self.ell, self.ell], (2, 1))
        else:   # translate from polar to cardinal
            pos = pos_polar[0][0] * np.array([np.cos(pos_polar[0][1]), np.sin(pos_polar[0][1])])

        # Add third dimension for coherency with RIS
        pos = np.array([[pos[0], pos[1], self.bs_height]])

        # Append nodes
        self.bs = BS(1, pos, ant, gain, max_pow, noise_power)

    def place_ue(self, n: int, pos_polar: np.ndarray = None, ant: int or np.ndarray = None,
                 gain: float or np.ndarray = None, max_pow: float or np.ndarray = None,
                 noise_power: float or np.ndarray = None):
        """Place a predefined number n UEs in the box. If a new set of UE is set the old one is canceled.

        :param n: int > 0
           Number of UEs to be placed.

        :param pos_polar: ndarray of shape (n, 2)
            Radius and positional angle of each UE with respect to the positive horizontal axis. If None, the position
            is randomly selected.

        :param ant: ndarray of int and shape (n, )
            Represent the number of antennas of each node. If a single value, each UE will have same number of antennas.

        :param gain: ndarray of shape (n, )
            Represent the antenna gain used in the path loss computation. If a single value, each UE will have same
            gain values.

        :param max_pow: ndarray of shape (n, )
            Represent the maximum power available on each UE. If a single value, each UE will have same max_pow.

        :param noise_power: ndarray of shape (n, )
            Represent the noise power in dBm of the RF chain. If a single vale, each bs will have same noise_power.
        """
        # Control on INPUT
        if not isinstance(n, int) or (n < 0):   # Cannot add a negative number od nodes
            raise ValueError('N must be int >= 0')
        elif n == 0:    # No node to be added
            return

        # Compute position
        if pos_polar is None:
            # if the position is not given, a random position inside the box is computed
            pos = np.hstack((self.rng.uniform(self.ell0, self.ell, (n, 2)), np.zeros((n, 1))))

        else:  # translate from polar to cardinal
            try:
                pos = np.vstack((pos_polar[:, 0] * np.cos(pos_polar[:, 1]), pos_polar[:, 0] * np.sin(pos_polar[:, 1]))).T
                # Add third dimension for coherency with RIS
                pos = np.hstack((pos[:, 0], pos[:, 1], np.zeros((n, 1))))
            except IndexError:
                # Add third dimension for coherency with RIS
                pos = pos[0] * np.array([[np.cos(pos[1]), np.sin(pos[1]), 0]])

        # Append nodes
        self.ue = UE(n, pos, ant, gain, max_pow, noise_power)

    def place_ris(self, pos_polar: np.ndarray = None, v_els: list or int = None,
                  h_els: list or int = None, configs: list or int = None):
        """Place a single RIS in the environment. If a new RIS is set the old one is canceled.

        :param n: int > 0
            Number of RIS to be placed.

        :param pos_polar: ndarray of shape (2,)
            Radius and positional angle of the RIS with respect to the positive horizontal axis. If None, the position
            is randomly selected.

        :param v_els: list of int > 0
            Number of vertical elements of the RIS.

        :param h_els: list of int > 0
            Number of horizontal elements of each RIS.

        :param configs: list of int > 0
            Maximum number of configuration available.
        """
        # Compute the position
        if pos_polar is None:
            # if the position is not given, the origin is adopted
            pos = np.array([[0, 0, 0]])
        else:  # translate from polar to cardinal
            pos = pos_polar[0] * np.array([[np.cos(pos_polar[1]), np.sin(pos_polar[1]), 0]])

        # Append nodes
        self.ris = RIS(1, pos, v_els, h_els, configs, self.wavelength)

    # Channel build
    def get_channel_model(self):
        """Compute downlink and uplink channel gains and phase shifts due to wave propagation.

        Returns
        -------
        channel_gains_dl : ndarray of shape (K, )
            Downlink channel gain between the BS and each UE for each RIS element and K UEs.

        channel_gains_ul : ndarray of shape (K, )
            Uplink channel gain between the BS and each UE for each RIS element and K UEs.

        phase_shifts_bs : ndarray of shape (N, )
            Propagation phase shift between the BS and each RIS element for N elements.

        phase_shifts_ue : ndarray of shape (N, )
            Propagation phase shifts between each UE and each RIS element for K UEs and N elements.
        """
        # Compute distance BS-RIS
        dist_bs = np.linalg.norm(self.bs.pos - self.ris.pos)

        # Compute distance RIS-UE of shape (K,)
        dist_ue = np.linalg.norm(self.ue.pos - self.ris.pos, axis=-1)

        # Compute constant Omega_k
        Omega_ue = self.bs.gain * self.ue.gain * self.ris.area**2 / (4 * np.pi * dist_bs * self.ris.num_els)**2

        # Common factor
        common = Omega_ue / (dist_ue**2)

        # DL channel gains
        channel_gains_dl = common * np.cos(self.bs.incidence_angle)**2

        # UL channel gains
        channel_gains_ul = common * np.cos(self.ue.incidence_angle)**2

        # Compute distance BS-RIS-elements of shape (N,)
        dist_bs_el = np.linalg.norm(self.bs.pos - self.ris.pos_els, axis=-1)

        # Compute distance BS-RIS-elements of shaspe (K,N)
        dist_ue_el = np.linalg.norm(self.ue.pos[:, np.newaxis, :] - self.ris.pos_els, axis=-1)

        # BS phase shifts
        phase_shifts_bs = 2 * np.pi * np.mod(dist_bs_el / self.wavenumber, 1)

        # UE phase shifts
        phase_shifts_ue = 2 * np.pi * np.mod(dist_ue_el / self.wavenumber, 1)

        return channel_gains_dl, channel_gains_ul, phase_shifts_bs, phase_shifts_ue

    # Visualization methods
    def list_nodes(self, label=None):
        if label is None:
            ls = '\n'.join(f'{i:2} {n}' for i, n in enumerate(self.node))
        elif label in common.node_labels:
            ls = '\n'.join(f'{i:2} {n}' for i, n in enumerate(self.node)
                           if n.type == label)
        else:
            raise ValueError(f'Node type must be in {common.node_labels}')
        return print(ls)

    def plot_scenario(self):
        """This method will plot the scenario of communication
        """
        # LaTeX type definitions
        rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
        rc('text', usetex=True)
        # Open axes
        fig, ax = plt.subplots()

        # Box positioning
        box = plt.Rectangle((self.ell0, self.ell0), self.ell, self.ell, ec="black", ls="--", lw=1, fc='#45EF0605')
        ax.add_patch(box)
        # User positioning
        delta = self.ell0 / 100
        # BS
        plt.scatter(self.bs.pos[:, 0], self.bs.pos[:, 1], c=common.node_color['BS'], marker=common.node_mark['BS'], label='BS')
        # plt.text(self.bs.pos[:, 0], self.bs.pos[:, 1] + delta, s='BS', fontsize=10)
        # UE
        plt.scatter(self.ue.pos[:, 0], self.ue.pos[:, 1], c=common.node_color['UE'], marker=common.node_mark['UE'], label='UE')
        for k in np.arange(self.ue.n):
            plt.text(self.ue.pos[k, 0], self.ue.pos[k, 1] + delta, s=f'{k}', fontsize=10)
        # RIS
        plt.scatter(self.ris.pos[:, 0], self.ris.pos[:, 1], c=common.node_color['RIS'], marker=common.node_mark['RIS'], label='RIS')
        # plt.text(self.ris.pos[:, 0], self.ris.pos[:, 1] + delta, s='RIS', fontsize=10)
        # Set axis
        # ax.axis('equal')
        ax.set_xlabel('$x$ [m]')
        ax.set_ylabel('$y$ [m]')
        # limits
        ax.set_ylim(ymin=-self.ell0/2)
        # Legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        # Finally
        plt.grid(color='#E9E9E9', linestyle='--', linewidth=0.5)
        plt.show(block=False)

    # def show_chan(self, subs: (int, list) = None):
    #     """Shows the chan_gain in a pretty and readable way.
    #     # TODO: re-update it
    #
    #     Parameters
    #     __________
    #     FR : (int, list)
    #         the subcarriers to be visualized; it can be a single or a list of subcarriers
    #     """
    #     # Control if channel gain tensor is built
    #     if self.chan_gain is None:
    #         warnings.warn('Channel gain tensor not instantiated.')
    #         return
    #     # Control on input
    #     if subs is None:
    #         subs = list(range(self.FR))
    #     elif isinstance(subs, int):
    #         subs = [subs]
    #     with np.errstate(divide='ignore'):
    #         user_grid = 20 * np.log10(np.abs(np.mean(self.chan_gain, axis=(3, 4))))
    #     out = str()
    #     nodes = list(range(self.chan_gain.shape[1]))
    #     for ind, f in enumerate(subs):
    #         head = [f'f={f}'] + nodes
    #         out += tabulate(user_grid[f], headers=head, floatfmt=".2f",
    #                         showindex="always", numalign="right",
    #                         tablefmt="github")
    #         out += '\n\n'
    #     print(out)

    # TODO: simplify the representation
    # def count_elem(self):
    #     count = [[0] * 3 for _ in range(len(dic.node_types))]
    #     ind = self.ind()[0]
    #     for j in range(len(dic.node_types)):
    #         count[j][0] = len(ind[j])
    #         if count[j][0]:
    #             ls = (list(dic.bs_dir) if j in range(len(dic.bs_dir))
    #                   else list(dic.user_dir))
    #             count[j][1] = [self.node[i].dir for i in ind[j]].count(ls[0])
    #             count[j][2] = [self.node[i].dir for i in ind[j]].count(ls[1])
    #     return count
    # def __repr__(self):
    #     count = self.count_elem()
    #     line = []
    #     for i in range(len(dic.node_types)):
    #         ls = (list(dic.bs_dir) if i in range(len(dic.bs_dir))
    #               else list(dic.user_dir))
    #         line.append(f'{list(dic.node_types)[i]:3} = {count[i][0]:02}, '
    #                     f'{ls[0]:2} = {count[i][1]:02}, '
    #                     f'{ls[1]:2} = {count[i][2]:02};')
    #     counter = '\n\t\t'.join(str(n) for n in line)
    #     return (f'cell #{self.id}: '
    #             f'\n\tdim:\t{self.ell0}-{self.ell} [m]'
    #             f'\n\tcoord:\t{self.pos[0]:+04.0f},{self.pos[1]:+04.0f} [m]'
    #             f'\n\tcoef:\tPL_exp = {self.pl_exp:1},'
    #             f'\tSH = {self.sh_std:02} [dB]'
    #             f'\n\tnodes:\t{counter}\n')

    # Properties
    @property
    def noise_vector(self):
        return np.array([n.noise.linear for n in self.node])
