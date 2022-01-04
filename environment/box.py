#!/usr/bin/env python3
# filename "cells.py"

import numpy as np
import environment.common as common
from environment.nodes import UE, BS
import matplotlib.pyplot as plt
import matplotlib.lines as mln
import warnings
from collections import OrderedDict
from matplotlib.patches import Circle
from matplotlib import rc
from tabulate import tabulate


class Box:
    """Class Box creates an environment square box with specific parameters and nodes."""
    def __init__(self, ell: float, ell0: float,
                 pl_exp: float = 2, sh_std: float = -np.inf, nid: int = 0,
                 rng: np.random.RandomState = None):
        """Constructor of the cell.

        :param ell : float, side length of the users (nodes can be placed between outer and inner).
        :param ell0 : float, distance between the x-axis and the lowest y-coordinate of the box.
        :param pl_exp: float, path loss exponent in the cell (standard is 2 as free space).
        :param sh_std: float, standard deviation of the shadowing phenomena (standard is 0 ) .
        :param nid: int, number representing the id of the cell (for multi-cell processing).
        """
        if (ell < ell0) or (ell0 <= 0) or (ell <= 0):
            raise ValueError('ell and ell0 must be >= 0 and ell > ell0')
        elif pl_exp <= 0:
            raise ValueError('pl_exp must be >= 0')
        # Physical attributes
        self.pos = np.sqrt(2) * (ell0 + ell / 2) * np.array([1, 1])  # center coordinates (I don't know if useful)
        self.ell = ell
        self.ell0 = ell0
        self.pl_exp = pl_exp
        self.sh_std = sh_std
        self.id = nid
        # Random State generator
        self.rng = np.random.RandomState() if rng is None else rng
        # List of cells data (multi-cell processing)
        self.prev = 0       # store the number of nodes of all previous instantiated cells
        self.next = None    # a pointer to the next instantiated cell
        # BS
        self.node = []

    @property
    def noise_vector(self):
        return np.array([n.noise.linear for n in self.node])

    def __lt__(self, other):    # (multi-cell processing)
        if isinstance(other, self.__class__):
            return True if self.id < other.id else False
        else:
            raise TypeError('< applied for element of different classes')

    def place_bs(self,
                 n: int = 1, pos: np.ndarray = None, ant: int or list = None, gain: float or list = None,
                 max_pow: float or list = None, noise_power: float or list = None):
        """Place a predefined number n of bs in the environment. Following the paper environment,
        the BS is always in the second quadrant of the coordinate system.

        :param n: int > 0, number of nodes to be placed.
        :param pos: N x 2 ndarray; row i represents the r, theta polar coordinates of node i.
            If None, the position is randomly selected.
        :param ant: list of int > 0 representing the number of antennas of each bs;
                if a single int, each bs will have same number of antennas.
        :param gain: sequence of int or float, representing the antenna gain used in the path loss computation;
                if a single value, each bs will have same gain.
        :param max_pow: sequence of float, representing the maximum power available on the bs;
                if a single vale, each bs will have same max_pow.
        :param noise_power: list of float, representing the noise power in dBm of the RF chain;
                if a single vale, each bs will have same noise_power.
        """
        # Control on INPUT
        if not isinstance(n, int) or (n < 0):
            raise ValueError('N must be int >= 0')
        elif n == 0:
            return
        # Input reformat
        if not isinstance(ant, list):
            ant = [ant] * n
        if not isinstance(gain, list):
            gain = [gain] * n
        if not isinstance(max_pow, list):
            max_pow = [max_pow] * n
        if not isinstance(noise_power, list):
            noise_power = [noise_power] * n
        # Counting the actual nodes
        n_old = len(self.node)
        if pos is None:
            # if the position is not given, a random position inside a specular box in second quadrant computed
            pos = self.rng.uniform([-self.ell0, self.ell0], [-self.ell, self.ell], (n, 2))
        else:   # translate from polar to cardinal
            pos = np.hstack((pos[:, 0] * np.cos(pos[:, 1]), pos[:, 0] * np.sin(pos[:, 1])))
        # Append nodes
        for i in range(n):
            self.node.append(BS(pos=pos[i], ant=ant[i], gain=gain[i], max_pow=max_pow[i], noise_power=noise_power[i]))
        # Order nodes
        self.order_nodes()
        self.update_id()

    def wipe_bs(self):
        """This method wipe out the bs in the node list"""
        users = []
        for n in self.node:
            if isinstance(n, UE):
                users.append(n)
        self.node = users
        self.order_nodes()
        self.update_id()

    def place_ue(self, n: list or int, pos: np.ndarray = None, ant: list or int = None, gain: list or float = None,
                 max_pow: list or float = None, noise_power: list or float = None):
        """Place a predefined number n of nodes in the cell.

        :param n: int > 0, representing the number of user to be placed.
        :param pos: N x 2 ndarray; row i represents the r, theta polar coordinates of node i.
        :param ant: list of int > 0 representing the number of antennas of each node;
                    if a single value, each user will have same number of antennas.
        :param gain: list of int or float, representing the antenna gain used in the path loss computation;
                    if a single value, each user will have same gain values.
        :param max_pow: list of int or float, representing the maximum power available on the node;
                    if a single value, each node will have same max_pow.
        """
        # Control on INPUT
        if not isinstance(n, int) or (n < 0):   # Cannot add a negative number od nodes
            raise ValueError('N must be int >= 0')
        elif n == 0:    # No node to be added
            return
        # Counting the present nodes
        n_old = len(self.node)
        # Input reformat
        if not isinstance(ant, list):
            ant = [ant] * n
        if not isinstance(gain, list):
            gain = [gain] * n
        if not isinstance(max_pow, list):
            max_pow = [max_pow] * n
        if not isinstance(noise_power, list):
            noise_power = [noise_power] * n
        # In case of UE
        if pos is None:
            # if the position is not given, a random position inside the box is computed.
            pos = self.rng.uniform(self.ell0, self.ell, (n, 2))
        else:  # translate from polar to cardinal
            pos = np.hstack((pos[:, 0] * np.cos(pos[:, 1]), pos[:, 0] * np.sin(pos[:, 1])))
        # Append nodes
        for i in range(n):
            self.node.append(UE(pos=pos[i], ant=ant[i], gain=gain[i], max_pow=max_pow[i], noise_power=noise_power[i]))
        # Order nodes
        self.order_nodes()
        self.update_id()

    def wipe_ue(self):
        """This method wipe out the users in the node list"""
        bs = []
        for n in self.node:
            if isinstance(n, BS):
                bs.append(n)
        self.node = bs
        self.order_nodes()
        self.update_id()

    def order_nodes(self):
        """The method orders the nodes following the order given by type: 'BS', 'UE'
        """
        self.node.sort(key=lambda x: (common.node_labels[x.label]))

    def update_id(self):
        """ The method updates the id of each node in the cell.
        It propagates the updating to the next cell instantiated.

        Returns
        ______
        node.id : tuple,
            the id of the node which is formed by the id of the cell and
            the number of the node in the cell, following the order given
            by order_nodes.
        node.ord : int,
            an ordered counter of the node built according to the channel
            gain tensor order.
        """
        for i, node in enumerate(self.node):
            node.id = (self.id, i)
            node.ord = self.prev + i
        if self.next is not None:
            self.next.prev = self.prev + len(self.node)
            self.next.update_id()

    # Channel build
    def build_chan_gain(self):
        """Create the list of all possible channel gain in the simulation.
        TODO: REDO it completely. We must insert the pathloss computation and the phase modeling HERE!
        Returns
        -------
        h : ndarray[f, j, i][k, l]
            contains all the channel gains, where:
            [k, l] is the element of the matrix Nr x Nt for the MIMO setting
            j is the transmitter
            i is the receiver
            f is the subcarrier
        """
        # collect data
        cells = self.cell
        nodes = self.nodes
        subs = self.FR
        d0 = self.frau
        # Data struct definition
        data = np.array([[[np.zeros((i.ant, j.ant), dtype=complex)
                           for i in nodes]
                          for j in nodes]
                         for _ in range(subs)])
        # For loop
        for f in range(subs):
            for i, n_i in enumerate(nodes):  # i is the receiver
                c_i = cells[n_i.id[0]]  # c_i is the cell of the receiver
                # Channel reciprocity
                for j, _ in enumerate(nodes[:i]):  # j is the transmitter
                    data[f, j, i] = data[f, i, j].T
                # Channel computation
                for j, n_j in enumerate(nodes[i:], i):  # j is the transmitter
                    c_j = cells[n_j.id[0]]  # c_j is the cell of the transmitter
                    # Creating the seed as a function of position and sub. In
                    # this way users in the same position will experience same
                    # fading coefficient
                    s = np.abs(np.sum((f + c_i.coord + n_i.coord + c_j.coord + n_j.coord) * 1e4, dtype=int))
                    if j != i:  # channel is modeled as a Rayleigh fading
                        # Computing Path Loss
                        d = np.linalg.norm(c_i.coord + n_i.coord - c_j.coord - n_j.coord)
                        pl = 20 * np.log10(4 * np.pi / self.RB.wavelength[f]) - n_i.gain - n_j.gain \
                             + 10 * c_i.pl_exp * np.log10(d) \
                             + 10 * ((2 - c_j.pl_exp) * np.log10(d0) + (c_j.pl_exp - c_i.pl_exp) * np.log10(d0))
                        # Computing Shadowing
                        sh = common.fading("Shadowing", shape=c_i.sh_std, seed=s)
                        # Computing fading matrix
                        fad = common.fading("Rayleigh", seed=s, dim=(n_i.ant, n_j.ant))
                        # Let the pieces fit
                        data[f, j, i][np.ix_(range(n_i.ant), range(n_j.ant))] = fad * np.sqrt(
                            10 ** (-(pl + sh) / 10))
                    elif n_j.dir == 'FD':  # j == i
                        # Full Duplex fading is Rice distributed
                        fad = common.fading("Rice", dim=(n_i.ant, n_j.ant), shape=n_j.r_shape, seed=s)
                        data[f, j, i][np.ix_(range(n_i.ant), range(n_j.ant))] = fad * np.sqrt(
                            10 ** (n_i.si_coef / 10))
                    else:  # j == 1 dir != 'FD'
                        data[f, j, i][np.ix_(range(n_i.ant), range(n_j.ant))] = np.zeros((n_i.ant, n_j.ant))
        self.chan_gain = data

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
        TODO: re-update it
        """
        # LaTeX type definitions
        rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
        rc('text', usetex=True)
        # Open axes
        fig, ax = plt.subplots()

        # Box positioning
        cell_out = Circle(c.coord, c.r_outer, facecolor='#45EF0605', edgecolor=(0, 0, 0), ls="--", lw=1)
        cell_in = Circle(c.coord, c.r_inner, facecolor='#37971310', edgecolor=(0, 0, 0), ls="--", lw=0.8)
        ax.add_patch(cell_out)
        ax.add_patch(cell_in)
        # User positioning
        delta = c.r_outer / 100
        plt.text(c.coord[0] + c.r_outer / 2, c.coord[1] + c.r_outer / 2, s=f'cell {c.id}', fontsize=11,
                 c='#D7D7D7')
        for n in c.node:
            plt.scatter(c.coord[0] + n.coord[0], c.coord[1] + n.coord[1],
                        c=dic.color[n.type], marker=dic.mark[n.dir], label=f'{n.type} ({n.dir})')
            plt.text(c.coord[0] + n.coord[0], c.coord[1] + n.coord[1] + delta, s=f'{n.id[1]}', fontsize=11)
        # Plot channel gain link from node to node.useful
        for n in c.node:
            if n.type in dic.user_types:
                ax = plt.gca()
                x = c.coord[0] + [n.coord[0], n.useful.coord[0]]
                y = c.coord[1] + [n.coord[1], n.useful.coord[1]]
                line = mln.Line2D(x, y, color='#CACACA', linewidth=0.4, linestyle='--')
                ax.add_line(line)
        # Set axis
        ax.axis('equal')
        ax.set_xlabel('$x$ [m]')
        ax.set_ylabel('$y$ [m]')
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
