from environment.box import Box

from operator import itemgetter

import numpy as np
from scipy.special import softmax

import matplotlib.pyplot as plt
from matplotlib import rc

import networkx as nx
from networkx.algorithms import bipartite

import time

########################################
# General parameters
########################################
seed = 42
np.random.seed(seed)

# Square length
ell = 100
ell0 = 10

# Number of elements
Nz = 10  # vertical
Nx = 10  # horizontal

# Number of pilots
num_pilots = 1

# Noise power
sigma2 = 10 ** (-94.0 / 10)  # mW

########################################
# Simulation parameters
########################################

# Range of configurations
num_configs_range = np.array([2, 4, 8, 16])

# Number of setups
num_setups = int(1e4)

########################################
# Simulation
########################################

# Prepare to store probabilities
dict_probabilities = {config: [] for config in num_configs_range}


#####


# Create a box
box = Box(ell, ell0, rng=np.random.RandomState(seed))

# Place BS (fixed)
box.place_bs(distance=10, zenith_angle_deg=45)

# Place UEs
box.place_ue(num_setups)

# Go through all number of configurations
for cc, num_configs in enumerate(num_configs_range):

    # Place RIS
    box.place_ris(num_configs=num_configs, num_els_v=Nz, num_els_h=Nx)

    ##################################################
    ## Step 01. DL - Training Phase
    ##################################################

    # Get DL channel model
    channel_gains_dl, phase_shifts_dl = box.get_channel_model_dl()

    # Compute received DL beacon of shape (num_configs, num_ues)
    rx_dl_beacon = np.sqrt(box.bs.max_pow * num_pilots * channel_gains_dl)[np.newaxis, :] * \
                      np.exp(1j * phase_shifts_dl).sum(axis=-1)

    # Compute angles of received DL beacon for each configuration for each UE of shape (num_configs, num_ues)
    angle_dl_beacon = np.abs(np.angle(rx_dl_beacon))

    # Compute configuration probabilities
    config_probabilities = softmax(angle_dl_beacon, axis=0)

    # Store probabilities
    dict_probabilities[cc] = config_probabilities

########################################
# Plot
########################################

# LaTeX type definitions
rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Open axes
fig, axes = plt.subplots(nrows=2, ncols=2)

axes[0][0].bar(np.arange(0, 2), dict_probabilities[0].mean(axis=-1))
axes[0][1].bar(np.arange(0, 4), dict_probabilities[1].mean(axis=-1))
axes[1][0].bar(np.arange(0, 8), dict_probabilities[2].mean(axis=-1))
axes[1][1].bar(np.arange(0, 16), dict_probabilities[3].mean(axis=-1))

# Set axis
axes[1][0].set_xlabel('configuration')
axes[1][1].set_xlabel('configuration')

axes[0][0].set_ylabel('avg. prob. of choosing a config.')
axes[1][0].set_ylabel('avg. prob. of choosing a config.')

# Set limits
axes[0][0].set_ylim([0, 1])
axes[0][1].set_ylim([0, 1])
axes[1][0].set_ylim([0, 1])
axes[1][1].set_ylim([0, 1])

axes[0][0].set_title('$S=2$')
axes[0][1].set_title('$S=4$')
axes[1][0].set_title('$S=16$')
axes[1][1].set_title('$S=32$')

# Finally
plt.grid(color='#E9E9E9', linestyle='--', linewidth=0.5)
plt.show(block=False)

# # Open axes
# fig, axes = plt.subplots(nrows=2, ncols=2)
#
# list = [(0, 0), (0, 1), (1, 0), (1, 1)]
#
# # Go through all number of configurations
# for cc, num_configs in enumerate(num_configs_range):
#
#     for nn in range(num_configs):
#
#         axes[list[cc]].hist(x=dict_probabilities[cc][nn] + nn, bins='auto', alpha=0.7, rwidth=0.85, density=True)


# # Set axis
# ax.set_xlabel('probability of access, $P_a$')
# ax.set_ylabel('normalized throughput')
#
# # # limits
# ax.set_ylim([0, 1])
# ax.set_xscale('log')
#
# # Legend
# ax.legend()

# # Finally
# plt.grid(color='#E9E9E9', linestyle='--', linewidth=0.5)
# plt.show(block=False)

#####

fig, axes = plt.subplots(nrows=2, ncols=2)

list = [(0, 0), (0, 1), (1, 0), (1, 1)]

for cc, num_configs in enumerate(num_configs_range):
    axes[list[cc]].hist(x=dict_probabilities[cc][0], bins='auto', alpha=0.7, rwidth=0.85, density=True)

# Set axis
axes[1][0].set_xlabel('choice prob.')
axes[1][1].set_xlabel('choice prob.')

axes[0][0].set_ylabel('likelihood')
axes[1][0].set_ylabel('likelihood')

# Set limits
axes[0][0].set_xlim([0, 1])
axes[0][1].set_xlim([0, 1])
axes[1][0].set_xlim([0, 1])
axes[1][1].set_xlim([0, 1])

axes[0][0].set_title('$S=2$')
axes[0][1].set_title('$S=4$')
axes[1][0].set_title('$S=16$')
axes[1][1].set_title('$S=32$')

# Finally
plt.grid(color='#E9E9E9', linestyle='--', linewidth=0.5)
plt.show(block=False)

#####

fig, axes = plt.subplots(nrows=2, ncols=2)

list = [(0, 0), (0, 1), (1, 0), (1, 1)]

for cc, num_configs in enumerate(num_configs_range):
    axes[list[cc]].plot(np.sort(dict_probabilities[cc][0]), np.linspace(0, 1, dict_probabilities[cc][0].size))

# Set axis
axes[1][0].set_xlabel('choice prob.')
axes[1][1].set_xlabel('choice prob.')

axes[0][0].set_ylabel('likelihood cdf')
axes[1][0].set_ylabel('likelihood cdf')

# Set limits
axes[0][0].set_xlim([0, 1])
axes[0][1].set_xlim([0, 1])
axes[1][0].set_xlim([0, 1])
axes[1][1].set_xlim([0, 1])

axes[0][0].set_title('$S=2$')
axes[0][1].set_title('$S=4$')
axes[1][0].set_title('$S=16$')
axes[1][1].set_title('$S=32$')

# Finally
plt.grid(color='#E9E9E9', linestyle='--', linewidth=0.5)
plt.show(block=False)




