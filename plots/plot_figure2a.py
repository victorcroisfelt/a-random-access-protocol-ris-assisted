"""Plot Figure 2a

This script plots Fig. 2a of the paper:

V. Croisfelt, F. Saggese, I. Leyva-Mayorga, R. Kotaba, G. Gradoni and P. Popovski, "A Random Access Protocol for RIS-
Aided Wireless Communications," 2022 IEEE 23rd International Workshop on Signal Processing Advances in Wireless
Communication (SPAWC), 2022, pp. 1-5, doi: 10.1109/SPAWC51304.2022.9833984.

Authors: @victorcroisfelt, @lostinafro
Date: 28/07/2022

Specific dependencies:
    - ../data/figure2.npz: data file generated by running sim_figure2.py
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc

# LaTeX type definitions
rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern'], 'size': 14})
rc('text', usetex=True)

########################################
# Loading
########################################

# Load data
data = np.load('../data/figure2.npz')

num_ues_range = data['num_ues_range']
num_configs_range = data['num_configs_range']

total_num_successful_attempts = data['total_num_successful_attempts']

# Take average
avg_num_successful_attempts = np.nanmean(total_num_successful_attempts, axis=-1)

# Mask S==4 and S==8
num_configs_range_ = [4, 8]
avg_num_successful_attempts_ = avg_num_successful_attempts[:, [3, 7], :]

########################################
# Plot
########################################
fig, ax = plt.subplots()

markers = ['o', 's', '*']
marker_sizes = [4, 6, 8]
marker_starts = [0, 1, 2]

colors = ['black', '#ef5675', '#7a5195']
styles = ['-', '--', ':']
methods = ['URP', 'CARP', 'SCP']

# Go through all methods
for mm in range(3):

    # Go through all number of configurations
    for cc, num_configs in enumerate(num_configs_range_):

        ax.plot(num_ues_range, avg_num_successful_attempts_[mm, cc] / num_configs, linewidth=2,
                marker=markers[cc], markevery=2, markersize=marker_sizes[cc], linestyle=styles[mm], color=colors[mm])

# Legend use
for cc, num_configs in enumerate(num_configs_range_):

    ax.plot(num_ues_range, avg_num_successful_attempts_[0, cc] / num_configs, linewidth=None, linestyle=None,
            markevery=1, marker=markers[cc], markersize=marker_sizes[cc], color='black',
            label='$S =' + str(num_configs) + '$')

# Set axis
ax.set_xlabel(r'number of contending UEs, $K$')
ax.set_ylabel(r'avg. $\mathrm{SA}$ [pkt./slot]')

# Legend
ax.legend(fontsize='x-small', framealpha=0.5, loc=1)

#ax.set_xticks(np.arange(1, 11))

# Pop out some useless legend curves
ax.lines.pop(-1)
ax.lines.pop(-1)

# Finally
plt.grid(color='#E9E9E9', linestyle='--', linewidth=0.5)

plt.tight_layout()

plt.show()