import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc

########################################
# Loading
########################################

# Load data
aloha_data = np.load('data/aloha.npz')
strongest_data = np.load('data/ris_strongest_N100.npz')
rand_data = np.load('data/ris_rand_N100.npz')

# Number of successful access attempts
num_successful_attempts_aloha = aloha_data['num_successful_attempts']
num_successful_attempts_strongest = strongest_data['num_successful_attempts']
num_successful_attempts_rand = rand_data['num_successful_attempts']

# Common parameters
num_configs_range = strongest_data['num_configs_range']
num_ues_range = strongest_data['num_ues_range']

# Slot duration
T = 1

# Config. duration
Tconfig = T / 10

# Prepare to store total access times
T_access_aloha = np.zeros(num_configs_range.size)
T_access_ris = np.zeros(num_configs_range.size)

# Go through all number of access frames
for aa, num_access_frame in enumerate(num_configs_range):
    T_access_aloha[aa] = (1 + num_access_frame) * T
    T_access_ris[aa] = num_access_frame * (Tconfig + 2*T)

# Prepare to compute the throughput
throughput_aloha = np.zeros(num_successful_attempts_aloha.shape)

throughput_strongest = np.zeros(num_successful_attempts_strongest.shape)
throughput_rand = np.zeros(num_successful_attempts_rand.shape)

# Go through all number of access frames
for aa, num_access_frame in enumerate(num_configs_range):

    throughput_aloha[aa] = num_successful_attempts_aloha[aa] / T_access_aloha[aa, np.newaxis, np.newaxis]

    throughput_strongest[aa] = num_successful_attempts_strongest[aa] / T_access_ris[aa, np.newaxis, np.newaxis]
    throughput_rand[aa] = num_successful_attempts_rand[aa] / T_access_ris[aa, np.newaxis, np.newaxis]

# Stack throughputs
throughput = np.stack((throughput_aloha, throughput_strongest, throughput_rand), axis=0)

########################################
# Plot
########################################

# LaTeX type definitions
rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Open axes
fig, ax = plt.subplots(figsize=(3.15, 4))

markers = ['o', 's', 'd']
colors = ['black', '#7a5195', '#ef5675']
styles = ['-', '--', '-.']
methods = ['Slotted ALOHA', 'Strongest', 'Random']

# Go through all methods
for mm in range(3):

    # Legend use
    ax.plot(num_ues_range, np.nanmean(throughput[mm, 0], axis=-1), linewidth=2.0, linestyle=styles[mm],
            color=colors[mm], label=methods[mm])

    # Go through all number of configurations
    for cc, num_configs in enumerate(num_configs_range):

        ax.plot(num_ues_range, np.nanmean(throughput[mm, cc], axis=-1), linewidth=2.0, marker=markers[cc],
                markevery=20, linestyle=styles[mm], color=colors[mm])

# Go through all number of configurations
for cc, num_configs in enumerate(num_configs_range):

    # Legend use
    ax.plot(num_ues_range, np.nanmean(throughput[0, cc], axis=-1), linewidth=None, linestyle=None,
            marker=markers[cc], color='black', label='$S =' + str(num_configs) + '$')

# Set axis
ax.set_xlabel(r'total number of UEs, $|\mathcal{K}|$')
ax.set_ylabel('average system throughput [packets/s]')

# Legend
#ax.legend(fontsize='x-small')

# Pop out some useless legend curves
ax.lines.pop(-1)
ax.lines.pop(-1)
ax.lines.pop(-1)

# Finally
plt.grid(color='#E9E9E9', linestyle='--', linewidth=0.5)

plt.tight_layout()

plt.savefig('figs/system_throughput.pdf', dpi=300)

plt.show(block=False)
