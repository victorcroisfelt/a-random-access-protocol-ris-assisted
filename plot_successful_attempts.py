import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc

########################################
# Loading
########################################
#trial
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

# Stack
num_successful_attempts = np.stack((num_successful_attempts_aloha, num_successful_attempts_strongest, num_successful_attempts_rand), axis=0)

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
    ax.plot(num_ues_range, np.nanmean(num_successful_attempts[mm, 0], axis=-1), linewidth=2.0, linestyle=styles[mm],
            color=colors[mm], label=methods[mm])

    # Go through all number of configurations
    for cc, num_configs in enumerate(num_configs_range):

        ax.plot(num_ues_range, np.nanmean(num_successful_attempts[mm, cc], axis=-1), linewidth=2.0, marker=markers[cc],
                markevery=20, linestyle=styles[mm], color=colors[mm])

# Go through all number of configurations
for cc, num_configs in enumerate(num_configs_range):

    # Legend use
    ax.plot(num_ues_range, np.nanmean(num_successful_attempts[0, cc], axis=-1), linewidth=None, linestyle=None,
            marker=markers[cc], color='black', label='$S =' + str(num_configs) + '$')

# Set axis
ax.set_xlabel(r'total number of UEs, $|\mathcal{K}|$')
ax.set_ylabel(r'average number of successful attempts [packets]')

# Legend
ax.legend(fontsize='small', framealpha=0.5)

# Pop out some useless legend curves
ax.lines.pop(-1)
ax.lines.pop(-1)
ax.lines.pop(-1)

# Finally
plt.grid(color='#E9E9E9', linestyle='--', linewidth=0.5)

plt.tight_layout()

plt.savefig('figs/num_successful_attempts.pdf', dpi=300)

plt.show(block=False)






# #colors = ['#003f5c', '#bc5090', '#ffa600']
#

#
# colors = ['black', '#7a5195', '#ef5675', '#ffa600']
# styles = ['-', '--', '-.', ':']
# methods = ['Slotted ALOHA', 'Strongest', 'Strongest + Minimum', 'Random']
# ax.fill_between(num_inactive_ue_range, np.percentile(throughput[mm, cc-1], 25, axis=-1),
#                np#ax.text(2000, .30, 'RIS-assisted $S=2$ coincides with\nSlotted-ALOHA for all methods')
#
# #ax.set_title('chosen configs. = ' + str(num_chosen_configs))
#
# # Limits
# #ax.set_ylim([0, 1]).percentile(throughput[mm, cc-1], 75, axis=-1), linewidth=0, alpha=0.25, color=colors[mm])