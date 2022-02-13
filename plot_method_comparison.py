import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc

########################################
# Load data
########################################

# Load data
best_data = np.load('data/data_best_Sc1_N100.npz')
rand_data = np.load('data/data_rand_Sc2_N100.npz')
mini_data = np.load('data/data_min_Sc2_N100.npz')

# Common parameters
num_configs_range = best_data['num_configs_range']
num_inactive_ue_range = best_data['num_inactive_ue_range']

# Throughput
throughput_best = best_data['normalized_throughput']
throughput_rand = rand_data['normalized_throughput']
throughput_mini = mini_data['normalized_throughput']

throughput_aloha = throughput_best[0]

throughput = np.stack((throughput_best[1:], throughput_rand[1:], throughput_mini[1:]), axis=0)
breakpoint()
########################################
# Plot
########################################

# LaTeX type definitions
rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Open axes
fig, ax = plt.subplots()

#colors = ['#003f5c', '#bc5090', '#ffa600']

markers = ['o', 's', 'd']

colors = ['#7a5195', '#ef5675', '#ffa600']
styles = ['--', '-.', ':']
methods = ['Best', 'Random', 'Minimum']

# Slotted ALOHA
ax.plot(num_inactive_ue_range, np.nanmean(throughput_aloha, axis=-1), linewidth=2.0, linestyle='-', color='black',
        label='Slotted-ALOHA')

# Go through all methods
for mm in range(3):

    ax.plot(num_inactive_ue_range, np.nanmean(throughput[mm, 0], axis=-1), linewidth=2.0, linestyle=styles[mm],
            color=colors[mm], label=methods[mm])

    # Go through all number of configurations
    for cc, num_configs in enumerate(num_configs_range):

        if cc == 0:
            continue

        ax.plot(num_inactive_ue_range, np.nanmean(throughput[mm, cc-1], axis=-1), linewidth=2.0, marker=markers[cc-1],
                markevery=10, linestyle=styles[mm], color=colors[mm])

        #ax.fill_between(num_inactive_ue_range, np.percentile(throughput[mm, cc-1], 25, axis=-1),
        #                np.percentile(throughput[mm, cc-1], 75, axis=-1), linewidth=0, alpha=0.25, color=colors[mm])

# Go through all number of configurations
for cc, num_configs in enumerate(num_configs_range):

    if cc == 0:
        continue

    ax.plot(num_inactive_ue_range, np.nanmean(throughput[0, cc - 1], axis=-1), linewidth=None, linestyle=None,
            marker=markers[cc-1], color='black', label='RIS-assisted: $S =' + str(num_configs) + '$')

# Set axis
ax.set_xlabel(r'total number of UEs, $|\mathcal{K}|$')
ax.set_ylabel('normalized throughput')

ax.text(2000, .30, 'RIS-assisted $S=2$ coincides with\nSlotted-ALOHA for all methods')

#ax.set_title('chosen configs. = ' + str(num_chosen_configs))

# Limits
#ax.set_ylim([0, 1])

# Legend
ax.legend()

ax.lines.pop(-1)
ax.lines.pop(-1)
ax.lines.pop(-1)

# Finally
plt.grid(color='#E9E9E9', linestyle='--', linewidth=0.5)

plt.tight_layout()

plt.savefig('figs/method_comparison.pdf', dpi=300)

plt.show(block=False)
