from environment.box import Box

import numpy as np
from scipy.special import softmax

import matplotlib.pyplot as plt
from matplotlib import rc

from randomaccessfunc import throughput_evaluation

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
num_configs_range = np.array([1, 2, 4, 8, 16, 32])

# Probability of access
prob_access_range = np.logspace(-3, -1)

# Range of number of inactive users
num_inactive_ue = 1000

# Number of chosen configurations
num_chosen_configs = 2

# Minimum threshold value

# Number of setups
num_setups = int(1e3)

########################################
# Simulation
########################################

# Prepare to store throughput
throughput = np.zeros((num_configs_range.size, prob_access_range.size, num_setups))


#####


# Create a box
box = Box(ell, ell0, rng=np.random.RandomState(seed))

# Place BS (fixed)
box.place_bs(distance=10, zenith_angle_deg=45)

# Go through all different probabilities of access
for ii, prob_access in enumerate(prob_access_range):

    timer_start = time.time()

    # Print current data point
    print(f"\tprobability: {ii}/{prob_access_range.size-1}")

    # Generate the number of UEs that wish to access the network
    num_active_ue_range = np.random.binomial(num_inactive_ue, prob_access, size=num_setups).astype(int)

    # Go through all setups
    for ss, num_active_ue in enumerate(num_active_ue_range):

        if not num_active_ue:
            continue

        # Place UEs
        box.place_ue(int(num_active_ue))

        # Pilot selection
        pilot_selections = np.random.randint(0, num_pilots, size=num_active_ue).astype(int)

        # Go through all number of configurations
        for cc, num_configs in enumerate(num_configs_range):

            # Store enumeration of configs
            enumeration_configs = np.arange(0, num_configs).astype(int)

            # Place RIS
            box.place_ris(num_configs=num_configs, num_els_v=Nz, num_els_h=Nx)

            ##################################################
            ## Step 01. DL - Training Phase
            ##################################################

            # Get Downlink channel model
            channel_gains_dl, phase_shifts_dl = box.get_channel_model_dl()

            # Generate noise vector at UEs
            #noise_ue = np.sqrt(sigma2 / 2) * (np.random.randn(num_configs, num_active_ue) + 1j * np.random.randn(num_configs, num_active_ue))

            # Compute received DL beacon of shape (num_configs, num_ues)
            rx_dl_beacon = np.sqrt(box.bs.max_pow * num_pilots * channel_gains_dl)[np.newaxis, :] * \
                              np.exp(1j * phase_shifts_dl).sum(axis=-1)

            # Compute angles of received DL beacon for each configuration for each UE of shape (num_configs, num_ues)
            angle_dl_beacon = np.abs(np.angle(rx_dl_beacon))

            if num_chosen_configs == 1 or num_configs == 1:

                # Select the best configuration
                chosen_configs = np.argmax(angle_dl_beacon, axis=0)

            else:

                # Compute vector of probabilities
                config_probabilities = softmax(angle_dl_beacon, axis=0)

                # Sampling
                chosen_configs = np.zeros((num_chosen_configs, num_active_ue)).astype(int)

                for kk in range(num_active_ue):
                    chosen_configs[:, kk] = np.random.choice(enumeration_configs, size=num_chosen_configs,
                                                             replace=False, p=config_probabilities[:, kk]).astype(int)

            # Create a dictionary to store UE choices
            ue_choices = {}

            for kk in range(num_active_ue):

                if num_chosen_configs == 1 or num_configs == 1:

                    if chosen_configs.shape == ():
                        ue_choices[kk] = (chosen_configs.item(), pilot_selections[kk])

                    else:
                        ue_choices[kk] = (chosen_configs[kk], pilot_selections[kk])

                else:

                    ue_choices[kk] = (list(chosen_configs[:, kk]), pilot_selections[kk])

            ##################################################
            ## Step 02. UL - Naive UL response
            ##################################################

            # Get Uplink channel model
            channel_gains_ul, phase_shifts_ul = box.get_channel_model_ul()

            # Generate noise vector at BS
            noise_bs = np.sqrt(sigma2 / 2) * (np.random.randn(num_configs) + 1j * np.random.randn(num_configs))

            # Get configs chosen by each UE
            mask_configs = np.array(list(ue_choices.values()), dtype=object)[:, 0]

            # Compute UL received signal response
            rx_ul_access_attempt = np.zeros((num_active_ue, num_chosen_configs))

            for kk in range(num_active_ue):
                rx_ul_access_attempt = np.sqrt(box.ue.max_pow * num_pilots * channel_gains_ul[kk]) *\
                                       np.exp(1j * phase_shifts_ul[mask_configs[kk], kk, :]).sum(axis=-1)

            ##################################################
            ## Step 03. BS analysis received signals
            ##################################################
            throughput[cc, ii, ss] = throughput_evaluation(ue_choices)

    print('\t[p_access] elapsed ' + str(np.round(time.time() - timer_start, 4)) + ' seconds.\n')

# Compute average waiting time
avg_waiting_time = num_chosen_configs/num_configs_range
avg_waiting_time[avg_waiting_time > 1.0] = 1.0

# Normalize throughput by the waiting time
normalized_throughput = avg_waiting_time[:, np.newaxis, np.newaxis] * throughput

########################################
# Plot
########################################

# LaTeX type definitions
rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Open axes
fig, ax = plt.subplots()

# Go through all number of configurations
for cc, num_configs in enumerate(num_configs_range):

    if num_configs == 1:
        ax.plot(prob_access_range, np.nanmean(normalized_throughput[cc], axis=-1), label='Slotted ALOHA')

    else:
        ax.plot(prob_access_range, np.nanmean(normalized_throughput[cc], axis=-1), label='RIS-assisted: $S =' + str(num_configs) + '$')

# Set axis
ax.set_xlabel('probability of access, $P_a$')
ax.set_ylabel('normalized throughput')

# # limits
ax.set_ylim([0, 1])
ax.set_xscale('log')

# Legend
ax.legend()

# Finally
plt.grid(color='#E9E9E9', linestyle='--', linewidth=0.5)
plt.show(block=False)
