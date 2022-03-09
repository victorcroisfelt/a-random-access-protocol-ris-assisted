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
prob_access = 0.001

# Range of number of inactive users
num_inactive_ue_range = np.arange(100, 10100, step=100)

# Number of chosen configurations
num_chosen_configs = 2

# Minimum threshold value

# Number of setups
num_setups = int(1e3)

########################################
# Simulation
########################################

# Prepare to store throughput
throughput = np.zeros((num_configs_range.size, num_inactive_ue_range.size, num_setups))


#####


# Create a box
box = Box(ell, ell0, rng=np.random.RandomState(seed))

# Place BS (fixed)
box.place_bs(distance=10, zenith_angle_deg=45)

# Go through all different number of inactive UEs
for ii, num_inactive_ue in enumerate(num_inactive_ue_range):

    timer_start = time.time()

    # Print current data point
    print(f"\tinactive: {ii}/{num_inactive_ue_range.size-1}")

    # Generate the number of UEs that wish to access the network
    num_active_ue_range = np.random.binomial(num_inactive_ue, prob_access, size=num_setups).astype(int)

    # Go through all setups
    for ss, num_active_ue in enumerate(num_active_ue_range):

        if not num_active_ue:
            continue

        # Store enumeration of active UEs
        enumeration_active_ue = np.arange(0, num_active_ue).astype(int)

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

            # Get channel gains
            channel_gains_dl, channel_gains_ul = box.get_channel_model()

            ##################################################
            ## Step 01. DL - Training Phase
            ##################################################

            # Generate noise vector at UEs
            #noise_ue = np.sqrt(sigma2 / 2) * (np.random.randn(num_configs, num_active_ue) + 1j * np.random.randn(num_configs, num_active_ue))

            # Compute received DL beacon of shape (num_configs, num_ues)
            rx_dl_beacon = np.sqrt(box.bs.max_pow * num_pilots * channel_gains_dl)

            # Compute received DL SNR of shape (num_configs, num_ues)
            gamma_dl = np.abs(rx_dl_beacon)**2

            if num_chosen_configs == 1 or num_configs == 1:

                # Select the best configuration
                chosen_configs = np.argmax(gamma_dl, axis=0)

            else:

                # Compute vector of probabilities
                config_probabilities = softmax(gamma_dl, axis=0)

                # Sampling
                chosen_configs = np.zeros((num_chosen_configs, num_active_ue)).astype(int)

                for kk in range(num_active_ue):
                    chosen_configs[:, kk] = np.random.choice(enumeration_configs, size=2,
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

            # Generate noise vector at BS
            noise_bs = np.sqrt(sigma2 / 2) * (np.random.randn(num_configs) + 1j * np.random.randn(num_configs))

            # Get configs chosen by each UE
            mask_configs = np.array(list(ue_choices.values()), dtype=object)[:, 0]

            # Compute UL received signal response
            rx_ul_access_attempt = np.zeros((num_active_ue, num_chosen_configs))

            for kk in range(num_active_ue):
                rx_ul_access_attempt = np.sqrt(box.ue.max_pow * num_pilots * channel_gains_ul[mask_configs[kk], kk]) + noise_bs[mask_configs[kk]]

            ##################################################
            ## Step 03. BS analysis received signals
            ##################################################
            throughput[cc, ii, ss] = throughput_evaluation(ue_choices)

    print('\t[inactive] elapsed ' + str(np.round(time.time() - timer_start, 4)) + ' seconds.\n')

# Compute average waiting time
avg_waiting_time = num_chosen_configs/num_configs_range
avg_waiting_time[avg_waiting_time > 1.0] = 1.0

# Normalize Throughput by the waiting time
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
        ax.plot(num_inactive_ue_range, np.nanmean(normalized_throughput[cc], axis=-1), label='Slotted-ALOHA')

    else:
        ax.plot(num_inactive_ue_range, np.nanmean(normalized_throughput[cc], axis=-1), label='RIS-assisted: $S =' + str(num_configs) + '$')

# Set axis
ax.set_xlabel('number of inactive UEs')
ax.set_ylabel('normalized throughput')

ax.set_title('chosen configs. = ' + str(num_chosen_configs))

# Limits
#ax.set_ylim([0, 1])

# Legend
ax.legend()

# Finally
plt.grid(color='#E9E9E9', linestyle='--', linewidth=0.5)
plt.show(block=False)
