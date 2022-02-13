from environment.box import Box
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc

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
num_pilots = 10

# Noise power
sigma2 = 10 ** (-94.0 / 10)  # mW

########################################
# Simulation parameters
########################################

# Range of configurations
num_configs_range = np.array([4, 8, 16, 32])

# Probability of access
prob_access = 0.001

# Range of number of inactive users
num_inactive_ue_range = np.arange(100, 10100, step=100)

# Number of setups
num_setups = int(1e3)

########################################
# Simulation
########################################

# Prepare to count number of collisions
num_collisions = np.zeros((num_inactive_ue_range.size, num_setups))
num_collisions_ris_assisted_ideal = np.zeros((num_configs_range.size, num_inactive_ue_range.size, num_setups))
num_collisions_ris_assisted_corrupted = np.zeros((num_configs_range.size, num_inactive_ue_range.size, num_setups))

# Prepare to save probability of collisions
prob_collisions = np.zeros((num_inactive_ue_range.size, num_setups))
prob_collisions_ris_assisted_ideal = np.zeros((num_configs_range.size, num_inactive_ue_range.size, num_setups))
prob_collisions_ris_assisted_corrupted = np.zeros((num_configs_range.size, num_inactive_ue_range.size, num_setups))


#####


# Create a box
box = Box(ell, ell0, rng=np.random.RandomState(seed))

# Place BS (fixed)
box.place_bs(distance=10, zenith_angle_deg=45)

# Go through all different number of inactive UEs
for ii, num_inactive_ue in enumerate(num_inactive_ue_range):

    # Generate the number of UEs that wish to access the network
    num_active_ue_range = np.random.binomial(num_inactive_ue, prob_access, size=num_setups).astype(int)

    # Go through all setups
    for ss, num_active_ue in enumerate(num_active_ue_range):
        # TODO: this is not the most efficient to do this, but just to implement the basics and get a sense

        if not num_active_ue:
            continue
        num_active_ue = 4

        # Place UEs
        box.place_ue(int(num_active_ue))  # TODO: needed to use int here since Ka is of type numpy.int32

        # Pilot selection
        pilot_selections = np.random.randint(0, num_pilots, size=num_active_ue).astype(int)

        # Go through all number of configurations
        for cc, num_configs in enumerate(num_configs_range):

            # Place RIS
            box.place_ris(num_configs=num_configs, num_els_v=Nz, num_els_h=Nx)


            ## Step 01. DL - Training Phase


            # Get Downlink channel model
            channel_gains_dl, phase_shifts_dl = box.get_channel_model_dl()

            # Generate noise vector at UEs
            noise_ue = np.sqrt(sigma2 / 2) * (np.random.randn(num_configs, num_active_ue) + 1j * np.random.randn(num_configs, num_active_ue))

            # Compute received DL beacon of shape (num_configs, num_ues)
            rx_dl_beacon_ue = np.sqrt(box.bs.max_pow * num_pilots * channel_gains_dl)[np.newaxis, :] * \
                              np.exp(1j * phase_shifts_dl).sum(axis=-1)

            # Compute angles of received DL beacon for each configuration for each UE
            angle_dl_beacon_ue = np.abs(np.angle(rx_dl_beacon_ue))

            # Find the best configuration for each UE
            best_magnitude_config_ue = np.argmax(angle_dl_beacon_ue, axis=0)

            # Create a dictionary to store choices of UEs
            ue_choices = {}

            for k in range(num_active_ue):
                ue_choices[k] = (best_magnitude_config_ue[k], pilot_selections[k])


            ## Step 02. UL - Naive UL response


            # Get Uplink channel model
            channel_gains_ul, phase_shifts_ul = box.get_channel_model_ul()

            # Generate noise vector at BS
            noise_bs = np.sqrt(sigma2 / 2) * (np.random.randn(num_configs) + 1j * np.random.randn(num_configs))

            # Prepare to save received UL access attempt of shape (num_configs, num_pilots)
            rx_ul_access_attempt = np.zeros((num_configs, num_pilots))


            # Go through ue_choices
            for k, choice in ue_choices.items():

                # Compute UL received signal response
                rx_ul_access_attempt[choice[0], choice[1]] += np.sqrt(box.ue.max_pow * num_pilots * channel_gains_ul[k]) * \
                              np.exp(1j * phase_shifts_ul[choice[0], k, :]).sum(axis=-1)

            rx_ul_access_attempt[rx_ul_access_attempt == 0.0] = np.nan




            breakpoint()


    # Compute average probabilities
    prob_collisions[ii, :] = num_collisions[ii, :] / num_active_ue_range[np.newaxis, :]

    prob_collisions_ris_assisted_ideal[:, ii, :] = num_collisions_ris_assisted_ideal[:, ii, :] / num_active_ue_range[np.newaxis, np.newaxis, :]
    prob_collisions_ris_assisted_corrupted[:, ii, :] = num_collisions_ris_assisted_corrupted[:, ii, :] / num_active_ue_range[np.newaxis, np.newaxis, :]

########################################
# Plot
########################################

# LaTeX type definitions
rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Open axes
fig, ax = plt.subplots()

ax.plot(num_inactive_ue_range, num_collisions.mean(axis=-1), label='Classical')

ax.plot(num_inactive_ue_range, num_collisions_ris_assisted_ideal[0].mean(axis=-1), label='RIS-assisted: $S = 4$')
ax.plot(num_inactive_ue_range, num_collisions_ris_assisted_ideal[1].mean(axis=-1), label='RIS-assisted: $S = 8$')
ax.plot(num_inactive_ue_range, num_collisions_ris_assisted_ideal[2].mean(axis=-1), label='RIS-assisted: $S = 16$')
ax.plot(num_inactive_ue_range, num_collisions_ris_assisted_ideal[3].mean(axis=-1), label='RIS-assisted: $S = 32$')

# Set axis
ax.set_xlabel('number of inactive UEs')
ax.set_ylabel('average number of collisions')

# # limits
# ax.set_ylim(ymin=-self.ell0 / 2)

# Legend
ax.legend()

# Finally
plt.grid(color='#E9E9E9', linestyle='--', linewidth=0.5)
plt.show(block=False)


##

fig, ax = plt.subplots()

ax.plot(num_inactive_ue_range, np.nanmean(prob_collisions, axis=-1), label='Classical')

ax.plot(num_inactive_ue_range, np.nanmean(prob_collisions_ris_assisted_ideal[0], axis=-1), label='RIS-assisted: $S = 4$')
ax.plot(num_inactive_ue_range, np.nanmean(prob_collisions_ris_assisted_ideal[1], axis=-1), label='RIS-assisted: $S = 8$')
ax.plot(num_inactive_ue_range, np.nanmean(prob_collisions_ris_assisted_ideal[2], axis=-1), label='RIS-assisted: $S = 16$')
ax.plot(num_inactive_ue_range, np.nanmean(prob_collisions_ris_assisted_ideal[3], axis=-1), label='RIS-assisted: $S = 32$')

# Set axis
ax.set_xlabel('number of inactive UEs')
ax.set_ylabel('average probability of collisions')

# # limits
# ax.set_ylim(ymin=-self.ell0 / 2)

# Legend
ax.legend()

# Finally
plt.grid(color='#E9E9E9', linestyle='--', linewidth=0.5)
plt.show(block=False)

