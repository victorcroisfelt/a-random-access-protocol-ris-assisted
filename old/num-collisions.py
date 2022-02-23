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
Na = 10  # vertical
Nb = 10  # horizontal

# Number of configurations
S = 4

# Number of pilots
taup = 10

# Noise power
sigma2 = 10 ** (-94.0 / 10)  # mW

########################################
# Simulation parameters
########################################

# Initialize environment
# TODO: the generation of users uses an internal random generator as suggested by Numpy new "good practice".
#  To keep the coexistence for the seed you need  to add a specific RandomState as key argument (as below)
box = Box(ell, ell0, rng=np.random.RandomState(seed))

# Place BS and RIS (fixed entities)
box.place_bs(distance=10, zenith_angle_deg=45)

box.place_ris(num_configs=S, num_els_v=Na, num_els_h=Nb)

box.ris.plot()

breakpoint()
# Number of active UEs
K = 10

# Place UEs
box.place_ue(K)

#box.plot_scenario()

# Evaluate deterministic channel model
#   channel_gains_dl : ndarray of shape (num_ues, )
#   phase_shifts_dl : ndarray of shape (num_configs, num_ues, num_els)
channel_gains_dl, phase_shifts_dl = box.get_channel_model_dl()

# Pilot selection
pilot_selections = np.random.randint(0, taup, size=K).astype(int)

# Generate noise vector at UEs
noise_ue = ((sigma2 / 2) ** (1 / 2)) * (np.random.randn(K) + 1j * np.random.randn(K))

# Compute received DL beacon of shape (S,K)
rx_dl_beacon_ue = np.exp(1j * phase_shifts_dl).sum(axis=-1)

#np.sqrt(box.bs.max_pow * taup * channel_gains_dl)[np.newaxis, :] * np.exp(1j * phase_shifts_dl).sum(axis=-1)  # + noise_ue[np.newaxis, :]

# Compute angles of received DL beacon in each configuration for each UE
angle_dl_beacon_ue = np.abs(np.angle(rx_dl_beacon_ue))

# Find the best configuration for each UE in terms of minimum phase shift
best_magnitude_config_ue = np.argmax(angle_dl_beacon_ue, axis=0)

# Evaluates if this is making sense
for k in range(K):
    print("(best config, UE angle) = (" + str(np.round(np.rad2deg(box.ris.configs[best_magnitude_config_ue[k]]), 2))
          + "," +
          str(np.round(np.rad2deg(np.arctan(box.ue.pos[k, 0] / box.ue.pos[k, 1])), 2))
          + ")"
          )


#breakpoint()
########################################
# Actual simulation
########################################

# Range of configurations
num_configs_range = np.array([4, 8, 16, 32])

# Probability of access
prob_access = 0.001

# Range of number of inactive users
num_inactive_ue_range = np.arange(100, 10100, step=100)

# Number of setups
num_setups = int(1e4)

# Prepare to count number of collisions
num_collisions = np.zeros((num_inactive_ue_range.size, num_setups))
num_collisions_ris_assisted_ideal = np.zeros((num_configs_range.size, num_inactive_ue_range.size, num_setups))
num_collisions_ris_assisted_corrupted = np.zeros((num_configs_range.size, num_inactive_ue_range.size, num_setups))

# Prepare to save probability of collisions
prob_collisions = np.zeros((num_inactive_ue_range.size, num_setups))
prob_collisions_ris_assisted_ideal = np.zeros((num_configs_range.size, num_inactive_ue_range.size, num_setups))
prob_collisions_ris_assisted_corrupted = np.zeros((num_configs_range.size, num_inactive_ue_range.size, num_setups))


#####


# Place BS and RIS (fixed entities)
box.place_bs(distance=10, zenith_angle_deg=45)

# Go through all different number of inactive UEs
for ii, num_inactive_ue in enumerate(num_inactive_ue_range):

    # Generate the number of UEs that wish to access the network
    num_active_ue_range = np.random.binomial(num_inactive_ue, prob_access, size=num_setups).astype(int)

    # Go through all setups
    for ss, Ka in enumerate(num_active_ue_range):
        # TODO: this is not the most efficient to do this, but just to implement the basics and get a sense

        if not Ka:
            continue

        # Place UEs
        box.place_ue(int(Ka))  # TODO: needed to use int here since Ka is of type numpy.int32

        # Pilot selection
        pilot_selections = np.random.randint(0, taup, size=Ka).astype(int)

        # Count number of collisions w/o RIS assistance
        _, collision_counting = np.unique(pilot_selections, return_counts=True)
        collision_counting[collision_counting <= 1.0] = 0.0

        num_collisions[ii, ss] = collision_counting.sum()

        # Go through all number of configurations
        for cc, num_configs in enumerate(num_configs_range):

            # Place RIS
            box.place_ris(num_configs=num_configs, num_els_v=Na, num_els_h=Nb)

            # Evaluate deterministic channel model
            channel_gains_dl, phase_shifts_dl = box.get_channel_model_dl()

            # Generate noise vector at UEs
            noise_ue = np.sqrt(sigma2 / 2) * (np.random.randn(num_configs, Ka) + 1j * np.random.randn(num_configs, Ka))

            # Compute received DL beacon of shape (S,K)
            rx_dl_beacon_ue = np.sqrt(box.bs.max_pow * taup * channel_gains_dl)[np.newaxis, :] * \
                              np.exp(1j * phase_shifts_dl).sum(axis=-1)

            rx_dl_beacon_ue_corrupted = rx_dl_beacon_ue + noise_ue

            # Compute angles of received DL beacon for each configuration for each UE
            angle_dl_beacon_ue = np.abs(np.angle(rx_dl_beacon_ue))
            angle_dl_beacon_ue_corrupted = np.abs(np.angle(rx_dl_beacon_ue_corrupted))

            # Find the best configuration for each UE
            best_magnitude_config_ue = np.argmax(angle_dl_beacon_ue, axis=0)
            best_magnitude_config_ue_corrupted = np.argmax(angle_dl_beacon_ue_corrupted, axis=0)

            # Combine pilot selection and best configs
            temp1 = [(x, y) for (x, y) in zip(pilot_selections, best_magnitude_config_ue)]
            temp2 = [(x, y) for (x, y) in zip(pilot_selections, best_magnitude_config_ue_corrupted)]

            # Prepare and save
            angular_based_ue = np.empty(len(temp1), dtype=object)
            angular_based_ue_corrupted = np.empty(len(temp2), dtype=object)

            angular_based_ue = temp1
            angular_based_ue_corrupted = temp2

            del temp1, temp2

            # Count number of collisions: same pilot AND best configuration.
            # The -1 is to disregard the non-collision case
            _, collision_counting_ris_assisted = np.unique(angular_based_ue, return_counts=True, axis=0)
            collision_counting_ris_assisted[collision_counting_ris_assisted <= 1.0] = 0.0

            _, collision_counting_ris_assisted_corrupted = np.unique(angular_based_ue_corrupted, return_counts=True, axis=0)
            collision_counting_ris_assisted_corrupted[collision_counting_ris_assisted_corrupted <= 1.0] = 0.0

            num_collisions_ris_assisted_ideal[cc, ii, ss] = collision_counting_ris_assisted.sum()
            num_collisions_ris_assisted_corrupted[cc, ii, ss] = collision_counting_ris_assisted_corrupted.sum()

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