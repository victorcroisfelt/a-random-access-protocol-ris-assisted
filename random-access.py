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
box.place_bs(distance=50, zenith_angle_deg=30)

box.place_ris(num_configs=S, num_els_v=Na, num_els_h=Nb)

# Number of active UEs
K = 10

# Place UEs
box.place_ue(K)

box.plot_scenario()

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
best_magnitude_config_ue = np.argmin(angle_dl_beacon_ue, axis=0)

# Evaluates if this is making sense
for k in range(K):
    print("(best config, UE angle) = (" + str(np.round(np.rad2deg(box.ris.configs[best_magnitude_config_ue[k]]), 2))
          + "," +
          str(np.round(np.rad2deg(np.arctan(box.ue.pos[k, 0] / box.ue.pos[k, 1])), 2))
          + ")"
          )


breakpoint()
########################################
# Actual simulation
########################################

# Probability of access
prob_access = 0.001

# Range of number of inactive users
num_inactive_ue_range = np.arange(100, 10100, step=100)

# Number of setups
num_setups = 100

# Prepare to count number of collisions
num_collisions = np.zeros((num_inactive_ue_range.size, num_setups))
num_collisions_ris_assisted = np.zeros((num_inactive_ue_range.size, num_setups))

#####


# Get DL reflection coefficients (this is fixed)
reflection_coefficients_dl = box.get_reflection_coefficients_dl

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

        # Evaluate deterministic channel model
        channel_gains_dl, phase_shifts_dl = box.get_channel_model_dl()

        # Pilot selection
        pilot_selections = np.random.randint(0, taup, size=Ka).astype(int)

        # Generate noise vector at UEs
        noise_ue = ((sigma2 / 2) ** (1 / 2)) * (np.random.randn(Ka) + 1j * np.random.randn(Ka))

        # Compute received DL beacon of shape (S,K)
        rx_dl_beacon_ue = np.sqrt(box.bs.max_pow * taup * channel_gains_dl)[np.newaxis, :] * \
                          np.exp(1j * phase_shifts_dl).sum(axis=-1) + noise_ue[np.newaxis, :]

        # Compute angles of received DL beacon in each configuration for each UE
        angle_dl_beacon_ue = np.abs(np.angle(rx_dl_beacon_ue))

        # Find the best configuration for each UE in terms of minimum phase shift
        best_magnitude_config_ue = np.argmin(angle_dl_beacon_ue, axis=0)

        # Combine pilot selection and best configs
        temp = [(x, y) for (x, y) in zip(pilot_selections, best_magnitude_config_ue)]

        angular_based_ue = np.empty(len(temp), dtype=object)
        angular_based_ue = temp
        # TODO: why the magnitude_based_ue is defined but overwritten in one line? Might be better to simply use "magnitude_based_ue = temp.copy()" ?
        # TODO: ANN. Note that temp is a list of tuples [(,), (,)], and magnitude_based_ue is an array of tuples nd.array([(,), (,)]);  this was the best way that
        # I found to transform one to the other; if you simply do nd.array(temp), this shall return a 2 dimension array, where each col represent an entry of the tuple.
        # Note that using this construct (a,b) is desired because it is easy to compare (a,b) == (c,d) by overriding the == operator

        del temp

        _, collision_counting_ris_assisted = np.unique(angular_based_ue, return_counts=True, axis=0)

        # Count number of collisions: same pilot AND best configuration. The -1 is to disregard the non-collision case
        num_collisions_ris_assisted[ii, ss] = (collision_counting_ris_assisted - 1).sum()

        # Count number of collisions w/o RIS assistance
        _, collision_counting = np.unique(pilot_selections, return_counts=True)
        num_collisions[ii, ss] = (collision_counting - 1).sum()

########################################
# Plot
########################################

# LaTeX type definitions
rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Open axes
fig, ax = plt.subplots()

ax.plot(num_inactive_ue_range, num_collisions.mean(axis=-1), label='Classical')
ax.plot(num_inactive_ue_range, num_collisions_ris_assisted.mean(axis=-1), label='RIS-assisted: $S =' + str(S) + '$')

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
