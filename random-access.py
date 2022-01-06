from environment.box import Box
import numpy as np

import time

########################################
# General parameters
########################################

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
box = Box(ell, ell0)

# Place BS and RIS (fixed entities)
box.place_bs(pos_polar=np.array([[30, np.deg2rad(90 + 45)]]))
box.place_ris(num_configs=S, v_els=Na, h_els=Nb)

# Number of active UEs
K = 10

# Place UEs
box.place_ue(K)

# Evaluate deterministic channel model
channel_gains_dl, channel_gains_ul, phase_shifts_bs, phase_shifts_ue = box.get_channel_model()

# Get DL reflection coefficients
reflection_coefficients_dl = box.get_reflection_coefficients_dl
#box.plot_reflection_coefficients(reflection_coefficients_dl)

# Pilot selection
pilot_selections = np.random.randint(0, taup, size=K).astype(int)

# Generate noise vector at UEs
noise_ue = ((sigma2 / 2) ** (1 / 2)) * (np.random.randn(K) + 1j * np.random.randn(K))

# Compute total DL phase shifts of shape (S,K,N)
phase_shifts_dl = reflection_coefficients_dl[:, np.newaxis, :] - phase_shifts_bs[np.newaxis, np.newaxis, :] - \
                  phase_shifts_ue[np.newaxis, :, :]  # TODO: Check if signs are correct

# Compute received DL beacon of shape (S,K)
rx_dl_beacon_ue = np.sqrt(box.bs.max_pow * taup * channel_gains_dl)[np.newaxis, :] * \
                  np.exp(1j * phase_shifts_dl.sum(axis=-1)) + noise_ue[np.newaxis, :]

# Compute magnitude of received DL beacon in each configuration for each UE
magnitude_dl_beacon_ue = np.abs(rx_dl_beacon_ue)

# Find the best configuration for each UE in terms of best magnitude
best_magnitude_config_ue = np.argmax(magnitude_dl_beacon_ue, axis=0)

# Evaluates if this is making sense

print("UE angles = ", np.round(np.rad2deg(np.arctan(box.ue.pos[:, 1] / box.ue.pos[:, 0])), 2))
print("Best configs = ", best_magnitude_config_ue)

for cc, config in enumerate(box.ris.set_configs):
    print("RIS config " + str(cc) + " is centered at = ", np.round(np.rad2deg(box.ris.set_configs[cc]), 2))

breakpoint()

box.plot_scenario()

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

#####


# Get DL reflection coefficients (this is fixed)
reflection_coefficients_dl = box.get_reflection_coefficients_dl

# Go through all different number of inactive UEs
for ii, num_inactive_ue in enumerate(num_inactive_ue_range):

    # Generate the number of UEs that wish to access the network
    num_active_ue_range = np.random.binomial(num_inactive_ue, prob_access, size=num_setups).astype(int)

    # Go through all setups
    for ss, Ka in enumerate(
            num_active_ue_range):  # TODO: this is not the most efficient to do this, but just to implement the basics and get a sense

        if not Ka:
            continue

        # Place UEs
        box.place_ue(int(Ka))  # TODO: needed to use int here since Ka is of type numpy.int32

        # Evaluate deterministic channel model
        channel_gains_dl, channel_gains_ul, phase_shifts_bs, phase_shifts_ue = box.get_channel_model()

        # Pilot selection
        pilot_selections = np.random.randint(0, taup, size=Ka).astype(int)

        # Generate noise vector at UEs
        noise_ue = ((sigma2 / 2) ** (1 / 2)) * (np.random.randn(Ka) + 1j * np.random.randn(Ka))

        # Compute total DL phase shifts of shape (S,Ka,N)
        phase_shifts_dl = reflection_coefficients_dl[:, np.newaxis, :] - phase_shifts_bs[np.newaxis, np.newaxis, :] - \
                          phase_shifts_ue[np.newaxis, :, :]  # TODO: Check if signs are correct
        phase_shifts_dl = np.mod(phase_shifts_dl, 2 * np.pi)

        # Compute received DL beacon of shape (S,K)
        rx_dl_beacon_ue = np.sqrt(box.bs.max_pow * taup * channel_gains_dl)[np.newaxis, :] * \
                          np.exp(1j * phase_shifts_dl.sum(axis=-1)) + noise_ue[np.newaxis, :]

        # Compute magnitude of received DL beacon in each configuration for each UE
        magnitude_dl_beacon_ue = np.abs(rx_dl_beacon_ue)

        # Find the best configuration for each UE in terms of best magnitude
        best_magnitude_config_ue = np.argmax(magnitude_dl_beacon_ue, axis=0)

        # Count number of collisions


        breakpoint()
