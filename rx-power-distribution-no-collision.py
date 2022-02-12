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

# Number of setups
num_setups = int(1e4)

########################################
# Simulation
########################################

# Prepare to store receiver powers
rx_powers = {config: [] for config in num_configs_range}


#####


# Create a box
box = Box(ell, ell0, rng=np.random.RandomState(seed))

# Place BS (fixed)
box.place_bs(distance=10, zenith_angle_deg=45)

# Place UEs
box.place_ue(num_setups)

# Go through all number of configurations
for cc, num_configs in enumerate(num_configs_range):

    # Place RIS
    box.place_ris(num_configs=num_configs, num_els_v=Nz, num_els_h=Nx)


    ## Step 01. DL - Training Phase


    # Get Downlink channel model
    channel_gains_dl, phase_shifts_dl = box.get_channel_model_dl()

    # Compute received DL beacon of shape (num_configs, num_ues)
    rx_dl_beacon_ue = np.sqrt(box.bs.max_pow * num_pilots * channel_gains_dl)[np.newaxis, :] * \
                      np.exp(1j * phase_shifts_dl).sum(axis=-1)

    # Compute angles of received DL beacon for each configuration for each UE
    angle_dl_beacon_ue = np.abs(np.angle(rx_dl_beacon_ue))

    # Find the best configuration for each UE
    best_magnitude_config_ue = np.argmax(angle_dl_beacon_ue, axis=0)


    ## Step 02. UL - Naive UL response


    # Get Uplink channel model
    channel_gains_ul, phase_shifts_ul = box.get_channel_model_ul()

    # Prepare to save received UL access attempt of shape (num_configs, num_pilots)
    rx_ul_access_attempt = np.zeros(num_setups)

    # Go through all choices
    for k, choice in enumerate(best_magnitude_config_ue):

        # Compute UL received signal response
        rx_ul_access_attempt[k] += np.sqrt(box.ue.max_pow * num_pilots * channel_gains_ul[k]) * \
                      np.exp(1j * phase_shifts_ul[choice, k, :]).sum(axis=-1)

    # Store received powers
    rx_powers[num_configs].append(list(rx_ul_access_attempt))

# Processed rx powers
rx_powers_proc = {config: [] for config in num_configs_range}

# Go through all number of configurations
for cc, num_configs in enumerate(num_configs_range):
    rx_powers_proc[num_configs] = np.array([item for sublist in rx_powers[num_configs] for item in sublist])

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
    ax.plot(10 * np.log10(np.sort(rx_powers_proc[num_configs]**2)) + 94, np.linspace(0, 1, num=rx_powers_proc[num_configs].size),
            label='RIS-assisted: $S=' + str(num_configs) + '$')

# Set axis
ax.set_xlabel('UL received power [dB]')
ax.set_ylabel('CDF')

# Legend
ax.legend()

# Finally
plt.grid(color='#E9E9E9', linestyle='--', linewidth=0.5)
plt.show(block=False)
