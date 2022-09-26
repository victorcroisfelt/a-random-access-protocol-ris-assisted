from environment.box import Box

import numpy as np
from scipy.special import softmax
from scipy.constants import speed_of_light

from randomaccessfunc import collision_resolution

import time

########################################
# Preamble
########################################
seed = 42
np.random.seed(seed)

########################################
# Parameters
########################################

# Define maximum distance
maximum_distance = 100

# Noise power
noise_power = 10 ** (-94.0 / 10)  # mW

# Range of configurations
num_configs_range = np.array([2, 4, 8])

# Probability of access
prob_access = 0.001

# Total number of UEs
num_ues = 1000

# Minimum threshold value
gamma_th = 1

# Number of setups
num_setups = int(1e4)

# Range of number of elements
num_els_range = np.arange(1, 17)

# Size of each element
carrier_frequency = 3e9
wavelength = speed_of_light / carrier_frequency

size_el = wavelength

# Compute minimum distance
ris_size = num_els_range.max() * size_el
minimum_distance = (2 / wavelength) * ris_size ** 2

########################################
# Simulation
########################################
print('--------------------------------------------------')
print('Slotted ALOHA: varying elements')
print('--------------------------------------------------')

# Prepare to store number of successful attempts
num_successful_attempts = np.empty((num_configs_range.size, num_els_range.size, num_setups))
num_successful_attempts[:] = np.nan


#####


# Create a box
box = Box(maximum_distance=maximum_distance, minimum_distance=minimum_distance, rng=np.random.RandomState(seed))

# Place BS (fixed)
box.place_bs(distance=25, zenith_angle_deg=45)

# Generate the number of UEs that wish to access the network
num_active_ues_range = np.random.binomial(num_ues, prob_access, size=num_setups).astype(int)

# Go through all setups
for ss, num_active_ues in enumerate(num_active_ues_range):

    if ss % 1000 == 0:

        timer_start = time.time()

        # Print current data point
        print(f"\tSetup: {ss}/{num_active_ues_range.size - 1}")

    if not num_active_ues:
        continue

    # Place UEs
    box.place_ue(int(num_active_ues))

    # Store enumeration of active UEs
    enumeration_active_ues = np.arange(0, num_active_ues).astype(int)

    # Go through all number of configurations
    for cc, num_configs in enumerate(num_configs_range):

        # Store enumeration of configs
        enumeration_configs = np.arange(0, num_configs).astype(int)

        # Obtain channel gains
        channel_gains_dl, channel_gains_ul = box.get_channel_model_slotted_aloha()

        ##################################################
        ## Deciding access frames to transmit
        ##################################################

        # Define number of access frames
        num_access_frames = num_configs

        # Create a dictionary to store UE choices
        ue_choices = {}

        # Go through all active UEs
        for ue in enumeration_active_ues:

            # Prepare to save selected access frames
            selected = np.zeros(num_access_frames).astype(int)

            while True:

                # Flipping coins
                tosses = np.random.rand(num_access_frames)

                # Selected access frames
                selected[tosses <= 1/2] = True

                # Store list of choices
                choices = [access_frame for access_frame in enumeration_configs if selected[access_frame] == 1]

                if len(choices) >= 1:
                    break

            # Store choices
            ue_choices[ue] = choices

            del choices

        ##################################################
        ## UL - Naive UEs' responses
        ##################################################

        # Generate noise vector at the BS
        noise_bs = np.sqrt(noise_power / 2) * (np.random.randn(num_access_frames) + 1j * np.random.randn(num_access_frames))

        # Store buffered UL received signal responses as a dictionary
        buffered_access_attempts = {access_frame: {} for access_frame in range(num_access_frames)}

        # Go through all access frames
        for aa in enumeration_configs:

            # Create another dictionary
            buffered_access_attempts[aa] = {}

            # Go through all active UEs
            for ue in enumeration_active_ues:

                if aa in ue_choices[ue]:
                    buffered_access_attempts[aa][ue] = np.sqrt(box.ue.max_pow) * channel_gains_ul[0, ue] * np.sqrt(1 / 2) * (1 + 1j)

            # Add noise
            buffered_access_attempts[aa]['noise'] = noise_bs[aa]

        ##################################################
        ## Step 03. Collision Resolution Strategy
        ##################################################
        num_successful_attempts[cc, :, ss] = collision_resolution(ue_choices, buffered_access_attempts, gamma_th)

    if ss % 1000 == 0:
        print('\t[setup] elapsed ' + str(np.round(time.time() - timer_start, 4)) + ' seconds.\n')

# Save data
np.savez('data/aloha_varyingN.npz',
         num_configs_range=num_configs_range,
         num_els_range=num_els_range,
         num_successful_attempts=num_successful_attempts
         )