"""Figure2a

This script obtains the data needed to plot Fig. 2a of the paper:

V. Croisfelt, F. Saggese, I. Leyva-Mayorga, R. Kotaba, G. Gradoni and P. Popovski, "A Random Access Protocol for RIS-
Aided Wireless Communications," 2022 IEEE 23rd International Workshop on Signal Processing Advances in Wireless
Communication (SPAWC), 2022, pp. 1-5, doi: 10.1109/SPAWC51304.2022.9833984.

Authors: @victorcroisfelt, @lostinafro
Date: 28/07/2022

Specific dependencies:
    - src/box.py
    - src/randomaccessfunc.py
"""

import numpy as np
from tqdm import trange
from scipy.constants import speed_of_light

from src.box import Box
from src.randomaccessfunc import *

########################################
# Preamble
########################################
seed = 42
np.random.seed(seed)

########################################
# Define system setup
########################################

# Wave parameters
carrier_frequency = 3e9
wavelength = speed_of_light / carrier_frequency

# Number of RIS elements
num_els_ver = 10  # vertical
num_els_hor = 10  # horizontal

# Size of each element
size_el = wavelength/2

# RIS size along one of the dimensions
ris_size = num_els_hor * size_el

# Distances
maximum_distance = 100
minimum_distance = (2/wavelength) * ris_size**2

# DL transmit power
ap_tx_power_dbm = 20 # [dBm]
ap_tx_power = 10**(ap_tx_power_dbm/10) / 1000

# UL transmit power
ue_tx_power_dbm = 10 # [dBm]
ue_tx_power = 10**(ue_tx_power_dbm/10) / 1000

# Noise power
noise_power_dbm = -94 # [dBm]
noise_power = 10**(noise_power_dbm/10) / 1000

########################################
# Simulation parameters
########################################

# Minimum threshold value
decoding_snr = 10**(3/10)

# Number of setups
num_setups = int(1e4)

# Range of configurations
num_configs_range = np.arange(1, 101)

# Range of number of contending UEs
num_ues_range = np.arange(1, 21)

# Define the access policies
access_policies = ['RCURAP', 'RCARAP', 'RGSCAP']

########################################
# Simulation
########################################

# Prepare to store number of successful attempts
total_num_successful_attempts = np.empty((len(access_policies), num_configs_range.size, num_ues_range.size, num_setups))
total_num_successful_attempts[:] = np.nan


#####


# Create a box
box = Box(maximum_distance=maximum_distance, minimum_distance=minimum_distance, rng=np.random.RandomState(seed))

# Place BS
box.place_bs(distance=minimum_distance, zenith_angle_deg=45)

# Place RIS
box.place_ris(num_els_ver=num_els_ver, num_els_hor=num_els_hor, size_el=size_el)

# Go through all setups
for ss in trange(num_setups, desc="Simulating", unit="setups"):

    # Go through all different total number of UEs
    for ii, num_ues in enumerate(num_ues_range):

        # Store enumeration of active UEs
        enumeration_active_ues = np.arange(0, num_ues).astype(int)

        # Place UEs
        box.place_ue(int(num_ues))

        # Generate UEs messages
        ue_messages = messages(num_ues)

        # Go through all number of configurations
        for cc, num_configs in enumerate(num_configs_range):

            # Get configurations
            configs = np.linspace(0, np.pi/2, num_configs)

            # Get channel gains
            channel_gains_ul = box.get_channels(ue_tx_power, noise_power, configs, direction='ul')

            # Go through all access policies
            for ap in range(len(access_policies)):

                # Extract current access policy
                access_policy = access_policies[ap]

                # Apply access policy
                ue_choices = get_access_policy(
                    channel_gains_ul,
                    num_configs=num_configs,
                    access_policy=access_policy,
                    decoding_snr=decoding_snr
                )

                # Get UL transmitted messages and received signals
                access_attempts, bigraph = ul_transmission(channel_gains_ul, ue_messages, ue_choices)

                # AP decoder
                access_result = decoder(
                    channel_gains_ul,
                    ue_messages,
                    access_attempts,
                    bigraph,
                    mvu_error_ul=0,
                    decoding_snr=decoding_snr
                )

                # Access number of successful UEs
                ac_num_successful_ues = len(access_result)

                # Store simulation result
                total_num_successful_attempts[ap, cc, ii, ss] = ac_num_successful_ues

# Save data
np.savez('data/figure2.npz',
         access_policies=access_policies,
         num_configs_range=num_configs_range,
         num_ues_range=num_ues_range,
         total_num_successful_attempts=total_num_successful_attempts
         )