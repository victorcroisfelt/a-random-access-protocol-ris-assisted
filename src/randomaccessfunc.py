"""Random access functions

Implementations of functions related to the random access procedure.

Authors: @victorcroisfelt, @lostinafro
Date: 28/07/2022
"""

import numpy as np


########################################
# Private functions
########################################
def bigraph_degree(edge_list: list):
    """Compute the degrees of nodes of type B stored in the second dimensions of a
    bipartite graph represented by a list of edges.

    Parameters
    ----------

    edge_list : array of tuples
        Each tuple connects a node type A to a node type B.

    degrees : dict
        Dictionary containing nodes of type B as keys and values representing
        their respective degrees.
    """

    degrees = {}

    for edge in edge_list:

        if edge[1] in degrees.keys():
            degrees[edge[1]] += 1

        else:
            degrees[edge[1]] = 1

    return degrees


########################################
# Public functions
########################################
def messages(num_ues):
    """Generate random messages for the UEs.

    Parameters
    ----------

    num_ues : int
        Number of UEs.

    Returns
    -------

    ue_messages : ndarray of complex floats with shape (num_ues, 1)
        Gaussian random generated messages.

    """

    return np.sqrt(1 / 2) * (np.random.randn(num_ues, 1) + 1j * np.random.randn(num_ues, 1))


def get_access_policy(ac_info, num_configs, num_packets=1, access_policy=None, decoding_snr=1, rng=None):
    """Implement heuristic access policies.

    Parameters
    ----------

    ac_info : ndarray of complex floats with shape (num_ues, num_configs)
        Information of the access phase obtained by the UEs through training.

    num_configs : int
        Number of RIS configurations.

    num_packets : int
        Number of packets repeated per UE.

    access_policy : str
        Chosen access policy:
            - 'RCURAP': regular repetition slotted ALOHA
            - 'RCARAP': R-configuration-aware random policy
            - 'RGSCAP': R-greedy-strongest-configuration policy
            - 'SMAP': strongest-minimum policy

    decoding_snr : float
        Threshold for successful decoding by the BS.

    rng : RandomState
        Instance of a random generator.

    Returns
    -------

    ue_choices : dict with keys being UEs and values their chosen access slots
        Choice of the UEs.
    """

    if rng is None:
        rng = np.random.RandomState()

    # Extract number of UEs
    num_ues = ac_info.shape[0]

    # Number of repeated packets
    num_packets = num_packets if num_configs >= num_packets else num_configs

    # Prepare to save chosen access slots
    ue_choices = {ue: None for ue in range(num_ues)}

    # Go through all UEs
    for ue in range(num_ues):

        # Choose access policy
        if access_policy == 'RCURAP':

            # Uniformly sample w/o replacement
            ue_choices[ue] = list(rng.choice(num_configs, size=num_packets, replace=False))

        elif access_policy == 'RCARAP':

            # Get probability mass function
            pmf = (np.abs(ac_info[ue, :])) / (np.abs(ac_info[ue, :])).sum()

            # Sample w/o replacement according to pmf
            ue_choices[ue] = list(rng.choice(num_configs, size=num_packets, replace=False, p=pmf))

        elif access_policy == 'RGSCAP':

            # Get channel qualities
            channel_qualities = np.abs(ac_info[ue, :])

            # Sorting
            argsort_channel_qualities = np.flip(np.argsort(channel_qualities))

            # Store choices
            ue_choices[ue] = list(argsort_channel_qualities[:num_packets])

        elif access_policy == 'SMAP':

            # Get channel qualities
            channel_qualities = np.abs(ac_info[ue, :])

            # Take the best
            best_idx = np.argmax(channel_qualities)

            # NaNing
            channel_qualities[best_idx] = np.nan

            if len(channel_qualities[~np.isnan(channel_qualities)]) != 0:

                # Compute inequality
                inequality = channel_qualities ** 2 - decoding_snr
                inequality[inequality < 0.0] = np.nan

                if len(inequality[~np.isnan(inequality)]) != 0:

                    # Get minimum idx
                    min_idx = np.nanargmin(inequality)

                    # Store choices
                    ue_choices[ue] = [best_idx, min_idx]

                else:
                    ue_choices[ue] = [best_idx, ]

            else:
                ue_choices[ue] = [best_idx, ]

    return ue_choices


def ul_transmission(channels_ul, ue_messages, ue_choices):
    """Get UL transmitted signal and received by the BS.

    Parameters
    ----------

    channels_ul : ndarray of complex floats with shape (num_ues, num_configs)
        UL channels.

    ue_messages : ndarray of complex floats with shape (num_ues, 1)
        Gaussian random generated messages.

    ue_choices : dict with keys being UEs and values their chosen access slots
        Choice of the UEs.

    Returns
    -------

    access_attempts : ndarray of complex floats with shape (num_configs)
        Signal received at the BS.

    bigraph : list of tuples
        Edge list of a Bipartite graph relating the UEs and the slotted that each one of them transmitted.
    """

    # Extract number of UEs and of configurations
    num_ues, num_configs = channels_ul.shape

    # Generate noise
    noise = np.sqrt(1 / 2) * (np.random.randn(num_configs, 1) + 1j * np.random.randn(num_configs, 1))

    # Prepare to save access attempts
    access_attempts = np.zeros((num_configs, 1), dtype=np.complex_)

    # Prepare to save bipartite graph
    bigraph = []

    # Go through each UE
    for ue in range(num_ues):

        # Go through UE's choice
        for ac in ue_choices[ue]:

            # Obtain received signal at the AP
            access_attempts[ac] += channels_ul[ue, ac] * ue_messages[ue, :]

            # Store in the graph
            bigraph.append((ue, ac))

    # Add noise
    access_attempts += noise

    return access_attempts, bigraph


def decoder(channels_ul, ue_messages, access_attempts, bigraph, mvu_error_ul=0, decoding_snr=1):
    """Evaluates the number of successful access attempts of the random access method given the choices made by the UEs
    and the power received by the BS.

    Parameters
    ----------

    channels_ul : ndarray of complex floats with shape (num_ues, num_configs)
        UL channels.

    ue_messages : ndarray of complex floats with shape (num_ues, 1)
        Gaussian random generated messages.

    access_attempts : ndarray of complex floats with shape (num_configs)
        Signal received at the BS.

    bigraph : list of tuples
        Edge list of a Bipartite graph relating the UEs and the slotted that each one of them transmitted.

    mvu_error_ul : float
        Error in estimating channel gains at the BS.

    decoding_snr : float
    Threshold for successful decoding by the BS.

    Returns
    -------

    access_result : dict with keys being UEs and values a list of the slots in which they successfully access
        Result after application of the BS decoder.

    """

    # Prepare to save decoding results
    access_result = {}

    while True:

        # Get graph degree as a dictionary
        degrees = bigraph_degree(bigraph)

        # No singletons, we cannot decode -> break
        if not (1 in degrees.values()):
            break

        # Get a singleton
        singleton = [(ue_idx, ac_idx) for (ue_idx, ac_idx) in bigraph if degrees[ac_idx] == 1][0]

        # Corresponding indexes
        (ue_idx, ac_idx) = singleton

        # Compute SNR of the buffered signal
        buffered_snr = np.linalg.norm(access_attempts[ac_idx])**2

        # Check SIC condition
        if buffered_snr >= decoding_snr:

            # Store results
            if ue_idx not in access_result.keys():
                access_result[ue_idx] = []

            access_result[ue_idx].append(ac_idx)

            # Reconstruct UE's signal
            reconstruction_noise = (np.sqrt(mvu_error_ul / 2) * (np.random.randn() + 1j * np.random.randn()))
            reconstructed_signal = (channels_ul[ue_idx, ac_idx] * ue_messages[ue_idx]) + reconstruction_noise

            # Identify other edges with the UE of interest
            ue_edges = [(ue, aa) for ue, aa in bigraph if ue == ue_idx]

            # Apply SIC
            for edge in ue_edges:

                # Extract access slot
                other_ac_idx = edge[1]

                # Update buffered signal
                access_attempts[other_ac_idx] -= reconstructed_signal

                # Remove edges
                bigraph.remove(edge)

        else:

            bigraph.remove((ue_idx, ac_idx))

    return access_result
