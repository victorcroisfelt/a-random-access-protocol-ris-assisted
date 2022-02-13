import numpy as np

import networkx as nx
from networkx.algorithms import bipartite


def throughput_evaluation(ue_choices):
    """Evaluates the throughput of the random access method given the choices made by the UEs.

    Parameters
    ----------
    ue_choices : dict
        Dictionary containing choices of the UEs.
            keys -> UE indexes
            values -> tuple,
                1st-dimension contains a list or an integer with the configurations chosen by a UE.
                2nd-dimension contains the pilot chosen by a UE.

    Returns
    -------
    throughput : float
        Ratio of success attempts per number of UEs trying to access.

    """

    # Get number of active UEs
    num_active_ue = len(list(ue_choices.keys()))

    # Nothing to compute
    if num_active_ue == 1:
        return 1

    # Get mask of pilots
    mask_pilots = np.array(list(ue_choices.values()), dtype=object)[:, 1]

    # Get active pilots
    active_pilots = np.unique(mask_pilots)

    # Pool of success UEs
    success_pool = []

    # Go through all active pilots
    for pilot in active_pilots:

        # Create a Bipartite graph
        B = nx.Graph()

        # Get index of colliding UEs
        colliding_ues = np.array(list(ue_choices.keys()))[mask_pilots == pilot]

        # Add colliding UEs
        B.add_nodes_from(colliding_ues, bipartite=0)

        # Create edge list
        edge_list = []

        # Create list with chosen configurations
        chosen_configs_list = []

        # Go through all colliding UEs
        for kk in colliding_ues:

            # Chosen configurations
            chosen_configs = ue_choices[kk][0]

            if isinstance(chosen_configs, (int, np.int64)):
                edge_list.append((kk, 'S' + str(chosen_configs)))

                if not ('S' + str(chosen_configs) in chosen_configs_list):
                    chosen_configs_list.append('S' + str(chosen_configs))

            else:

                # Go through all configurations chosen by a UE
                for config in chosen_configs:
                    edge_list.append((kk, 'S' + str(config)))

                    if not ('S' + str(config) in chosen_configs_list):
                        chosen_configs_list.append('S' + str(config))

        # Add configuration nodes
        B.add_nodes_from(chosen_configs_list, bipartite=1)

        # Add edges
        B.add_edges_from(edge_list)

        # Capture effect
        while True:

            # Obtain node degrees
            deg_ue, deg_config = bipartite.degrees(B, colliding_ues)
            dict_deg_ue = dict(deg_ue)

            # No singletons, we cannot decode
            if not (1 in dict_deg_ue.values()):
                break

            # Go through the degree dictionary, if not break
            for config, deg in dict_deg_ue.items():

                # Is there a singleton?
                if deg == 1:

                    # Find respective UE
                    ue = [ue for ue, cc in B.edges if cc == config][0]

                    # Remove edge
                    B.remove_edge(ue, config)

                    # Check UE
                    if not (ue in success_pool):

                        # Add UE to the pool
                        success_pool.append(ue)

    # Compute success attempts
    success_attempts = len(success_pool)

    # Compute performance metric: Throughput
    assert success_attempts <= num_active_ue

    return success_attempts / num_active_ue


def throughput_evaluation_with_power(ue_choices, gamma_ul, gamma_th):
    """Evaluates the throughput of the random access method given the choices made by the UEs and the power received by
    the BS.

    Parameters
    ----------
    ue_choices : dict
        Dictionary containing choices of the UEs.
            keys -> UE indexes
            values -> tuple,
                1st-dimension contains a list or an integer with the configurations chosen by a UE.
                2nd-dimension contains the pilot chosen by a UE.

    gamma_ul : ndarray of shape (num_chosen_configs, num_ues)
        Received uplink SNR for each UE for each config.

    gamma_th : float
        Threshold SNR for SIC.


    Returns
    -------
    throughput : float
        Ratio of success attempts per number of UEs trying to access.

    """

    # Get number of active UEs
    num_active_ue = len(list(ue_choices.keys()))

    # Nothing to compute
    if num_active_ue == 1:
        return 1

    try:
        gamma_ul.shape[1]
    except:
        gamma_ul = gamma_ul[np.newaxis, :]

    # Get mask of pilots
    mask_pilots = np.array(list(ue_choices.values()), dtype=object)[:, 1]

    # Get active pilots
    active_pilots = np.unique(mask_pilots)

    # Pool of success UEs
    success_pool = []

    # Go through all active pilots
    for pilot in active_pilots:

        # Create a Bipartite graph
        B = nx.Graph()

        # Get index of colliding UEs
        colliding_ues = np.array(list(ue_choices.keys()))[mask_pilots == pilot]

        # Add colliding UEs
        B.add_nodes_from(colliding_ues, bipartite=0)

        # Create edge list
        edge_list = []

        # Create list with chosen configurations
        chosen_configs_list = []

        # Go through all colliding UEs
        for kk in colliding_ues:

            # Chosen configurations
            chosen_configs = ue_choices[kk][0]

            if isinstance(chosen_configs, (int, np.int64)):
                edge_list.append((kk, 'S' + str(chosen_configs)))

                if not ('S' + str(chosen_configs) in chosen_configs_list):
                    chosen_configs_list.append('S' + str(chosen_configs))

            else:

                # Go through all configurations chosen by a UE
                for config in chosen_configs:
                    edge_list.append((kk, 'S' + str(config)))

                    if not ('S' + str(config) in chosen_configs_list):
                        chosen_configs_list.append('S' + str(config))

        # Add configuration nodes
        B.add_nodes_from(chosen_configs_list, bipartite=1)

        # Add edges
        B.add_edges_from(edge_list)

        # Obtain node degrees
        deg_ue, deg_config = bipartite.degrees(B, colliding_ues)
        dict_deg_ue = dict(deg_ue)

        # Capture effect
        while True:

            # No singletons, we cannot decode -> break
            if not (1 in dict_deg_ue.values()):
                break

            # Go through the degree dictionary
            for config, deg in dict_deg_ue.items():

                # Is there a singleton?
                if deg == 1:

                    # Find respective UE
                    ue = [ue for ue, cc in B.edges if cc == config][0]

                    # Get absolute config index
                    abs_index_config = int(config[1:])

                    # Get relative config index
                    rel_index_config = np.where(np.array(ue_choices[ue][0]) == abs_index_config)[0].item()

                    # Check power
                    if gamma_ul[rel_index_config, ue] > gamma_th:

                        # Remove all edges with that UE
                        for user, configuration in B.edges():

                            if user == ue:
                                B.remove_edge(ue, configuration)

                                # Update entries
                                dict_deg_ue[configuration] -= 1

                        # Check UE
                        if not (ue in success_pool):

                            # Add UE to the pool
                            success_pool.append(ue)
                    else:

                        # Put a -1 to indicate we cannot that singleton decode
                        dict_deg_ue[config] = -1

    # Compute success attempts
    success_attempts = len(success_pool)

    # Compute performance metric: Throughput
    assert success_attempts <= num_active_ue

    throughput = success_attempts / num_active_ue

    return throughput