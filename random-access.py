from environment.box import Box
import numpy as np

########################################
# General parameters
########################################

# Square length
ell = 100
ell0 = 10

# Number of pilots
taup = 10

########################################
# Simulation parameters
########################################

# Initialize environment
box = Box(ell, ell0)

# Place BS and RIS (fixed entities)
box.place_bs(pos_polar=np.array([[30, np.deg2rad(90 + 45)]]))
box.place_ris()

# Number of active UEs
K = 10

# Place UEs
box.place_ue(K)

# Evaluate deterministic channel model
channel_gains_dl, channel_gains_ul, phase_shifts_bs, phase_shifts_ue = box.get_channel_model()



box.plot_scenario()



def dl_reflection_coefficients(self):
    """
    Define vectors of DL reflection coefficients that steer the signal towards each configuration, having the
    azimuth angle of the BS as the incidence angle.

    Returns
    -------
    None.

    """
    # Go along x dimension
    x_range = np.arange(-self.size_h / 2, + self.size_h / 2, self.size_el)  # [m]

    # Check if the size of x_range meets the number of horizonal els
    if len(x_range) != self.num_els_h:
        raise Exception("Range over x-axis does not meet number of horizontal Nb elements.")

    # Prepare to save the reflection coefficients for each configuration
    local_surface_phase = np.zeros((self.num_configs, self.num_els_h))

    # Go through all configurations
    for config, theta_s in enumerate(self.set_configs):
        local_surface_phase[config, :] = (
                    2 * np.pi * np.mod(self.wavenumber * (-np.sin(theta_s) + np.sin(self.incidence_angle)) * x_range,
                                       1))

    # Store the reflection coefficient of each element for each by repeating
    # the same local surface phase along x-axis
    Phi_dl = np.tile(local_surface_phase, rep=self.num_els_h)

    return Phi_dl


