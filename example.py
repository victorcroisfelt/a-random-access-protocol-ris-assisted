from environment.box import Box
import numpy as np

# General parameters
ell = 100
ell0 = 10

K = 5

#

taup = 10

# Initialize the environment
box = Box(ell, ell0, taup=taup)

box.place_bs(pos_polar=np.array ( [30, np.deg2rad(90 + 45)] ) )
box.place_ue(K)     # place K users
box.place_ris()

box.plot_scenario()
# box.build_chan_gain()   # still work in progress


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


