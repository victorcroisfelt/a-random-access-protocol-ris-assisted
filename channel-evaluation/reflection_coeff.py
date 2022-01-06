########################################
#   phase_shift.py
#
#   Description. This script exemplifies how to obtain the reflection coefficient
#   needed to be imposed along the RIS in order to steer an incoming signal towards
#   a desired direction.
#
#   Authors.
#           @lostinafro 
#           @victorcroisfelt
#
#   Date. Jan 03, 2022
#
########################################
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

########################################
# Preamble
########################################

axis_font = {'size': '8'}

plt.rcParams.update({'font.size': 8})

matplotlib.rc('xtick', labelsize=8)
matplotlib.rc('ytick', labelsize=8)

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"

########################################
# Private functions
########################################
def quant(x, bits):
    """
    Quantize a signal x considering the given number of bits.

    Parameters
    ----------

    x : array of floats
        input signal

    bits : integer
        number of bits that defines the step size (resolution) of the 
        quantization process.

    Returns
    -------

    yk : array of floats
        quantized version of the input signal
    """

    # Obtain step size
    Delta = (1/2)**(bits-1)

    # Re-scale the signal
    x_zeros = x * (1 - 1e-12)
    x_scaled = x_zeros - Delta/2

    # Quantization stage
    k = np.round(x/Delta) * Delta

    # Reconstruction stage
    yk = k + Delta/2

    return yk

########################################
# Parameters
########################################

# Physical parameters
c = 3e8     # speed of light
fc = 3e9    # carrier frequency
wavelengh = c/fc    # wavelength
omega = 2*np.pi/wavelengh   # wavenumber

# Angle of incidence
theta_i = np.radians(45)

# Desired direction of reflection
#configs = np.radians(np.array([30, 45, 60]))
configs = np.array([0.19634954, 0.58904862, 0.9817477])

# Define x-axis of the surface: [-5\lambda,5\lambda] corresponds to the case
# that the surface is 10\lambda long along the x-axis
x_range = wavelengh * np.arange(-5, 5, 1e-4)

########################################
# Simulation
########################################

# Prepare to save simulation results
phi_r_dl = np.zeros((x_range.size, configs.size))
phi_r_ul = np.zeros((x_range.size, configs.size))

# Go through all configurations
for cc, theta_s in enumerate(configs):

    # Go through x dimension of the RIS
    for xi, x in enumerate(x_range):

        # Calculate and normalize the result with respect to 2*pi
        phi_r_dl[xi, cc] = 2*np.pi * np.mod(( (omega * ( -np.sin(theta_s) + np.sin(theta_i)) ) * x ), 1)
        phi_r_ul[xi, cc] = 2*np.pi * np.mod(( (omega * ( +np.sin(theta_s) - np.sin(theta_i)) ) * x ), 1)

########################################
# Plot
########################################
lines = ["-", "--", ":"]
legends = [30, 45, 60]

fig, ax = plt.subplots(figsize=(3.15, 3), nrows=2)

# Go through all configs
for cc, theta_s in enumerate(configs):

    ax[0].plot(x_range/wavelengh, np.degrees(phi_r_dl[:, cc]), linewidth=1.5, linestyle=lines[cc], label=r"$\theta^{\text{DL}}_s=" + str(legends[cc]) + "^\circ$")
    ax[1].plot(x_range/wavelengh, np.degrees(phi_r_ul[:, cc]), linewidth=1.5, linestyle=lines[cc], label=r"$\theta^{\text{UL}}_s=" + str(legends[cc]) + "^\circ$")

ax[0].set_xlabel(r"$x/\lambda$")
ax[1].set_xlabel(r"$x/\lambda$")

ax[0].set_ylabel(r"downlink $\phi^{\text{DL}}_s(x)$")
ax[1].set_ylabel(r"uplink $\phi^{\text{UL}}_s(x)$")

ax[0].legend(loc='lower right')
ax[1].legend(loc='lower right')

ax[0].set_yticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
ax[1].set_yticks([0, 45, 90, 135, 180, 225, 270, 315, 360])

plt.show()

#####

fig, ax = plt.subplots(figsize=(3.15, 3))

# Go through all configs
for cc, theta_s in enumerate(configs):

    # Compute discretize version
    discretized = quant((np.degrees(phi_r_dl[:, cc])/180) - 1, bits=3)

    ax.plot(x_range/wavelengh, np.degrees(phi_r_dl[:, cc]), linewidth=1.5, linestyle=lines[cc], label=r"$\theta_r=" + str(legends[cc]) + r"^\circ$")
    ax.plot(x_range/wavelengh, 180*(discretized + 1), linewidth=1.5, linestyle=lines[cc], label=r"$\theta_r=" + str(legends[cc]) + r"^\circ$")

ax.set_xlabel(r"$x/\lambda$")
ax.set_ylabel(r"local surface phase $\phi^{\text{DL}}_r(x)$")

ax.legend(loc='lower right')
ax.set_yticks([0, 45, 90, 135, 180, 225, 270, 315, 360])

plt.show()
