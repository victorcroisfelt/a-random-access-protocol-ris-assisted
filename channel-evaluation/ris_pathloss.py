########################################
#   ris_pathloss.py
#
#   Description. This script evaluates the pathloss of the DL and UL phases when 
#   the communication of a BS and a user is aided by a RIS.
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

def pathloss_dl(theta_b, theta_k, theta_s, a, b, wavelength, db=20, dk=20, Gb=10**(5/10), Gk=10**(5/10)):
    """
    Compute the pathloss for the DL. The calculations are as follows:

        1. Omega = (Gb * Gk) * (a * b)^2 / (4 * pi * db * dk)^2

        2. argument = pi * b / lambda * (sin(theta_k) - sin(theta_s))

        3. Pathloss = Omega * cos^2(theta_b) * (sin(argument)/\argument)^2

    Parameters
    ----------

    theta_b : BS angle

    theta_k : UE angle

    theta_s : desired direction

    a, b : vertical and horizontal dimensions of the RIS

    wavelength : wavelength

    db : distance BS-RIS

    dk : distance RIS-UE

    Gb : BS antenna gain

    Gk : UE antenna gain


    Returns
    ----------

    beta_dl_k : DL pathloss for UE k

    """

    # Compute constant
    Omega = (Gb * Gk) * (a * b / 4 / np.pi / db / dk)**2

    # Compute argument
    argument = np.pi * b / wavelength * (np.sin(theta_k) - np.sin(theta_s))

    # Compute pathloss
    beta_dl_k = Omega * np.cos(theta_b)**2 * (np.sin(argument)/argument)**2

    return beta_dl_k


def pathloss_ul(theta_b, theta_k, theta_s, a, b, wavelength, db=20, dk=20, Gb=10**(5/10), Gk=10**(5/10)):
    """
    Compute the pathloss for the UL. The calculations are as follows:

        1. Omega = (Gb * Gk) * (a * b)^2 / (4 * pi * db * dk)^2

        2. argument = pi * b / lambda * (sin(theta_s) - sin(theta_b))

        3. Pathloss = Omega * cos^2(theta_k) * (sin(argument)/\argument)^2

    Parameters
    ----------

    theta_b : BS angle

    theta_k : UE angle

    theta_s : desired direction

    a, b : vertical and horizontal dimensions of the RIS

    wavelength : wavelength

    db : distance BS-RIS

    dk : distance RIS-UE

    Gb : BS antenna gain

    Gk : UE antenna gain


    Returns
    ----------

    beta_ul_k : DL pathloss for UE k

    """

    # Compute constant
    Omega = (Gb * Gk) * (a * b / 4 / np.pi / db / dk)**2

    # Compute argument
    #argument = np.pi * b / wavelength * (np.sin(theta_s) - np.sin(theta_b))
    argument = np.pi * b / wavelength * (-np.sin(theta_b) + np.sin(theta_s))

    # Compute pathloss
    beta_ul_k = Omega * np.cos(theta_k)**2 * (np.sin(argument)/argument)**2

    return beta_ul_k

########################################
# Parameters
########################################

# Physical parameters
c = 3e8 # speed of light
fc = 3e9 # carrier frequency
wavelength = c/fc # wavelength

# RIS size (square RIS)
a = b = 10 * wavelength

# RIS configurations (theta_s)
configs = np.radians(np.array([30, 45, 60]))

# BS elevation angle
theta_b = np.radians(45)

# UE elevation angle
theta_k = np.radians(15)

# Observation angle
theta_obs = np.linspace(0, np.pi/2)

########################################
# Simulation
########################################

# Prepare to save simulation results
beta_dl = np.zeros((configs.size, theta_obs.size))
beta_ul = np.zeros((configs.size, theta_obs.size))

# Go through all configurations
for cc, theta_s in enumerate(configs):

    # Compute pathloss
    beta_dl[cc, :] = pathloss_dl(theta_b, theta_obs, theta_s, a, b, wavelength)
    beta_ul[cc, :] = pathloss_ul(theta_obs, theta_k, theta_s, a, b, wavelength)

########################################
# Plot
########################################

legends = ["$30^\circ$", "$45^\circ$", "$60^\circ$"]
lines = ["-", "--", "-."]

#fig, ax = plt.subplots(figsize=(3.15, 3), nrows=2)
fig, ax = plt.subplots(nrows=2)

ax[0].plot([np.degrees(theta_k), np.degrees(theta_k)], [-150, 50], color="black", label=r"UE: $\theta_k = 15^\circ$", linestyle=':')
ax[1].plot([-np.degrees(theta_b), -np.degrees(theta_b)], [-150, 50], color="black", label=r"BS: $\theta_b = 45^\circ$", linestyle=':')

# Go through all configurations
for cc, theta_s in enumerate(configs):

    ax[0].plot(np.degrees(theta_obs), 10*np.log10(beta_dl[cc, :]), linewidth=1.5, linestyle=lines[cc], label=r"$\theta^{\rm DL}_s=$" + str(legends[cc]))
    ax[1].plot(-np.degrees(theta_obs), 10*np.log10(beta_ul[cc, :]), linewidth=1.5, linestyle=lines[cc], label=r"$\theta^{\rm UL}_s=$" + str(legends[cc]))


ax[0].set_xlabel(r"observation angle $\theta_s$")
ax[1].set_xlabel(r"observation angle $\theta_s$")

ax[0].set_ylabel(r"normalized $S_{\text{dl}}$ [dB]")
ax[1].set_ylabel(r"normalized $S_{\text{ul}}$ [dB]")

ax[0].set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
ax[1].set_xticks([-90, -80, -70, -60, -50, -40, -30, -20, -10, 0])

ax[0].set_ylim([-150, -50])
ax[1].set_ylim([-150, -50])

ax[0].legend(loc='lower right')
ax[1].legend(loc='lower right')

plt.show()
