# file: pathloss.py
# This evaluates the correctness of the analytical solution for the CDF and PDF computation of the DL and UL pathloss

import numpy as np
import scipy as sp
from scipy.constants import speed_of_light as c
import matplotlib.pyplot as plt
from matplotlib import rc


# LaTeX type definitions
rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('lines', **{'markerfacecolor': "None", 'markersize': 5})
rc('axes.grid', which='minor')


def pathloss(incidence_angle: float,
             g_bs: float, g_ue: float,
             area_ris: float, el_ris: int,
             d_bs: float, d_ue: float):
    """ Compute the pathloss given the environmental parameters.

    :param incidence_angle: the incidence angle created by the arriving LOS path
    :param g_bs: antenna gain of the BS, in dBi
    :param g_ue: antenna gain of the UE, in dBi
    :param area_ris: area of the RIS (equivalent to the product a x b in the paper)
    :param el_ris: number of element of the RIS
    :param d_bs: distance of the BS from RIS, in meters
    :param d_ue: distance of the UE from RIS, in meters
    """
    return omega(g_bs, g_ue, area_ris, el_ris, d_bs) * (np.cos(incidence_angle) / d_ue) ** 2


def omega(g_bs: float, g_ue: float,
          area_ris: float, el_ris: int,
          d_bs: float):
    """ Compute the pathloss given the environmental parameters.

    :param g_bs: antenna gain of the BS, in dBi
    :param g_ue: antenna gain of the UE, in dBi
    :param area_ris: area of the RIS (equivalent to the product a x b in the paper)
    :param el_ris: number of element of the RIS
    :param d_bs: distance of the BS from RIS, in meters
    """
    gain_tot = 10 ** ((g_bs + g_ue) / 10)
    return gain_tot * (area_ris / el_ris / 4 / np.pi / d_bs) ** 2


def ecdf(x):
    return np.sort(x), np.arange(len(x)) / float(len(x))


# Deterministic parameters
gain_ue = 2.15          # [dBi]
gain_bs = 4.85          # [dBi]
dist_bs = 30            # [m]
angle_bs = np.pi / 4    # [rad]
N = 100                 # RIS elements (10 x 10)
height_ris = 1          # [m]
width_ris = 1           # [m]
ell = 100               # UE room side length [m]
ell0 = 15               # distance between room and origin of axis [m]
fc = 3e9                # Working frequency [Hz]
wavelength = c / fc     # wavelength

# Random parameters
seed = None     # Needed for reproducibility
rng = np.random.default_rng(seed)
samples = int(1e4)


if __name__ == "__main__":
    # Data
    x_ue = rng.uniform(ell0, ell + ell0, samples)
    y_ue = rng.uniform(ell0, ell + ell0, samples)
    dist_ue = np.sqrt(x_ue ** 2 + y_ue ** 2)
    angle_ue = np.arctan2(y_ue, x_ue)
    om = omega(gain_bs, gain_ue, height_ris * width_ris, N, dist_bs)
    pl_ul = pathloss(angle_ue, gain_bs, gain_ue, height_ris * width_ris, N, dist_bs, dist_ue)
    pl_dl = pathloss(angle_bs, gain_bs, gain_ue, height_ris * width_ris, N, dist_bs, dist_ue)

    # Analytical computation
    # DL-CDF
    lo_lim = om * np.cos(angle_bs) ** 2 / 2 / (ell0 + ell) ** 2
    mi_lim = om * np.cos(angle_bs) ** 2 / (ell0 ** 2 + (ell0 + ell) ** 2)
    up_lim = om * np.cos(angle_bs) ** 2 / 2 / ell0 ** 2

    beta_dl = np.linspace(lo_lim, up_lim, samples)
    om_cos = om * np.cos(angle_bs) ** 2
    cdf_dl = np.piecewise(beta_dl, [beta_dl <= lo_lim, (beta_dl > lo_lim) * (beta_dl <= mi_lim), (beta_dl > mi_lim) * (beta_dl <= up_lim), beta_dl > up_lim],
                          [0,
                           lambda x: 1 / ell ** 2 * ((ell0 + ell) ** 2 - (ell0 + ell) * np.sqrt(om_cos / x - (ell0 + ell) ** 2) - om_cos / x / 2 * np.arctan((2 * (ell0+ell) ** 2 - om_cos / x) / (2 * (ell0 + ell) * np.sqrt(om_cos / x - (ell0+ell) ** 2)))),
                           lambda x: 1 / ell ** 2 * ((ell ** 2 - ell0 ** 2) + ell0 * np.sqrt(om_cos / x - ell0 ** 2) + om_cos / x / 2 * np.arctan((2 * ell0 ** 2 - om_cos / x) / (2 * ell0 * np.sqrt(om_cos / x - ell0 ** 2)))),
                          1])
    # DL-PDF
    pdf_dl = np.piecewise(beta_dl, [beta_dl <= lo_lim, (beta_dl > lo_lim) * (beta_dl <= mi_lim), (beta_dl > mi_lim) * (beta_dl <= up_lim), beta_dl > up_lim],
                          [0,
                           lambda x: om_cos / (2 * ell ** 2 * x ** 2) * np.arctan((2 * x * (ell0+ell) ** 2 - om_cos) / (2 * np.sqrt(x) * (ell0 + ell) * np.sqrt(om_cos - x * (ell0+ell) ** 2))),
                           lambda x: - om_cos / (2 * ell ** 2 * x ** 2) * np.arctan((2 * x * ell0 ** 2 - om_cos) / (2 * np.sqrt(x) * ell0 * np.sqrt(om_cos - x * ell0 ** 2))),
                          0])

    # UL-CDF
    limits = om * np.array([ell0 / (ell0 ** 2 + (ell0 + ell) ** 2),
                            1 / 2 / (ell0 + ell),
                            (ell0 + ell) / (ell0 ** 2 + (ell0 + ell) ** 2),
                            1 / 2 / ell0]) ** 2
    beta_ul = np.linspace(limits[0], limits[-1], samples)
    cdf_ul = np.piecewise(beta_ul, [beta_ul <= limits[0], (beta_ul > limits[0]) * (beta_ul <= limits[1]), (beta_ul > limits[1]) * (beta_ul <= limits[2]), (beta_ul > limits[2]) * (beta_ul <= limits[3]), beta_ul > limits[3]],
                          [0,
                          lambda x: 1 / ell ** 2 * (- ell0 * (ell0 + ell) - om / 8 / x * (2 * np.sqrt(x / om) * (ell0 + ell) * (np.sqrt(1 - 4 * x * (ell0 + ell) ** 2 / om) - 2) + 2 * (x / om) ** (1/4) * np.sqrt(ell0 * (1 - np.sqrt(x/om) * ell0)) * (1 - 2 * np.sqrt(x / om) * ell0) + np.arcsin(2 * np.sqrt(x / om) * (ell0 + ell)) - np.arcsin(2 * (x / om) ** (1/4) * np.sqrt(ell0 * (1 - np.sqrt(x / om) * ell0))))),
                          lambda x: 1 + ell0 / ell - om / 8 / x / ell ** 2 * (np.pi + 2 * (x / om) ** (1/4) * (np.sqrt(ell0 * (1 - np.sqrt(x / om) * ell0)) * (1 - 2 * np.sqrt(x / om) * ell0) + np.sqrt((ell0 + ell) * (1 - np.sqrt(x / om) * (ell0 + ell))) * (2 * (ell0 + ell) * np.sqrt(x / om) - 1)) - np.arcsin(2 * (x / om) ** (1/4) * np.sqrt(ell0 * (1 - np.sqrt(x / om) * ell0))) - np.arcsin(2 * (x / om) ** (1/4) * np.sqrt((ell0+ell) * (1 - np.sqrt(x / om) * (ell0+ell))))),
                          lambda x: 1 - (ell0 / ell) ** 2 - om / 8 / x / ell ** 2 * (np.pi - 2 * np.sqrt(x / om) * ell0 * (2 + np.sqrt(1 - 4 * x * ell0 ** 2 / om)) + 2 * (x / om) ** (1/4) * np.sqrt(ell0 * (1 - np.sqrt(x / om) * ell0)) * (1 - 2 * np.sqrt(x / om) * ell0) - np.arcsin(2 * np.sqrt(x / om) * ell0) - np.arcsin(2 * (x / om) ** (1/4) * np.sqrt(ell0 * (1 - np.sqrt(x / om) * ell0)))),
                          1])
    # UL-PDF
    pdf_ul = np.piecewise(beta_ul, [beta_ul <= limits[0], (beta_ul > limits[0]) * (beta_ul <= limits[1]), (beta_ul > limits[1]) * (beta_ul <= limits[2]), (beta_ul > limits[2]) * (beta_ul <= limits[3]), beta_ul > limits[3]],
                          [0,
                          lambda x: 1 / 4 / ell ** 2 / x * (np.sqrt(om / x) * (np.sqrt(ell0) * np.sqrt(np.sqrt(om / x) - ell0) - (ell0+ell)) + om / x / 2 * (np.arcsin(2 * np.sqrt(x / om) * (ell0 + ell)) - np.arcsin(2 * (x / om) ** (1/4) * np.sqrt(ell0 * (1 - np.sqrt(x / om) * ell0))))),
                          0, # lambda x: 1 + ell0 / ell - om / 8 / x / ell ** 2 * (np.pi + 2 * (x / om) ** (1/4) * (np.sqrt(ell0 * (1 - np.sqrt(x / om) * ell0)) * (1 - 2 * np.sqrt(x / om) * ell0) + np.sqrt((ell0 + ell) * (1 - np.sqrt(x / om) * (ell0 + ell))) * (2 * (ell0 + ell) * np.sqrt(x / om) - 1)) - np.arcsin(2 * (x / om) ** (1/4) * np.sqrt(ell0 * (1 - np.sqrt(x / om) * ell0))) - np.arcsin(2 * (x / om) ** (1/4) * np.sqrt((ell0+ell) * (1 - np.sqrt(x / om) * (ell0+ell))))),
                          0, # lambda x: 1 - (ell0 / ell) ** 2 - om / 8 / x / ell ** 2 * (np.pi - 2 * np.sqrt(x / om) * ell0 * (2 + np.sqrt(1 - 4 * x * ell0 ** 2 / om)) + 2 * (x / om) ** (1/4) * np.sqrt(ell0 * (1 - np.sqrt(x / om) * ell0)) * (1 - 2 * np.sqrt(x / om) * ell0) - np.arcsin(2 * np.sqrt(x / om) * ell0) - np.arcsin(2 * (x / om) ** (1/4) * np.sqrt(ell0 * (1 - np.sqrt(x / om) * ell0)))),
                          0])
    



    # Plotting
    # DL-CDF
    _, ax = plt.subplots()
    ax.plot(*ecdf(pl_dl), label='empirical')
    ax.plot(beta_dl, cdf_dl, label='analytical', linestyle='--')
    plt.ylabel(r'$F_{B^{\mathrm{DL}}}(\beta)$')
    plt.xlabel(r'$\beta$')
    ax.grid()
    ax.legend()
    plt.show()

    # DL-PDF
    _, ax = plt.subplots()
    ax.hist(pl_dl, bins=100, density=True, label='empirical')
    ax.plot(beta_dl, pdf_dl, label='analytical', linestyle='--')
    plt.ylabel(r'$f_{B^{\mathrm{DL}}}(\beta)$')
    plt.xlabel(r'$\beta$')
    ax.grid()
    ax.legend()
    plt.show()
    plt.show()

    # UL-CDF
    _, ax = plt.subplots()
    ax.plot(*ecdf(pl_ul), label='empirical')
    ax.plot(beta_ul, cdf_ul, label='analytical', linestyle='--')
    plt.ylabel(r'$f_{B^{\mathrm{UL}}}(\beta)$')
    plt.xlabel(r'$\beta$')
    ax.grid()
    ax.legend()
    plt.show()

    # UL-PDF
    _, ax = plt.subplots()
    ax.hist(pl_ul, bins=100, density=True, label='empirical')
    ax.plot(beta_ul, pdf_ul, label='analytical', linestyle='--')
    plt.ylabel(r'$f_{B^{\mathrm{UL}}}(\beta)$')
    plt.xlabel(r'$\beta$')
    ax.grid()
    ax.legend()
    plt.show()
    plt.show()
