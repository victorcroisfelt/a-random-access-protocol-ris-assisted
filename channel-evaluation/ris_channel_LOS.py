import numpy as np
import scipy as sp
from scipy.constants import speed_of_light as c
import matplotlib.pyplot as plt
from matplotlib import rc
import os
from datetime import date


# LaTeX type definitions
rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('lines', **{'markerfacecolor': "None", 'markersize': 5})
rc('axes.grid', which='minor')


def pathloss2d(src_angle: float, dst_angle: float,
               src_gain: float, dst_gain: float, src_dist: float, dst_dist: float,
               el_dist_x: float, el_dist_z: float,
               w_length: float):
    """ Compute the pathloss given the environmental parameters.

    :param src_angle: the incidence angle created by the arriving LOS path (positive) [second quadrant]
    :param dst_angle: the destination angle created by the arriving LOS path (positive) [first quadrant]
    :param src_gain: antenna gain of the source, in dBi
    :param dst_gain: antenna gain of the destination, in dBi
    :param src_dist: distance of the source from RIS, in meters
    :param dst_dist: distance of the destination from RIS, in meters
    :param el_dist_x: distance between elements, x-dimension
    :param el_dist_z: distance between elements, z-dimension
    :param w_length: wavelength of the central frequency
    """
    gain_tot = 10 ** ((src_gain + dst_gain) / 10)
    # Remember that np.sinc is the normalized one, so sin(pi x) / (pi x)
    return gain_tot / (4 * np.pi) ** 2 * (el_dist_x * el_dist_z / src_dist / dst_dist) ** 2 * np.cos(src_angle) ** 2 * np.sinc(el_dist_x / w_length * (np.sin(src_angle) - np.sin(dst_angle))) ** 2


def array_factor(src_angle: float, dst_angle: float, conf_x: float,
                 el_dist_x: float, el_num_x: int, w_length: float):
    """ Compute the array factor (squared) given the environmental parameters

    :param src_angle: the incidence angle created by the arriving LOS path (positive) [second quadrant]
    :param dst_angle: the destination angle created by the arriving LOS path (positive) [first quadrant]
    :param conf_x: configuration coefficient on the x-axis
    :param el_dist_x: distance between elements, x-dimension
    :param el_num_x: number of elements on the x-axis
    :param w_length: wavelength of the central frequency
    """
    w_number = 2 * np.pi / w_length
    a = np.piecewise(conf_x - np.sin(src_angle) + np.sin(dst_angle), [conf_x - np.sin(src_angle) + np.sin(dst_angle) == 0, conf_x - np.sin(src_angle) + np.sin(dst_angle) != 0],
                     [el_num_x,
                     lambda x: np.sin(el_num_x * el_dist_x * w_number * x / 2) / np.sin(el_dist_x * w_number * x / 2)])
    # np.sin(el_num_x * el_dist_x * w_number * (conf_x - np.sin(dst_angle) + np.sin(src_angle)) / 2) / np.sin(el_dist_x * w_number * (conf_x - np.sin(dst_angle) + np.sin(src_angle)) / 2)])
    return np.abs(a) ** 2


def ff_dist(el_num_x, el_dist_x, w_length):
    return 2 / w_length * (el_num_x * el_dist_x) ** 2

# Deterministic parameters
gain_ue = 2.15          # [dBi]
gain_bs = 4.85          # [dBi]
dist_bs = 30            # [m]
dist_ue = 30
angle_bs = np.pi / 4    # [rad]
N_x = 10                # RIS elements on x-axis
N_z = 10                # RIS elements on z-axis
fc = 3e9                # Working frequency [Hz]
wavelength = c / fc     # wavelength

# Random parameters
seed = None     # Needed for reproducibility
rng = np.random.default_rng(seed)
samples = int(10000)

# Simulation parameters
output_dir = os.path.join(os.path.expanduser('~'), 'uni/plots/ris', str(date.today()))
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
render = False
src_ang = np.pi / 4      # assumed known
dst_angles = np.pi / 2 * np.array([1/3, 1/2, 2/3])

if __name__ == "__main__":
    # Element spacing and far-field approximation
    d_x = np.array([4 * wavelength, 2 * wavelength, wavelength, wavelength / 2,  wavelength / 4])

    # Plot
    _, ax = plt.subplots()
    ax.plot(d_x / wavelength, ff_dist(d_x, N_x, wavelength))
    plt.ylabel(r'$d_{\mathrm{ff}}$ [m]')
    plt.xlabel(r'$d_x / \lambda$')
    title = r'Minimum distance for far field approximation'
    ax.grid()
    if not render:
        plt.title(title)
        plt.show(block=False)
    else:
        filename = os.path.join(output_dir, f'ff_distance')
        # tikzplotlib.save(filename + '.tex')
        plt.title(title)
        plt.savefig(filename + '.png', dpi=300)
        plt.close()

    # Pathloss varying element distance
    d_x_label = [r'$4 \lambda$', r'$2 \lambda$', r'$\lambda$', r'$\lambda / 2$', r'$\lambda / 4$']
    dst_angles_full = np.pi / 2 * np.linspace(0, 1, samples)
    pathloss = np.zeros((len(dst_angles_full), len(d_x)))
    for i in range(len(d_x)):
        pathloss[:, i] = 10 * np.log10(pathloss2d(src_ang, dst_angles_full, gain_bs, gain_ue, dist_bs, dist_ue, d_x[i], d_x[i], wavelength))

    # Plot
    _, ax = plt.subplots()
    for i in range(len(d_x)):
        ax.plot(np.rad2deg(dst_angles_full), pathloss[:, i], label=f'$d_x$ = {d_x_label[i]}')
    plt.ylabel(r'$\beta^{\mathrm{DL}}$ [dB]')
    plt.xlabel(r'$\theta_k$ [degrees]')
    title = r'Pathloss varying element spacing ($\theta_b = 45$°)'
    ax.grid()
    ax.legend()

    if not render:
        plt.title(title)
        plt.show(block=False)
    else:
        filename = os.path.join(output_dir, f'pathloss_incident45')
        # tikzplotlib.save(filename + '.tex')
        plt.title(title)
        plt.savefig(filename + '.png', dpi=300)
        plt.close()

    # Array_factor
    phi = -np.sin(np.linspace(0, np.pi, samples)) + np.sin(src_ang)    # compensating the known angle
    Nd_struct = [(10, wavelength / 2), (10, wavelength / 4), (100, wavelength / 2), (100, wavelength / 4)]

    for iteration in Nd_struct:
        array = np.zeros((len(phi), len(dst_angles)))  # initialization
        N_x = iteration[0]
        d_x = iteration[1]
        ff_d = np.around((ff_dist(d_x, N_x, wavelength)))
        title_values = f'($N_x = {N_x}$,' + r'$d_x = \lambda /' + f'{np.around(wavelength / d_x):.0f}$,' + r'$d_{\mathrm{ff}} =' + f' {ff_d:.0f}$)'
        title = r'Array factor compensating the source angle ' + title_values

        # Compute array factor
        for i in range(len(dst_angles)):
            array[:, i] = array_factor(src_ang, dst_angles[i], phi, d_x, N_x, wavelength)

        # Plotting
        _, ax = plt.subplots()
        for i in range(len(dst_angles)):
            ax.plot(-np.rad2deg(np.arcsin(phi - np.sin(src_ang))), array[:, i], label=r'$\theta_k$ =' + f'{np.around(np.rad2deg(dst_angles[i]))}°')
        plt.ylabel(r'$|\mathcal{A}^{\mathrm{DL}}_k|^2$')
        plt.xlabel(r'$-\arcsin(\psi)$')
        ax.grid()
        ax.legend()

        if not render:
            plt.title(title)
            plt.show(block=False)
        else:
            filename = os.path.join(output_dir, f'array_factor_N{N_x}_lamVSd{np.around(wavelength / d_x):.0f}')
            # tikzplotlib.save(filename + '.tex')
            plt.title(title)
            plt.savefig(filename + '.png', dpi=300)
            plt.close()
