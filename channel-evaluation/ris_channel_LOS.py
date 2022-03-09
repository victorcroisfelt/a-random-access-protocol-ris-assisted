import numpy as np
import scipy as sp
from scipy.constants import speed_of_light as c
from scipy import special
import matplotlib.pyplot as plt
from matplotlib import rc
import os
from datetime import date
import sympy as sy
import argparse


# LaTeX type definitions
rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('lines', **{'markerfacecolor': "None", 'markersize': 5})
rc('axes.grid', which='minor')


def command_parser():
    """Parse command line using arg-parse and get user data to run the render.

        :return: the parsed arguments
    """
    # Parse depending on the boolean watch flag
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--render", action="store_true", default=False)
    args: dict = vars(parser.parse_args())
    return list(args.values())


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
    return (np.abs(a) / el_num_x) ** 2


def gen_array_factor(src_angle: float, dst_angle: float, conf_x: float,
                     el_dist_x: float, el_num_x: int,
                     central_frequency: float, num_resources: int, sub_spacing: float):
    """ Compute the array factor (squared) given the environmental parameters

    :param src_angle: the incidence angle created by the arriving LOS path (positive) [second quadrant]
    :param dst_angle: the destination angle created by the arriving LOS path (positive) [first quadrant]
    :param conf_x: configuration coefficient on the x-axis
    :param el_dist_x: distance between elements, x-dimension
    :param el_num_x: number of elements on the x-axis
    :param central_frequency: initial frequency of operation [Hz]
    :param num_resources: number of frequencies usable
    :param sub_spacing: spacing between frequency resources [Hz]
    """
    constant_factor = el_dist_x * np.pi / c * central_frequency
    f = np.arange(num_resources)
    argument = conf_x + (1 + f * sub_spacing / central_frequency) * (np.sin(dst_angle) - np.sin(src_angle))
    a = np.piecewise(argument, [argument == 0, argument != 0],
                     [el_num_x,
                     lambda x: np.sin(el_num_x * constant_factor * x) / np.sin(constant_factor * x)])
    return (np.abs(a) / el_num_x) ** 2


def qfunc(x):
    return 0.5-0.5*special.erf(x / np.sqrt(2))


def ff_dist(el_num_x, el_dist_x, w_length):
    return 2 / w_length * (el_num_x * el_dist_x) ** 2


def degarcsin(x):
    return np.rad2deg(np.arcsin(x))


def degarccos(x):
    return np.rad2deg(np.arccos(x))


# Deterministic parameters
gain_ue = 2.15          # [dBi]
gain_bs = 4.85          # [dBi]
dist_bs = 30            # [m]
dist_ue = 30
N_x = 10                # RIS elements on x-axis
N_z = 10                # RIS elements on z-axis
fc = 3.8e9                # Working frequency [Hz]
src_ang = np.pi / 4  # assumed known
wavelength = c / fc     # wavelength

# Random parameters
seed = None     # Needed for reproducibility
rng = np.random.default_rng(seed)
samples = int(10000)

# Standard output for rendering
output_dir = os.path.join(os.path.expanduser('~'), 'uni/plots/ris', str(date.today()))

if __name__ == "__main__":
    render = command_parser()[0]
    # Crate the output folder if missing
    if render:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

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
    colors = ['blue', 'orange', 'green', 'purple', 'darkred']
    dst_angles = np.pi / 2 * np.array([1 / 6, 1 / 2, 5 / 6])
    phi = -np.sin(np.linspace(0, np.pi, samples)) + np.sin(src_ang)    # compensating the known angle
    # Nd_struct is a list of tuple. Each tuple contains N_x and d_x of the RIS
    Nd_struct = [(8, wavelength), (16, wavelength), (32, wavelength / 2)]

    for iteration in Nd_struct:
        array = np.zeros((len(phi), len(dst_angles)))  # initialization
        N_x = iteration[0]
        d_x = iteration[1]
        ff_d = np.around((ff_dist(d_x, N_x, wavelength)))
        lam_denominator_str = f'/{np.around(wavelength / d_x):.0f}' if wavelength / d_x > 1 else ''
        title_values = f'($N_x = {N_x}$, ' + r'$d_x = \lambda' + lam_denominator_str + r'$, $d_{\mathrm{ff}} =' + f' {ff_d:.0f}$)'
        title = r'Array factor compensating the source angle ' + title_values
        # compute the FNBM
        FNBM_right = wavelength / d_x / N_x - np.sin(dst_angles)
        FNBM_left = -wavelength / d_x / N_x - np.sin(dst_angles)
        a = 1.391
        HPBM_left = wavelength * a / N_x / d_x / np.pi - np.sin(dst_angles)
        HPBM_right = -wavelength * a / N_x / d_x / np.pi - np.sin(dst_angles)

        # Compute array factor
        for i in range(len(dst_angles)):
            array[:, i] = array_factor(src_ang, dst_angles[i], phi, d_x, N_x, wavelength)
            # compute the HPBW

        # Plotting
        _, ax = plt.subplots()
        for i in range(len(dst_angles)):
            ax.plot(degarcsin(-(phi - np.sin(src_ang))), array[:, i], label=r'$\theta_k$ =' + f'{np.around(np.rad2deg(dst_angles[i])):.0f}°', c=colors[i])
            plt.axvline(x=degarcsin(-FNBM_right[i]), ymin=0, ymax=N_x, lw=0.7, ls=':', c=colors[i])
            plt.axvline(x=degarcsin(-FNBM_left[i]), ymin=0, ymax=N_x, lw=0.7, ls=':', c=colors[i])
            plt.axvline(x=degarcsin(-HPBM_right[i]), ymin=0, ymax=N_x, lw=0.7, ls=':', c=colors[i])
            plt.axvline(x=degarcsin(-HPBM_left[i]), ymin=0, ymax=N_x, lw=0.7, ls=':', c=colors[i])
        plt.ylabel(r'$\frac{|\mathcal{A}^{\mathrm{DL}}_k|^2}{N_x^2}$')
        plt.xlabel(r'$\theta_s$')
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

        # Configurations check
        conf_num = int(np.ceil(N_x * d_x * np.pi / 2 / wavelength / a))
        conf_values = 1 - wavelength * a / N_x / d_x / np.pi * (2 * np.arange(1, conf_num + 1) - 1)
        conf_HP_minus = np.arcsin(1 - wavelength * a / N_x / d_x / np.pi * (2 * np.arange(1, conf_num + 1)))
        conf_HP_plus = np.arcsin(1 - wavelength * a / N_x / d_x / np.pi * (2 * np.arange(1, conf_num + 1) - 2))
        conf_directions = degarcsin(conf_values)
        title = r'AF coverage compensating the source angle ' + title_values

        # Configuration coverage
        conf_tot = - conf_values + np.sin(src_ang)
        af = np.zeros((len(dst_angles_full), len(conf_tot)))
        for i in range(conf_num):
            af[:, i] = array_factor(src_ang, dst_angles_full, conf_tot[i], d_x, N_x, wavelength)

        _, ax = plt.subplots()
        for i in range(conf_num - 1):
            ax.plot(np.rad2deg(dst_angles_full), af[:, i],  label=r'$\theta_s$ =' + f'{np.around(conf_directions[i]):.0f}°')
        plt.ylabel(r'$\frac{|\mathcal{A}_k|^2}{N_x^2}$')
        plt.xlabel(r'$\theta_k$')
        ax.grid()
        ax.legend(loc='right')

        if not render:
            plt.title(title)
            plt.show(block=False)
        else:
            filename = os.path.join(output_dir, f'conf_coverage_N{N_x}_lamVSd{np.around(wavelength / d_x):.0f}')
            # tikzplotlib.save(filename + '.tex')
            plt.title(title)
            plt.savefig(filename + '.png', dpi=300)
            plt.close()

        # Error mass probability
        # Let us assume a user has position theta_k = theta_k_0 + error
        ue_angle_0 = dst_angles_full
        error_degree = 5
        std_error = np.deg2rad(error_degree)       # 4 degree of error for the 95%
        # Computing the mass probability for every configuration
        ue_angle_0_mat = np.tile(ue_angle_0[np.newaxis], (conf_num, 1))
        mass_probability = qfunc((conf_HP_minus[np.newaxis].T - ue_angle_0_mat) / std_error) - qfunc((conf_HP_plus[np.newaxis].T - ue_angle_0_mat) / std_error)
        # Plotting probability for some determined angular positions dst_angles
        width_tot = 0.4
        width_each = width_tot / len(dst_angles)
        xaxis = np.arange(0, conf_num)
        xlabels = np.char.mod('%d', conf_directions)  # [str(i) for i in (xaxis + 1)]
        threshold = 1e-1
        _, ax = plt.subplots()
        for i in range(len(dst_angles)):
            closest_angle_idx = np.abs(dst_angles_full - dst_angles[i]).argmin()
            bar = ax.bar(xaxis - i * width_each, mass_probability[:, closest_angle_idx], width_each,
                         color=colors[i], label=r'$\theta_k^{(0)}$=' + f'{np.around(np.rad2deg(dst_angles[i])):.0f}°')
            top_label_value = mass_probability[:, closest_angle_idx]
            # bar_top_label = [f'{i:.2f}' if i > threshold else '' for i in top_label_value]
            # ax.bar_label(bar, label=bar_top_label, frmpadding=3)
        plt.ylabel(r'$p_k(s)$')
        plt.xlabel(r'$\theta_s$')
        ax.set_xticks(xaxis, xlabels)
        ax.legend()
        plt.grid(axis='y')
        title = r'Mass probability having $\sigma =' + f'{error_degree}$° ' + title_values

        if not render:
            plt.title(title)
            plt.show(block=False)
        else:
            filename = os.path.join(output_dir, f'mass_prob_N{N_x}_lamVSd{np.around(wavelength / d_x):.0f}')
            # tikzplotlib.save(filename + '.tex')
            plt.title(title)
            plt.savefig(filename + '.png', dpi=300)
            plt.close()

        # Mass probability on the best configuration (to the right)
        error_degree = np.linspace(0.5, 22.5, samples)
        std_error = np.deg2rad(error_degree)
        # Taking best configuration per angular position
        _, ax = plt.subplots()
        for i in range(len(dst_angles)):
            best_conf_idx = np.abs(np.arcsin(conf_values) - dst_angles[i]).argmin()
            mass_probability_nearest = qfunc((conf_HP_minus[best_conf_idx] - dst_angles[i]) / std_error) - qfunc((conf_HP_plus[best_conf_idx] - dst_angles[i]) / std_error)
            ax.plot(error_degree, mass_probability_nearest, color=colors[i], label=r'$\theta_k^{(0)}$=' + f'{np.around(np.rad2deg(dst_angles[i])):.0f}°')
        plt.ylabel(r'$p_k(s)$')
        plt.xlabel(r'$\sigma$ (°)')
        ax.legend()
        plt.grid()
        title = r'Best configuration vs error deviation ' + title_values

        if not render:
            plt.title(title)
            plt.show(block=False)
        else:
            filename = os.path.join(output_dir, f'bestconf_N{N_x}_lamVSd{np.around(wavelength / d_x):.0f}')
            # tikzplotlib.save(filename + '.tex')
            plt.title(title)
            plt.savefig(filename + '.png', dpi=300)
            plt.close()

        # Mass probability on the nearest configuration (to the right)
        _, ax = plt.subplots()
        for i in range(len(dst_angles)):
            best_conf_idx = np.abs(np.arcsin(conf_values) - dst_angles[i]).argmin()
            mass_probability_nearest = qfunc((conf_HP_minus[best_conf_idx + 1] - dst_angles[i]) / std_error) - qfunc((conf_HP_plus[best_conf_idx + 1] - dst_angles[i]) / std_error)
            ax.plot(error_degree, mass_probability_nearest, color=colors[i], label=r'$\theta_k^{(0)}$=' + f'{np.around(np.rad2deg(dst_angles[i])):.0f}°')
        plt.ylabel(r'$p_k(s)$')
        plt.xlabel(r'$\sigma$ [deg]')
        ax.legend()
        plt.grid()
        title = r'Nearest configuration vs error deviation ' + title_values

        if not render:
            plt.title(title)
            plt.show(block=False)
        else:
            filename = os.path.join(output_dir, f'nearestconf_N{N_x}_lamVSd{np.around(wavelength / d_x):.0f}')
            # tikzplotlib.save(filename + '.tex')
            plt.title(title)
            plt.savefig(filename + '.png', dpi=300)
            plt.close()

        # Frequency selectivity and hopping
        carrier_spacing = 300e3
        F = 50000

        # What happens to the second-best configuration (right)
        array_frequency = np.zeros((F, len(dst_angles)))
        _, ax = plt.subplots()
        for i in range(len(dst_angles)):
            best_conf_idx = abs(np.arcsin(conf_values) - dst_angles[i]).argmin()
            # best
            conf = - conf_values[best_conf_idx] + np.sin(src_ang)
            array_frequency[:, i] = gen_array_factor(src_ang, dst_angles[i], conf, d_x, N_x, fc, F, carrier_spacing)
            # plot
            ax.plot(array_frequency[:, i], color=colors[i], label=r'$\theta_k^{(0)}$=' + f'{np.around(np.rad2deg(dst_angles[i])):.0f}°')
            # second
            conf = - conf_values[best_conf_idx + 1] + np.sin(src_ang)
            array_frequency[:, i] = gen_array_factor(src_ang, dst_angles[i], conf, d_x, N_x, fc, F, carrier_spacing)
            # plot
            ax.plot(array_frequency[:, i], ls='--', color=colors[i], label='_nolabel_')
            # third
            conf = - conf_values[best_conf_idx + 2] + np.sin(src_ang)
            array_frequency[:, i] = gen_array_factor(src_ang, dst_angles[i], conf, d_x, N_x, fc, F, carrier_spacing)
            # plot
            ax.plot(array_frequency[:, i], ls='--', color=colors[i], label='_nolabel_')
            # forth
            conf = - conf_values[best_conf_idx + 3] + np.sin(src_ang)
            array_frequency[:, i] = gen_array_factor(src_ang, dst_angles[i], conf, d_x, N_x, fc, F, carrier_spacing)
            # plot
            ax.plot(array_frequency[:, i], ls='--', color=colors[i], label='_nolabel_')


        plt.ylabel(r'$\frac{|\mathcal{A}_k|^2}{N_x^2}$')
        plt.xlabel(r'$f$')
        ax.legend()
        plt.grid()
        title = r'Frequency hopping ' + title_values

        if not render:
            plt.title(title)
            plt.show(block=False)
        else:
            filename = os.path.join(output_dir, f'frequencyhop_N{N_x}_lamVSd{np.around(wavelength / d_x):.0f}')
            # tikzplotlib.save(filename + '.tex')
            plt.title(title)
            plt.savefig(filename + '.png', dpi=300)
            plt.close()

