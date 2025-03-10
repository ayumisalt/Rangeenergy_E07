from statistics import harmonic_mean, mean, stdev
import matplotlib.pyplot as plt
import numpy as np
import copy
import math
from scipy.optimize import fsolve
import rangeenergy as reen

# import kinema_impl
import kinema
import sys

from sympy import arg

sys.path.append("..")

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18

if __name__ == "__main__":
    args = sys.argv
    filename = args[1]
    density_alpha = args[2]
    # density[i] = float(args[3])

    reen.RANGEENERGY_MODEL = "Mishina"

    proton_mass = 938.272
    He3_mass = 2808.391
    He4_mass = 3727.379
    pi_mass = 139.57
    H3L_mass = 1875.613 + 1115.683
    H4L_mass = 2808.921 + 1115.683

    # ID  Range_1  Range_error_1  Theta_1  Theta_error_1  Phi_1  Phi_error_1  Range_2  Range_error_2  Theta_2  Theta_error_2  Phi_2  Phi_error_2  vertex_e  chi  label;
    ID = []
    Range_1 = []
    Range_error_1 = []
    Theta_1 = []
    Theta_error_1 = []
    Phi_1 = []
    Phi_error_1 = []
    Range_2 = []
    Range_error_2 = []
    Theta_2 = []
    Theta_error_2 = []
    Phi_2 = []
    Phi_error_2 = []
    Range_3 = []
    Range_error_3 = []
    Theta_3 = []
    Theta_error_3 = []
    Phi_3 = []
    Phi_error_3 = []
    label = []
    density = []
    density_error = []

    for line in open(filename, "r"):
        if len(line) == 1:
            continue
        if line[0] == "#":
            continue
        item_list = line.split()
        ID.append(item_list[0])
        Range_1.append(float(item_list[1]))
        Range_error_1.append(float(item_list[2]))
        # Range_error_1.append(0.5)
        Theta_1.append(float(item_list[3]))
        Theta_error_1.append(float(item_list[4]))
        Phi_1.append(float(item_list[5]))
        Phi_error_1.append(float(item_list[6]))
        Range_2.append(float(item_list[7]))
        Range_error_2.append(float(item_list[8]))
        Theta_2.append(float(item_list[9]))
        Theta_error_2.append(float(item_list[10]))
        Phi_2.append(float(item_list[11]))
        Phi_error_2.append(float(item_list[12]))
        Range_3.append(float(item_list[13]))
        Range_error_3.append(float(item_list[14]))
        Theta_3.append(float(item_list[15]))
        Theta_error_3.append(float(item_list[16]))
        Phi_3.append(float(item_list[17]))
        Phi_error_3.append(float(item_list[18]))
        label.append(item_list[19])
        if item_list[20] == "XXX":
            density.append(float(density_alpha))
        else:
            density.append(float(item_list[20]))
        if item_list[21] == "XXX":
            density_error.append(float(0.03))
        else:
            density_error.append(float(item_list[21]))

    He4_range = []
    He_moms = []
    pi_moms = []
    diff1 = []
    diff2 = []

    for i in range(len(ID)):
        dx1 = math.sin(Theta_1[i] * (math.pi / 180.0)) * math.cos(
            Phi_1[i] * (math.pi / 180.0)
        )
        dy1 = math.sin(Theta_1[i] * (math.pi / 180.0)) * math.sin(
            Phi_1[i] * (math.pi / 180.0)
        )
        dz1 = math.cos(Theta_1[i] * (math.pi / 180.0))

        dx2 = math.sin(Theta_2[i] * (math.pi / 180.0)) * math.cos(
            Phi_2[i] * (math.pi / 180.0)
        )
        dy2 = math.sin(Theta_2[i] * (math.pi / 180.0)) * math.sin(
            Phi_2[i] * (math.pi / 180.0)
        )
        dz2 = math.cos(Theta_2[i] * (math.pi / 180.0))

        dx3 = math.sin(Theta_3[i] * (math.pi / 180.0)) * math.cos(
            Phi_3[i] * (math.pi / 180.0)
        )
        dy3 = math.sin(Theta_3[i] * (math.pi / 180.0)) * math.sin(
            Phi_3[i] * (math.pi / 180.0)
        )
        dz3 = math.cos(Theta_3[i] * (math.pi / 180.0))

        # Coplanarity cut
        cop = (
            ((dy3 * dz1) - (dz3 * dy1)) * dx2
            + ((dz3 * dx1) - (dx3 * dz1)) * dy2
            + ((dx3 * dy1) - (dy3 * dx1)) * dz2
        )

        dcop_dtheta0 = (
            (
                math.cos(Theta_1[i] * (math.pi / 180.0))
                * math.cos(Phi_1[i] * (math.pi / 180.0))
            )
            * dy2
            * dz3
            + +dx2 * dy3 * (-1 * math.sin(Theta_1[i] * (math.pi / 180.0)))
            + +dx3
            * (
                math.cos(Theta_1[i] * (math.pi / 180.0))
                * math.sin(Phi_1[i] * (math.pi / 180.0))
            )
            * dz2
            + -(
                math.cos(Theta_1[i] * (math.pi / 180.0))
                * math.cos(Phi_1[i] * (math.pi / 180.0))
            )
            * dy3
            * dz2
            + -dx2
            * (
                math.cos(Theta_1[i] * (math.pi / 180.0))
                * math.sin(Phi_1[i] * (math.pi / 180.0))
            )
            * dz3
            + -dx3 * dy2 * (-1 * math.sin(Theta_1[i] * (math.pi / 180.0)))
        )

        dcop_dphi0 = (
            (
                -1
                * math.sin(Theta_1[i] * (math.pi / 180.0))
                * math.sin(Phi_1[i] * (math.pi / 180.0))
            )
            * dy2
            * dz3
            + +0
            + +dx3
            * (
                math.sin(Theta_1[i] * (math.pi / 180.0))
                * math.cos(Phi_1[i] * (math.pi / 180.0))
            )
            * dz2
            + -(
                -1
                * math.sin(Theta_1[i] * (math.pi / 180.0))
                * math.sin(Phi_1[i] * (math.pi / 180.0))
            )
            * dy3
            * dz2
            + -dx2
            * (
                math.sin(Theta_1[i] * (math.pi / 180.0))
                * math.cos(Phi_1[i] * (math.pi / 180.0))
            )
            * dz3
            + -0
        )

        dcop_dtheta1 = (
            (
                math.cos(Theta_2[i] * (math.pi / 180.0))
                * math.cos(Phi_2[i] * (math.pi / 180.0))
            )
            * dy3
            * dz1
            + +dx3 * dy1 * (-1 * math.sin(Theta_2[i] * (math.pi / 180.0)))
            + +dx1
            * (
                math.cos(Theta_2[i] * (math.pi / 180.0))
                * math.sin(Phi_2[i] * (math.pi / 180.0))
            )
            * dz3
            + -(
                math.cos(Theta_2[i] * (math.pi / 180.0))
                * math.cos(Phi_2[i] * (math.pi / 180.0))
            )
            * dy1
            * dz3
            + -dx3
            * (
                math.cos(Theta_2[i] * (math.pi / 180.0))
                * math.sin(Phi_2[i] * (math.pi / 180.0))
            )
            * dz1
            + -dx1 * dy3 * (-1 * math.sin(Theta_2[i] * (math.pi / 180.0)))
        )

        dcop_dphi1 = (
            (
                -1
                * math.sin(Theta_2[i] * (math.pi / 180.0))
                * math.sin(Phi_2[i] * (math.pi / 180.0))
            )
            * dy3
            * dz1
            + +0
            + +dx1
            * (
                math.sin(Theta_2[i] * (math.pi / 180.0))
                * math.cos(Phi_2[i] * (math.pi / 180.0))
            )
            * dz3
            + -(
                -1
                * math.sin(Theta_2[i] * (math.pi / 180.0))
                * math.sin(Phi_2[i] * (math.pi / 180.0))
            )
            * dy1
            * dz3
            + -dx3
            * (
                math.sin(Theta_2[i] * (math.pi / 180.0))
                * math.cos(Phi_2[i] * (math.pi / 180.0))
            )
            * dz1
            + -0
        )

        dcop_dtheta2 = (
            (
                math.cos(Theta_3[i] * (math.pi / 180.0))
                * math.cos(Phi_3[i] * (math.pi / 180.0))
            )
            * dy1
            * dz2
            + +dx1 * dy2 * (-1 * math.sin(Theta_3[i] * (math.pi / 180.0)))
            + +dx2
            * (
                math.cos(Theta_3[i] * (math.pi / 180.0))
                * math.sin(Phi_3[i] * (math.pi / 180.0))
            )
            * dz1
            + -(
                math.cos(Theta_3[i] * (math.pi / 180.0))
                * math.cos(Phi_3[i] * (math.pi / 180.0))
            )
            * dy2
            * dz1
            + -dx1
            * (
                math.cos(Theta_3[i] * (math.pi / 180.0))
                * math.sin(Phi_3[i] * (math.pi / 180.0))
            )
            * dz2
            + -dx2 * dy1 * (-1 * math.sin(Theta_3[i] * (math.pi / 180.0)))
        )

        dcop_dphi2 = (
            (
                -1
                * math.sin(Theta_3[i] * (math.pi / 180.0))
                * math.sin(Phi_3[i] * (math.pi / 180.0))
            )
            * dy1
            * dz2
            + +0
            + +dx2
            * (
                math.sin(Theta_3[i] * (math.pi / 180.0))
                * math.cos(Phi_3[i] * (math.pi / 180.0))
            )
            * dz1
            + -(
                -1
                * math.sin(Theta_3[i] * (math.pi / 180.0))
                * math.sin(Phi_3[i] * (math.pi / 180.0))
            )
            * dy2
            * dz1
            + -dx1
            * (
                math.sin(Theta_3[i] * (math.pi / 180.0))
                * math.cos(Phi_3[i] * (math.pi / 180.0))
            )
            * dz2
            + -0
        )

        cop_error = math.sqrt(
            pow(dcop_dtheta0 * Theta_error_1[i] * (math.pi / 180.0), 2.0)
            + pow(dcop_dphi0 * Phi_error_1[i] * (math.pi / 180.0), 2.0)
            + pow(dcop_dtheta1 * Theta_error_2[i] * (math.pi / 180.0), 2.0)
            + pow(dcop_dphi1 * Phi_error_2[i] * (math.pi / 180.0), 2.0)
            + pow(dcop_dtheta2 * Theta_error_3[i] * (math.pi / 180.0), 2.0)
            + pow(dcop_dphi2 * Phi_error_3[i] * (math.pi / 180.0), 2.0)
        )

        if abs(cop) > 0.26373:
            continue

        InP = dx1 * dx2 + dy1 * dy2 + dz1 * dz2

        if InP >= -0.95764:
            continue

        if label[i] == "4":
            He4_range.append(Range_1[i])
            He_ke = reen.KEfromRange(He4_mass, Range_1[i], 2, density[i])
            He_mom = kinema.ke2mom(He4_mass, He_ke)
            He_moms.append(He_mom)
            # pi_mom = He_mom
            pi_ke = reen.KEfromRange(pi_mass, Range_2[i], -1, density[i])
            pi_mom = kinema.ke2mom(pi_mass, pi_ke)

            pi_moms.append(pi_mom)
            diff2.append(pi_mom - He_mom)

        else:
            print(ID[i], label[i])

    cmap = copy.copy(plt.cm.jet)
    cmap.set_under("w", 1)

    counts, xedges, yedges, im = plt.hist2d(
        He_moms, pi_moms, range=[[125, 145], [125, 145]], bins=[20, 20], cmap=cmap
    )
    plt.xlabel("He momentum")
    plt.ylabel("$\pi$ momentum")
    plt.grid()
    im.set_clim(1, 5)
    plt.colorbar(im, label="Entries")
    plt.show()

    diff = diff1 + diff2
    plt.hist(diff1, range=(-15, 15), bins=20, histtype="stepfilled", alpha=0.8)
    plt.hist(diff2, range=(-15, 15), bins=20, histtype="step")
    plt.xlabel("He mom - $\pi$ mom")
    plt.show()

    plt.hist(diff, range=(-15, 15), bins=20)
    plt.xlabel("$\pi$ mom - He mom [MeV/c]")
    plt.show()

    plt.scatter(He_moms, pi_moms)
    plt.xlim(100, 150)
    plt.ylim(100, 150)
    plt.grid()

    plt.show()
