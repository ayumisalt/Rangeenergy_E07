import enum
from statistics import harmonic_mean, mean, stdev
import os
from time import process_time_ns

# from turtle import color
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import warnings
import copy
import json
import math
from scipy.optimize import fsolve
import rangeenergy as reen
import nuclide

# import kinema_impl
import kinema
import sys


# from scipy.optimize.minpack import _fixed_point_helper
# from sympy import arg
# sys.path.append('..')


def gaussian_func(arr, constant, mean, sigma):
    return (
        constant
        * np.exp(-((arr - mean) ** 2) / (2 * sigma**2))
        / (2 * np.pi * sigma**2) ** 0.5
    )


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["figure.figsize"] = (7.5, 6)

if __name__ == "__main__":

    args = sys.argv
    filename_H3L = args[1]

    BL_H3L = []
    BL_H3L_stat = []
    BL_H3L_syst = []
    BL_H3L_error = []

    for line in open(filename_H3L, "r"):
        if len(line) == 1:
            continue
        if line[0] == "#":
            continue
        item_list = line.split()
        BL_H3L.append(float(item_list[0]))
        BL_H3L_stat.append(float(item_list[1]))
        BL_H3L_syst.append(float(item_list[2]))
        BL_H3L_error.append(
            math.sqrt(
                float(item_list[1]) * float(item_list[1])
                + float(item_list[2]) * float(item_list[2])
            )
        )

    x = 0
    w = 0
    for i, j in zip(BL_H3L, BL_H3L_error):
        w += 1 / (j * j)
        x += i * (1 / (j * j))

    weighted_ave_H3L = x / w
    # print(weighted_ave_H3L_H3L)

    x = 0
    w = 0
    for i, j in zip(BL_H3L, BL_H3L_error):
        x += (1 / (j * j)) * (i - weighted_ave_H3L) ** 2
        w += 1 / (j * j)

    weighted_error_H3L = 1 / math.sqrt(w)

    BL_H3L_axis = np.linspace(-0.5, 1, 100)

    fig, ax = plt.subplots()
    events = list(range(len(BL_H3L)))
    plt.axvline(
        x=weighted_ave_H3L,
        ymin=0,
        ymax=len(BL_H3L),
        color="b",
        label="{0:.3f} +/- {1:.3f} MeV".format(weighted_ave_H3L, weighted_error_H3L),
    )
    plt.errorbar(
        BL_H3L,
        events,
        yerr=None,
        xerr=BL_H3L_stat,
        capsize=5,
        fmt="o",
        markersize=3,
        ecolor="black",
        markeredgecolor="black",
        color="k",
    )

    plt.errorbar(
        BL_H3L[-1],
        events[-1],
        yerr=None,
        xerr=BL_H3L_stat[-1],
        capsize=5,
        fmt="o",
        markersize=3,
        ecolor="red",
        markeredgecolor="red",
        color="r",
    )

    event_counter = 0
    for m, e in zip(BL_H3L, BL_H3L_syst):
        # print(m, e)
        r = patches.Rectangle(
            xy=(m - e, event_counter - 0.2), width=2 * e, height=0.4, ec="k", fill=False
        )
        ax.add_patch(r)
        event_counter += 1

    r = patches.Rectangle(
        xy=(BL_H3L[-1] - BL_H3L_syst[-1], event_counter - 1.2),
        width=2 * BL_H3L_syst[-1],
        height=0.4,
        ec="r",
        fill=False,
    )
    ax.add_patch(r)

    b = patches.Rectangle(
        xy=(weighted_ave_H3L - weighted_error_H3L, -1),
        width=weighted_error_H3L * 2,
        height=len(BL_H3L) + 2,
        ec="gray",
        fill="gray",
        alpha=0.5,
    )

    ax.add_patch(b)

    # y = 0
    # for m, e in zip(BL_H3L, BL_H3L_error):
    #     y += gaussian_func(BL_H3L_axis, 1, m, e)

    # # y = y / 4
    # plt.plot(BL_H3L_axis, y, color="r", label="H3L")

    plt.xlabel("B$_{\u039b}$ [MeV]")
    plt.xlim(-0.25, 1.25)
    plt.ylim(-1, len(BL_H3L) + 1)
    plt.legend(loc="upper right", bbox_to_anchor=(1, 1), borderaxespad=0)
    plt.show()
    plt.close()

    filename_H4L = args[2]

    BL_H4L = []
    BL_H4L_stat = []
    BL_H4L_syst = []
    BL_H4L_error = []

    for line in open(filename_H4L, "r"):
        if len(line) == 1:
            continue
        if line[0] == "#":
            continue
        item_list = line.split()
        BL_H4L.append(float(item_list[0]))
        BL_H4L_stat.append(float(item_list[1]))
        BL_H4L_syst.append(float(item_list[2]))
        BL_H4L_error.append(
            math.sqrt(
                float(item_list[1]) * float(item_list[1])
                + float(item_list[2]) * float(item_list[2])
            )
        )

    x = 0
    w = 0
    for i, j in zip(BL_H4L, BL_H4L_error):
        w += 1 / (j * j)
        x += i * (1 / (j * j))

    weighted_ave_H4L = x / w
    # print(weighted_ave_H4L_H4L)

    x = 0
    w = 0
    for i, j in zip(BL_H4L, BL_H4L_error):
        x += (1 / (j * j)) * (i - weighted_ave_H4L) ** 2
        w += 1 / (j * j)

    weighted_error_H4L = 1 / math.sqrt(w)

    BL_H4L_axis = np.linspace(1.5, 3, 100)

    fig, ax = plt.subplots()
    events = list(range(len(BL_H4L)))
    plt.axvline(
        x=weighted_ave_H4L,
        ymin=0,
        ymax=len(BL_H4L),
        color="b",
        label="{0:.3f} +/- {1:.3f} MeV".format(weighted_ave_H4L, weighted_error_H4L),
    )
    plt.errorbar(
        BL_H4L,
        events,
        yerr=None,
        xerr=BL_H4L_stat,
        capsize=5,
        fmt="o",
        markersize=3,
        ecolor="black",
        markeredgecolor="black",
        color="k",
    )

    plt.errorbar(
        BL_H4L[-1],
        events[-1],
        yerr=None,
        xerr=BL_H4L_stat[-1],
        capsize=5,
        fmt="o",
        markersize=3,
        ecolor="red",
        markeredgecolor="red",
        color="r",
    )

    event_counter = 0
    for m, e in zip(BL_H4L, BL_H4L_syst):
        # print(m, e)
        r = patches.Rectangle(
            xy=(m - e, event_counter - 0.2), width=2 * e, height=0.4, ec="k", fill=False
        )
        ax.add_patch(r)
        event_counter += 1

    r = patches.Rectangle(
        xy=(BL_H4L[-1] - BL_H4L_syst[-1], event_counter - 1.2),
        width=2 * BL_H4L_syst[-1],
        height=0.4,
        ec="r",
        fill=False,
    )
    ax.add_patch(r)

    b = patches.Rectangle(
        xy=(weighted_ave_H4L - weighted_error_H4L, -1),
        width=weighted_error_H4L * 2,
        height=len(BL_H4L) + 2,
        ec="gray",
        fill="gray",
        alpha=0.5,
    )

    ax.add_patch(b)

    # y = 0
    # for m, e in zip(BL_H4L, BL_H4L_error):
    #     y += gaussian_func(BL_H4L_axis, 1, m, e)
    # # y = y / 4
    # plt.plot(BL_H4L_axis, y, color="r", label="H4L")

    plt.xlabel("B$_{\u039b}$ [MeV]")
    plt.xlim(1.8, 3.3)
    plt.ylim(-1, len(BL_H4L) + 1)
    plt.legend(loc="upper right", bbox_to_anchor=(1, 1), borderaxespad=0)
    plt.show()
    plt.close()
