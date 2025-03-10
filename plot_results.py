# plot_results.py

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["figure.figsize"] = (7.5, 6)

def gaussian_func(x_array, constant, mean, sigma):
    """
    Returns a Gaussian distribution evaluated at x_array with
    amplitude 'constant', center 'mean', and standard deviation 'sigma'.
    """
    return (
        constant
        * np.exp(-((x_array - mean) ** 2) / (2 * sigma**2))
        / math.sqrt(2 * math.pi * sigma**2)
    )

def plot_BL_distribution(
    BL_values,
    BL_stat_errors,
    BL_syst_errors,
    x_min=-8,
    x_max=10,
    gaussian_range=(-20, 20),
):
    """
    Plots the binding-energy (BΛ) distribution for hypernuclei, replicating the logic
    from the original script:
      - Weighted average and error
      - Chi-squared calculation
      - Per-event statistical error bars
      - Systematic error boxes
      - Summed Gaussian distribution

    Parameters
    ----------
    BL_values : list of float
        List of binding-energy values (BΛ) for each event.
    BL_stat_errors : list of float
        List of statistical errors for each event (same length as BL_values).
    BL_syst_errors : list of [float, float]
        List of systematic error pairs [upper_shift, lower_shift] per event,
        same length as BL_values.
    label_str : str, optional
        A label to identify this distribution (e.g., "H3Λ" or "H4Λ").
        Used in the legend and axis label.
    x_min : float, optional
        Left boundary for the x-axis (default = -8).
    x_max : float, optional
        Right boundary for the x-axis (default = 10).
    gaussian_range : tuple of (float, float), optional
        Range over which the summed Gaussians are calculated for plotting.
        Default is (-20, 20).

    Returns
    -------
    weighted_ave : float
        Weighted average of the BΛ distribution.
    weighted_error : float
        Total weighted error (statistical only, in this plotting logic).
        (If you want to include systematic contributions, you could adjust accordingly.)
    chi2 : float
        The chi-squared value relative to the weighted average.
    ndf : int
        The number of degrees of freedom (len(BL_values) - 1).
    """

    # --- 1) Compute weighted average and total error (statistical) ---
    # weighted average formula:  ave = sum( B_i / err_i^2 ) / sum( 1 / err_i^2 )
    # error = 1 / sqrt( sum(1 / err_i^2) )
    if len(BL_values) == 0:
        # Avoid zero-division in case of empty lists
        print(f"[plot_BL_distribution] No events for")
        return None, None, None, None

    BL_error = []
    for i, j in zip(BL_stat_errors, BL_syst_errors):
        e_sys = abs(j[0]-j[1])/2
        e = math.sqrt(i * i + e_sys * e_sys)
        BL_error.append(e)

    w_sum = 0.0
    x_sum = 0.0
    for b_i, e_i in zip(BL_values, BL_error):
        weight = 1.0 / (e_i * e_i)
        w_sum += weight
        x_sum += b_i * weight

    weighted_ave = x_sum / w_sum
    weighted_error = 1.0 / math.sqrt(w_sum)
    ndf = len(BL_values) - 1

    # --- 2) Compute Chi-squared for the distribution ---
    chi2 = 0.0
    for b_i, e_i in zip(BL_values, BL_error):
        chi2 += ((weighted_ave - b_i) / e_i) ** 2

    # --- 3) Start plotting ---
    fig, ax = plt.subplots()

    # (a) Vertical line at the weighted average
    plt.axvline(
        x=weighted_ave,
        ymin=0,
        ymax=len(BL_values),
        color="b",
        label=(
            f"Weighted Ave.\n"
            f"{weighted_ave:.2f} $\pm$ {weighted_error:.2f}"
        ),
    )

    # (b) Plot the data points (per-event BΛ) with statistical error bars
    #     Each event is placed on the y-axis at integer steps
    event_indices = list(range(len(BL_values)))
    plt.errorbar(
        BL_values,
        event_indices,
        xerr=BL_stat_errors,
        fmt="o",
        ecolor="black",
        color="black",
        capsize=5,
        markersize=3,
        label=None,  # we already have a label for the line
    )

    # (c) Draw systematic error "boxes" around each point
    #     For each event, we have [ +syst_up, +syst_down ] around the central value
    event_counter = 0
    for b_i, e_stat, e_syst in zip(BL_values, BL_stat_errors, BL_syst_errors):
        syst_up, syst_down = e_syst
        # rectangle from (b_i - syst_down) to (b_i + syst_up), centered on event_counter
        rect = patches.Rectangle(
            xy=(b_i - syst_down, event_counter - 0.4),
            width=(syst_up + syst_down),
            height=0.8,
            fill=False,
            ec="black",
        )
        ax.add_patch(rect)
        event_counter += 1

    # (d) Shaded box for weighted-average ± error
    #     Extend the height from y=-1 to y=len(BL_values)+1 so it covers all data
    shaded_box = patches.Rectangle(
        xy=(weighted_ave - weighted_error, -1),
        width=2.0 * weighted_error,
        height=len(BL_values) + 1,
        alpha=0.5,
        ec="gray",
        color="gray",
    )
    ax.add_patch(shaded_box)

    # (e) Sum up Gaussians for each data point as a visual representation
    BL_axis = np.linspace(gaussian_range[0], gaussian_range[1], 1000)
    y_sum = np.zeros_like(BL_axis)
    for b_i, e_i in zip(BL_values, BL_error):
        y_sum += gaussian_func(BL_axis, 1.0, b_i, e_i)

    plt.plot(
        BL_axis,
        y_sum,
        color="r", 
        label=f"\u03c7$^{2}$/Ndf\n {chi2:.2f}/{ndf:d}"
    )

    # --- 4) Final plot styling ---
    plt.xlabel("B$_{\u039b}$ [MeV]")
    plt.xlim(x_min, x_max)
    plt.ylim(-1, len(BL_values))
    plt.legend(borderaxespad=0)
    plt.xticks(np.arange(x_min, x_max + 1, step=2))

    plt.show()
    plt.close(fig)

    # Return the weighting results in case they're needed outside
    return weighted_ave, weighted_error, chi2, ndf
