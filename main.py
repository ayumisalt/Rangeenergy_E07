# main.py
import sys
sys.path.append("..")

from read_data import read_data
from cut_events import apply_angle_cuts
from calc_invariant_mass import compute_invariant_masses
from plot_results import plot_BL_distribution


if __name__ == "__main__":
    args = sys.argv
    filename = args[1]
    density_ave = float(args[2])

    # Step 1: Read the data
    data = read_data(filename, density_ave)

    # Step 2: Apply angle-based cuts
    filtered_data = apply_angle_cuts(data)

    # Step 3: Compute invariant masses / BΛ
    results = compute_invariant_masses(filtered_data)

    # Step 4: Plot results for H3Λ
    wa3, we3, chi3, ndf3 = plot_BL_distribution(
        BL_values=results["BL_H3L"],
        BL_stat_errors=results["BL_H3L_stat"],
        BL_syst_errors=results["BL_H3L_syst"],
    )

    # Plot results for H4Λ
    wa4, we4, chi4, ndf4 = plot_BL_distribution(
        BL_values=results["BL_H4L"],
        BL_stat_errors=results["BL_H4L_stat"],
        BL_syst_errors=results["BL_H4L_syst"],
    )

    # Optionally, print or log the final weighted averages, errors, etc.
    print(f"H3\u039b Weighted Average: {wa3:.2f} $\pm$ {we3:.2f}  (\u03c7$^{2}$/Ndf = {chi3:.2f}/{ndf3})")
    print(f"H4\u039b Weighted Average: {wa4:.2f} $\pm$ {we4:.2f}  (\u03c7$^{2}$/Ndf = {chi4:.2f}/{ndf4})")
