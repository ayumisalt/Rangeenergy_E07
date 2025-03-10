# calc_invariant_mass.py

import math
import numpy as np
import kinema
import rangeenergy as reen

reen.RANGEENERGY_MODEL = "ATIMA"

def compute_invariant_masses(
    data,
    range_shift_H3L=0,
    range_shift_H4L=0,
    mom_shift=0,
    ke_shift=0.043, # Correction from Barkas effect
    density_shift=0,
    range_factor_H3L=1,
    range_factor_H4L=1,
    mom_factor_H3L=1,
    mom_factor_H4L=1,
    density_factor=1,
    ke_factor=1,
):
    """
    Computes the invariant masses and binding energies (BΛ) for H3Λ and H4Λ
    from the given data.

    Parameters
    ----------
    data : dict
        A dictionary of lists (e.g., filtered by angle cuts) containing
        at least the following keys:
          "Range_1", "Range_error_1", "Theta_1", "Theta_error_1",
          "Phi_1",   "Phi_error_1",   "Range_2", "Range_error_2",
          "Theta_2", "Theta_error_2", "Phi_2",   "Phi_error_2",
          "Range_3", "Range_error_3", "Theta_3", "Theta_error_3",
          "Phi_3",   "Phi_error_3",   "label",
          "density", "density_error"
        The arrays must be of the same length (one entry per event).
    range_shift_H3L : float
        Range shift for H3Λ events (default=0).
    range_shift_H4L : float
        Range shift for H4Λ events (default=0).
    mom_shift : float
        Momentum shift applied after momentum calculations (default=0).
    ke_shift : float
        Kinetic-energy shift subtracted after KE calculations (default=0.043).
    density_shift : float
        Density shift subtracted from the measured density (default=0).
    range_factor_H3L : float
        Range scale factor for H3Λ events (default=1).
    range_factor_H4L : float
        Range scale factor for H4Λ events (default=1).
    mom_factor_H3L : float
        Momentum scale factor for H3Λ events (default=1).
    mom_factor_H4L : float
        Momentum scale factor for H4Λ events (default=1).
    density_factor : float
        Density scale factor (default=1).
    ke_factor : float
        Kinetic-energy scale factor (default=1).

    Returns
    -------
    results : dict
        A dictionary containing lists of computed results:
          {
            "BL_H3L": list[float],
            "BL_H3L_stat": list[float],
            "BL_H3L_syst": list[list[float]],   # [upper_shift, lower_shift] per event
            "BL_H4L": list[float],
            "BL_H4L_stat": list[float],
            "BL_H4L_syst": list[list[float]],   # [upper_shift, lower_shift] per event
          }

        - BL_H3L and BL_H4L are the binding energies (BΛ) for each valid event.
        - BL_H3L_stat and BL_H4L_stat are the statistical errors computed from
          range/pion-momentum uncertainties.
        - BL_H3L_syst and BL_H4L_syst are the lists of [positive shift, negative shift],
          representing systematic-like shifts from density uncertainties.
    """

    # Mass constants (in MeV/c^2)
    proton_mass = 938.272
    He3_mass = 2808.391
    He4_mass = 3727.379
    pi_mass = 139.57
    # Mass of Lambda hypernuclei:
    #   H3Λ = (2H + Λ) => 1875.613 + 1115.683
    #   H4Λ = (3He + Λ) => 2808.921 + 1115.683
    H3L_mass = 1875.613 + 1115.683
    H4L_mass = 2808.921 + 1115.683

    # Prepare lists to store results
    BL_H3L = []
    BL_H3L_stat = []
    BL_H3L_syst = []

    BL_H4L = []
    BL_H4L_stat = []
    BL_H4L_syst = []

    # Retrieve data arrays
    Range_1 = data["Range_1"]
    Range_error_1 = data["Range_error_1"]
    Theta_1 = data["Theta_1"]
    Theta_error_1 = data["Theta_error_1"]
    Phi_1 = data["Phi_1"]
    Phi_error_1 = data["Phi_error_1"]

    Range_2 = data["Range_2"]
    Range_error_2 = data["Range_error_2"]
    Theta_2 = data["Theta_2"]
    Theta_error_2 = data["Theta_error_2"]
    Phi_2 = data["Phi_2"]
    Phi_error_2 = data["Phi_error_2"]

    Range_3 = data["Range_3"]
    Range_error_3 = data["Range_error_3"]
    Theta_3 = data["Theta_3"]
    Theta_error_3 = data["Theta_error_3"]
    Phi_3 = data["Phi_3"]
    Phi_error_3 = data["Phi_error_3"]

    label = data["label"]
    density = data["density"]
    density_error = data["density_error"]

    # Loop over each event
    for i in range(len(label)):
        # We only compute if label is "3" (H3Λ) or "4" (H4Λ)
        if label[i] == "3":
            # Compute kinetic energy from range (pion track is assumed to be Range_2)
            ke_c = (
                reen.KEfromRange(
                    pi_mass,
                    (Range_2[i] - range_shift_H3L) * range_factor_H3L,
                    -1,
                    (density[i] * density_factor) - density_shift
                ) * ke_factor
            ) - ke_shift

            # + / - density error for systematic
            ke_cu = (
                reen.KEfromRange(
                    pi_mass,
                    (Range_2[i] - range_shift_H3L) * range_factor_H3L,
                    -1,
                    (density[i] * density_factor) - density_shift + density_error[i]
                ) * ke_factor
            ) - ke_shift
            ke_cl = (
                reen.KEfromRange(
                    pi_mass,
                    (Range_2[i] - range_shift_H3L) * range_factor_H3L,
                    -1,
                    (density[i] * density_factor) - density_shift - density_error[i]
                ) * ke_factor
            ) - ke_shift

            # Convert KE to momentum
            mom_c = kinema.ke2mom(pi_mass, ke_c) * mom_factor_H3L - mom_shift
            mom_cu = kinema.ke2mom(pi_mass, ke_cu) * mom_factor_H3L - mom_shift
            mom_cl = kinema.ke2mom(pi_mass, ke_cl) * mom_factor_H3L - mom_shift

            # Evaluate range straggling for error
            range_straggling = reen.RangeStragglingFromRange(
                pi_mass, Range_2[i], -1, density[i]
            )
            range_e = math.sqrt(Range_error_2[i] ** 2 + range_straggling**2)

            range_l = (Range_2[i] - range_shift_H3L) * range_factor_H3L - range_e
            ke_l = (
                reen.KEfromRange(
                    pi_mass,
                    range_l,
                    -1,
                    (density[i] * density_factor) - density_shift
                ) * ke_factor
            ) - ke_shift
            mom_l = kinema.ke2mom(pi_mass, ke_l) * mom_factor_H3L - mom_shift

            range_r = (Range_2[i] - range_shift_H3L) * range_factor_H3L + range_e
            ke_r = (
                reen.KEfromRange(
                    pi_mass,
                    range_r,
                    -1,
                    (density[i] * density_factor) - density_shift
                ) * ke_factor
            ) - ke_shift
            mom_r = kinema.ke2mom(pi_mass, ke_r) * mom_factor_H3L - mom_shift

            # Momentum error (statistical)
            mom_e = np.fabs(mom_l - mom_r) / 2.0

            # Invariant mass of He-3 + pion
            invariant_mass = math.sqrt(mom_c**2 + He3_mass**2) + math.sqrt(
                mom_c**2 + pi_mass**2
            )

            # BΛ = (H3Λ mass) - (invariant mass)
            BL_value = H3L_mass - invariant_mass
            BL_H3L.append(BL_value)

            # Approx. error propagation from momentum error (stat)
            # d(M) = dM/dp * dp,  where M = sqrt(p^2 + MHe3^2) + sqrt(p^2 + Mπ^2)
            # => dM/dp = p / sqrt(MHe3^2 + p^2) + p / sqrt(Mπ^2 + p^2)
            dMdp = (
                mom_c / math.sqrt(He3_mass**2 + mom_c**2) +
                mom_c / math.sqrt(pi_mass**2 + mom_c**2)
            )
            invariant_mass_error = dMdp * mom_e
            BL_H3L_stat.append(invariant_mass_error)

            # Systematic shift due to density uncertainty
            invariant_mass_cu = math.sqrt(mom_cu**2 + He3_mass**2) + math.sqrt(
                mom_cu**2 + pi_mass**2
            )
            invariant_mass_cl = math.sqrt(mom_cl**2 + He3_mass**2) + math.sqrt(
                mom_cl**2 + pi_mass**2
            )
            BL_H3L_syst.append([
                abs(invariant_mass_cu - invariant_mass),
                abs(invariant_mass_cl - invariant_mass),
            ])

        elif label[i] == "4":
            # For H4Λ
            ke_c = (
                reen.KEfromRange(
                    pi_mass,
                    (Range_2[i] - range_shift_H4L) * range_factor_H4L,
                    -1,
                    (density[i] * density_factor) - density_shift
                ) * ke_factor
            ) - ke_shift

            ke_cu = (
                reen.KEfromRange(
                    pi_mass,
                    (Range_2[i] - range_shift_H4L) * range_factor_H4L,
                    -1,
                    (density[i] * density_factor) - density_shift + density_error[i]
                ) * ke_factor
            ) - ke_shift
            ke_cl = (
                reen.KEfromRange(
                    pi_mass,
                    (Range_2[i] - range_shift_H4L) * range_factor_H4L,
                    -1,
                    (density[i] * density_factor) - density_shift - density_error[i]
                ) * ke_factor
            ) - ke_shift

            mom_c = kinema.ke2mom(pi_mass, ke_c) * mom_factor_H4L - mom_shift
            mom_cu = kinema.ke2mom(pi_mass, ke_cu) * mom_factor_H4L - mom_shift
            mom_cl = kinema.ke2mom(pi_mass, ke_cl) * mom_factor_H4L - mom_shift

            range_straggling = reen.RangeStragglingFromRange(
                pi_mass, Range_2[i], -1, density[i]
            )
            range_e = math.sqrt(Range_error_2[i] ** 2 + range_straggling**2)
            range_l = (Range_2[i] - range_shift_H4L) * range_factor_H4L - range_e
            ke_l = (
                reen.KEfromRange(
                    pi_mass,
                    range_l,
                    -1,
                    (density[i] * density_factor) - density_shift
                ) * ke_factor
            ) - ke_shift
            mom_l = kinema.ke2mom(pi_mass, ke_l) * mom_factor_H4L - mom_shift

            range_r = (Range_2[i] - range_shift_H4L) * range_factor_H4L + range_e
            ke_r = (
                reen.KEfromRange(
                    pi_mass,
                    range_r,
                    -1,
                    (density[i] * density_factor) - density_shift
                ) * ke_factor
            ) - ke_shift
            mom_r = kinema.ke2mom(pi_mass, ke_r) * mom_factor_H4L - mom_shift
            mom_e = np.fabs(mom_l - mom_r) / 2.0

            # Invariant mass of He-4 + pion
            invariant_mass = math.sqrt(mom_c**2 + He4_mass**2) + math.sqrt(
                mom_c**2 + pi_mass**2
            )
            BL_value = H4L_mass - invariant_mass
            BL_H4L.append(BL_value)

            # Error propagation
            dMdp = (
                mom_c / math.sqrt(He4_mass**2 + mom_c**2) +
                mom_c / math.sqrt(pi_mass**2 + mom_c**2)
            )
            invariant_mass_error = dMdp * mom_e
            BL_H4L_stat.append(invariant_mass_error)

            invariant_mass_cu = math.sqrt(mom_cu**2 + He4_mass**2) + math.sqrt(
                mom_cu**2 + pi_mass**2
            )
            invariant_mass_cl = math.sqrt(mom_cl**2 + He4_mass**2) + math.sqrt(
                mom_cl**2 + pi_mass**2
            )
            BL_H4L_syst.append([
                abs(invariant_mass_cu - invariant_mass),
                abs(invariant_mass_cl - invariant_mass),
            ])

    # Prepare a dictionary with all results
    results = {
        "BL_H3L": BL_H3L,
        "BL_H3L_stat": BL_H3L_stat,
        "BL_H3L_syst": BL_H3L_syst,
        "BL_H4L": BL_H4L,
        "BL_H4L_stat": BL_H4L_stat,
        "BL_H4L_syst": BL_H4L_syst,
    }
    return results
