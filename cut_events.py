# cut_events.py

import math

def apply_angle_cuts(
    data,
    cop_threshold=0.26373,
    inP_threshold=-0.95764
): # thresholds were determined with experimental data (±3 sigma) 
    """
    Applies angle-related cuts based on coplanarity (cop) and the inner product (InP).

    The function calculates the direction vectors (dx, dy, dz) for each track,
    then computes:
      - cop (coplanarity): a scalar that measures the out-of-plane factor
      - InP (inner product): a measure of the angle between two tracks

    Events that do not meet the specified thresholds (|cop| <= cop_threshold
    and InP <= inP_threshold) are excluded from the output.

    Parameters
    ----------
    data : dict
        A dictionary of lists returned by read_data, containing:
          "ID", "Range_1", "Range_error_1", "Theta_1", "Theta_error_1", ...
          "Phi_3", "Phi_error_3", "label", "density", "density_error".
    cop_threshold : float, optional
        The maximum allowed absolute value of coplanarity (default=0.26373).
    inP_threshold : float, optional
        The maximum allowed value of the inner product (default=-0.95764).
        If InP > inP_threshold, the event is cut.

    Returns
    -------
    filtered_data : dict
        A dictionary with the same keys as `data`, containing only events
        that pass the angle cuts.
    """

    # Prepare empty lists to hold events that pass the cuts
    filtered_data = {key: [] for key in data.keys()}

    # Retrieve references to original data arrays for readability
    ID = data["ID"]
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

    # Loop over all events
    for i in range(len(ID)):
        # Calculate direction cosines for each track
        dx1 = math.sin(Theta_1[i] * math.pi / 180.0) * math.cos(Phi_1[i] * math.pi / 180.0)
        dy1 = math.sin(Theta_1[i] * math.pi / 180.0) * math.sin(Phi_1[i] * math.pi / 180.0)
        dz1 = math.cos(Theta_1[i] * math.pi / 180.0)

        dx2 = math.sin(Theta_2[i] * math.pi / 180.0) * math.cos(Phi_2[i] * math.pi / 180.0)
        dy2 = math.sin(Theta_2[i] * math.pi / 180.0) * math.sin(Phi_2[i] * math.pi / 180.0)
        dz2 = math.cos(Theta_2[i] * math.pi / 180.0)

        dx3 = math.sin(Theta_3[i] * math.pi / 180.0) * math.cos(Phi_3[i] * math.pi / 180.0)
        dy3 = math.sin(Theta_3[i] * math.pi / 180.0) * math.sin(Phi_3[i] * math.pi / 180.0)
        dz3 = math.cos(Theta_3[i] * math.pi / 180.0)

        # Calculate coplanarity (cop)
        # cop is essentially the scalar triple product of (track3) x (track1) dot (track2)
        # but here it's explicitly expanded.
        cop = (
            ((dy3 * dz1) - (dz3 * dy1)) * dx2
            + ((dz3 * dx1) - (dx3 * dz1)) * dy2
            + ((dx3 * dy1) - (dy3 * dx1)) * dz2
        )

        # Calculate inner product (InP) between track1 and track2
        InP = dx1 * dx2 + dy1 * dy2 + dz1 * dz2

        # Apply the cuts
        if abs(cop) > cop_threshold:
            continue
        if InP > inP_threshold:
            continue

        # If the event passes both cuts, append it to the filtered_data
        filtered_data["ID"].append(ID[i])
        filtered_data["Range_1"].append(Range_1[i])
        filtered_data["Range_error_1"].append(Range_error_1[i])
        filtered_data["Theta_1"].append(Theta_1[i])
        filtered_data["Theta_error_1"].append(Theta_error_1[i])
        filtered_data["Phi_1"].append(Phi_1[i])
        filtered_data["Phi_error_1"].append(Phi_error_1[i])
        filtered_data["Range_2"].append(Range_2[i])
        filtered_data["Range_error_2"].append(Range_error_2[i])
        filtered_data["Theta_2"].append(Theta_2[i])
        filtered_data["Theta_error_2"].append(Theta_error_2[i])
        filtered_data["Phi_2"].append(Phi_2[i])
        filtered_data["Phi_error_2"].append(Phi_error_2[i])
        filtered_data["Range_3"].append(Range_3[i])
        filtered_data["Range_error_3"].append(Range_error_3[i])
        filtered_data["Theta_3"].append(Theta_3[i])
        filtered_data["Theta_error_3"].append(Theta_error_3[i])
        filtered_data["Phi_3"].append(Phi_3[i])
        filtered_data["Phi_error_3"].append(Phi_error_3[i])
        filtered_data["label"].append(label[i])
        filtered_data["density"].append(density[i])
        filtered_data["density_error"].append(density_error[i])

    return filtered_data
