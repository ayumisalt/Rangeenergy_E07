# read_data.py

def read_data(filename, density_ave):
    """
    Reads a text file and stores each column of data in corresponding lists,
    then returns a dictionary containing all these lists.

    Parameters
    ----------
    filename : str
        The path to the input text file.
    density_ave : float
        The default density value to use when a line contains 'XXX' in place
        of actual density data.

    Returns
    -------
    data_dict : dict
        A dictionary containing lists of parsed data. The keys are:
        {
            "ID": list[str],
            "Range_1": list[float],
            "Range_error_1": list[float],
            "Theta_1": list[float],
            "Theta_error_1": list[float],
            "Phi_1": list[float],
            "Phi_error_1": list[float],
            "Range_2": list[float],
            "Range_error_2": list[float],
            "Theta_2": list[float],
            "Theta_error_2": list[float],
            "Phi_2": list[float],
            "Phi_error_2": list[float],
            "Range_3": list[float],
            "Range_error_3": list[float],
            "Theta_3": list[float],
            "Theta_error_3": list[float],
            "Phi_3": list[float],
            "Phi_error_3": list[float],
            "label": list[str],
            "density": list[float],
            "density_error": list[float],
        }
    """

    # Prepare lists to hold each column of data
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

    # Open the file and read it line by line
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines or lines starting with '#'
            if not line or line.startswith("#"):
                continue

            # Split the line by whitespace into a list
            item_list = line.split()

            # Append each column value to its corresponding list
            ID.append(item_list[0])
            Range_1.append(float(item_list[1]))
            Range_error_1.append(float(item_list[2]))
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

            # For density and density_error, replace 'XXX' with density_ave or 0.0
            if item_list[20] == "XXX":
                density.append(density_ave)
            else:
                density.append(float(item_list[20]))

            if item_list[21] == "XXX":
                density_error.append(0.0)
            else:
                density_error.append(float(item_list[21]))

    # Create a dictionary to organize all the lists
    data_dict = {
        "ID": ID,
        "Range_1": Range_1,
        "Range_error_1": Range_error_1,
        "Theta_1": Theta_1,
        "Theta_error_1": Theta_error_1,
        "Phi_1": Phi_1,
        "Phi_error_1": Phi_error_1,
        "Range_2": Range_2,
        "Range_error_2": Range_error_2,
        "Theta_2": Theta_2,
        "Theta_error_2": Theta_error_2,
        "Phi_2": Phi_2,
        "Phi_error_2": Phi_error_2,
        "Range_3": Range_3,
        "Range_error_3": Range_error_3,
        "Theta_3": Theta_3,
        "Theta_error_3": Theta_error_3,
        "Phi_3": Phi_3,
        "Phi_error_3": Phi_error_3,
        "label": label,
        "density": density,
        "density_error": density_error,
    }

    return data_dict
