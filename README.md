# Range-Energy relation for E07 nuclear emulsion

This repository contains the source code for the Range-Energy calclation for the E07 nuclear emulsion.
It includes experimental data and code to calculate the binding energy of $\Lambda\$ hypernuclei.
Stopping-power calculations based on [ATIMA](https://web-docs.gsi.de/~weick/atima/) and [pycatima](https://github.com/hrosiak/pycatima).
[Link for the paper](https://arxiv.org/).

## Contents

- main: Calclate binding energy of H3 $\Lambda$ and H4 $\Lambda\$
  - rangeenergy
    - ATIMA_RangeEnergy
  - kinema
  - kinema_impl
  - ims
  - nuclide
  - read_data
  - cut_events
  - calc_invariant_mass
  - plot_results
- mom_Helium_vs_pi: Compare deduced momentum from Helium and pion
- BL_weighted: Calclate world average from experimental data
- rangeenergy_gui: GUI tool
- data
  - alldata.txt: Measurement data
  - alldata_alpha.txt: Calibrated densities with $\alpha$ particle
  - _sumamry: Summary of experimental data

## Dependencies

Required:

- python3
- pycatima
- numpy
- scipy
- matplotlib
- uncertainties
- colorama
- sympy
- pyqt5

You can install dependences using pip:

```sh
pip install -r pip_requirements.txt
```

## Usage

```sh
python main.py data/alldata.txt <density>
# We used density = 3.379 in the analysis.
# (Figure. 3)
```

```sh
python mom_Helium_vs_pi.py data/alldata_alpha.txt 1
# (Extended Data Figure. 3)
```

```sh
python BL_weighted.py data/H3L_BL_summary.txt data/H4L_BL_summary.txt
# (Figure. 4)
```

To use Range-Energy relation from GUI:

```sh
python rangeenergy_gui.py
```

- Mishina: Calclation based on Traditional Range-Energy Barkas et.al
- ATIMA: Updated Range-Energy relation for E07 nuclear emulsion with ATIMA
