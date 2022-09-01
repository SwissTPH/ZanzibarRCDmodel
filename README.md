# ZanzibarRCD
Malaria model for studying the impact of reactive case detection and human mobility in Zanzibar

Copyright (C) 2021 Swiss Tropical and Public Health Institute

This malaria model is free software; you can redistribute it and/or modify it under the terms of version 2 of the GNU General Public License as published by the Free Software Foundation.

## Data
This folder contains two files:
* parameters.csv: This contains the fixed parameter values derived from the RADZEC study and other sources.
* param_100.csv: This contains 100 draws from various parameter distributions based on the data from the RADZEC study. This was used for estimating the uncertainty in the final malaria prevalence reached after 40 years, based on the uncertainty in the input parameters and the stochasticity of the model.

## Deterministic Model
The deterministic model does not contain RCD. It can be used to estimate the transmission parameter, beta, that is compatible with the data available.

## Stochastic Model
This folder contains the files used to simulate interventions following calibration.
* RCD_stochastic_3day.py: This includes changing the proportion of index cases followed up from 35% to a range of other values. Otherwise, all interventions remain at the baseline value
* RCD_neighbours_3day.py: Adding screening and treatment of 4 neighbouring households
* RCD_double_3day.py: Doubling the treatment seeking rate
* RCD_rfmda_3day.py: Switching from RCD to rfMDA
* RCD_rfmda_neighbours_3day.py: Switching from RCD to rfMDA and including 4 neighbouring households
* RCD_imp_treat_30_3day.py: Treating 50% of imported cases coming onto both Pemba and Unguja
