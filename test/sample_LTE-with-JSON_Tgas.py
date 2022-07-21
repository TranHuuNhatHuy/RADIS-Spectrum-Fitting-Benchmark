# -*- coding: utf-8 -*-

from modules.fitting_module import fit_spectrum
from radis import load_spec


# Get JSON file path (note that the experimental spectrum file MUST BE IN THE SAME FOLDER containing JSON file)
JSON_path = "./test_dir/test_JSON_file.json"


# Conduct the fitting process!
s_best, result, log = fit_spectrum(input_file = JSON_path)


# Now investigate the log

print("\nResidual history: \n")
print(log["residual"])

print("\nFitted values history: \n")
for fit_val in log["fit_vals"]:
    print(fit_val)

print("\nTotal fitting time: ")
print(log["time_fitting"], end = " s\n")