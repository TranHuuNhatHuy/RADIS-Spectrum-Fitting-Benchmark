# -*- coding: utf-8 -*-

from modules.fitting_module import fit_spectrum
import matplotlib.pyplot as plt

JSON_path = "./test_dir/test_JSON_file.json"
s_best, result, log = fit_spectrum(JSON_path)

# Now investigate the log

print("\nResidual history: \n")
print(log["residual"])

print("\nFitted values history: \n")
for fit_val in log["fit_vals"]:
    print(fit_val)

print("\nTotal fitting time: ")
print(log["time_fitting"], end = " s\n")