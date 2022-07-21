# -*- coding: utf-8 -*-

from modules.fitting_module import fit_spectrum
from radis import load_spec


# Load experimental spectrum. You can prepare yours, or fetch one of them in the ground-truth folder like below.
s_experimental = load_spec("./data/LTE/ground-truth/synth-NH3-1-500-2000-cm-1-P10-t1000-v-r-mf0.01-p1-sl1nm.spec")

# Experimental conditions which will be used for spectrum modeling. Basically, these are known ground-truths.
experimental_conditions = {
    "molecule" : "NH3",         # Molecule ID
    "isotope" : "1",            # Isotope ID, can have multiple at once
    "wmin" : 1000,              # Starting wavelength/wavenumber to be cropped out from the original experimental spectrum.
    "wmax" : 1050,              # Ending wavelength/wavenumber for the cropping range.
    "wunit" : "cm-1",           # Accompanying unit of those 2 wavelengths/wavenumbers above.
    "mole_fraction" : 0.01,     # Species mole fraction, from 0 to 1.
    "pressure" : 10,            # Partial pressure of gas, in "bar" unit.
    "path_length" : 1,          # Experimental path length, in "cm" unit.
    "slit" : "1 nm",            # Experimental slit, must be a blank space separating slit amount and unit.
    "offset" : "-0.2 nm"        # Experimental offset, must be a blank space separating offset amount and unit.
}

# List of parameters to be fitted.
fit_parameters = {
    "Tgas" : 700,               # Fit parameter, accompanied by its initial value.
}

# List of bounding ranges applied for those fit parameters above.
bounding_ranges = {
    "Tgas" : [600, 2000],       # Bounding ranges for each fit parameter stated above. You can skip this step, but not recommended.
}

# Fitting pipeline setups.
fit_properties = {
    "method" : "lbfgsb",        # Preferred fitting method from the 17 confirmed methods of LMFIT stated in week 4 blog. By default, "leastsq".
    "fit_var" : "radiance",     # Spectral quantity to be extracted for fitting process, such as "radiance", "absorbance", etc.
    "normalize" : False,        # Either applying normalization on both spectra or not.
    "max_loop" : 150,           # Max number of loops allowed. By default, 100.
    "tol" : 1e-10               # Fitting tolerance, only applicable for "lbfgsb" method.
}


# Conduct the fitting process!
s_best, result, log = fit_spectrum(
    s_exp = s_experimental,
    fit_params = fit_parameters,
    bounds = bounding_ranges,
    model = experimental_conditions,
    pipeline = fit_properties
)


# Now investigate the log

print("\nResidual history: \n")
print(log["residual"])

print("\nFitted values history: \n")
for fit_val in log["fit_vals"]:
    print(fit_val)

print("\nTotal fitting time: ")
print(log["time_fitting"], end = " s\n")