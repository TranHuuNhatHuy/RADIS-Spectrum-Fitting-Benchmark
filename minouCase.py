from modules.fitting_module import fit_spectrum
from radis import load_spec, Spectrum
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
# Load experimental spectrum. You can prepare yours, or fetch one of them in the ground-truth folder like below.
# s_experimental = load_spec("./data/LTE/ground-truth/synth-NH3-1-500-2000-cm-1-P10-t1000-v-r-mf0.01-p1-sl1nm.spec")
data =  scipy.io.loadmat('/Users/tranhuunhathuy/Documents/GitHub/RADIS-Spectrum-Fitting-Benchmark/1857_VoigtCO_Minesi.mat', simplify_cells=True)['CO_resu_Voigt']
index = 20
### Fit performed with another code
plt.figure('reference fit')
plt.title('T = {} K'.format(data['Boltzmann']['T_K'][index]))
plt.plot(data['nu'], data['A_exp'][:,index], 'k-', label='Data')
plt.plot(data['nu'], data['A_fit'][:,index], 'r-', label='Fit')

###
# s_experimental = experimental_spectrum(data['nu'], data['A_exp'][:,index], Iunit='absorbance');
s_experimental = Spectrum.from_array(data['nu'], data['A_exp'][:,index], 'absorbance',
                           wunit='cm-1', unit='') # adimensioned    # CHANGED FROM nm TO cm-1
s_experimental.plot()

# Experimental conditions which will be used for spectrum modeling. Basically, these are known ground-truths.
experimental_conditions = {
    "molecule" : "CO",         # Molecule ID
    "isotope" : "1",            # Isotope ID, can have multiple at once
    "wmin" : 2010.6,              # Starting wavelength/wavenumber to be cropped out from the original experimental spectrum.
    "wmax" : 2011.6,              # Ending wavelength/wavenumber for the cropping range.
    "wunit" : "cm-1",           # Accompanying unit of those 2 wavelengths/wavenumbers above.
    #"mole_fraction" : 0.05638,     # Species mole fraction, from 0 to 1.
    "pressure" : 1,            # Partial pressure of gas, in "bar" unit.
    "path_length" : 10,          # Experimental path length, in "cm" unit.
    # "slit" : "1 nm",            # Experimental slit, must be a blank space separating slit amount and unit.
    #"offset" : "-0.01783 cm-1",        # Experimental offset, must be a blank space separating offset amount and unit.
    "wstep" : 0.001,
    "databank" : "hitemp"         # Databank used for calculation. Must be stated.
}

# List of parameters to be fitted.
fit_parameters = {
    "Tgas" : 7170,               # Fit parameter, accompanied by its initial value.
    "mole_fraction" : 0.07,     # Species mole fraction, from 0 to 1.
    "offset" : "0 cm-1"        # Experimental offset, must be a blank space separating offset amount and unit.
}

# List of bounding ranges applied for those fit parameters above.
bounding_ranges = {
    "Tgas" : [2000, 9000],       # Bounding ranges for each fit parameter stated above. You can skip this step, but not recommended.
    "mole_fraction" : [0, 1],     # Species mole fraction, from 0 to 1.
    "offset" : [-0.1, 0.1]        # Experimental offset, must be a blank space separating offset amount and unit
}

# Fitting pipeline setups.
fit_properties = {
    "method" : "lbfgsb",        # Preferred fitting method from the 17 confirmed methods of LMFIT stated in week 4 blog. By default, "leastsq".
    "fit_var" : "absorbance",     # Spectral quantity to be extracted for fitting process, such as "radiance", "absorbance", etc.
    "normalize" : False,        # Either applying normalization on both spectra or not.
    "max_loop" : 300,           # Max number of loops allowed. By default, 100.
    "tol" : 1e-20               # Fitting tolerance, only applicable for "lbfgsb" method.
}




# Conduct the fitting process!
s_best, result, log = fit_spectrum(
    s_exp = s_experimental,
    fit_params = fit_parameters,
    bounds = bounding_ranges,
    model = experimental_conditions,
    pipeline = fit_properties,
)


# Now investigate the log

print("\nResidual history: \n")
print(log["residual"])

print("\nFitted values history: \n")
for fit_val in log["fit_vals"]:
    print(fit_val)

print("\nTotal fitting time: ")
print(log["time_fitting"], end = " s\n")