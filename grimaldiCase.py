from modules.fitting_module import fit_spectrum
from radis import load_spec
import numpy as np
import matplotlib.pyplot as plt

# The following script is for Mr. Correnti Grimaldi's cases, used for his own slit settings. This is not native to current fitting modules.
# ----------------- Begin of Mr. Grimaldi's script ----------------- #
def slit_dispersion(w):
    phi = -6.33
    f = 750
    gr = 300
    m = 1
    phi *= - 2*np.pi/360
    d = 1e-3/gr
    disp = w/(2*f)*(-np.tan(phi)+np.sqrt((2*d/m/(w*1e-9)*np.cos(phi))**2-1))
    return disp  # nm/mm
slit = 1500  # µm
pitch = 20   # µm
top_slit_um = slit - pitch   # µm
base_slit_um = slit + pitch  # µm
center_slit = 5090
dispersion = slit_dispersion(center_slit)
top_slit_nm = top_slit_um*1e-3*dispersion
base_slit_nm = base_slit_um*1e-3*dispersion*1.33
# ----------------- End of Mr. Grimaldi's script ----------------- #

# Load experimental spectrum.
specName = [
    "0_100cm Down Sampled - 10cm_10pctCO2_1-wc-gw450-gr300-sl1500-acc5000-.spec",
    "0_100cm Down Sampled - 20cm_10pctCO2_1-wc-gw450-gr300-sl1500-acc5000-.spec",
    "0_200cm Down Sampled - 35cm_10pctCO2_1-wc-gw450-gr300-sl1500-acc5000-.spec",
    "0_300cm Down Sampled - 10cm_10pctCO2_1-wc-gw450-gr300-sl1500-acc5000-.spec",
]
# Change the number in order of spectrum name listed above
s_experimental = load_spec(f"/Users/tranhuunhathuy/Documents/GitHub/RADIS-Spectrum-Fitting-Benchmark/CorrentiGrimaldi/{specName[0]}").offset(-0.6, "nm")

# Experimental conditions which will be used for spectrum modeling. Basically, these are known ground-truths.
experimental_conditions = {
    "molecule" : "CO",         # Molecule ID
    "isotope" : "1,2,3",            # Isotope ID, can have multiple at once
    "wmin" : 2270,              # Starting wavelength/wavenumber to be cropped out from the original experimental spectrum.
    "wmax" : 2700,              # Ending wavelength/wavenumber for the cropping range.
    "wunit" : "nm",           # Accompanying unit of those 2 wavelengths/wavenumbers above.
    #"mole_fraction" : 0.05638,     # Species mole fraction, from 0 to 1.
    "pressure" : 1.01325,            # Partial pressure of gas, in "bar" unit.
    "path_length" : 1/195,          # Experimental path length, in "cm" unit.
    "slit" : {          # Experimental slit. In simple form: "[value] [unit]", i.e. "-0.2 nm". In complex form: a dict with parameters of apply_slit()
        "slit_function" : (top_slit_nm, base_slit_nm),
        "unit" : "nm",
        "shape" : 'trapezoidal',
        "center_wavespace" : center_slit,
        "slit_dispersion" : slit_dispersion,
        "inplace" : False,
    },
    #"offset" : "-0.6 nm",        # Experimental offset, must be a blank space separating offset amount and unit.
    "wstep" : 0.001,
    "cutoff" : 0,
    "databank" : "hitemp"         # Databank used for calculation. Must be stated.
}

# List of parameters to be fitted, accompanied by its initial value
fit_parameters = {
    "Tvib" : 5000,               # Fit parameter, accompanied by its initial value.
    "Trot" : 5000,
    "mole_fraction" : 0.05, 
}

# List of bounding ranges applied for those fit parameters above.
bounding_ranges = {
    "Tvib" : [2000, 7000],       # Bounding ranges for each fit parameter stated above. You can skip this step, but not recommended.
    "Trot" : [2000, 7000],
    "mole_fraction" : [0, 0.1],     # Species mole fraction, from 0 to 1.
}

# Fitting pipeline setups.
fit_properties = {
    "method" : "lbfgsb",        # Preferred fitting method from the 17 confirmed methods of LMFIT stated in week 4 blog. By default, "leastsq".
    "fit_var" : "radiance",     # Spectral quantity to be extracted for fitting process, such as "radiance", "absorbance", etc.
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