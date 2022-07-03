# -*- coding: utf-8 -*-
"""

RADIS Fitting Benchmarking Module

This module is to conduct benchmarking process on spectra (LET or Non-LTE) with
temperatures as fitting parameters.

Fitting spectra are stored in ./data/<spectrum-type>/ground-truth, while their
corresponding fitting results are stored in ./data/<spectrum-type>/result. All the
information will be in JSON format.

For each spectrum type, after acquiring most stable fitting pipeline, the model,
the algorithm and other specific settings will be stored as a model file in
./fitting-module/<spectrum-type>/model.py.

"""

from math import nan
import numpy as np
import time
import astropy.units as u
import json
import os
from os.path import join
from typing import Union

from radis import calc_spectrum, plot_diff, Spectrum, SpectrumFactory, experimental_spectrum
from lmfit import minimize, Parameters, fit_report
from lmfit.minimizer import MinimizerResult
from radis.tools.database import load_spec

# Load models
from model_LTE import residual_LTE
from model_nonLTE import NonLTEModel



def get_JSON(
    input_file,
    verbose = True,
) -> Union[dict, dict, Parameters]:
    """Read the JSON file containing ground-truth information and fit parameters.
    Return a dict parameter containing fixed conditions for spectrum calculation,
    and an LMFIT.Parameters object that contains fit parameters for later use.

    Parameters
    ----------
    input_file : str
        path to the JSON input file from ./data/, will be changed when implementing
        to RADIS codebase. For now the format is <spectrum-type>/ground-truth/<name>.json.
        For example:: "large/ground-truth/CO2_measured_spectrum_4-5um.json"

    Other Parameters
    ----------------
    verbose: bool, or ``2``
        use ``2`` for high verbose level.

    Returns
    -------
    conditions: dict
        Dictionary containing all ground-truth information.
    params: Parameters
        LMFIT.Parameters object that contains fit params.

    """

    # Log the input file
    with open("/".join(["../data", input_file]), 'r') as f:

        conditions = json.load(f)         # Load JSON as dict
        params = Parameters()       # Initiate Parameters object

        # Get fit parameters first and load them into params

        fit_params = conditions.pop("fit")
        dhb = 500              # Default half-bound for fit temperatures, in case user don't specify bounds


        if "Tgas" in fit_params:
            Tgas = fit_params.pop("Tgas")
            params.add("Tgas", value = Tgas)

            bound_Tgas = [(Tgas - dhb) if Tgas > dhb else 0, Tgas + dhb] # Default fitting bounds
            if "bound_Tgas" in fit_params:
                bound_Tgas = fit_params.pop("bound_Tgas") # Update user-defined bounds
            params["Tgas"].set(min = bound_Tgas[0], max = bound_Tgas[1])    # Update params attributes


        if "Tvib" in fit_params:
            Tvib = fit_params.pop("Tvib")
            params.add("Tvib", value = Tvib)

            bound_Tvib = [(Tvib - dhb) if Tvib > dhb else 0, Tvib + dhb] # Default bounds
            if "bound_Tvib" in fit_params:
                bound_Tvib = fit_params.pop("bound_Tvib") # Update user-defined bounds
            params["Tvib"].set(min = bound_Tvib[0], max = bound_Tvib[1])    # Update params attributes


        if "Trot" in fit_params:
            Trot = fit_params.pop("Trot")
            params.add("Trot", value = Trot)

            bound_Trot = [(Trot - dhb) if Trot > dhb else 0, Trot + dhb] # Default bounds
            if "bound_Trot" in fit_params:
                bound_Trot = fit_params.pop("bound_Trot") # Update user-defined bounds
            params["Trot"].set(min = bound_Trot[0], max = bound_Trot[1])    # Update params attributes


        if "mole_fraction" in fit_params:
            mole_fraction = fit_params.pop("mole_fraction")
            params.add("mole_fraction", value = mole_fraction)

            bound_molefrac = [0, 1] # Default bounds
            if "bound_mole_fraction" in fit_params:
                bound_molefrac = fit_params.pop("bound_mole_fraction") # Update user-defined bounds
            params["mole_fraction"].set(min = bound_molefrac[0], max = bound_molefrac[1])    # Update params attributes

        
        # Return the remaining dict and the Parameters

        return conditions, params


def fit_spectrum(input_file, verbose = True) -> Union[Spectrum, MinimizerResult]:
    """Fit an experimental spectrum (from here referred as "data spectrum") with a modeled one,
    then derive the fit results. Data spectrum is loaded from the path stated in the JSON file,
    while model spectrum is generated based on the conditions stated in the JSON file, too.

    Parameters
    ----------
    input_file : str
        path to the JSON input file from ./data/, will be changed when implementing
        to RADIS codebase. For now the format is <spectrum-type>/ground-truth/<name>.json.
        For example:: "large/ground-truth/CO2_measured_spectrum_4-5um.json"
    
    Other parameters
    ----------
    verbose : bool
        by default, True, print details about the fitting progress.

    Returns
    -------
    s_best: Spectrum
        visualization of the best fit results obtained, as a spectrum.
    best_fit: MinimizerResult
        best fit results, output of LMFIT MinimizeResult.

    """

    begin = time.time()             # Start the fitting time counter

    
    # Load data from JSON file, then create a Parameters object
    conditions, params = get_JSON(input_file)

    # Get s_data spectrum from the path stated in acquired JSON data
    spec_path = "/".join([
        "..",
        "data/large/spectrum",
        conditions["fileName"]
    ])

    s_data = (
        load_spec(spec_path)
        .crop(
            conditions["wmin"], 
            conditions["wmax"], 
            conditions["wunit"]
        )
    )

    if verbose:
        end_exp_load = time.time()
        time_exp_load = end_exp_load - begin
        print(f"\nSuccessfully retrieved the experimental data in {time_exp_load}s.")


    # Further refine the data spectrum before calculating diff
    fit_var = s_data.get_vars()[0]
    s_data_wav, s_data_val = s_data.get(fit_var)

    # A wise man once said, "Nan is good but only in India"
    s_data_mtr = np.vstack((s_data_wav, s_data_val))

    # Purge wav-val pairs containing NaN values.
    s_data_mtr = s_data_mtr[:, ~np.isnan(s_data_mtr).any(axis = 0)]            

    s_data = experimental_spectrum(          # Recreate the data spectrum
        s_data_mtr[0],
        s_data_mtr[1],
        wunit = conditions["wunit"],
        Iunit = s_data.units[fit_var]
    )

    # Further refinement
    s_data = (
        s_data
        .take(fit_var)
        .normalize()
        .sort()
        .offset(-0.2, conditions["wunit"])
    )

    if verbose:
        end_exp_refine = time.time()
        time_exp_refine = end_exp_refine - end_exp_load
        s_data.plot(show = True)
        print(f"\nSuccessfully refined the experimental data in {time_exp_refine}s.")


    # Decide the type of spectrum among 4 types.
    # For now it's just for LTE spectra. Non-LTE will be updated soon.

    LTE = True                                         # LTE == True means it's LTE
    if ("Trot" in params) or ("Tvib" in params):
        LTE = False                                    # LTE == False means it's non-LTE


    # Commence fitting process
    
    method = "leastsq"

    # For LTE spectra
    if LTE:
        print("\nCommence fitting process for LTE spectrum!")
        result = minimize(residual_LTE, params, method = method, args = [conditions, s_data])

    # For non-LTE spectra
    #if not(LTE):
    #    result = minimize(residual_NonLTE, params, method = "leastsq", args = [data, s_data])

    
    if verbose:
        end_fitting = time.time()
        time_fitting = end_fitting - end_exp_refine
        print(f"\nSuccesfully finished the fitting process in {time_fitting}s.")


    # Display result
    print(fit_report(result))

    if verbose:
        # Load initial values of fit parameters
        for name, param in result.params.items():
            fit_show = {name : float(param.value)}
        
        # Load conditions (fixed parameters)

        slit_info = conditions.pop("slit")      # Because calc_spectrum() does not support slit parameters this way
        slit, slit_unit = slit_info.split()     # But it should be this way instead.

        conditions.pop("fileName")              # Because calc_spectrum() does not have "fileName" parameter
        conditions["name"] = "best_fit"         # But it only has "name".
        
        fit_show = {**fit_show, **conditions}

        # Generate best fitted spectrum result

        s_result = calc_spectrum(
            **fit_show,
            cutoff = 1e-25,
            wstep = 0.001,
            truncation = 1,
            verbose = False,
            warnings = "ignore"
        )
        s_result = (
            s_result
            .apply_slit(float(slit), slit_unit)     # Simulate slit
            .take(fit_var)
            .normalize()                            # Normalize
            .resample(                              # Downgrade to data spectrum's resolution
                s_data, 
                energy_threshold = 2e-2
            )
        )

        # Plot the difference between the two
        plot_diff(
            s_data, 
            s_result, 
            fit_var, 
            method=['diff', 'ratio'], 
            show = True
        )


    return s_result, result



if __name__ == "__main__":

    input_path = "large/ground-truth/CO2_measured_spectrum_4-5um.json"

    fit_spectrum(input_path)