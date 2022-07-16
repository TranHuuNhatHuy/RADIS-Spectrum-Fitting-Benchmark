# -*- coding: utf-8 -*-
"""

RADIS Fitting Benchmarking Module

This module is to conduct benchmarking process on spectra (LTE or Non-LTE) with
temperatures (and sometimes mole fraction) as fitting parameters.

Fitting spectra are stored in ./data/<spectrum-type>/ground-truth, while their
corresponding fitting results are stored in ./data/<spectrum-type>/result. All the
information will be in JSON format.

For each spectrum type, after acquiring most stable fitting pipeline, the model,
the algorithm and other specific settings will be stored as a model file in
./fitting-module/<spectrum-type>/model.py.

"""

from logging import warning
from math import nan
import numpy as np
import time
import astropy.units as u
import json
import os
import matplotlib.pyplot as plt
from os.path import join
from typing import Union
from pathlib import Path

from radis import calc_spectrum, plot_diff, Spectrum, SpectrumFactory, experimental_spectrum, calculated_spectrum
from lmfit import minimize, Parameters, fit_report
from lmfit.minimizer import MinimizerResult
from radis.tools.database import load_spec

# Load models
from .model_LTE import residual_LTE
from .model_nonLTE import residual_NonLTE



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
        For example:: "LTE/ground-truth/CO2_measured_spectrum_4-5um.json"

    Other Parameters
    ----------------
    verbose: bool, or ``2``
        by default, True, print details about the fitting progress.

    Returns
    -------
    conditions: dict
        Dictionary containing all ground-truth information.
    params: Parameters
        LMFIT.Parameters object that contains fit params.

    """

    # LOG THE INPUT FILE AND INITIATE PARAMETERS OBJECT FROM IT

    with open(input_file, 'r') as f:

        conditions = json.load(f)         # Load JSON as dict
        params = Parameters()             # Initiate Parameters object

        # Get fit parameters first and load them into params

        fit_params = conditions.pop("fit")
        dhb = 500              # Default half-bound for fit temperatures, in case user don't specify bounds

        for param in fit_params:

            fit_val = fit_params[param]

            if (type(fit_val) == int) or (type(fit_val) == float):     # User states an initial value
                print(f"User stated an initial fit value {param} = {fit_val}.")
                init_val = fit_val
                if param == "mole_fraction":
                    init_bound = [0, 1]
                else:
                    init_bound = [
                        (init_val - dhb) if init_val > dhb else 0, 
                        init_val + dhb
                    ]

            elif (type(fit_val) == list) or (type(fit_val) == tuple):  # User states a bounding range
                print(f"User stated an initial bounding range value {param} = {fit_val}.")
                init_bound = [fit_val[0], fit_val[1]]
                init_val = fit_val[0] + (fit_val[1] - fit_val[0]) / 2

            #elif (type(fit_val) == str):                              # User states an equation
                # print(f"User stated a relative equation for {param} = {fit_val}.")
                # Will be updated later

            else:
                print(f"Unable to recognize initial statement of {param}.")
                break

            params.add(                              # Add fit parameter into Parameters object
                param, 
                value = init_val,
                min = init_bound[0],
                max = init_bound[1]
            )

        # Return the remaining dict and the Parameters

        return conditions, params


def spectrum_refinement(s_data, conditions, verbose = True) -> Union[Spectrum, dict]:
    """Receive an experimental spectrum and further refine it according to the provided pipeline.
    Refinement process includes extracting the desired spectrum quantity, removing NaN values,
    and other additional convolutions. Finally, a refined Spectrum object is returned, along with
    the original condition.

    Parameters
    ----------
    s_data : Spectrum
        experimental spectrum loaded from the path given in input JSON file.
    conditions : dict
        a Dictionary containing all ground-truth information, and also spectrum refinements or
        convolutions (desired spectrum quantity, normalization, etc.), as well as fitting process
        (max loop allowed, terminal tolerance, etc.).

    Other parameters
    ----------------
    verbose : bool
        by default, True, print details about the fitting progress.

    Returns
    -------
    s_refined : Spectrum
        the refined spectrum.
    conditions : dict
        the input conditions that might have been added fit_var (in case user didn't).
    
    """

    # Extract spectrum quantity

    pipeline = conditions["pipeline"]

    if "fit_var" in pipeline:
        fit_var = pipeline["fit_var"]   # Acquire the stated quantity
    else:
        fit_var = s_data.get_vars()[0]                  # If not stated, take first quantity
        conditions["pipeline"]["fit_var"] = fit_var     # And add to the dict

    s_data_wav, s_data_val = s_data.get(fit_var)

    if verbose:
        print(f"Acquired spectral quantity \'{fit_var}\' from the spectrum.")

    # Remove NaN values. A wise man once said, "Nan is good but only in India"

    s_data_mtr = np.vstack((s_data_wav, s_data_val))
    s_data_mtr = s_data_mtr[:, ~np.isnan(s_data_mtr).any(axis = 0)]     # Purge NaN pairs

    if verbose:
        print(f"NaN values successfully purged.")           

    # Recreate the data spectrum with the spectral quantity

    s_refined = Spectrum(
        {
            fit_var : (s_data_mtr[0], s_data_mtr[1])
        },
        wunit = conditions["wunit"],
        units = {
            fit_var : s_data.units[fit_var]
        }
    ).take(fit_var).sort()

    # Further refinement

    if "normalize" in pipeline:
        if pipeline["normalize"]:
            s_refined = s_refined.normalize()
            if verbose:
                print("Normalization applied.")


    return s_refined, conditions


def fit_spectrum(input_file, verbose = True) -> Union[Spectrum, MinimizerResult, dict]:
    """Fit an experimental spectrum (from here referred as "data spectrum") with a modeled one,
    then derive the fit results. Data spectrum is loaded from the path stated in the JSON file,
    while model spectrum is generated based on the conditions stated in the JSON file, too.

    Parameters
    ----------
    input_file : str
        path to the JSON input file from the directory containing this module, will be changed
        when implementing to RADIS codebase. For now it will be something like this::
        "../data/LTE/ground-truth/CO2_measured_spectrum_4-5um.json"
    
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
    log: dict
        a Dictionary storing runtime log of the fitting process that are 
        not quite covered by the Minimizer, including: residual and fit 
        values after each fitting loop, and total time elapsed.

    """

    begin = time.time()             # Start the fitting time counter

    
    # ACQUIRE AND REFINE EXPERIMENTAL SPECTRUM s_data

    # Load data from JSON file, then create a Parameters object
    conditions, params = get_JSON(input_file)

    # Get s_data spectrum from the path stated in acquired JSON data, assuming the JSON
    # and the .spec files are in the same directory (it SHOULD be!)

    fileName = conditions.pop("fileName")
    if (fileName[-5:] != ".spec"):             # fileName doesn't have ".spec" extension
        print("Warning: fileName must include \".spec\" extension!")

    input_dir = "/".join(input_file.split("/")[0 : -1]) + "/"
    spec_path = input_dir + fileName

    # Load and crop the experimental spectrum
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
        print(f"\nSuccessfully retrieved the experimental data in {time_exp_load}s.\n")

    # Further refine the data spectrum before calculating diff

    s_data, conditions = spectrum_refinement(s_data, conditions)
    pipeline = conditions["pipeline"]
    modeling = conditions["modeling"]
    fit_var = pipeline["fit_var"]

    if verbose:
        end_exp_refine = time.time()
        time_exp_refine = end_exp_refine - end_exp_load
        s_data.plot(show = True)
        print(f"Successfully refined the experimental data in {time_exp_refine}s.")
    

    # PRE-MINIMIZATION SETUP
    
    # Create a Spectrum Factory object for modeling spectra

    kwargs = {}
    for cond in conditions:
        if (cond != "pipeline") and (cond != "modeling"):
            kwargs[cond] = conditions[cond]
    
    sf = SpectrumFactory(
        **kwargs,
        verbose = False,
        warnings = "ignore"
    )
    sf.load_databank("HITRAN-" + conditions["molecule"])

    # Decide the type of model - LTE or Non-LTE
    LTE = True                                         # LTE == True means it's LTE
    if ("Trot" in params) or ("Tvib" in params):
        LTE = False                                    # LTE == False means it's non-LTE
    
    # Determine fitting method, either stated by user or "lbfgsb" by default
    if "method" in pipeline:
        method = pipeline["method"]     # User-defined
    else:
        method = "leastsq"              # By default

    # Determine additional fitting conditions for the minimizer

    fit_kws = {
        'max_nfev': pipeline["max_loop"] if "max_loop" in pipeline else 200,
    }

    if "tol" in pipeline:
        if pipeline["method"] == "lbfgsb":
            fit_kws["tol"] = pipeline["tol"]
        else:
            print("\"tol\" parameter spotted but \"method\" is not \"lbfgsb\"!")

    # Prepare fitting log
    log = {
        "residual": [],
        "fit_vals": [],
        "time_fitting": 0
    }


    # COMMENCE THE FITTING PROCESS

    # For LTE spectra
    if LTE:
        print("\nCommence fitting process for LTE spectrum!")
        result = minimize(
            residual_LTE, 
            params, 
            method = method, 
            args = [conditions, s_data, sf, log], 
            **fit_kws
        )

    # For non-LTE spectra
    if not(LTE):
        print("\nCommence fitting process for non-LTE spectrum!")
        result = minimize(
            residual_NonLTE, 
            params, 
            method = method, 
            args = [conditions, s_data, sf, log], 
            **fit_kws
        )

    if verbose:
        end_fitting = time.time()
        time_fitting = end_fitting - end_exp_refine
        print(f"\nSuccesfully finished the fitting process in {time_fitting}s.")
        log["time_fitting"] = time_fitting


    # POST-MINIMIZATION REPORT

    print(fit_report(result))       # Report the fitting result

    if verbose:

        # REGENERATE THE BEST-FIT SPECTRUM, AS A RESULT FROM THE MINIMIZER

        # Load initial values of fit parameters
        for name, param in result.params.items():
            fit_show = {name : float(param.value)}
        fit_show["name"] = "best_fit"

        # Generate best fitted spectrum result
        if LTE:
            s_result = sf.eq_spectrum(**fit_show)
            result_subfdr = "LTE"
        else:
            s_result = sf.non_eq_spectrum(**fit_show)
            result_subfdr = "nonLTE"

        # Apply slit
        if "slit" in modeling:
            slit, slit_unit = modeling["slit"].split()
            s_result = s_result.apply_slit(float(slit), slit_unit)

        # Take spectral quantity
        s_result = s_result.take(fit_var)

        # Apply normalization
        if "normalize" in pipeline:
            if pipeline["normalize"]:
                s_result = s_result.normalize()


        # PLOT THE DIFFERENCE BETWEEN THE TWO AND SAVE THE FIGURE 
        specName = fileName.replace(".spec", "")
        fig_loc = input_dir + specName + ".png"     # Save fig at same JSON/spec directory
        plot_diff(
            s_data, 
            s_result, 
            fit_var, 
            method=['diff', 'ratio'], 
            show = True,
            save = fig_loc
        )

        if verbose:
            print(f"\nComparison figure successfully saved at {fig_loc}")

    return s_result, result, log



# if __name__ == "__main__":

#     spec_list = [
#         "CO2_measured_spectrum_4-5um",                                      # 0
#         "synth-CO-1-1800-2300-cm-1-P3-t1500-v-r-mf0.1-p1-sl1nm",            # 1
#         "synth-CO2-1-500-1100-cm-1-P2-t900-v-r-mf0.5-p1-sl1nm",             # 2
#         "synth-CO2-1-500-3000-cm-1-P93-t740-v-r-mf0.96-p1-sl1nm",           # 3
#         "synth-CO2-1-3300-3700-cm-1-P0.005-t3000-v-r-mf0.01-p1-sl1.4nm",    # 4
#         "synth-H2O-1-1000-2500-cm-1-P0.5-t1500-v-r-mf0.5-p1-sl1nm",         # 5
#         "synth-NH3-1-500-2000-cm-1-P10-t1000-v-r-mf0.01-p1-sl1nm",          # 6
#         "synth-O2-1-7500-8000-cm-1-P1.01325-t298.15-v-r-mf0.21-p1-sl1nm",   # 7
#     ]

#     for i in range(len(spec_list)):
#         input_path = f"../data/LTE/ground-truth/{spec_list[i]}.json"
#         _, result, log, pipeline = fit_spectrum(input_path)
#         json_data = {
#             "fileName": f"{spec_list[i]}.spec",
#             "pipeline": {
#                 "method": pipeline["method"],
# 	            "fit_var": "radiance",
# 	            "normalize": pipeline["normalize"],
# 	            "max_loop": 100
#             },
#             "result": {
#                 "last_residual": log["residual"][-1],
#                 "loops": result.nfev,
#                 "time": log["time_fitting"]
#             }
#         }
        
#         with open(f"../data/LTE/result/{spec_list[i]}/best_fit/pipeline.json", 'w') as f:
#             json.dump(json_data, f, indent = 2)
#             print("JSON file successfully created.")

#         with open(f"../data/LTE/result/{spec_list[i]}/best_fit/log_residuals.txt", 'w') as f:
#             for resi in log["residual"]:
#                 f.write(f"{resi}\n")
#             f.close()