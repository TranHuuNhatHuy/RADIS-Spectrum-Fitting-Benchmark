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



def get_conditions(
    input_conditions,
    verbose = True,
) -> Union[dict, dict, Parameters]:
    """Get the ground-truth information by reading from input - either a JSON file
    or a dict parameter. Return a dict parameter containing fixed conditions for
    spectrum calculation, and an LMFIT.Parameters object containing fit parameters.

    Parameters
    ----------
    input_conditions : str or dict
        If str, the code assumes this is path to the JSON input file from the script
        calling it, for example:: "./test_dir/CO2_measured_spectrum_4-5um.json", and
        then parses the JSON data as a dict.
        If dict, the code just simply parses the dict, yeah, that simple.

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

    # LOG THE INPUT CONDITIONS (JSON OR DICT)

    if (type(input_conditions) == str):         # Must be a JSON path
        
        with open(input_conditions, 'r') as f:  # Load JSON as dict
            conditions = json.load(f)
            f.close()

    elif (type(input_conditions) == dict):      # Must be just a dick

        conditions = input_conditions

    else:

        print(f"Input unrecognizable. Got type {type(input_conditions)}, expecting str or dict.")

    params = Parameters()                       # Initiate Parameters object

    # READ "fit" KEY AND LOAD IT INTO NEW PARAMETERS OBJECT

    fit_params = conditions.pop("fit")
    dhb = 500       # Default half-bound for fit temperatures, for cases user don't specify bounds

    # Deal with "Tvib", especially if it has multiple temperatures

    if "Tvib" in fit_params:

        Tvib = fit_params["Tvib"]

        if (type(Tvib) == list) or (type(Tvib) == tuple):   # Multiple temps detected

            # Split Tvib[] into Tvib0, Tvib1, Tvib2,... for later being put into Parameters
            for i in range(len(Tvib)):
                fit_params[f"Tvib{i}"] = Tvib[i]

            # Destroy the original "Tvib" key
            fit_params.pop("Tvib")

    # Put every fit parameter into Parameters, with the default bounding range

    for param in fit_params:

        init_val = fit_params[param]

        if (param == "mole_fraction"):
            init_bound = [0, 1]
        else:
            init_bound = [
                (init_val - dhb) if init_val > dhb else 0, 
                init_val + dhb
            ]

        params.add(
            param,
            value = init_val,
            min = init_bound[0],
            max = init_bound[1]
        )

    # READ "bounds" KEY AND RE-ASSIGN DEFAULT BOUNDING RANGES WITH USER-DEFINED ONES

    if ("bounds" in conditions):

        bounds = conditions.pop("bounds")

        # Deal with "Tvib", especially if it has multiple temperature-bounding ranges

        if "Tvib" in bounds:

            bound_Tvib = bounds["Tvib"]

            if (len(bound_Tvib) > 1):           # Not sure if it's multiple ranges, or just 1 range with 2 ends
                if (type(bound_Tvib[0]) in [list, tuple]):      # Now we sure it's multiple ranges

                    # Split Tvib[] into Tvib0, Tvib1, Tvib2,... for later being put into Parameters
                    for i in range(len(bound_Tvib)):
                        bounds[f"Tvib{i}"] = bound_Tvib[i]

                    # Destroy the original key
                    bounds.pop("Tvib")

        # Set user-defined bounding ranges
        
        for bound_param in bounds:

            bound_val = bounds[bound_param]

            params[bound_param].set(
                min = bound_val[0],
                max = bound_val[1]
            )


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

    # Extract spectrum 
    
    pipeline = conditions["pipeline"]

    if "fit_var" in pipeline:
        fit_var = pipeline["fit_var"]                   # Acquire the stated quantity
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
        wunit = conditions["model"]["wunit"],
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


def fit_spectrum( 
    s_exp = None,
    fit_params = None,
    model = None,
    pipeline = None,
    bounds = None,
    input_file = None,
    verbose = True) -> Union[Spectrum, MinimizerResult, dict]:
    """Fit an experimental spectrum (from here referred as "data spectrum") with a modeled one,
    then derive the fit results. Data spectrum is loaded from the path stated in the JSON file,
    while model spectrum is generated based on the conditions stated in the JSON file, too.

    Parameters
    ----------
    s_exp : Spectrum
        a Spectrum object, containing the experimental spectrum.
    fit_params : dict
        a dict object, containing fit parameters and their initial values.
    bounds : dict
        a dict object, optional, containing bounding ranges of those fit parameters. If not stated,
        all fit parameters will receive default bounding ranges.
    model : dict
        a dict object, containing experimental conditions and some additional convolutions for
        modeling the spectrum.
    pipeline : dict
        a dict object, containing some preference properties of fitting process.
    input_file : str
        path to the JSON input file from the script calling it, by default, None. If this parameter
        is defined, it means user uses JSON file for inputing everything, and so will discard all
        other input parameters. For example:: "./test_dir/CO2_measured_spectrum_4-5um.json"
    
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

    if (input_file != None):        # The user uses JSON file

        # Load data from JSON file, then create a Parameters object
        conditions, params = get_conditions(input_file)

        # Get s_data spectrum from the path stated in acquired JSON data, assuming the JSON
        # and the .spec files are in the same directory (it SHOULD be!)

        fileName = conditions["model"].pop("fileName")
        if (fileName[-5:] != ".spec"):             # fileName doesn't have ".spec" extension
            print("Warning: fileName must include \".spec\" extension!")

        input_dir = "/".join(input_file.split("/")[0 : -1]) + "/"
        spec_path = input_dir + fileName

        # Load and crop the experimental spectrum
        s_data = (
            load_spec(spec_path)
            .crop(
                conditions["model"]["wmin"], 
                conditions["model"]["wmax"], 
                conditions["model"]["wunit"]
            )
        )

    else:                           # The user states a bunch of dicts separately

        # Merge all separated dict inputs into a single dict first

        conditions = {
            "model" : {},
            "fit" : {},
            "bounds" : {},
            "pipeline" : {}
        }

        for key in model:
            conditions["model"][key] = model[key]
        for key in fit_params:
            conditions["fit"][key] = fit_params[key]
        if (bounds != None):
            for key in bounds:
                conditions["bounds"][key] = bounds[key]
        for key in pipeline:
            conditions["pipeline"][key] = pipeline[key]

        # Load data, create Parameters object, from the above dict
        conditions, params = get_conditions(conditions)

        # Load and crop the experimental spectrum
        s_data = (
            s_exp
            .crop(
                conditions["model"]["wmin"], 
                conditions["model"]["wmax"], 
                conditions["model"]["wunit"]
            )
        )

    if verbose:
        end_exp_load = time.time()
        time_exp_load = end_exp_load - begin
        print(f"\nSuccessfully retrieved the experimental data in {time_exp_load}s.\n")

    # Further refine the data spectrum before calculating diff

    s_data, conditions = spectrum_refinement(s_data, conditions)

    pipeline = conditions["pipeline"]
    fit_var = pipeline["fit_var"]

    if verbose:
        end_exp_refine = time.time()
        time_exp_refine = end_exp_refine - end_exp_load
        s_data.plot(show = True)
        print(f"Successfully refined the experimental data in {time_exp_refine}s.")
    

    # PRE-MINIMIZATION SETUP
    
    # Create a Spectrum Factory object for modeling spectra

    kwargs = {}
    model = conditions["model"]

    # List of keys that are included in "model", but not included in SpectrumFactory syntax
    ignore_keys = [
        "slit",
        "offset",
    ]

    for cond in model:
        if cond not in ignore_keys:
            kwargs[cond] = model[cond]
    
    sf = SpectrumFactory(
        **kwargs,
        verbose = False,
        warnings = {
            'MissingSelfBroadeningWarning' : 'ignore',
            'UserWarning' : 'ignore',
            'NegativeEnergiesWarning' : 'ignore',
            'HighTemperatureWarning' : 'ignore'
        }
    )

    # Decide the type of model - LTE or Non-LTE
    LTE = True                                         # LTE == True means it's LTE
    if ("Tvib" in params) or ("Tvib0" in params):
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

        sf.fetch_databank(
            "hitran",
            load_columns = "equilibrium"
        )

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

        sf.fetch_databank(
            "hitemp",
            load_columns = "noneq"
        )

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
        fit_show = {}
        for name, param in result.params.items():
            fit_show[name] = float(param.value)
        fit_show["name"] = "best_fit"

        # Generate best fitted spectrum result
        if LTE:
            s_result = sf.eq_spectrum(**fit_show)
        else:
            s_result = sf.non_eq_spectrum(**fit_show)

        # Apply slit
        if "slit" in model:
            slit, slit_unit = model["slit"].split()
            s_result = s_result.apply_slit(float(slit), slit_unit)

        # Take spectral quantity
        s_result = s_result.take(fit_var)

        # Apply offset
        if "offset" in model:
            off_val, off_unit = model["offset"].split()
            s_result = s_result.offset(float(off_val), off_unit)

        # Apply normalization
        if "normalize" in pipeline:
            if pipeline["normalize"]:
                s_result = s_result.normalize()


        # PLOT THE DIFFERENCE BETWEEN THE TWO AND SAVE THE FIGURE 
        plot_diff(
            s_data, 
            s_result, 
            fit_var, 
            method=['diff', 'ratio'], 
            show = True,
        )

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