# -*- coding: utf-8 -*-

import numpy as np
import time
import astropy.units as u
import json
import os
from os.path import join
from typing import Union

from radis import calc_spectrum, plot_diff, Spectrum, SpectrumFactory, get_diff, get_residual
from lmfit import minimize, Parameters, report_fit
from lmfit.minimizer import MinimizerResult
from radis.test.utils import getTestFile
from radis.tools.database import load_spec

# General model for LTE spectra. This model will be heavily modified during
# benchmarking process to see which one works best.

def residual_LTE(params, conditions, s_data, sf, log, verbose = True):
    """A cost function that calculates an LTE spectrum based on the
    initial conditions and values of fit parameters, then returning a
    scalar containing difference between experimental and data spectra.

    Parameters
    ----------
    params : LMFIT.Parameters
        basically a dict containing fit parameters that will be varied
        during the minimization.
    conditions: dict
        a dict containing conditions (fixed parameters) for generating
        model spectra.
    s_data: Spectrum
        data spectrum, loaded from the directory stated in JSON file.
    sf: SpectrumFactory
        the SpectrumFactory object used for modeling spectra, generated
        before the minimize loop occurs.
    log: list
        a Dictionary storing runtime log of the fitting process that are 
        not quite covered by the Minimizer, including: residual and fit 
        values after each fitting loop, and total time elapsed.

    Other parameters
    ----------------
    verbose : bool
        by default, True, will print out information of fitting process.

    Returns
    -------
    residual: float
        residuals of the two spectra, using RADIS's get_residual().

    """

    # GENERATE LTE SPECTRUM BASED ON THOSE PARAMETERS

    # Load initial values of fit parameters
    ignore_keys = [
        "offsetnm",
        "offsetcm-1",
    ]
    kwargs = {}
    for param in params:
        if param not in ignore_keys:
            kwargs[param] = float(params[param])
    
    # Spectrum calculation
    s_model = sf.eq_spectrum(**kwargs)

    # Deal with "offset"
    if "offsetnm" in params:
        offset_value = float(params["offsetnm"])
        s_model = s_model.offset(offset_value, "nm")
    if "offsetcm-1" in params:
        offset_value = float(params["offsetcm-1"])
        s_model = s_model.offset(offset_value, "cm-1")



    # FURTHER REFINE THE MODELED SPECTRUM BEFORE CALCULATING DIFF
    
    pipeline = conditions["pipeline"]
    model = conditions["model"]

    # Apply slit
    if "slit" in model:
        slit, slit_unit = model["slit"].split()
        s_model = s_model.apply_slit(float(slit), slit_unit)

    # Take spectral quantity
    fit_var = pipeline["fit_var"]
    s_model = s_model.take(fit_var)

    # Apply offset
    if "offset" in model:
        off_val, off_unit = model["offset"].split()
        s_model = s_model.offset(float(off_val), off_unit)

    # Apply normalization
    if "normalize" in pipeline:
        if pipeline["normalize"]:
            s_model = s_model.normalize()


    # ACQUIRE AND RETURN DIFF, ALSO LOG FITTING HISTORY

    # Acquire diff
    residual = get_residual(
        s_data, 
        s_model, 
        fit_var, 
        norm = "L2",
        ignore_nan = "True"
    )
    
    # Log the current residual
    log["residual"].append(residual)
    
    # Log the current fitting values of fit parameters
    current_fitvals = []
    for param in params:
        current_fitvals.append(float(params[param]))
    log["fit_vals"].append(current_fitvals)

    # Print information of fitting process
    if verbose:
        for param in params:
            print(f"{param} = {float(params[param])}")
        print(f"\nResidual = {residual}\n")


    return residual