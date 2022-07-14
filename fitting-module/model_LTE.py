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
    initial conditions and values of fit parameters, then returning a 1D
    array (if least-squared method is used) or a scalar (if other methods
    are used) containing the difference between it and data spectrum.

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
        a two-dimensional list, containing fitting history. log[0] has
        residual history, and log[1] has fitting values. This is for the
        fitting result reporting only.

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
    kwargs = {}
    for param in params:
        kwargs[param] = float(params[param])
    
    # Spectrum calculation
    s_model = sf.eq_spectrum(**kwargs)


    # FURTHER REFINE THE MODELED SPECTRUM BEFORE CALCULATING DIFF
    
    pipeline = conditions["pipeline"]
    modeling = conditions["modeling"]

    # Apply slit
    if "slit" in modeling:
        slit, slit_unit = modeling["slit"].split()
        s_model = s_model.apply_slit(float(slit), slit_unit)

    # Take spectral quantity
    fit_var = pipeline["fit_var"]
    s_model = s_model.take(fit_var)

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
            print(f"\n{param} = {float(params[param])}")
        print(f"Residual = {residual}\n")
        #s_model.plot(show = True)


    return residual