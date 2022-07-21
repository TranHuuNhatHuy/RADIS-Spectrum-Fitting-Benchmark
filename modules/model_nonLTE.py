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

# General model for non-LTE spectra. This model will be heavily modified during
# benchmarking process to see which one works best.

def residual_NonLTE(params, conditions, s_data, sf, log, verbose = True):
    """A cost function that calculates a non-LTE spectrum based on the
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

    # GENERATE NON-LTE SPECTRUM BASED ON THOSE PARAMETERS

    # Load initial values of fit parameters
    kwargs = {}
    for param in params:
        kwargs[param] = float(params[param])

    # Deal with the case of multiple Tvib temperatures
    if "Tvib0" in kwargs:       # There is trace of a "Tvib fragmentation" before
        Tvib = []
        for kw in kwargs:
            if "Tvib" in kw:                # Such as "Tvib0", "Tvib1" or so
                Tvib.append(kwargs[kw])     # Bring them altogether, uwu
                kwargs.pop(kw)              # Dispose the fragmented one in kwargs
        kwargs["Tvib"] = tuple(Tvib)        # Finally, we have the tuple of Tvib
    
    # Spectrum calculation
    s_model = sf.non_eq_spectrum(**kwargs)


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