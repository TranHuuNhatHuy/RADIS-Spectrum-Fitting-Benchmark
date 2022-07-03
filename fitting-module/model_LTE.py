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

def residual_LTE(params, conditions, s_data, verbose = True):
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

    Other parameters
    ----------
    verbose : bool
        by default, True, will print out information of fitting process.

    Returns
    -------
    diff: list
        a 1D-array containing difference between two spectra.

    """

    # Load initial values of fit parameters
    for param in params:
        kwargs = {param : float(params[param])}
    
    # Load conditions (fixed parameters)
    conds = conditions.copy()
    slit_info = conds.pop("slit")      # Hide the slit info first, will use it later below
    fileName = conds.pop("fileName")
    conds["name"] = fileName           # Because calc_spectrum() requires "name" parameter
    kwargs = {**kwargs, **conds}
    fit_var = s_data.get_vars()[0]

    # Generate LTE spectrum based on those parameters
    s_model = calc_spectrum(
        **kwargs,
        cutoff = 1e-25,
        wstep = 0.001,
        truncation = 1,
        verbose = False,
        warnings = "ignore"
    )

    # Further refine the modeled spectrum before calculating diff
    
    slit, slit_unit = slit_info.split()         # Now get the slit info

    s_model = (
        s_model
        .apply_slit(float(slit), slit_unit)     # Simulate slit
        .take(fit_var)
        .normalize()                            # Normalize
        .resample(                              # Downgrade to data spectrum's resolution
            s_data, 
            energy_threshold = 2e-2
        )
    )


    # Acquire diff
    residual = get_residual(
        s_model, 
        s_data, 
        fit_var, 
        norm = "L2",
        ignore_nan = "True"
    )
    #residual = get_residual(s_model, s_data, "radiance")


    # Print information of fitting process
    if verbose:
        for param in params:
            print(f"\n{param} = {float(params[param])}")
        print(f"Residual = {residual}\n")


    return residual