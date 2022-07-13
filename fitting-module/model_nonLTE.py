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

# General model for Non-LTE spectra. This model will be heavily modified during
# benchmarking process to see which one works best.
def residual_NonLTE(factory, fit_params, name, fixed_params = {}) -> Spectrum:
    """Returning a single-slab non-LTE spectrum with Tvib = (T12, T12, T3), Trot

    Currently appliable for Non-LTE CO2 modeling.

    Parameters
    ----------
    fit_params: dict
        dictionary containing fit parameters, loaded from JSON ground-
        truth file while calling the fit_spectrum function. Example
        ::
            {'T12': 500,
             'T3': 500,
             'Trot': 500}
    fixed_params: dict
        dictionary containing fixed parameters, loaded from JSON ground-
        truth file while calling the fit_spectrum function. Will override
        input conditions of non_eq_spectrum.

    Returns
    -------
    Spectrum: calculated spectrum

    """

    fit_params = fit_params.copy()

    # Acquire the temperatures respectively
    T12 = fit_params.pop("T12")
    T3 = fit_params.pop("T3")
    Trot = fit_params.pop("Trot")
    # Specifically add them into the fittable list
    kwargs = {
        "Tvib": (T12, T12, T3),
        "Trot": Trot,
        "Ttrans": Trot,
    }

    kwargs = {"name": name}

    # Load remaining fittable and fixed parameters
    kwargs = {**kwargs, **fit_params}
    kwargs = {**kwargs, **fixed_params}

    # Generate LTE spectrum based on those parameters
    s = factory.non_eq_spectrum(**kwargs)

    return s