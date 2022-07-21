# -*- coding: utf-8 -*-
"""
Modules for conducting spectrum fitting and modeling (LTE and non-LTE).
"""

from .fitting_module import (
    get_conditions, 
    spectrum_refinement, 
    fit_spectrum
)
from .model_LTE import residual_LTE
from .model_nonLTE import residual_NonLTE

__all__ = [
    "get_conditions",
    "spectrum_refinement",
    "fit_spectrum",
    "residual_LTE",
    "residual_NonLTE",
]