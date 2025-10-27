"""
GammaFlow: A Python library for working with time series of gamma ray spectra.
"""

from gammaflow.core.spectrum import Spectrum
from gammaflow.core.time_series import SpectralTimeSeries
from gammaflow.core.listmode import ListMode
from gammaflow.utils.exceptions import (
    GammaFlowError,
    SpectrumError,
    CalibrationError,
    IncompatibleBinningError,
)

# Make operations easily accessible
from gammaflow import operations

__version__ = "0.1.0"
__all__ = [
    "Spectrum",
    "SpectralTimeSeries",
    "ListMode",
    "GammaFlowError",
    "SpectrumError",
    "CalibrationError",
    "IncompatibleBinningError",
    "operations",
]

