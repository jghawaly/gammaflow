"""Synthetic data effects for gamma-ray spectral time series.

Currently: temperature-induced gain shift, offset drift, and resolution
degradation, with configurable temperature profiles.
"""

from gammaflow.simulation import temperature
from gammaflow.simulation.gain import (
    GainModel,
    LinearGainModel,
    PolynomialGainModel,
    CallableGainModel,
    ResolutionModel,
    TemperatureResponseModel,
    apply_gain_shift,
    apply_temperature_drift,
)

__all__ = [
    "temperature",
    "GainModel",
    "LinearGainModel",
    "PolynomialGainModel",
    "CallableGainModel",
    "ResolutionModel",
    "TemperatureResponseModel",
    "apply_gain_shift",
    "apply_temperature_drift",
]
