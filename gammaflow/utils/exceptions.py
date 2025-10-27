"""
Custom exceptions for GammaFlow.
"""


class GammaFlowError(Exception):
    """Base exception for all GammaFlow errors."""
    pass


class SpectrumError(GammaFlowError):
    """Exception raised for errors in Spectrum operations."""
    pass


class CalibrationError(SpectrumError):
    """Exception raised for energy calibration errors."""
    pass


class IncompatibleBinningError(SpectrumError):
    """Exception raised when trying to operate on spectra with incompatible binning."""
    pass


class TimeSeriesError(GammaFlowError):
    """Exception raised for errors in SpectralTimeSeries operations."""
    pass

