"""
Algorithms for advanced spectral analysis.

This module provides implementations of advanced algorithms for gamma-ray
spectroscopy.

Note: Some advanced algorithms (like CEW) may be available in development
versions but not included in the released package.
"""

__all__ = []

# Try to import CEW if available (may not be present in released versions)
try:
    from gammaflow.algorithms.censored_energy_window import (
        optimize_cew_windows,
        fit_cew_predictor,
        CEWPredictor,
    )
    __all__.extend([
        'optimize_cew_windows',
        'fit_cew_predictor',
        'CEWPredictor',
    ])
except ImportError:
    # CEW not available in this version
    pass

