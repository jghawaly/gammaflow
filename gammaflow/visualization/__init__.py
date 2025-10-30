"""
Visualization tools for spectra and time series.

Publication-quality plotting functions using matplotlib and seaborn.
"""

from gammaflow.visualization.plotting import (
    plot_spectrum,
    plot_count_rate_time_series,
    plot_waterfall,
    plot_roi_time_series,
    plot_spectrum_comparison,
)

__all__ = [
    'plot_spectrum',
    'plot_count_rate_time_series',
    'plot_waterfall',
    'plot_roi_time_series',
    'plot_spectrum_comparison',
]
