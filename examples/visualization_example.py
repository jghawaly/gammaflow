"""
Example: Visualization with GammaFlow

Demonstrates various plotting functions for spectra and time series
using matplotlib and seaborn.
"""

import numpy as np
import matplotlib.pyplot as plt
from gammaflow import Spectrum, SpectralTimeSeries
from gammaflow.visualization import (
    plot_spectrum,
    plot_count_rate_time_series,
    plot_waterfall,
    plot_roi_time_series,
    plot_spectrum_comparison,
)
from gammaflow.operations import EnergyROI

print("=" * 70)
print("GammaFlow Visualization Examples")
print("=" * 70)

# =============================================================================
# Example 1: Plot a Single Spectrum
# =============================================================================

print("\n" + "=" * 70)
print("Example 1: Single Spectrum Plot")
print("=" * 70)

# Create a synthetic spectrum with Cs-137 and K-40 peaks
np.random.seed(42)
energy_edges = np.linspace(0, 3000, 1025)  # 0-3000 keV
counts = np.random.poisson(50, 1024)  # Background

# Add Cs-137 peak at 661.7 keV
cs137_center = 661.7
cs137_width = 20
cs137_idx = np.argmin(np.abs(energy_edges[:-1] - cs137_center))
counts[cs137_idx-5:cs137_idx+5] += np.random.poisson(500, 10)

# Add K-40 peak at 1460.8 keV
k40_center = 1460.8
k40_width = 30
k40_idx = np.argmin(np.abs(energy_edges[:-1] - k40_center))
counts[k40_idx-7:k40_idx+7] += np.random.poisson(200, 14)

spectrum = Spectrum(counts, energy_edges=energy_edges, real_time=600.0)

print("\nPlotting spectrum with log y-axis...")
fig, ax = plot_spectrum(spectrum, log_y=True, show_uncertainty=True)
ax.set_title('Synthetic Gamma Spectrum (Cs-137 + K-40)')
plt.savefig('/tmp/spectrum_plot.png', dpi=150, bbox_inches='tight')
print("  Saved to /tmp/spectrum_plot.png")
plt.close()

# Plot count rate instead of counts
print("\nPlotting count rate...")
fig, ax = plot_spectrum(spectrum, mode='count_rate', log_y=True)
ax.set_title('Count Rate Spectrum')
plt.savefig('/tmp/spectrum_count_rate.png', dpi=150, bbox_inches='tight')
print("  Saved to /tmp/spectrum_count_rate.png")
plt.close()

# =============================================================================
# Example 2: Spectrum Comparison
# =============================================================================

print("\n" + "=" * 70)
print("Example 2: Comparing Multiple Spectra")
print("=" * 70)

# Create background and source spectra
background_counts = np.random.poisson(30, 1024)
background = Spectrum(background_counts, energy_edges=energy_edges, real_time=600.0)

source_counts = background_counts.copy()
source_counts[cs137_idx-5:cs137_idx+5] += np.random.poisson(800, 10)
source = Spectrum(source_counts, energy_edges=energy_edges, real_time=600.0)

print("\nComparing background and source spectra...")
fig, ax = plot_spectrum_comparison(
    [background, source],
    labels=['Background', 'Source (with Cs-137)'],
    log_y=True
)
ax.set_title('Background vs Source Comparison')
plt.savefig('/tmp/spectrum_comparison.png', dpi=150, bbox_inches='tight')
print("  Saved to /tmp/spectrum_comparison.png")
plt.close()

# =============================================================================
# Example 3: Time Series Count Rate Plot
# =============================================================================

print("\n" + "=" * 70)
print("Example 3: Time Series Count Rate")
print("=" * 70)

# Create a time series with a source appearing in the middle
print("\nGenerating time series with source appearing at t=40-60s...")
spectra_list = []
for i in range(100):
    counts_base = np.random.poisson(50, 1024)
    
    # Inject source between t=40 and t=60
    if 40 <= i < 60:
        counts_base[cs137_idx-5:cs137_idx+5] += np.random.poisson(400, 10)
    
    spectra_list.append(
        Spectrum(
            counts_base,
            energy_edges=energy_edges,
            timestamp=float(i),
            real_time=1.0
        )
    )

time_series = SpectralTimeSeries(spectra_list)

print("\nPlotting total count rate vs time...")
fig, ax = plot_count_rate_time_series(time_series, show_uncertainty=True)
ax.set_title('Total Count Rate Over Time')
ax.axvspan(40, 60, alpha=0.2, color='red', label='Source Present')
ax.legend()
plt.savefig('/tmp/count_rate_time_series.png', dpi=150, bbox_inches='tight')
print("  Saved to /tmp/count_rate_time_series.png")
plt.close()

# =============================================================================
# Example 4: Waterfall Plot
# =============================================================================

print("\n" + "=" * 70)
print("Example 4: Waterfall Plot (2D Spectral Evolution)")
print("=" * 70)

print("\nCreating waterfall plot...")
fig, ax = plot_waterfall(
    time_series,
    mode='count_rate',
    log_scale=True,
    cmap='viridis'
)
plt.savefig('/tmp/waterfall_plot.png', dpi=150, bbox_inches='tight')
print("  Saved to /tmp/waterfall_plot.png")
plt.close()

# Waterfall with energy range focused on Cs-137 peak
print("\nCreating zoomed waterfall plot (Cs-137 region)...")
fig, ax = plot_waterfall(
    time_series,
    mode='count_rate',
    log_scale=True,
    energy_range=(600, 750),
    cmap='hot'
)
ax.set_title('Spectral Evolution - Cs-137 Region (661.7 keV)')
plt.savefig('/tmp/waterfall_zoomed.png', dpi=150, bbox_inches='tight')
print("  Saved to /tmp/waterfall_zoomed.png")
plt.close()

# =============================================================================
# Example 5: ROI Time Series
# =============================================================================

print("\n" + "=" * 70)
print("Example 5: ROI Time Evolution")
print("=" * 70)

# Define regions of interest
rois = [
    EnergyROI(e_min=100, e_max=500, label="Low Energy Background"),
    EnergyROI(e_min=655, e_max=668, label="Cs-137 Peak (661.7 keV)"),
    EnergyROI(e_min=1450, e_max=1470, label="K-40 Peak (1460.8 keV)"),
]

print("\nPlotting ROI time evolution...")
fig, ax = plot_roi_time_series(
    time_series,
    rois,
    mode='count_rate',
    show_uncertainty=True
)
ax.set_title('ROI Count Rates Over Time')
ax.axvspan(40, 60, alpha=0.1, color='red')
plt.savefig('/tmp/roi_time_series.png', dpi=150, bbox_inches='tight')
print("  Saved to /tmp/roi_time_series.png")
plt.close()

# =============================================================================
# Example 6: Multi-Panel Figure
# =============================================================================

print("\n" + "=" * 70)
print("Example 6: Multi-Panel Figure")
print("=" * 70)

print("\nCreating multi-panel figure...")
fig = plt.figure(figsize=(14, 10))

# Top: Spectrum
ax1 = plt.subplot(3, 2, (1, 2))
plot_spectrum(spectrum, log_y=True, fig=fig, ax=ax1)
ax1.set_title('Spectrum', fontweight='bold')

# Middle left: Count rate time series
ax2 = plt.subplot(3, 2, 3)
plot_count_rate_time_series(time_series, fig=fig, ax=ax2)
ax2.set_title('Total Count Rate', fontweight='bold')

# Middle right: ROI time series
ax3 = plt.subplot(3, 2, 4)
plot_roi_time_series(
    time_series,
    [rois[1]],  # Just Cs-137
    mode='count_rate',
    fig=fig,
    ax=ax3
)
ax3.set_title('Cs-137 ROI Evolution', fontweight='bold')

# Bottom: Waterfall
ax4 = plt.subplot(3, 2, (5, 6))
plot_waterfall(
    time_series,
    mode='count_rate',
    log_scale=True,
    energy_range=(600, 750),
    fig=fig,
    ax=ax4
)
ax4.set_title('Waterfall Plot - Cs-137 Region', fontweight='bold')

plt.tight_layout()
plt.savefig('/tmp/multi_panel_figure.png', dpi=150, bbox_inches='tight')
print("  Saved to /tmp/multi_panel_figure.png")
plt.close()

print("\n" + "=" * 70)
print("All visualization examples completed!")
print("=" * 70)
print("\nGenerated files:")
print("  - /tmp/spectrum_plot.png")
print("  - /tmp/spectrum_count_rate.png")
print("  - /tmp/spectrum_comparison.png")
print("  - /tmp/count_rate_time_series.png")
print("  - /tmp/waterfall_plot.png")
print("  - /tmp/waterfall_zoomed.png")
print("  - /tmp/roi_time_series.png")
print("  - /tmp/multi_panel_figure.png")
print("\n" + "=" * 70)

