"""
Examples of using Energy Regions of Interest (ROIs) for spectral analysis.

This demonstrates defining ROIs, rebinning spectra by ROI, handling overlapping
and non-consecutive ROIs, and analyzing time evolution of specific energy ranges.
"""

import numpy as np
from gammaflow import Spectrum, SpectralTimeSeries
from gammaflow.operations import (
    EnergyROI,
    rebin_spectrum_rois,
    rebin_time_series_rois,
    create_roi_collection,
    check_roi_overlaps,
    print_roi_summary
)

print("=" * 70)
print("Energy ROI Examples")
print("=" * 70)

# =============================================================================
# Example 1: Basic ROI Definition
# =============================================================================

print("\n" + "=" * 70)
print("Example 1: Basic ROI Definition")
print("=" * 70)

# Define ROIs for common gamma-ray peaks
k40_roi = EnergyROI(
    e_min=1450,
    e_max=1470,
    label="K-40 Peak",
    method="manual"
)

cs137_roi = EnergyROI(
    e_min=655,
    e_max=668,
    label="Cs-137 Peak",
    method="manual"
)

print(f"K-40 ROI: {k40_roi}")
print(f"  Width: {k40_roi.width} keV")
print(f"  Center: {k40_roi.center} keV")

print(f"\nCs-137 ROI: {cs137_roi}")
print(f"  Width: {cs137_roi.width} keV")
print(f"  Center: {cs137_roi.center} keV")

# =============================================================================
# Example 2: Rebinning a Spectrum Using ROIs
# =============================================================================

print("\n" + "=" * 70)
print("Example 2: Rebinning a Spectrum Using ROIs")
print("=" * 70)

# Create a simulated spectrum
energy_edges = np.linspace(0, 1500, 1501)
counts = np.ones(1500) * 50  # Background

# Add Cs-137 peak
cs137_idx = int(661.7)
counts[cs137_idx-5:cs137_idx+5] += 1000

# Add K-40 peak
k40_idx = int(1460.8)
counts[k40_idx-5:k40_idx+5] += 500

spec = Spectrum(counts, energy_edges=energy_edges)

# Define ROIs
rois = [
    EnergyROI(e_min=100, e_max=200, label="Low Energy Background"),
    EnergyROI(e_min=655, e_max=668, label="Cs-137"),
    EnergyROI(e_min=1450, e_max=1470, label="K-40")
]

# Rebin spectrum
roi_counts, labels = rebin_spectrum_rois(spec, rois, return_labels=True)

print("ROI Integration Results:")
for label, count in zip(labels, roi_counts):
    print(f"  {label:25s}: {count:8.0f} counts")

# =============================================================================
# Example 3: Overlapping ROIs
# =============================================================================

print("\n" + "=" * 70)
print("Example 3: Overlapping ROIs")
print("=" * 70)

# Define overlapping ROIs (useful for peak fitting, background estimation)
overlapping_rois = [
    EnergyROI(e_min=650, e_max=670, label="Cs-137 + Background"),
    EnergyROI(e_min=655, e_max=665, label="Cs-137 Core"),
    EnergyROI(e_min=645, e_max=655, label="Lower Background"),
    EnergyROI(e_min=665, e_max=675, label="Upper Background")
]

# Check for overlaps
overlaps = check_roi_overlaps(overlapping_rois)
print(f"Found {len(overlaps)} overlaps:")
for i, j in overlaps:
    print(f"  '{overlapping_rois[i].label}' overlaps with '{overlapping_rois[j].label}'")

# Rebin with overlapping ROIs
roi_counts = rebin_spectrum_rois(spec, overlapping_rois)
print("\nIntegrated counts (note: overlaps count events multiple times):")
for roi, count in zip(overlapping_rois, roi_counts):
    print(f"  {roi.label:25s}: {count:8.0f} counts")

# =============================================================================
# Example 4: Non-Consecutive ROIs
# =============================================================================

print("\n" + "=" * 70)
print("Example 4: Non-Consecutive ROIs")
print("=" * 70)

# Define non-consecutive ROIs (gaps between them)
peak_rois = create_roi_collection([
    (100, 150, "Low Energy"),
    (655, 668, "Cs-137"),      # Large gap
    (1450, 1470, "K-40")        # Another large gap
], method="manual")

print_roi_summary(peak_rois, check_overlaps=True)

roi_counts = rebin_spectrum_rois(spec, peak_rois)
print("\nROI counts:")
for roi, count in zip(peak_rois, roi_counts):
    print(f"  {roi.label}: {count:.0f}")

# =============================================================================
# Example 5: Time Series ROI Analysis
# =============================================================================

print("\n" + "=" * 70)
print("Example 5: Time Series ROI Analysis")
print("=" * 70)

# Create time series with time-varying peak intensities
n_spectra = 50
counts_array = np.zeros((n_spectra, 1500))

for i in range(n_spectra):
    # Background
    counts_array[i] = 50
    
    # Cs-137 peak (decreasing over time - simulating decay)
    cs137_intensity = 1000 * np.exp(-i / 20)
    counts_array[i, cs137_idx-5:cs137_idx+5] += cs137_intensity
    
    # K-40 peak (constant)
    counts_array[i, k40_idx-5:k40_idx+5] += 500

timestamps = np.arange(n_spectra) * 10.0  # Every 10 seconds
ts = SpectralTimeSeries.from_array(counts_array, energy_edges=energy_edges, timestamps=timestamps)

# Define ROIs
rois = [
    EnergyROI(e_min=655, e_max=668, label="Cs-137"),
    EnergyROI(e_min=1450, e_max=1470, label="K-40")
]

# Rebin time series
roi_counts_ts, labels = rebin_time_series_rois(ts, rois, return_labels=True)

print(f"Time series rebinned from {ts.counts.shape} to {roi_counts_ts.shape}")
print(f"Rows: {roi_counts_ts.shape[0]} time points")
print(f"Columns: {roi_counts_ts.shape[1]} ROIs ({', '.join(labels)})")

# Analyze time evolution
print("\nTime evolution:")
print(f"  Cs-137 (first 3): {roi_counts_ts[:3, 0]}")
print(f"  Cs-137 (last 3):  {roi_counts_ts[-3:, 0]}")
print(f"  K-40 (first 3):   {roi_counts_ts[:3, 1]}")
print(f"  K-40 (last 3):    {roi_counts_ts[-3:, 1]}")

# =============================================================================
# Example 6: Censored Energy Windows Method
# =============================================================================

print("\n" + "=" * 70)
print("Example 6: Censored Energy Windows for Background Estimation")
print("=" * 70)

# Define censored windows (avoiding known peaks)
# Note: spectrum only goes to 1500 keV, so we stay within range
censored_rois = create_roi_collection([
    (100, 600, "Low Energy Window"),
    (700, 1400, "Mid Energy Window")
], method="Censored Energy Windows", shared_metadata={'purpose': 'background_estimation'})

print("Censored Energy Windows:")
for roi in censored_rois:
    print(f"  {roi.label}: [{roi.e_min:.0f}, {roi.e_max:.0f}] keV, method='{roi.method}'")

# Integrate over censored windows
bg_counts = rebin_spectrum_rois(spec, censored_rois)
print("\nBackground counts in censored windows:")
for roi, count in zip(censored_rois, bg_counts):
    print(f"  {roi.label}: {count:.0f} counts ({count/roi.width:.1f} counts/keV)")

# =============================================================================
# Example 7: Creating ROIs from Peak Search Results
# =============================================================================

print("\n" + "=" * 70)
print("Example 7: Creating ROIs from Peak Search Results")
print("=" * 70)

# Simulate peak search results
peak_energies = [661.7, 1460.8, 511.0]  # keV
peak_fwhms = [2.5, 3.0, 2.0]  # keV

# Create ROIs around peaks (±2 FWHM)
roi_width_factor = 2
roi_defs = []
for energy, fwhm in zip(peak_energies, peak_fwhms):
    width = roi_width_factor * fwhm
    roi_defs.append((energy - width, energy + width, f"Peak at {energy:.1f} keV"))

peak_search_rois = create_roi_collection(
    roi_defs,
    method="peak_search",
    shared_metadata={'algorithm': 'auto_peak_finder', 'fwhm_factor': roi_width_factor}
)

print("ROIs from peak search:")
for roi in peak_search_rois:
    print(f"  {roi.label}: [{roi.e_min:.1f}, {roi.e_max:.1f}] keV")
    print(f"    method: {roi.method}, metadata: {roi.metadata}")

# =============================================================================
# Example 8: ROI Serialization (Save/Load)
# =============================================================================

print("\n" + "=" * 70)
print("Example 8: ROI Serialization")
print("=" * 70)

# Create ROI
original_roi = EnergyROI(
    e_min=655,
    e_max=668,
    label="Cs-137",
    method="manual",
    metadata={'confidence': 0.95, 'peak_fwhm': 2.5}
)

# Convert to dict (for JSON serialization, database storage, etc.)
roi_dict = original_roi.to_dict()
print("ROI as dictionary:")
print(f"  {roi_dict}")

# Reconstruct from dict
restored_roi = EnergyROI.from_dict(roi_dict)
print(f"\nRestored ROI: {restored_roi}")
print(f"  Metadata preserved: {restored_roi.metadata}")

# =============================================================================
# Example 9: Comprehensive Gamma Spectroscopy Workflow
# =============================================================================

print("\n" + "=" * 70)
print("Example 9: Complete Gamma Spectroscopy Workflow")
print("=" * 70)

# Create realistic spectrum
energy_edges = np.linspace(0, 2000, 2001)
counts = np.random.poisson(30, size=2000)  # Noisy background

# Add multiple peaks
peaks = {
    'Co-60 (1)': (1173.2, 800),
    'Co-60 (2)': (1332.5, 700),
    'Cs-137': (661.7, 1500),
    'K-40': (1460.8, 600)
}

for peak_name, (energy, intensity) in peaks.items():
    idx = int(energy)
    if idx < len(counts):
        counts[idx-3:idx+3] += intensity

spec = Spectrum(counts, energy_edges=energy_edges)

# Define ROIs for all peaks
analysis_rois = []
for peak_name, (energy, _) in peaks.items():
    roi = EnergyROI(
        e_min=energy - 10,
        e_max=energy + 10,
        label=peak_name,
        method="manual"
    )
    analysis_rois.append(roi)

# Add background ROIs
analysis_rois.extend([
    EnergyROI(e_min=100, e_max=200, label="Background 1"),
    EnergyROI(e_min=1800, e_max=1900, label="Background 2")
])

print_roi_summary(analysis_rois)

# Perform analysis
roi_counts, labels = rebin_spectrum_rois(spec, analysis_rois, return_labels=True)

print("\nAnalysis Results:")
print(f"{'ROI':25s} {'Counts':>10s} {'Rate':>10s}")
print("-" * 50)
for label, count in zip(labels, roi_counts):
    rate = count / 20  # Assuming 20 keV width
    print(f"{label:25s} {count:10.0f} {rate:10.1f}")

print("\n" + "=" * 70)
print("Examples Complete!")
print("=" * 70)
print("\nKey Features Demonstrated:")
print("  1. Basic ROI definition with labels and methods")
print("  2. Rebinning spectra using ROIs")
print("  3. Overlapping ROIs (useful for peak fitting)")
print("  4. Non-consecutive ROIs (gaps allowed)")
print("  5. Time series analysis with ROIs")
print("  6. Censored Energy Windows method")
print("  7. Creating ROIs from peak search results")
print("  8. ROI serialization (save/load)")
print("  9. Complete gamma spectroscopy workflow")

