"""
Basic usage examples for GammaFlow.

This script demonstrates the core functionality of the Spectrum and
SpectralTimeSeries classes.
"""

import numpy as np
from gammaflow import Spectrum, SpectralTimeSeries

print("=" * 60)
print("GammaFlow Basic Usage Examples")
print("=" * 60)

# ============================================
# Example 1: Creating Uncalibrated Spectra
# ============================================
print("\n1. Creating Uncalibrated Spectrum")
print("-" * 60)

# Simulate counts data
counts = np.random.poisson(lam=100, size=1024)
spectrum = Spectrum(counts, live_time=10.0)

print(f"Created: {spectrum}")
print(f"Is calibrated: {spectrum.is_calibrated}")
print(f"Energy range: {spectrum.energy_edges[0]} to {spectrum.energy_edges[-1]} (channels)")
print(f"Total counts: {np.sum(spectrum.counts):.0f}")

# ============================================
# Example 2: Creating Calibrated Spectra
# ============================================
print("\n2. Creating Calibrated Spectrum")
print("-" * 60)

# Create calibrated spectrum (E = 0 + 0.5*channel)
calibrated = spectrum.apply_calibration([0, 0.5])  # Linear calibration

print(f"Created: {calibrated}")
print(f"Is calibrated: {calibrated.is_calibrated}")
print(f"Energy range: {calibrated.energy_edges[0]:.1f} to {calibrated.energy_edges[-1]:.1f} keV")

# ============================================
# Example 3: Arithmetic Operations
# ============================================
print("\n3. Arithmetic Operations")
print("-" * 60)

spec1 = Spectrum(np.random.poisson(100, size=512))
spec2 = Spectrum(np.random.poisson(100, size=512))

# Add spectra
sum_spec = spec1 + spec2
print(f"Spec1 total: {np.sum(spec1.counts):.0f}")
print(f"Spec2 total: {np.sum(spec2.counts):.0f}")
print(f"Sum total: {np.sum(sum_spec.counts):.0f}")

# Subtract spectra
diff_spec = spec1 - spec2
print(f"Difference total: {np.sum(diff_spec.counts):.0f}")

# Multiply by scalar
scaled_spec = spec1 * 2.0
print(f"Scaled (x2) total: {np.sum(scaled_spec.counts):.0f}")

# ============================================
# Example 4: Energy Slicing
# ============================================
print("\n4. Energy Slicing")
print("-" * 60)

# Create calibrated spectrum
spec = Spectrum(
    counts=np.random.poisson(100, size=1024),
    energy_edges=np.linspace(0, 1000, 1025)  # 0 to 1000 keV
)

# Extract energy slice
roi = spec.slice_energy(e_min=200, e_max=400)
print(f"Original spectrum: {spec.n_bins} bins, range [0, 1000] keV")
print(f"ROI [200-400] keV: {roi.n_bins} bins")
print(f"ROI total counts: {np.sum(roi.counts):.0f}")

# Integrate over range
total_roi = spec.integrate(e_min=200, e_max=400)
print(f"Integrated counts in ROI: {total_roi:.0f}")

# ============================================
# Example 5: Creating Time Series
# ============================================
print("\n5. Creating SpectralTimeSeries")
print("-" * 60)

# Method 1: From list of Spectrum objects
spectra_list = []
for i in range(100):
    counts = np.random.poisson(lam=50 + i, size=512)
    spec = Spectrum(counts, timestamp=float(i), live_time=1.0)
    spec.metadata['measurement_id'] = i
    spectra_list.append(spec)

time_series1 = SpectralTimeSeries(spectra_list)
print(f"Method 1 (from list): {time_series1}")

# Method 2: Directly from 2D numpy array (simpler!)
counts_array = np.random.poisson(100, size=(100, 512))
time_series2 = SpectralTimeSeries.from_array(counts_array)
print(f"Method 2 (from array): {time_series2}")
print(f"Counts array shape: {time_series2.counts.shape}")

# Method 2 with calibration
edges = np.linspace(0, 3000, 513)
timestamps = np.arange(100) * 10.0  # Every 10 seconds
time_series3 = SpectralTimeSeries.from_array(
    counts_array,
    energy_edges=edges,
    timestamps=timestamps,
    live_times=9.5  # Same for all
)
print(f"Method 2 with calibration: {time_series3}")

# ============================================
# Example 6: Vectorized Operations
# ============================================
print("\n6. Vectorized Operations on Time Series")
print("-" * 60)

# Get counts as 2D array
counts_array = time_series.counts
print(f"Counts array shape: {counts_array.shape}")

# Compute statistics across time
total_per_spectrum = np.sum(counts_array, axis=1)
mean_spectrum = np.mean(counts_array, axis=0)
std_spectrum = np.std(counts_array, axis=0)

print(f"Total counts per spectrum (first 5): {total_per_spectrum[:5]}")
print(f"Mean spectrum total: {np.sum(mean_spectrum):.0f}")
print(f"Mean uncertainty: {np.mean(std_spectrum):.2f}")

# Background subtraction (vectorized)
background = np.median(counts_array, axis=0)
time_series.counts[:] = counts_array - background
print(f"Background subtracted (median of all spectra)")

# ============================================
# Example 7: Object-Oriented Access
# ============================================
print("\n7. Object-Oriented Access")
print("-" * 60)

# Access individual spectra
spec = time_series[42]
print(f"Spectrum 42: {spec}")
print(f"  Timestamp: {spec.timestamp}")
print(f"  Live time: {spec.live_time}")
print(f"  Metadata: {spec.metadata}")
print(f"  Total counts: {np.sum(spec.counts):.0f}")

# Iterate over spectra
high_count_spectra = []
for spec in time_series:
    if np.sum(spec.counts) > 2500:
        high_count_spectra.append(spec)
        spec.metadata['high_counts'] = True

print(f"Found {len(high_count_spectra)} spectra with > 2500 counts")

# ============================================
# Example 8: Shared Memory (Changes Propagate)
# ============================================
print("\n8. Shared Memory Demonstration")
print("-" * 60)

# Modify via time series array
original_value = time_series.counts[10, 100]
print(f"Original value at [10, 100]: {original_value:.0f}")

time_series.counts[10, 100] = 9999.0

# Check via spectrum object
spec = time_series[10]
print(f"Value in Spectrum object: {spec.counts[100]:.0f}")
print("Changes propagate! (shared memory)")

# Modify via spectrum object
spec.counts[101] = 8888.0

# Check via array
print(f"Value in array: {time_series.counts[10, 101]:.0f}")
print("Changes propagate both ways!")

# ============================================
# Example 9: Applying Calibration to Time Series
# ============================================
print("\n9. Applying Calibration to Time Series")
print("-" * 60)

# Create uncalibrated time series
uncal_spectra = [
    Spectrum(np.random.poisson(100, size=256)) 
    for _ in range(50)
]
uncal_ts = SpectralTimeSeries(uncal_spectra)

print(f"Before calibration: {uncal_ts.is_calibrated}")

# Apply calibration to all spectra at once
calibrated_ts = uncal_ts.apply_calibration([0, 0.5, 0.001])  # Quadratic

print(f"After calibration: {calibrated_ts.is_calibrated}")
print(f"Energy range: {calibrated_ts.energy_edges[0]:.2f} to {calibrated_ts.energy_edges[-1]:.2f} keV")

# ============================================
# Example 10: Time Series Analysis
# ============================================
print("\n10. Time Series Analysis")
print("-" * 60)

# Create time series with timestamps
ts_spectra = []
for i in range(100):
    spec = Spectrum(
        counts=np.random.poisson(lam=100, size=256),
        timestamp=i * 10.0,  # Every 10 seconds
        live_time=9.5,
    )
    ts_spectra.append(spec)

time_series = SpectralTimeSeries(ts_spectra)

# Time slicing
sliced = time_series.slice_time(t_min=200, t_max=500)
print(f"Original: {len(time_series)} spectra")
print(f"Sliced [200-500]: {len(sliced)} spectra")

# Integration over time
integrated = time_series.integrate_time(t_min=0, t_max=100)
print(f"Integrated spectrum (0-100s): {np.sum(integrated.counts):.0f} total counts")

# Mean spectrum
mean_spec = time_series.mean_spectrum()
print(f"Mean spectrum: {np.sum(mean_spec.counts):.0f} counts")

# ============================================
# Example 11: Metadata Usage
# ============================================
print("\n11. Using Metadata")
print("-" * 60)

# Create spectra with rich metadata
spec = Spectrum(
    counts=np.random.poisson(100, size=512),
    timestamp=1234567890.0,
    live_time=100.0,
    real_time=105.0,
)

# Add metadata
spec.metadata['detector'] = 'HPGe'
spec.metadata['source'] = 'Co-60'
spec.metadata['distance_cm'] = 10.0
spec.metadata['operator'] = 'John Doe'

print(f"Spectrum metadata:")
for key, value in spec.metadata.items():
    print(f"  {key}: {value}")

print(f"Dead time fraction: {spec.dead_time_fraction:.2%}")

# ============================================
# Example 12: Copy-on-Write Behavior
# ============================================
print("\n12. Copy-on-Write (COW) Behavior")
print("-" * 60)

# Create time series with shared calibration
spectra = [Spectrum(np.random.poisson(100, size=256)) for _ in range(10)]
ts = SpectralTimeSeries(spectra, shared_calibration=True)

# Get a spectrum
spec = ts[5]
print(f"Spectrum has shared calibration: {spec.has_shared_calibration}")

# Apply calibration in-place (detaches automatically)
spec.apply_calibration_([0, 1.0])

print(f"After modification, shared calibration: {spec.has_shared_calibration}")
print(f"Other spectra still shared: {ts[4].has_shared_calibration}")
print("COW: Automatically detached from shared calibration!")

print("\n" + "=" * 60)
print("Examples Complete!")
print("=" * 60)

