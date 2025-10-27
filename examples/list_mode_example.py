"""
Example: Working with List Mode Data

This example demonstrates how to convert list mode data (event-by-event)
into spectral time series using rolling time windows.
"""

import numpy as np
import matplotlib.pyplot as plt
from gammaflow import SpectralTimeSeries


# ============================================
# Example 1: Basic List Mode Conversion
# ============================================
print("1. Basic List Mode Conversion")
print("-" * 60)

# Generate synthetic list mode data
# - Time deltas: time since last event (exponential distribution for Poisson process)
# - Energies: detected energies (or pulse heights)
np.random.seed(42)
n_events = 100000
time_deltas = np.random.exponential(0.001, size=n_events)  # ~1000 Hz rate
energies = np.random.gamma(2, 500, size=n_events)  # Gamma-like spectrum

print(f"Generated {n_events} events")
print(f"Average rate: {1 / np.mean(time_deltas):.1f} Hz")
print(f"Total time: {np.sum(time_deltas):.1f} seconds")
print(f"Energy range: {energies.min():.1f} - {energies.max():.1f} keV")

# Convert to spectral time series with non-overlapping 10-second windows
ts = SpectralTimeSeries.from_list_mode(
    time_deltas, energies,
    integration_time=10.0,  # 10 second windows
    energy_bins=512         # 512 energy bins
)

print(f"\nCreated time series:")
print(f"  {ts.n_spectra} spectra")
print(f"  {ts.n_bins} bins per spectrum")
print(f"  Time range: {ts.timestamps[0]:.1f} - {ts.timestamps[-1]:.1f} seconds")


# ============================================
# Example 2: Overlapping Windows
# ============================================
print("\n2. Overlapping Windows for Temporal Analysis")
print("-" * 60)

# Create time series with overlapping windows
# This gives better temporal resolution for tracking changes
ts_overlap = SpectralTimeSeries.from_list_mode(
    time_deltas, energies,
    integration_time=20.0,  # 20 second integration
    stride_time=2.0,        # Move by 2 seconds (90% overlap)
    energy_bins=256
)

print(f"Non-overlapping: {ts.n_spectra} spectra")
print(f"Overlapping (90%): {ts_overlap.n_spectra} spectra")
print(f"Temporal resolution improved by {ts_overlap.n_spectra / ts.n_spectra:.1f}x")


# ============================================
# Example 3: Custom Energy Range
# ============================================
print("\n3. Custom Energy Range and Binning")
print("-" * 60)

# Focus on specific energy region
ts_focused = SpectralTimeSeries.from_list_mode(
    time_deltas, energies,
    integration_time=10.0,
    energy_bins=1024,
    energy_range=(0, 3000)  # Only 0-3000 keV
)

print(f"Energy range: {ts_focused.energy_edges[0]:.1f} - {ts_focused.energy_edges[-1]:.1f} keV")
print(f"Energy bin width: {(ts_focused.energy_edges[1] - ts_focused.energy_edges[0]):.2f} keV")


# ============================================
# Example 4: Accessing Event Counts
# ============================================
print("\n4. Accessing Event Metadata")
print("-" * 60)

# Each spectrum stores metadata about the window
for i, spec in enumerate(ts[:5]):  # First 5 spectra
    print(f"Spectrum {i}:")
    print(f"  Window: {spec.metadata['window_start']:.1f} - {spec.metadata['window_end']:.1f} s")
    print(f"  Events: {spec.metadata['n_events']}")
    print(f"  Total counts: {np.sum(spec.counts):.0f}")
    print(f"  Count rate: {spec.count_rate.sum():.1f} cps")


# ============================================
# Example 5: Time Series Analysis
# ============================================
print("\n5. Time Series Analysis")
print("-" * 60)

# Analyze count rates over time
count_rates = [np.sum(spec.counts) / spec.real_time for spec in ts]
print(f"Mean count rate: {np.mean(count_rates):.1f} ± {np.std(count_rates):.1f} cps")
print(f"Min/Max: {np.min(count_rates):.1f} / {np.max(count_rates):.1f} cps")

# Coefficient of variation (should be ~1/sqrt(N) for Poisson)
cv = np.std(count_rates) / np.mean(count_rates)
expected_cv = 1 / np.sqrt(np.mean(count_rates) * 10)  # 10 second windows
print(f"Coefficient of variation: {cv:.3f}")
print(f"Expected (Poisson): {expected_cv:.3f}")


# ============================================
# Example 6: Vectorized Background Subtraction
# ============================================
print("\n6. Vectorized Operations on List Mode Time Series")
print("-" * 60)

# Background subtraction
background = np.mean(ts.counts, axis=0)
ts.counts -= background

print(f"Subtracted background (mean spectrum)")
print(f"Background total: {np.sum(background):.0f} counts")

# Peak search in sum spectrum
sum_spectrum = np.sum(ts.counts, axis=0)
peak_bin = np.argmax(sum_spectrum)
peak_energy = ts.energy_centers[peak_bin]
print(f"Peak at bin {peak_bin}: {peak_energy:.1f} keV")


# ============================================
# Example 7: Realistic Example - Variable Rate
# ============================================
print("\n7. Variable Count Rate Example")
print("-" * 60)

# Simulate varying count rate (e.g., source moving)
n_events = 50000
base_rate = 1000  # 1000 Hz

# Time-varying rate
event_times = []
t = 0
for i in range(n_events):
    # Rate varies sinusoidally: 500 to 1500 Hz
    current_rate = base_rate + 500 * np.sin(2 * np.pi * t / 100)
    dt = np.random.exponential(1 / current_rate)
    event_times.append(t)
    t += dt

time_deltas = np.diff(np.concatenate([[0], event_times]))
energies = np.random.gamma(2, 500, size=n_events)

# Create time series
ts_variable = SpectralTimeSeries.from_list_mode(
    time_deltas, energies,
    integration_time=5.0,
    energy_bins=256
)

# Analyze rate variation
rates = [spec.metadata['n_events'] / spec.real_time for spec in ts_variable]
print(f"Count rate variation:")
print(f"  Mean: {np.mean(rates):.1f} Hz")
print(f"  Range: {np.min(rates):.1f} - {np.max(rates):.1f} Hz")
print(f"  Std dev: {np.std(rates):.1f} Hz")


# ============================================
# Example 8: Gap Detection
# ============================================
print("\n8. Stride Time > Integration Time (Gaps)")
print("-" * 60)

# Create windows with gaps between them
ts_gaps = SpectralTimeSeries.from_list_mode(
    time_deltas, energies,
    integration_time=5.0,   # 5 second windows
    stride_time=10.0,       # 10 second stride (5 second gaps)
    energy_bins=256
)

print(f"Integration time: 5.0 seconds")
print(f"Stride time: 10.0 seconds")
print(f"Gap between windows: 5.0 seconds")
print(f"Number of spectra: {ts_gaps.n_spectra}")

# Verify gaps
if ts_gaps.n_spectra > 1:
    timestamp_spacing = np.diff(ts_gaps.timestamps)
    print(f"Average timestamp spacing: {np.mean(timestamp_spacing):.1f} seconds")


print("\n" + "=" * 60)
print("List mode examples complete!")
print("=" * 60)

