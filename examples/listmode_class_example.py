"""
Example: Working with the ListMode Class

This example demonstrates the ListMode class for storing and manipulating
event-by-event gamma ray detection data.
"""

import numpy as np
from gammaflow.core.listmode import ListMode
from gammaflow.core.time_series import SpectralTimeSeries


# ============================================
# Example 1: Creating ListMode Objects
# ============================================
print("1. Creating ListMode Objects")
print("-" * 60)

# Generate synthetic event data
np.random.seed(42)
time_deltas = np.random.exponential(0.001, size=50000)  # ~1000 Hz
energies = np.random.gamma(2, 500, size=50000)  # Gamma-like spectrum

# Create ListMode object
lm = ListMode(time_deltas, energies)

print(f"Created: {lm}")
print(f"Total events: {lm.n_events}")
print(f"Duration: {lm.total_time:.2f} seconds")
print(f"Mean rate: {lm.mean_rate:.1f} Hz")
print(f"Energy range: {lm.energy_range[0]:.1f} - {lm.energy_range[1]:.1f} keV")


# ============================================
# Example 2: Adding Metadata
# ============================================
print("\n2. ListMode with Metadata")
print("-" * 60)

metadata = {
    'detector': 'HPGe',
    'run_id': 12345,
    'date': '2025-10-27',
    'source': 'Cs-137'
}

lm_with_meta = ListMode(time_deltas, energies, metadata=metadata)

print(f"Detector: {lm_with_meta.metadata['detector']}")
print(f"Run ID: {lm_with_meta.metadata['run_id']}")
print(f"Source: {lm_with_meta.metadata['source']}")


# ============================================
# Example 3: Energy Filtering
# ============================================
print("\n3. Energy Filtering")
print("-" * 60)

# Filter to energy ROI
roi_min, roi_max = 400, 800
lm_roi = lm.filter_energy(e_min=roi_min, e_max=roi_max)

print(f"Original: {lm.n_events} events")
print(f"ROI [{roi_min}-{roi_max} keV]: {lm_roi.n_events} events")
print(f"Fraction in ROI: {lm_roi.n_events / lm.n_events:.1%}")

# Filter with only minimum
lm_high = lm.filter_energy(e_min=1000)
print(f"Above 1000 keV: {lm_high.n_events} events")


# ============================================
# Example 4: Time Slicing
# ============================================
print("\n4. Time Slicing")
print("-" * 60)

# Extract first 10 seconds
lm_slice = lm.slice_time(t_min=0, t_max=10.0)

print(f"Original duration: {lm.total_time:.2f} seconds")
print(f"Sliced duration: {lm_slice.total_time:.2f} seconds")
print(f"Events in slice: {lm_slice.n_events}")
print(f"Mean rate in slice: {lm_slice.mean_rate:.1f} Hz")

# Extract middle portion
t_start, t_end = 20, 30
lm_middle = lm.slice_time(t_min=t_start, t_max=t_end)
print(f"\nMiddle slice [{t_start}-{t_end}s]: {lm_middle.n_events} events")


# ============================================
# Example 5: Chaining Filters
# ============================================
print("\n5. Chaining Filters")
print("-" * 60)

# Filter by energy, then slice time
lm_filtered = (lm
    .filter_energy(e_min=300, e_max=900)
    .slice_time(t_min=10, t_max=40))

print(f"Original: {lm.n_events} events, {lm.total_time:.1f}s")
print(f"After energy filter + time slice: {lm_filtered.n_events} events")
print(f"Duration: {lm_filtered.total_time:.1f}s")
print(f"Mean rate: {lm_filtered.mean_rate:.1f} Hz")


# ============================================
# Example 6: Converting to SpectralTimeSeries
# ============================================
print("\n6. Converting to SpectralTimeSeries")
print("-" * 60)

# Method 1: Direct conversion
ts1 = SpectralTimeSeries.from_list_mode(
    lm,
    integration_time=5.0,
    energy_bins=512
)

print(f"Method 1 - Direct conversion:")
print(f"  Created {ts1.n_spectra} spectra")
print(f"  {ts1.n_bins} energy bins")

# Method 2: Filter first, then convert
lm_filtered = lm.filter_energy(e_min=200, e_max=1200)
ts2 = SpectralTimeSeries.from_list_mode(
    lm_filtered,
    integration_time=10.0,
    energy_bins=256,
    energy_range=(200, 1200)
)

print(f"\nMethod 2 - With pre-filtering:")
print(f"  Filtered to {lm_filtered.n_events} events")
print(f"  Created {ts2.n_spectra} spectra")
print(f"  Energy range: {ts2.energy_edges[0]:.0f}-{ts2.energy_edges[-1]:.0f} keV")


# ============================================
# Example 7: Comparing Original vs Filtered
# ============================================
print("\n7. Comparing Original vs Filtered")
print("-" * 60)

# Create time series from full data
ts_full = SpectralTimeSeries.from_list_mode(
    lm,
    integration_time=10.0,
    energy_bins=256
)

# Create time series from high-energy events only
lm_high_e = lm.filter_energy(e_min=800)
ts_high = SpectralTimeSeries.from_list_mode(
    lm_high_e,
    integration_time=10.0,
    energy_bins=256
)

print(f"Full data time series:")
print(f"  Mean counts per spectrum: {np.mean([np.sum(s.counts) for s in ts_full]):.0f}")

print(f"\nHigh-energy (>800 keV) time series:")
print(f"  Mean counts per spectrum: {np.mean([np.sum(s.counts) for s in ts_high]):.0f}")
print(f"  Reduction factor: {np.mean([np.sum(s.counts) for s in ts_full]) / np.mean([np.sum(s.counts) for s in ts_high]):.1f}x")


# ============================================
# Example 8: Working with Absolute Times
# ============================================
print("\n8. Working with Absolute Times")
print("-" * 60)

# Access absolute times (computed once and cached)
abs_times = lm.absolute_times

print(f"First event at: {abs_times[0]:.6f} s")
print(f"Last event at: {abs_times[-1]:.2f} s")
print(f"Time span: {abs_times[-1] - abs_times[0]:.2f} s")

# Inter-event time statistics
print(f"\nTime delta statistics:")
print(f"  Mean: {np.mean(lm.time_deltas)*1000:.3f} ms")
print(f"  Std: {np.std(lm.time_deltas)*1000:.3f} ms")
print(f"  Min: {np.min(lm.time_deltas)*1000:.3f} ms")
print(f"  Max: {np.max(lm.time_deltas)*1000:.3f} ms")


# ============================================
# Example 9: Copying ListMode Objects
# ============================================
print("\n9. Copying ListMode Objects")
print("-" * 60)

# Create a copy
lm_copy = lm.copy()

print(f"Original: {lm.n_events} events")
print(f"Copy: {lm_copy.n_events} events")

# Modify copy doesn't affect original
lm_filtered_copy = lm_copy.filter_energy(e_min=500)
print(f"\nAfter filtering copy:")
print(f"  Original still has: {lm.n_events} events")
print(f"  Filtered copy has: {lm_filtered_copy.n_events} events")


# ============================================
# Example 10: Real-World Workflow
# ============================================
print("\n10. Real-World Workflow Example")
print("-" * 60)

# Simulate a real workflow
print("Workflow: Background subtraction using list mode")

# Step 1: Create background and source measurements
bg_time_deltas = np.random.exponential(0.01, size=5000)  # 100 Hz background
bg_energies = np.random.uniform(0, 1500, size=5000)
lm_background = ListMode(bg_time_deltas, bg_energies)

src_time_deltas = np.random.exponential(0.002, size=25000)  # 500 Hz with source
src_energies = np.random.gamma(2, 500, size=25000)  # Source peaks
lm_source = ListMode(src_time_deltas, src_energies)

print(f"Background: {lm_background.mean_rate:.1f} Hz, {lm_background.total_time:.1f}s")
print(f"Source: {lm_source.mean_rate:.1f} Hz, {lm_source.total_time:.1f}s")

# Step 2: Convert to spectra
ts_bg = SpectralTimeSeries.from_list_mode(
    lm_background,
    integration_time=lm_background.total_time,  # Single spectrum
    energy_bins=512
)

ts_src = SpectralTimeSeries.from_list_mode(
    lm_source,
    integration_time=lm_source.total_time,  # Single spectrum
    energy_bins=512
)

# Step 3: Background subtraction
bg_spectrum = ts_bg[0]
src_spectrum = ts_src[0]

# Normalize by time
bg_rate = bg_spectrum.counts / bg_spectrum.real_time
src_rate = src_spectrum.counts / src_spectrum.real_time
net_rate = src_rate - bg_rate

print(f"\nBackground rate: {np.sum(bg_rate):.1f} cps")
print(f"Source+BG rate: {np.sum(src_rate):.1f} cps")
print(f"Net source rate: {np.sum(net_rate):.1f} cps")


print("\n" + "=" * 60)
print("ListMode class examples complete!")
print("=" * 60)

