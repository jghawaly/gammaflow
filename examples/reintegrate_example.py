"""
Examples of using the reintegrate() method for SpectralTimeSeries.

The reintegrate() method allows you to combine spectra from a time series
(typically created from list mode) into larger time windows. This is useful
for:
- Creating multiple time resolution views of the same data
- Improving statistics at longer timescales
- Adaptive binning workflows

"""

import numpy as np
from gammaflow import SpectralTimeSeries

# =============================================================================
# Example 1: Basic Reintegration (Double the Integration Time)
# =============================================================================

print("=" * 70)
print("Example 1: Basic Reintegration")
print("=" * 70)

# Create simulated list mode data (1 minute, ~1000 Hz)
np.random.seed(42)
time_deltas = np.random.exponential(0.001, 60000)  # Inter-event times
energies = np.random.gamma(shape=3, scale=200, size=60000)  # Gamma spectrum

# Create time series with fine time resolution (0.1s windows)
ts_fine = SpectralTimeSeries.from_list_mode(
    time_deltas,
    energies,
    integration_time=0.1,
    stride_time=0.1,
    energy_bins=100,
    energy_range=(0, 1500)
)

print(f"Fine resolution: {ts_fine.n_spectra} spectra with {ts_fine.integration_time}s integration")
print(f"Total counts: {np.sum(ts_fine.counts):.0f}")

# Reintegrate to 0.2s windows (2x larger)
ts_2x = ts_fine.reintegrate(new_integration_time=0.2)
print(f"\n2x reintegration: {ts_2x.n_spectra} spectra with {ts_2x.integration_time}s integration")
print(f"Total counts: {np.sum(ts_2x.counts):.0f}")

# =============================================================================
# Example 2: Multiple Time Scales from Same Data
# =============================================================================

print("\n" + "=" * 70)
print("Example 2: Multiple Time Scales")
print("=" * 70)

# Create coarser views at multiple timescales
ts_1s = ts_fine.reintegrate(new_integration_time=1.0)
ts_5s = ts_fine.reintegrate(new_integration_time=5.0)
ts_10s = ts_fine.reintegrate(new_integration_time=10.0)

print(f"0.1s resolution: {ts_fine.n_spectra} spectra")
print(f"1.0s resolution: {ts_1s.n_spectra} spectra")
print(f"5.0s resolution: {ts_5s.n_spectra} spectra")
print(f"10.0s resolution: {ts_10s.n_spectra} spectra")

# All have same total counts (approximately)
print(f"\nCount conservation check:")
print(f"  0.1s: {np.sum(ts_fine.counts):.0f}")
print(f"  1.0s: {np.sum(ts_1s.counts):.0f}")
print(f"  5.0s: {np.sum(ts_5s.counts):.0f}")
print(f"  10.0s: {np.sum(ts_10s.counts):.0f}")

# =============================================================================
# Example 3: Overlapping Windows
# =============================================================================

print("\n" + "=" * 70)
print("Example 3: Overlapping Windows")
print("=" * 70)

# Create time series with 0.5s integration and stride
time_deltas = np.random.exponential(0.01, 5000)
energies = np.random.uniform(100, 1000, 5000)

ts = SpectralTimeSeries.from_list_mode(
    time_deltas,
    energies,
    integration_time=0.5,
    stride_time=0.5,
    energy_bins=50
)

print(f"Original: {ts.n_spectra} spectra (0.5s integration, 0.5s stride)")

# Reintegrate with overlapping windows
# Integration = 2.0s, Stride = 1.0s (windows overlap by 1.0s)
ts_overlap = ts.reintegrate(
    new_integration_time=2.0,
    new_stride_time=1.0
)

print(f"Overlapping: {ts_overlap.n_spectra} spectra (2.0s integration, 1.0s stride)")
print(f"Original total counts: {np.sum(ts.counts):.0f}")
print(f"Overlapping total counts: {np.sum(ts_overlap.counts):.0f}")
print("Note: Overlapping windows count events multiple times, increasing total")

# =============================================================================
# Example 4: Progressive Coarsening Workflow
# =============================================================================

print("\n" + "=" * 70)
print("Example 4: Progressive Coarsening")
print("=" * 70)

# Start with very fine resolution
time_deltas = np.random.exponential(0.01, 10000)
energies = np.random.uniform(100, 1000, 10000)

ts_0 = SpectralTimeSeries.from_list_mode(
    time_deltas,
    energies,
    integration_time=0.25,
    stride_time=0.25,
    energy_bins=64
)

# Progressively coarsen
ts_1 = ts_0.reintegrate(new_integration_time=0.5)
ts_2 = ts_1.reintegrate(new_integration_time=1.0)
ts_3 = ts_2.reintegrate(new_integration_time=2.0)
ts_4 = ts_3.reintegrate(new_integration_time=4.0)

print("Progressive coarsening:")
print(f"  0.25s: {ts_0.n_spectra} spectra")
print(f"  0.50s: {ts_1.n_spectra} spectra")
print(f"  1.00s: {ts_2.n_spectra} spectra")
print(f"  2.00s: {ts_3.n_spectra} spectra")
print(f"  4.00s: {ts_4.n_spectra} spectra")

# =============================================================================
# Example 5: Metadata Preservation
# =============================================================================

print("\n" + "=" * 70)
print("Example 5: Metadata Preservation")
print("=" * 70)

time_deltas = np.random.exponential(0.01, 1000)
energies = np.random.uniform(100, 1000, 1000)

ts = SpectralTimeSeries.from_list_mode(
    time_deltas,
    energies,
    integration_time=0.5,
    stride_time=0.5,
    energy_bins=32
)

ts_reint = ts.reintegrate(new_integration_time=1.0)

# Check metadata for first spectrum
spec = ts_reint.spectra[0]
print(f"First reintegrated spectrum metadata:")
print(f"  Window start: {spec.metadata['window_start']:.3f}s")
print(f"  Window end: {spec.metadata['window_end']:.3f}s")
print(f"  Window width: {spec.metadata['window_end'] - spec.metadata['window_start']:.3f}s")
print(f"  Spectra combined: {spec.metadata['n_spectra_combined']}")
print(f"  Timestamp (center): {spec.timestamp:.3f}s")

# =============================================================================
# Example 6: Validation Examples
# =============================================================================

print("\n" + "=" * 70)
print("Example 6: Validation")
print("=" * 70)

time_deltas = np.random.exponential(0.01, 1000)
energies = np.random.uniform(100, 1000, 1000)

ts = SpectralTimeSeries.from_list_mode(
    time_deltas,
    energies,
    integration_time=0.5,
    stride_time=0.5
)

print("Valid reintegrations (even multiples):")
valid_times = [1.0, 1.5, 2.0, 2.5, 5.0]
for t in valid_times:
    ts_reint = ts.reintegrate(new_integration_time=t)
    print(f"  {t}s: ✓ ({ts_reint.n_spectra} spectra)")

print("\nInvalid reintegrations will raise errors:")
try:
    # Try to reduce integration time (not allowed)
    ts.reintegrate(new_integration_time=0.25)
except ValueError as e:
    print(f"  Reducing time: ✗ ({e})")

try:
    # Try non-multiple (0.7 / 0.5 = 1.4, not integer)
    ts.reintegrate(new_integration_time=0.7)
except ValueError as e:
    print(f"  Non-multiple: ✗ (must be even multiple)")

# =============================================================================
# Example 7: Energy Calibration Preservation
# =============================================================================

print("\n" + "=" * 70)
print("Example 7: Energy Calibration Preservation")
print("=" * 70)

time_deltas = np.random.exponential(0.01, 5000)
energies = np.random.uniform(100, 1000, 5000)

# Create with specific energy calibration
energy_edges = np.linspace(0, 1200, 121)  # 10 keV bins
ts = SpectralTimeSeries.from_list_mode(
    time_deltas,
    energies,
    integration_time=0.5,
    stride_time=0.5,
    energy_bins=energy_edges
)

ts_reint = ts.reintegrate(new_integration_time=2.0)

print(f"Original energy bins: {ts.n_bins}")
print(f"Reintegrated energy bins: {ts_reint.n_bins}")
print(f"Energy edges match: {np.allclose(ts.energy_edges, ts_reint.energy_edges)}")
print("✓ Energy calibration preserved during reintegration")

# =============================================================================
# Example 8: Adaptive Workflow Based on Count Statistics
# =============================================================================

print("\n" + "=" * 70)
print("Example 8: Adaptive Binning Based on Statistics")
print("=" * 70)

# Create data with varying count rates
time_deltas = np.random.exponential(0.01, 10000)
energies = np.random.uniform(100, 1000, 10000)

ts = SpectralTimeSeries.from_list_mode(
    time_deltas,
    energies,
    integration_time=0.1,
    stride_time=0.1,
    energy_bins=64
)

print(f"Initial: {ts.n_spectra} spectra at 0.1s resolution")
print(f"Mean counts per spectrum: {np.mean(np.sum(ts.counts, axis=1)):.1f}")

# Check if we have enough statistics
mean_counts = np.mean(np.sum(ts.counts, axis=1))
if mean_counts < 100:
    # Low counts - reintegrate to improve statistics
    target_counts = 100
    factor = int(np.ceil(target_counts / mean_counts))
    # Round to nearest power of 2 for cleaner multiples
    factor = 2 ** int(np.log2(factor) + 0.5)
    
    new_integration = ts.integration_time * factor
    ts_adaptive = ts.reintegrate(new_integration_time=new_integration)
    
    print(f"\nAdaptive binning applied:")
    print(f"  Factor: {factor}x")
    print(f"  New integration: {new_integration}s")
    print(f"  New spectrum count: {ts_adaptive.n_spectra}")
    print(f"  Mean counts per spectrum: {np.mean(np.sum(ts_adaptive.counts, axis=1)):.1f}")

print("\n" + "=" * 70)
print("Examples complete!")
print("=" * 70)

