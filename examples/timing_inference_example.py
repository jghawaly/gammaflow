"""
Examples demonstrating automatic timing inference and validation.

This shows how SpectralTimeSeries now automatically detects integration_time
and stride_time from the data, allowing reintegration to work with any
consistently-timed data (not just from list mode).
"""

import numpy as np
from gammaflow import Spectrum, SpectralTimeSeries

print("=" * 70)
print("Timing Inference and Validation Examples")
print("=" * 70)

# =============================================================================
# Example 1: Auto-detection from Spectrum List
# =============================================================================

print("\n" + "=" * 70)
print("Example 1: Auto-detection from Spectrum List")
print("=" * 70)

# Create spectra with constant real_time and evenly-spaced timestamps
spectra = [
    Spectrum(np.random.poisson(100, 64), real_time=1.0, timestamp=i*1.0)
    for i in range(20)
]

ts = SpectralTimeSeries(spectra)

print(f"Created {ts.n_spectra} spectra")
print(f"  integration_time: {ts.integration_time} (auto-detected from real_time)")
print(f"  stride_time: {ts.stride_time} (auto-detected from timestamps)")

# Reintegrate works!
ts_2x = ts.reintegrate(2.0)
print(f"\nReintegrated to 2.0s windows:")
print(f"  {ts.n_spectra} -> {ts_2x.n_spectra} spectra")
print("  ✓ SUCCESS!")

# =============================================================================
# Example 2: Auto-detection from from_array
# =============================================================================

print("\n" + "=" * 70)
print("Example 2: Auto-detection from from_array")
print("=" * 70)

counts = np.random.poisson(100, size=(50, 64))
timestamps = np.arange(50) * 0.5  # Every 0.5 seconds
real_times = np.ones(50) * 0.5     # All 0.5 seconds

ts = SpectralTimeSeries.from_array(
    counts,
    timestamps=timestamps,
    real_times=real_times
)

print(f"Created {ts.n_spectra} spectra from 2D array")
print(f"  integration_time: {ts.integration_time}")
print(f"  stride_time: {ts.stride_time}")

# Reintegrate to 2.0s (4x)
ts_4x = ts.reintegrate(2.0)
print(f"\nReintegrated from 0.5s to 2.0s:")
print(f"  {ts.n_spectra} -> {ts_4x.n_spectra} spectra")
print("  ✓ SUCCESS!")

# =============================================================================
# Example 3: Validation Prevents Errors
# =============================================================================

print("\n" + "=" * 70)
print("Example 3: Validation Prevents Configuration Errors")
print("=" * 70)

spectra = [
    Spectrum(np.random.poisson(100, 64), real_time=1.0, timestamp=i*1.0)
    for i in range(10)
]

# Correct values - works
print("Providing correct values:")
ts = SpectralTimeSeries(spectra, integration_time=1.0, stride_time=1.0)
print(f"  ✓ Accepted: integration_time={ts.integration_time}, stride_time={ts.stride_time}")

# Wrong integration_time - fails
print("\nProviding incorrect integration_time:")
try:
    ts = SpectralTimeSeries(spectra, integration_time=2.0)
    print("  ✗ ERROR: Should have raised ValueError")
except ValueError as e:
    print(f"  ✓ Correctly rejected:")
    print(f"    {str(e)[:80]}...")

# Wrong stride_time - fails
print("\nProviding incorrect stride_time:")
try:
    ts = SpectralTimeSeries(spectra, stride_time=0.5)
    print("  ✗ ERROR: Should have raised ValueError")
except ValueError as e:
    print(f"  ✓ Correctly rejected:")
    print(f"    {str(e)[:80]}...")

# =============================================================================
# Example 4: When Inference Isn't Possible
# =============================================================================

print("\n" + "=" * 70)
print("Example 4: When Timing Can't Be Inferred")
print("=" * 70)

# Case 1: Varying real_times
print("Case 1: Varying real_times (can't infer integration_time)")
spectra_varying = [
    Spectrum(
        np.random.poisson(100, 64),
        real_time=np.random.uniform(0.8, 1.2),
        timestamp=i*1.0
    )
    for i in range(10)
]

ts_varying = SpectralTimeSeries(spectra_varying)
print(f"  integration_time: {ts_varying.integration_time} (can't infer)")
print(f"  stride_time: {ts_varying.stride_time} (can infer from timestamps)")

try:
    ts_varying.reintegrate(2.0)
    print("  ✗ ERROR: Should have failed")
except Exception as e:
    print(f"  ✓ Correctly prevented reintegration")

# Case 2: Irregular timestamps
print("\nCase 2: Irregular timestamps (can't infer stride_time)")
timestamps_irregular = [0, 1, 2, 3, 5, 7, 10, 14, 19, 25]
spectra_irregular = [
    Spectrum(np.random.poisson(100, 64), real_time=1.0, timestamp=t)
    for t in timestamps_irregular
]

ts_irregular = SpectralTimeSeries(spectra_irregular)
print(f"  integration_time: {ts_irregular.integration_time} (can infer from real_time)")
print(f"  stride_time: {ts_irregular.stride_time} (can't infer)")

try:
    ts_irregular.reintegrate(2.0)
    print("  ✗ ERROR: Should have failed")
except Exception as e:
    print(f"  ✓ Correctly prevented reintegration")

# =============================================================================
# Example 5: Works with from_list_mode (original functionality)
# =============================================================================

print("\n" + "=" * 70)
print("Example 5: Still Works with from_list_mode")
print("=" * 70)

time_deltas = np.random.exponential(0.01, 5000)
energies = np.random.uniform(100, 1000, 5000)

ts_listmode = SpectralTimeSeries.from_list_mode(
    time_deltas,
    energies,
    integration_time=0.5,
    stride_time=0.5,
    energy_bins=64
)

print(f"Created {ts_listmode.n_spectra} spectra from list mode")
print(f"  integration_time: {ts_listmode.integration_time}")
print(f"  stride_time: {ts_listmode.stride_time}")

ts_2s = ts_listmode.reintegrate(2.0)
print(f"\nReintegrated from 0.5s to 2.0s:")
print(f"  {ts_listmode.n_spectra} -> {ts_2s.n_spectra} spectra")
print("  ✓ Original functionality preserved!")

# =============================================================================
# Example 6: Progressive Reintegration with Mixed Sources
# =============================================================================

print("\n" + "=" * 70)
print("Example 6: Progressive Reintegration")
print("=" * 70)

# Start with manual spectra
spectra_init = [
    Spectrum(np.random.poisson(100, 64), real_time=0.25, timestamp=i*0.25)
    for i in range(80)
]

ts_0 = SpectralTimeSeries(spectra_init)
print(f"Initial: {ts_0.n_spectra} spectra at {ts_0.integration_time}s")

# Progressive coarsening
ts_1 = ts_0.reintegrate(0.5)
print(f"Step 1: {ts_1.n_spectra} spectra at {ts_1.integration_time}s")

ts_2 = ts_1.reintegrate(1.0)
print(f"Step 2: {ts_2.n_spectra} spectra at {ts_2.integration_time}s")

ts_3 = ts_2.reintegrate(2.0)
print(f"Step 3: {ts_3.n_spectra} spectra at {ts_3.integration_time}s")

ts_4 = ts_3.reintegrate(4.0)
print(f"Step 4: {ts_4.n_spectra} spectra at {ts_4.integration_time}s")

print("\n✓ All steps successful with auto-detected timing!")

# =============================================================================
# Example 7: Explicit Control with Validation
# =============================================================================

print("\n" + "=" * 70)
print("Example 7: Explicit Control (with Validation)")
print("=" * 70)

# Create from array with explicit parameters
counts = np.random.poisson(100, size=(30, 64))
timestamps = np.arange(30) * 0.5
real_times = np.ones(30) * 0.5

print("Creating with explicit, matching parameters:")
ts_explicit = SpectralTimeSeries.from_array(
    counts,
    timestamps=timestamps,
    real_times=real_times,
    integration_time=0.5,  # Matches real_times
    stride_time=0.5        # Matches timestamp spacing
)
print(f"  ✓ Accepted: integration_time={ts_explicit.integration_time}")
print(f"  ✓ Accepted: stride_time={ts_explicit.stride_time}")

print("\nTrying with mismatched parameters:")
try:
    ts_bad = SpectralTimeSeries.from_array(
        counts,
        timestamps=timestamps,
        real_times=real_times,
        integration_time=1.0,  # Doesn't match real_times!
    )
    print("  ✗ ERROR: Should have been rejected")
except ValueError:
    print("  ✓ Correctly rejected mismatched integration_time")

print("\n" + "=" * 70)
print("All Examples Complete!")
print("=" * 70)
print("\nKey Takeaways:")
print("  1. integration_time auto-detected from constant real_time")
print("  2. stride_time auto-detected from evenly-spaced timestamps")
print("  3. Explicit values are validated against data")
print("  4. Reintegrate works with ANY consistently-timed data")
print("  5. Clear error messages when timing can't be inferred")
print("  6. Fully backward compatible with existing code")

