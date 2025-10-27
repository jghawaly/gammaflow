"""
Advanced usage examples for GammaFlow.

This script demonstrates advanced features like rebinning, filtering,
and complex time series operations.
"""

import numpy as np
from gammaflow import Spectrum, SpectralTimeSeries

print("=" * 60)
print("GammaFlow Advanced Usage Examples")
print("=" * 60)

# ============================================
# Example 1: Energy Rebinning
# ============================================
print("\n1. Energy Rebinning")
print("-" * 60)

# Create high-resolution spectrum
energy_edges_fine = np.linspace(0, 3000, 3001)  # 1 keV bins
counts = np.random.poisson(lam=50, size=3000)
spec_fine = Spectrum(counts, energy_edges=energy_edges_fine)

print(f"Fine spectrum: {spec_fine.n_bins} bins")

# Rebin to coarser resolution
energy_edges_coarse = np.linspace(0, 3000, 301)  # 10 keV bins
spec_coarse = spec_fine.rebin_energy(energy_edges_coarse)

print(f"Coarse spectrum: {spec_coarse.n_bins} bins")
print(f"Fine total counts: {np.sum(spec_fine.counts):.0f}")
print(f"Coarse total counts: {np.sum(spec_coarse.counts):.0f}")
print(f"Counts conserved: {np.isclose(np.sum(spec_fine.counts), np.sum(spec_coarse.counts))}")

# ============================================
# Example 2: Custom Vectorized Operations
# ============================================
print("\n2. Custom Vectorized Operations")
print("-" * 60)

# Create time series
n_times = 200
n_bins = 512
spectra = [
    Spectrum(np.random.poisson(lam=100, size=n_bins), timestamp=float(i))
    for i in range(n_times)
]
ts = SpectralTimeSeries(spectra)

# Apply custom vectorized function
def smooth_and_normalize(counts):
    """Apply smoothing and normalize."""
    from scipy.ndimage import gaussian_filter1d
    # Smooth along energy axis
    smoothed = gaussian_filter1d(counts, sigma=2.0, axis=1)
    # Normalize each spectrum
    normalized = smoothed / smoothed.sum(axis=1, keepdims=True)
    return normalized

ts_processed = ts.apply_vectorized(smooth_and_normalize)
print(f"Applied custom vectorized operation to {ts.n_spectra} spectra")
print(f"Each spectrum now normalized to sum = 1.0")
print(f"Example sum: {np.sum(ts_processed.counts[0]):.6f}")

# ============================================
# Example 3: Per-Spectrum Operations
# ============================================
print("\n3. Per-Spectrum Operations")
print("-" * 60)

# Create spectra with varying properties
spectra = []
for i in range(50):
    counts = np.random.poisson(lam=50 + i*2, size=256)
    spec = Spectrum(counts, live_time=1.0 + i*0.1)
    spec.metadata['quality'] = 'good' if i % 3 == 0 else 'fair'
    spectra.append(spec)

ts = SpectralTimeSeries(spectra)

# Apply function to each spectrum
def process_spectrum(spec: Spectrum) -> Spectrum:
    """Convert to count rate and smooth."""
    # Use count_rate property for rate normalization
    rate_spectrum = Spectrum(
        counts=spec.count_rate,
        energy_edges=spec.energy_edges,
        uncertainty=spec.uncertainty / (spec.live_time if spec.live_time else spec.real_time),
        metadata=spec.metadata
    )
    # Could add smoothing here
    return rate_spectrum

ts_processed = ts.apply_to_each(process_spectrum)
print(f"Applied per-spectrum processing to {ts.n_spectra} spectra")

# Filter based on metadata
ts_good = ts.filter_spectra(lambda s: s.metadata['quality'] == 'good')
print(f"Filtered to {len(ts_good)} 'good' quality spectra")

# Filter based on statistics
ts_high = ts.filter_spectra(lambda s: np.sum(s.counts) > 5000)
print(f"Filtered to {len(ts_high)} high-count spectra")

# ============================================
# Example 4: Time Rebinning
# ============================================
print("\n4. Time Rebinning")
print("-" * 60)

# Create time series with fine time resolution
spectra = []
for i in range(100):
    counts = np.random.poisson(lam=20, size=128)
    spec = Spectrum(counts, timestamp=float(i), live_time=1.0)
    spectra.append(spec)

ts_fine = SpectralTimeSeries(spectra)
print(f"Fine time series: {ts_fine.n_spectra} spectra")

# Rebin in time (integrate every 10 spectra)
ts_coarse = ts_fine.rebin_time(integration_time=10.0, stride=10.0)
print(f"Coarse time series: {ts_coarse.n_spectra} spectra")

# With overlap
ts_overlap = ts_fine.rebin_time(integration_time=10.0, stride=5.0)
print(f"Overlapping time series: {ts_overlap.n_spectra} spectra")

# ============================================
# Example 5: Background Estimation
# ============================================
print("\n5. Background Subtraction Methods")
print("-" * 60)

# Create time series with background + signal
spectra = []
background_level = 50
signal_start = 40
signal_end = 60

for i in range(100):
    bg = np.random.poisson(lam=background_level, size=256)
    
    # Add signal in middle period
    if signal_start <= i < signal_end:
        signal = np.random.poisson(lam=100, size=256)  # Strong signal
        counts = bg + signal
    else:
        counts = bg
    
    spectra.append(Spectrum(counts, timestamp=float(i)))

ts = SpectralTimeSeries(spectra)

# Method 1: Mean background
ts_mean_sub = ts.background_subtract('mean')
print("Background subtracted using mean")

# Method 2: Median background (robust to outliers)
ts_median_sub = ts.background_subtract('median')
print("Background subtracted using median")

# Method 3: Custom background (from known background region)
background_spectra = [ts[i] for i in range(20)]  # First 20 spectra
background_ts = SpectralTimeSeries(background_spectra)
background_spectrum = background_ts.mean_spectrum()

ts_custom_sub = ts.background_subtract(background_spectrum)
print("Background subtracted using custom spectrum")

# Compare
print(f"Original mean counts: {np.mean(np.sum(ts.counts, axis=1)):.0f}")
print(f"After mean subtraction: {np.mean(np.sum(ts_mean_sub.counts, axis=1)):.0f}")
print(f"After median subtraction: {np.mean(np.sum(ts_median_sub.counts, axis=1)):.0f}")

# ============================================
# Example 6: Spectrum Statistics
# ============================================
print("\n6. Spectrum Statistics")
print("-" * 60)

# Create spectrum with calibration
energy_edges = np.linspace(0, 2000, 1025)
counts = np.random.poisson(lam=100, size=1024)
# Add a peak
counts[500:520] += np.random.poisson(lam=500, size=20)

spec = Spectrum(counts, energy_edges=energy_edges)

# Compute statistics using numpy (more flexible than a rigid method)
total = np.sum(spec.counts)
mean_energy = np.average(spec.energy_centers, weights=spec.counts)
std_energy = np.sqrt(np.average((spec.energy_centers - mean_energy)**2, weights=spec.counts))
max_counts = np.max(spec.counts)
max_idx = np.argmax(spec.counts)
max_energy = spec.energy_centers[max_idx]

print("Spectrum statistics:")
print(f"  total_counts: {total:.2f}")
print(f"  mean_energy: {mean_energy:.2f}")
print(f"  std_energy: {std_energy:.2f}")
print(f"  max_counts: {max_counts:.2f}")
print(f"  max_energy: {max_energy:.2f}")

# ============================================
# Example 7: Normalization Methods
# ============================================
print("\n7. Normalization Methods")
print("-" * 60)

spec = Spectrum(np.random.poisson(lam=50, size=256))

# Area normalization
spec_area = spec.normalize('area')
print(f"Area normalized: {np.sum(spec_area.counts):.6f} (should be 1.0)")

# Peak normalization
spec_peak = spec.normalize('peak')
print(f"Peak normalized: {np.max(spec_peak.counts):.6f} (should be 1.0)")

# Count rate calculation (replaces live_time normalization)
count_rate = spec.count_rate
print(f"Count rate: {np.sum(count_rate):.2f} counts/sec")

# ============================================
# Example 8: Uncertainty Propagation
# ============================================
print("\n8. Uncertainty Propagation")
print("-" * 60)

# Create spectra with explicit uncertainties
counts1 = np.array([100, 200, 300])
uncertainty1 = np.array([10, 14, 17])  # sqrt of counts
spec1 = Spectrum(counts1, uncertainty=uncertainty1)

counts2 = np.array([50, 100, 150])
uncertainty2 = np.array([7, 10, 12])
spec2 = Spectrum(counts2, uncertainty=uncertainty2)

print(f"Spec1 counts: {spec1.counts}")
print(f"Spec1 uncertainty: {spec1.uncertainty}")

# Addition propagates uncertainty
spec_sum = spec1 + spec2
print(f"\nSum counts: {spec_sum.counts}")
print(f"Sum uncertainty: {spec_sum.uncertainty}")
print(f"Expected: sqrt(σ₁² + σ₂²) = {np.sqrt(uncertainty1**2 + uncertainty2**2)}")

# Subtraction
spec_diff = spec1 - spec2
print(f"\nDifference counts: {spec_diff.counts}")
print(f"Difference uncertainty: {spec_diff.uncertainty}")

# ============================================
# Example 9: Conversion Between Calibration Modes
# ============================================
print("\n9. Converting Between Calibration Modes")
print("-" * 60)

# Create time series with independent calibrations
spectra = [Spectrum(np.random.poisson(100, size=256)) for _ in range(50)]
ts_indep = SpectralTimeSeries(spectra, shared_calibration=False)
print(f"Independent mode: uses_shared = {ts_indep.uses_shared_calibration}")
print(f"Memory usage (approx): {ts_indep.counts.nbytes / 1024:.1f} KB for counts")

# Convert to shared calibration
ts_shared = ts_indep.to_shared_calibration()
print(f"\nShared mode: uses_shared = {ts_shared.uses_shared_calibration}")
print("Memory efficient for large time series!")

# Convert back
ts_indep2 = ts_shared.to_independent_calibration()
print(f"\nBack to independent: uses_shared = {ts_indep2.uses_shared_calibration}")

# ============================================
# Example 10: Complex Analysis Pipeline
# ============================================
print("\n10. Complex Analysis Pipeline")
print("-" * 60)

# Create simulated data
print("Creating simulated data...")
spectra = []
for i in range(200):
    # Simulate varying background and signal
    background = np.random.poisson(lam=30, size=512)
    
    # Occasional peaks
    if i % 20 == 0:
        peak_loc = np.random.randint(100, 400)
        background[peak_loc:peak_loc+5] += np.random.poisson(lam=200, size=5)
    
    spec = Spectrum(
        counts=background,
        timestamp=float(i * 10),  # 10 second intervals
        live_time=9.8,
        real_time=10.0
    )
    spec.metadata['run_number'] = i // 50  # 4 runs
    spectra.append(spec)

ts = SpectralTimeSeries(spectra)
print(f"Created time series: {ts}")

# Step 1: Apply calibration
print("\nStep 1: Applying calibration...")
ts = ts.apply_calibration([0, 0.5, 0.0001])

# Step 2: Convert to count rates
print("Step 2: Converting to count rates...")
# Use vectorized operation for count rate normalization
times = np.array([s.live_time if s.live_time else s.real_time for s in ts])
ts = ts.apply_vectorized(lambda counts: counts / times[:, np.newaxis])

# Step 3: Background subtraction
print("Step 3: Subtracting background...")
ts = ts.background_subtract('median')

# Step 4: Filter out low-quality spectra
print("Step 4: Filtering spectra...")
ts = ts.filter_spectra(lambda s: s.live_time / s.real_time > 0.95)
print(f"Remaining spectra: {ts.n_spectra}")

# Step 5: Group by run and compute means
print("Step 5: Computing mean spectra per run...")
runs = {}
for spec in ts:
    run = spec.metadata['run_number']
    if run not in runs:
        runs[run] = []
    runs[run].append(spec)

mean_spectra = {}
for run, specs in runs.items():
    run_ts = SpectralTimeSeries(specs)
    mean_spectra[run] = run_ts.mean_spectrum()
    print(f"  Run {run}: {len(specs)} spectra, mean counts = {np.sum(mean_spectra[run].counts):.0f}")

print("\n" + "=" * 60)
print("Advanced Examples Complete!")
print("=" * 60)

