# GammaFlow

A Python library for working with time series of gamma ray spectra and listmode data.

## Features

### Core Classes

- **Spectrum**: Comprehensive base class for gamma ray spectra
  - Optional energy calibration (uncalibrated = channel mode)
  - Arithmetic operations (add, subtract, scalar multiply/divide)
  - Energy operations (rebinning, calibration, slicing)
  - Analysis methods (integration, normalization)
  - Uncertainty propagation (Poisson statistics by default)
  - Optional live_time (falls back to real_time for count rates)
  
- **SpectralTimeSeries**: Efficient time series of spectra
  - Shared calibration mode (memory efficient)
  - Independent calibration mode (flexible)
  - NumPy array integration for vectorized operations
  - True Spectrum objects with individual metadata
  - Copy-on-write for safe modifications
  - Auto-detection of integration_time and stride_time
  - Reintegration to coarser time scales (multiples only)
  - Create from 2D arrays or list mode data

- **ListMode**: Event-by-event data handling
  - Encapsulates time deltas and energies
  - Filtering and slicing operations
  - Conversion to SpectralTimeSeries

### Operations

- **Energy ROIs (Regions of Interest)**
  - Define labeled energy windows for analysis
  - Rebin spectra by integrating over ROIs
  - Support for overlapping and non-consecutive ROIs
  - Track creation method (manual, peak search, censored windows, etc.)
  - Time series analysis with ROIs

### Performance

- **Vectorized Operations**: Leverage NumPy for performance
  - Direct array access via `.counts` property
  - Broadcasting and fancy indexing support
  - Efficient batch processing

## Installation

```bash
pip install -e .
```

## Quick Start

### Basic Spectrum Operations

```python
import numpy as np
from gammaflow import Spectrum, SpectralTimeSeries

# Create a spectrum
counts = np.random.poisson(100, size=1024)
spectrum = Spectrum(counts, real_time=10.0)

# Apply energy calibration
energy_edges = np.linspace(0, 3000, 1025)  # 0-3000 keV
calibrated = Spectrum(counts, energy_edges=energy_edges)

# Arithmetic operations
combined = spectrum1 + spectrum2
scaled = spectrum * 2.0

# Energy operations
roi = spectrum.slice_energy(e_min=500, e_max=1000)
integrated = spectrum.integrate(e_min=600, e_max=700)
normalized = spectrum.normalize(method='area')

# Count rate (uses live_time if available, else real_time)
rate = spectrum.count_rate
```

### Time Series Creation

```python
# Method 1: From list of Spectrum objects
spectra = [Spectrum(np.random.poisson(100, 1024), real_time=1.0) for _ in range(100)]
ts = SpectralTimeSeries(spectra)

# Method 2: From 2D array (most common)
counts = np.random.poisson(100, size=(100, 1024))
timestamps = np.arange(100) * 1.0
real_times = np.ones(100) * 1.0
energy_edges = np.linspace(0, 3000, 1025)

ts = SpectralTimeSeries.from_array(
    counts,
    energy_edges=energy_edges,
    timestamps=timestamps,
    real_times=real_times
)

# Method 3: From list mode data
from gammaflow import ListMode

time_deltas = np.random.exponential(0.001, size=100000)
energies = np.random.gamma(2, 500, size=100000)

# Direct from arrays
ts = SpectralTimeSeries.from_list_mode(
    time_deltas, energies,
    integration_time=10.0,
    stride_time=10.0,
    energy_bins=1024
)

# Or using ListMode object (for filtering first)
listmode = ListMode(time_deltas, energies)
filtered = listmode.filter_energy(e_min=200, e_max=800)
ts = SpectralTimeSeries.from_list_mode(filtered, integration_time=10.0)
```

### Time Series Operations

```python
# Timing is auto-detected from data
print(f"Integration time: {ts.integration_time}")  # Auto-detected from real_time
print(f"Stride time: {ts.stride_time}")            # Auto-detected from timestamps

# Reintegrate to coarser time resolution (must be even multiple)
ts_20s = ts.reintegrate(new_integration_time=20.0)  # 2x coarser
ts_40s = ts.reintegrate(new_integration_time=40.0)  # 4x coarser

# Vectorized operations
background = np.mean(ts.counts, axis=0)
ts.counts[:] -= background  # In-place modification

# Time slicing
ts_subset = ts.slice_time(t_min=100, t_max=500)

# Object-oriented access
for spec in ts:
    if spec.timestamp > 200:
        spec.metadata['processed'] = True
```

### ROI Analysis

```python
from gammaflow.operations import EnergyROI, rebin_spectrum_rois, rebin_time_series_rois

# Define regions of interest
rois = [
    EnergyROI(e_min=655, e_max=668, label="Cs-137 (661.7 keV)"),
    EnergyROI(e_min=1450, e_max=1470, label="K-40 (1460.8 keV)"),
    EnergyROI(e_min=1165, e_max=1180, label="Co-60 Peak 1"),
    EnergyROI(e_min=1325, e_max=1340, label="Co-60 Peak 2")
]

# Integrate single spectrum over ROIs
roi_counts, labels = rebin_spectrum_rois(spectrum, rois, return_labels=True)
print(f"Cs-137 counts: {roi_counts[0]}")

# Analyze time evolution of ROIs
roi_ts = rebin_time_series_rois(time_series, rois)
# Shape: (n_spectra, n_rois) - rows are time, columns are ROIs

# Plot K-40 evolution
import matplotlib.pyplot as plt
plt.plot(time_series.timestamps, roi_ts[:, 1])
plt.xlabel("Time (s)")
plt.ylabel("K-40 Peak Counts")
```

### Advanced Features

```python
# Overlapping ROIs (useful for peak fitting)
overlapping_rois = [
    EnergyROI(e_min=650, e_max=670, label="Peak + Background"),
    EnergyROI(e_min=655, e_max=665, label="Peak Core")
]

# Non-consecutive ROIs (gaps allowed - perfect for peaks)
peak_rois = [
    EnergyROI(e_min=661, e_max=663, label="Cs-137"),
    EnergyROI(e_min=1460, e_max=1462, label="K-40")  # Large gap
]

# Censored Energy Windows for background estimation
from gammaflow.operations import create_roi_collection

censored = create_roi_collection([
    (100, 600, "Low Window"),
    (700, 1400, "Mid Window"),
    (1500, 1900, "High Window")
], method="Censored Energy Windows")

# NumPy integration for advanced analysis
total_counts = np.sum(ts.counts, axis=1)
spectral_std = np.std(ts.counts, axis=0)
peak_locations = np.argmax(ts.counts, axis=1)
```

## Project Structure

```
gammaflow/
├── gammaflow/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── calibration.py
│   │   ├── spectrum.py
│   │   └── time_series.py
│   ├── operations/
│   │   ├── __init__.py
│   │   ├── energy.py
│   │   └── temporal.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── exceptions.py
│   └── visualization/
│       └── __init__.py
├── tests/
├── examples/
└── README.md
```

## Examples

See the `examples/` directory for comprehensive examples:
- `basic_usage.py` - Spectrum and time series basics
- `advanced_usage.py` - Advanced operations and workflows
- `list_mode_example.py` - Working with list mode data
- `listmode_class_example.py` - ListMode class usage
- `reintegrate_example.py` - Time resolution reintegration
- `timing_inference_example.py` - Automatic timing detection
- `roi_example.py` - Energy ROI analysis

## Documentation

- `PROJECT_OVERVIEW.md` - High-level architecture
- `INSTALLATION.md` - Installation instructions
- `REINTEGRATE_FEATURE.md` - Time reintegration details
- `ROI_FEATURE.md` - Energy ROI documentation
- `TIMING_INFERENCE_UPDATE.md` - Automatic timing detection
- `tests/README.md` - Test suite documentation

## Key Concepts

### Energy Calibration
- Spectra can be calibrated (with energy_edges) or uncalibrated (channel mode)
- Shared calibration in time series for memory efficiency
- Auto-detection of compatible calibrations

### Timing
- `real_time`: Clock time (including dead time)
- `live_time`: Actual counting time (excluding dead time)
- `integration_time`: Time window width for binning
- `stride_time`: Time between consecutive windows
- Auto-detection from consistent data

### ROIs (Regions of Interest)
- Define labeled energy windows for analysis
- Can overlap (for peak fitting, background estimation)
- Can be non-consecutive (for peak-only analysis)
- Track creation method and metadata

### List Mode
- Event-by-event data: time deltas + energies
- Filter and slice before binning
- Convert to time series with flexible windowing

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Current status: **287 tests passing** ✓

## License

MIT

