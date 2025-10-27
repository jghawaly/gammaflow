"""
Basic tests for GammaFlow functionality.
"""

import numpy as np
import pytest
from gammaflow import Spectrum, SpectralTimeSeries


def test_spectrum_creation():
    """Test basic spectrum creation."""
    counts = np.array([100, 200, 150, 300])
    spec = Spectrum(counts)
    
    assert spec.n_bins == 4
    assert not spec.is_calibrated
    assert np.array_equal(spec.counts, counts)


def test_spectrum_calibration():
    """Test spectrum calibration."""
    counts = np.array([100, 200, 150, 300])
    spec = Spectrum(counts)
    
    # Apply calibration
    calibrated = spec.apply_calibration([0, 0.5])
    
    assert calibrated.is_calibrated
    assert calibrated.energy_edges[0] == 0.0
    assert calibrated.energy_edges[-1] == 2.0  # 4 bins * 0.5


def test_spectrum_arithmetic():
    """Test spectrum arithmetic operations."""
    spec1 = Spectrum(np.array([100, 200, 300]))
    spec2 = Spectrum(np.array([50, 100, 150]))
    
    # Addition
    sum_spec = spec1 + spec2
    assert np.allclose(sum_spec.counts, [150, 300, 450])
    
    # Subtraction
    diff_spec = spec1 - spec2
    assert np.allclose(diff_spec.counts, [50, 100, 150])
    
    # Scalar multiplication
    mult_spec = spec1 * 2
    assert np.allclose(mult_spec.counts, [200, 400, 600])
    
    # Scalar division
    div_spec = spec1 / 2
    assert np.allclose(div_spec.counts, [50, 100, 150])


def test_time_series_creation():
    """Test time series creation."""
    spectra = [Spectrum(np.random.poisson(100, size=128)) for _ in range(10)]
    ts = SpectralTimeSeries(spectra)
    
    assert ts.n_spectra == 10
    assert ts.n_bins == 128
    assert ts.counts.shape == (10, 128)


def test_time_series_shared_calibration():
    """Test shared calibration mode."""
    spectra = [Spectrum(np.random.poisson(100, size=64)) for _ in range(20)]
    ts = SpectralTimeSeries(spectra, shared_calibration=True)
    
    assert ts.uses_shared_calibration
    
    # Check that spectra reference shared calibration
    spec = ts[0]
    assert spec.has_shared_calibration


def test_time_series_vectorized_operations():
    """Test vectorized operations on time series."""
    spectra = [Spectrum(np.ones(10) * i) for i in range(5)]
    ts = SpectralTimeSeries(spectra)
    
    # Background subtraction
    background = np.ones(10) * 2
    ts_sub = ts.background_subtract(background)
    
    assert ts_sub.n_spectra == 5
    assert np.allclose(ts_sub.counts[0], -2 * np.ones(10))


def test_shared_memory():
    """Test that shared memory works correctly."""
    spectra = [Spectrum(np.ones(10)) for _ in range(5)]
    ts = SpectralTimeSeries(spectra, shared_calibration=True)
    
    # Modify via array
    ts.counts[2, 5] = 999
    
    # Check via spectrum
    spec = ts[2]
    assert spec.counts[5] == 999
    
    # Modify via spectrum
    spec.counts[6] = 888
    
    # Check via array
    assert ts.counts[2, 6] == 888


def test_energy_slicing():
    """Test energy slicing."""
    energy_edges = np.linspace(0, 1000, 101)  # 0 to 1000 keV, 100 bins
    counts = np.ones(100) * 10
    spec = Spectrum(counts, energy_edges=energy_edges)
    
    # Slice 200-400 keV
    roi = spec.slice_energy(200, 400)
    
    assert roi.n_bins == 21  # Bins from 200-400 keV inclusive
    assert np.sum(roi.counts) == 210  # 21 bins * 10 counts


def test_spectrum_integration():
    """Test spectrum integration."""
    energy_edges = np.linspace(0, 100, 101)
    counts = np.ones(100) * 5
    spec = Spectrum(counts, energy_edges=energy_edges)
    
    # Integrate full range
    total = spec.integrate()
    assert total == 500  # 100 bins * 5 counts
    
    # Integrate partial range
    partial = spec.integrate(e_min=20, e_max=40)
    assert partial == 105  # 21 bins * 5 counts (inclusive range)


def test_normalization():
    """Test spectrum normalization."""
    counts = np.array([100, 200, 300, 400])
    spec = Spectrum(counts)
    
    # Area normalization
    norm_spec = spec.normalize('area')
    assert np.isclose(np.sum(norm_spec.counts), 1.0)
    
    # Peak normalization
    peak_spec = spec.normalize('peak')
    assert np.isclose(np.max(peak_spec.counts), 1.0)


def test_time_series_filtering():
    """Test filtering time series."""
    spectra = []
    for i in range(10):
        spec = Spectrum(np.ones(10) * i)
        spec.metadata['index'] = i
        spectra.append(spec)
    
    ts = SpectralTimeSeries(spectra)
    
    # Filter
    filtered = ts.filter_spectra(lambda s: s.metadata['index'] > 5)
    
    assert filtered.n_spectra == 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

