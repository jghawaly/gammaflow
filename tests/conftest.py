"""
Pytest configuration and fixtures for GammaFlow tests.
"""

import numpy as np
import pytest
from gammaflow import Spectrum, SpectralTimeSeries
from gammaflow.core.calibration import EnergyCalibration


# ============================================
# Fixtures for EnergyCalibration
# ============================================

@pytest.fixture
def uncalibrated_edges():
    """Fixture for uncalibrated (None) edges."""
    return None


@pytest.fixture
def simple_edges():
    """Fixture for simple energy edges."""
    return np.array([0.0, 1.0, 2.0, 3.0, 4.0])


@pytest.fixture
def calibration_linear():
    """Fixture for linear energy calibration."""
    return EnergyCalibration(np.linspace(0, 511.5, 1025))


@pytest.fixture
def calibration_quadratic():
    """Fixture for quadratic energy calibration."""
    n_bins = 512
    channels = np.arange(n_bins + 1)
    edges = 0 + 0.5 * channels + 0.001 * channels**2
    return EnergyCalibration(edges)


# ============================================
# Fixtures for Spectrum
# ============================================

@pytest.fixture
def simple_counts():
    """Fixture for simple counts array."""
    return np.array([100, 200, 150, 300], dtype=float)


@pytest.fixture
def random_counts():
    """Fixture for random Poisson counts."""
    np.random.seed(42)
    return np.random.poisson(lam=100, size=512)


@pytest.fixture
def spectrum_uncalibrated(simple_counts):
    """Fixture for uncalibrated spectrum."""
    return Spectrum(simple_counts, live_time=10.0)


@pytest.fixture
def spectrum_calibrated(simple_counts):
    """Fixture for calibrated spectrum."""
    edges = np.array([0, 100, 200, 300, 400])
    return Spectrum(simple_counts, energy_edges=edges, live_time=10.0)


@pytest.fixture
def spectrum_with_uncertainty(simple_counts):
    """Fixture for spectrum with explicit uncertainty."""
    edges = np.array([0, 1, 2, 3, 4])
    uncertainty = np.array([10, 14, 12, 17], dtype=float)
    return Spectrum(simple_counts, energy_edges=edges, uncertainty=uncertainty)


@pytest.fixture
def spectrum_large():
    """Fixture for large spectrum."""
    np.random.seed(42)
    counts = np.random.poisson(lam=50, size=1024)
    edges = np.linspace(0, 3000, 1025)
    return Spectrum(counts, energy_edges=edges, live_time=100.0, real_time=105.0)


# ============================================
# Fixtures for SpectralTimeSeries
# ============================================

@pytest.fixture
def small_spectra_list():
    """Fixture for small list of spectra."""
    np.random.seed(42)
    spectra = []
    for i in range(10):
        counts = np.random.poisson(lam=50, size=64)
        spec = Spectrum(counts, timestamp=float(i), live_time=1.0)
        spec.metadata['index'] = i
        spectra.append(spec)
    return spectra


@pytest.fixture
def time_series_small(small_spectra_list):
    """Fixture for small time series."""
    return SpectralTimeSeries(small_spectra_list)


@pytest.fixture
def calibrated_spectra_list():
    """Fixture for calibrated spectra list."""
    np.random.seed(42)
    edges = np.linspace(0, 1000, 257)
    spectra = []
    for i in range(20):
        counts = np.random.poisson(lam=100, size=256)
        spec = Spectrum(counts, energy_edges=edges, timestamp=float(i * 10))
        spectra.append(spec)
    return spectra


@pytest.fixture
def time_series_calibrated(calibrated_spectra_list):
    """Fixture for calibrated time series."""
    return SpectralTimeSeries(calibrated_spectra_list)


@pytest.fixture
def mixed_calibration_spectra():
    """Fixture for spectra with different calibrations."""
    np.random.seed(42)
    spectra = []
    for i in range(5):
        counts = np.random.poisson(lam=50, size=128)
        edges = np.linspace(0, 1000 + i * 10, 129)  # Different edges
        spec = Spectrum(counts, energy_edges=edges)
        spectra.append(spec)
    return spectra


# ============================================
# Parametrize helpers
# ============================================

@pytest.fixture(params=[
    'polynomial',
    'linear',
])
def calibration_model(request):
    """Parametrize calibration models."""
    return request.param


@pytest.fixture(params=[
    'area',
    'peak',
])
def normalization_method(request):
    """Parametrize normalization methods."""
    return request.param


@pytest.fixture(params=[True, False])
def shared_calibration_mode(request):
    """Parametrize shared calibration mode."""
    return request.param

