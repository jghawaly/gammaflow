"""
Tests for Spectrum class.
"""

import numpy as np
import pytest
from gammaflow import Spectrum
from gammaflow.utils.exceptions import SpectrumError, IncompatibleBinningError, CalibrationError


class TestSpectrumCreation:
    """Test Spectrum creation and validation."""
    
    def test_create_uncalibrated(self, simple_counts):
        """Test creating uncalibrated spectrum."""
        spec = Spectrum(simple_counts)
        assert spec.n_bins == 4
        assert not spec.is_calibrated
        assert np.array_equal(spec.counts, simple_counts)
    
    def test_create_calibrated(self, simple_counts):
        """Test creating calibrated spectrum."""
        edges = np.array([0, 100, 200, 300, 400])
        spec = Spectrum(simple_counts, energy_edges=edges)
        assert spec.is_calibrated
        assert np.array_equal(spec.energy_edges, edges)
    
    def test_create_with_metadata(self, simple_counts):
        """Test creating spectrum with metadata."""
        metadata = {'detector': 'HPGe', 'source': 'Co-60'}
        spec = Spectrum(simple_counts, metadata=metadata)
        assert spec.metadata['detector'] == 'HPGe'
        assert spec.metadata['source'] == 'Co-60'
    
    def test_create_with_timestamp(self, simple_counts):
        """Test creating spectrum with timestamp."""
        spec = Spectrum(simple_counts, timestamp=12345.0)
        assert spec.timestamp == 12345.0
    
    def test_create_with_times(self, simple_counts):
        """Test creating spectrum with live and real time."""
        spec = Spectrum(simple_counts, live_time=10.0, real_time=12.0)
        assert spec.live_time == 10.0
        assert spec.real_time == 12.0
        assert np.isclose(spec.dead_time_fraction, 1 - 10/12)
    
    def test_validation_edge_count_mismatch(self, simple_counts):
        """Test validation catches edge count mismatch."""
        bad_edges = np.array([0, 1, 2])  # Only 3 edges for 4 bins!
        with pytest.raises(SpectrumError, match="Energy edges length"):
            Spectrum(simple_counts, energy_edges=bad_edges)
    
    def test_validation_negative_live_time(self, simple_counts):
        """Test validation catches negative live time."""
        with pytest.raises(SpectrumError, match="Live time must be non-negative"):
            Spectrum(simple_counts, live_time=-1.0)
    
    def test_validation_real_time_less_than_live(self, simple_counts):
        """Test validation catches real < live time."""
        with pytest.raises(SpectrumError, match="Real time must be >= live time"):
            Spectrum(simple_counts, live_time=10.0, real_time=5.0)


class TestSpectrumProperties:
    """Test Spectrum properties."""
    
    def test_counts_property(self, spectrum_uncalibrated):
        """Test counts property."""
        assert isinstance(spectrum_uncalibrated.counts, np.ndarray)
        assert len(spectrum_uncalibrated.counts) == 4
    
    def test_energy_edges_uncalibrated(self, spectrum_uncalibrated):
        """Test energy edges for uncalibrated returns channels."""
        edges = spectrum_uncalibrated.energy_edges
        assert np.array_equal(edges, np.array([0, 1, 2, 3, 4]))
    
    def test_energy_edges_calibrated(self, spectrum_calibrated):
        """Test energy edges for calibrated."""
        edges = spectrum_calibrated.energy_edges
        assert edges[0] == 0
        assert edges[-1] == 400
    
    def test_energy_centers(self, spectrum_calibrated):
        """Test energy centers computation."""
        centers = spectrum_calibrated.energy_centers
        expected = np.array([50, 150, 250, 350])
        assert np.array_equal(centers, expected)
    
    def test_energy_widths(self, spectrum_calibrated):
        """Test energy widths computation."""
        widths = spectrum_calibrated.energy_widths
        assert np.all(widths == 100)
    
    def test_uncertainty_poisson(self, simple_counts):
        """Test Poisson uncertainty (sqrt(N))."""
        spec = Spectrum(simple_counts)
        expected = np.sqrt(simple_counts)
        assert np.allclose(spec.uncertainty, expected)
    
    def test_uncertainty_explicit(self, spectrum_with_uncertainty):
        """Test explicit uncertainty."""
        assert np.array_equal(
            spectrum_with_uncertainty.uncertainty,
            np.array([10, 14, 12, 17])
        )
    
    def test_count_rate(self, spectrum_calibrated):
        """Test count rate calculation."""
        rate = spectrum_calibrated.count_rate
        expected = np.array([100, 200, 150, 300]) / 10.0
        assert np.allclose(rate, expected)
    
    def test_count_density(self, spectrum_calibrated):
        """Test count density calculation."""
        density = spectrum_calibrated.count_density
        # Counts / width (all widths = 100)
        expected = np.array([100, 200, 150, 300]) / 100.0
        assert np.allclose(density, expected)
    
    def test_is_view_false_by_default(self, spectrum_uncalibrated):
        """Test is_view is False by default."""
        assert spectrum_uncalibrated.is_view is False


class TestSpectrumArithmetic:
    """Test Spectrum arithmetic operations."""
    
    def test_add_spectra(self):
        """Test adding two spectra."""
        spec1 = Spectrum(np.array([100, 200, 300]))
        spec2 = Spectrum(np.array([50, 100, 150]))
        result = spec1 + spec2
        assert np.allclose(result.counts, [150, 300, 450])
    
    def test_add_scalar(self):
        """Test adding scalar to spectrum."""
        spec = Spectrum(np.array([100, 200, 300]))
        result = spec + 50
        assert np.allclose(result.counts, [150, 250, 350])
    
    def test_radd_scalar(self):
        """Test right addition."""
        spec = Spectrum(np.array([100, 200, 300]))
        result = 50 + spec
        assert np.allclose(result.counts, [150, 250, 350])
    
    def test_subtract_spectra(self):
        """Test subtracting spectra."""
        spec1 = Spectrum(np.array([100, 200, 300]))
        spec2 = Spectrum(np.array([50, 100, 150]))
        result = spec1 - spec2
        assert np.allclose(result.counts, [50, 100, 150])
    
    def test_subtract_scalar(self):
        """Test subtracting scalar."""
        spec = Spectrum(np.array([100, 200, 300]))
        result = spec - 50
        assert np.allclose(result.counts, [50, 150, 250])
    
    def test_multiply_scalar(self):
        """Test multiplying by scalar."""
        spec = Spectrum(np.array([100, 200, 300]))
        result = spec * 2
        assert np.allclose(result.counts, [200, 400, 600])
    
    def test_multiply_spectrum_raises_error(self):
        """Test multiplying two spectra raises error."""
        spec1 = Spectrum(np.array([2, 3, 4]))
        spec2 = Spectrum(np.array([5, 6, 7]))
        
        with pytest.raises(TypeError, match="Multiplying two spectra is not supported"):
            spec1 * spec2
    
    def test_divide_scalar(self):
        """Test dividing by scalar."""
        spec = Spectrum(np.array([100, 200, 300]))
        result = spec / 2
        assert np.allclose(result.counts, [50, 100, 150])
    
    def test_divide_spectrum_raises_error(self):
        """Test dividing two spectra raises error."""
        spec1 = Spectrum(np.array([100, 200, 300]))
        spec2 = Spectrum(np.array([10, 20, 30]))
        
        with pytest.raises(TypeError, match="Dividing two spectra is not supported"):
            spec1 / spec2
    
    def test_incompatible_binning_raises_error(self):
        """Test arithmetic with incompatible binning raises error."""
        spec1 = Spectrum(np.array([100, 200, 300]), energy_edges=[0, 1, 2, 3])
        spec2 = Spectrum(np.array([50, 100, 150]), energy_edges=[0, 2, 4, 6])
        
        with pytest.raises(IncompatibleBinningError):
            spec1 + spec2
    
    def test_uncertainty_propagation_addition(self):
        """Test uncertainty propagates correctly in addition."""
        spec1 = Spectrum(np.array([100, 200]), uncertainty=np.array([10, 14]))
        spec2 = Spectrum(np.array([50, 100]), uncertainty=np.array([7, 10]))
        result = spec1 + spec2
        
        # σ² = σ₁² + σ₂²
        expected = np.sqrt(10**2 + 7**2), np.sqrt(14**2 + 10**2)
        assert np.allclose(result.uncertainty, expected)


class TestSpectrumCalibration:
    """Test Spectrum calibration methods."""
    
    def test_apply_calibration_linear(self):
        """Test applying linear calibration."""
        spec = Spectrum(np.array([100, 200, 300, 400]))
        calibrated = spec.apply_calibration([0, 0.5])
        
        assert calibrated.is_calibrated
        assert calibrated.energy_edges[0] == 0.0
        assert calibrated.energy_edges[-1] == 2.0
    
    def test_apply_calibration_quadratic(self):
        """Test applying quadratic calibration."""
        spec = Spectrum(np.array([100, 200, 300]))
        calibrated = spec.apply_calibration([0, 1, 0.1])
        
        # E = 0 + 1*ch + 0.1*ch²
        expected = np.array([0, 1.1, 2.4, 3.9])
        assert np.allclose(calibrated.energy_edges, expected)
    
    def test_apply_calibration_inplace(self):
        """Test applying calibration in-place."""
        spec = Spectrum(np.array([100, 200, 300]))
        spec.apply_calibration_([0, 0.5])
        
        assert spec.is_calibrated
        assert spec.energy_edges[-1] == 1.5
    
    def test_from_channels_factory(self):
        """Test creating spectrum from channels."""
        counts = np.array([100, 200, 300, 400])
        spec = Spectrum.from_channels(counts, [0, 0.5], live_time=10.0)
        
        assert spec.is_calibrated
        assert spec.live_time == 10.0
    
    def test_to_channels(self):
        """Test removing calibration."""
        spec = Spectrum(np.array([100, 200]), energy_edges=[0, 100, 200])
        uncalibrated = spec.to_channels()
        
        assert not uncalibrated.is_calibrated
        assert np.array_equal(uncalibrated.energy_edges, [0, 1, 2])


class TestSpectrumEnergyOperations:
    """Test Spectrum energy operations."""
    
    def test_slice_energy(self):
        """Test energy slicing."""
        edges = np.linspace(0, 1000, 11)  # 0, 100, 200, ..., 1000
        counts = np.ones(10) * 5
        spec = Spectrum(counts, energy_edges=edges)
        
        sliced = spec.slice_energy(200, 500)
        assert sliced.n_bins == 4  # 200-300, 300-400, 400-500, 500-600 (inclusive)
        assert np.sum(sliced.counts) == 20  # 4 bins * 5 counts
    
    def test_slice_energy_no_bounds(self):
        """Test slicing without bounds returns full spectrum."""
        spec = Spectrum(np.ones(10), energy_edges=np.linspace(0, 100, 11))
        sliced = spec.slice_energy()
        assert sliced.n_bins == spec.n_bins
    
    def test_integrate(self):
        """Test integration."""
        edges = np.linspace(0, 100, 11)
        counts = np.ones(10) * 5
        spec = Spectrum(counts, energy_edges=edges)
        
        total = spec.integrate()
        assert total == 50  # 10 bins * 5 counts
    
    def test_integrate_range(self):
        """Test integration over range."""
        edges = np.linspace(0, 100, 11)
        counts = np.ones(10) * 5
        spec = Spectrum(counts, energy_edges=edges)
        
        partial = spec.integrate(e_min=20, e_max=50)
        assert partial == 20  # 4 bins * 5 counts (20-30, 30-40, 40-50, 50-60)
    
    def test_rebin_energy(self):
        """Test energy rebinning."""
        edges = np.linspace(0, 100, 11)  # 10 bins of width 10
        counts = np.ones(10) * 5
        spec = Spectrum(counts, energy_edges=edges)
        
        new_edges = np.linspace(0, 100, 6)  # 5 bins of width 20
        rebinned = spec.rebin_energy(new_edges)
        
        assert rebinned.n_bins == 5
        # Total counts should be approximately conserved
        assert np.isclose(np.sum(rebinned.counts), np.sum(spec.counts), rtol=0.2)
    
    def test_rebin_energy_inplace(self):
        """Test in-place rebinning."""
        edges = np.linspace(0, 100, 11)
        counts = np.ones(10) * 5
        spec = Spectrum(counts, energy_edges=edges)
        
        new_edges = np.linspace(0, 100, 6)
        original_total = np.sum(spec.counts)
        spec.rebin_energy_(new_edges)
        
        assert spec.n_bins == 5
        assert np.isclose(np.sum(spec.counts), original_total, rtol=0.2)


class TestSpectrumAnalysis:
    """Test Spectrum analysis methods."""
    
    def test_normalize_area(self):
        """Test area normalization."""
        spec = Spectrum(np.array([100, 200, 300]))
        normalized = spec.normalize('area')
        
        assert np.isclose(np.sum(normalized.counts), 1.0)
    
    def test_normalize_peak(self):
        """Test peak normalization."""
        spec = Spectrum(np.array([100, 200, 300]))
        normalized = spec.normalize('peak')
        
        assert np.isclose(np.max(normalized.counts), 1.0)
    
    def test_count_rate_property(self):
        """Test count rate property (replaces live_time normalization)."""
        spec = Spectrum(np.array([100, 200, 300]), live_time=10.0)
        
        # count_rate property gives counts per second
        expected = np.array([100, 200, 300]) / 10.0
        assert np.allclose(spec.count_rate, expected)
    
    def test_normalize_invalid_method(self):
        """Test invalid normalization method raises error."""
        spec = Spectrum(np.array([100, 200, 300]))
        
        with pytest.raises(ValueError, match="Unknown normalization method"):
            spec.normalize('invalid_method')
    


class TestSpectrumCopyDetach:
    """Test Spectrum copy and detach operations."""
    
    def test_copy_deep(self, spectrum_calibrated):
        """Test deep copy."""
        spec_copy = spectrum_calibrated.copy(deep=True)
        
        # Modify copy
        spec_copy.counts[0] = 9999
        
        # Original unchanged
        assert spectrum_calibrated.counts[0] != 9999
    
    def test_copy_shallow(self, spectrum_calibrated):
        """Test shallow copy shares calibration."""
        spec_copy = spectrum_calibrated.copy(deep=False)
        
        assert spec_copy._calibration is spectrum_calibrated._calibration
    
    def test_detach(self, spectrum_calibrated):
        """Test detach method."""
        spec = spectrum_calibrated
        spec.detach()
        
        assert not spec.is_view
        assert not spec.has_shared_calibration


class TestSpectrumNumpyInterface:
    """Test Spectrum numpy interface."""
    
    def test_array_protocol(self, spectrum_uncalibrated):
        """Test __array__ protocol."""
        arr = np.array(spectrum_uncalibrated)
        assert isinstance(arr, np.ndarray)
        assert np.array_equal(arr, spectrum_uncalibrated.counts)
    
    def test_len(self, spectrum_uncalibrated):
        """Test __len__."""
        assert len(spectrum_uncalibrated) == 4
    
    def test_getitem_single(self, spectrum_uncalibrated):
        """Test single index access."""
        value = spectrum_uncalibrated[2]
        assert value == 150
    
    def test_getitem_slice(self):
        """Test slice access."""
        spec = Spectrum(np.array([100, 200, 300, 400, 500]))
        sliced = spec[1:4]
        
        assert isinstance(sliced, Spectrum)
        assert sliced.n_bins == 3
        assert np.array_equal(sliced.counts, [200, 300, 400])


class TestSpectrumRepr:
    """Test Spectrum string representation."""
    
    def test_repr_uncalibrated(self, spectrum_uncalibrated):
        """Test repr for uncalibrated spectrum."""
        repr_str = repr(spectrum_uncalibrated)
        assert "Spectrum" in repr_str
        assert "uncalibrated" in repr_str
        assert "channels" in repr_str
    
    def test_repr_calibrated(self, spectrum_calibrated):
        """Test repr for calibrated spectrum."""
        repr_str = repr(spectrum_calibrated)
        assert "Spectrum" in repr_str
        assert "calibrated" in repr_str
        assert "keV" in repr_str

