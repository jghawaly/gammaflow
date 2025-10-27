"""
Tests for Region of Interest (ROI) operations.

Tests cover:
- EnergyROI creation and properties
- ROI integration with spectra
- ROI rebinning for spectra and time series
- Overlapping ROIs
- Non-consecutive ROIs
- ROI collections and utilities
"""

import numpy as np
import pytest

from gammaflow import Spectrum, SpectralTimeSeries
from gammaflow.operations import (
    EnergyROI,
    rebin_spectrum_rois,
    rebin_time_series_rois,
    create_roi_collection,
    check_roi_overlaps
)


class TestEnergyROICreation:
    """Test EnergyROI creation and validation."""
    
    def test_create_basic_roi(self):
        """Test creating a basic ROI."""
        roi = EnergyROI(e_min=100, e_max=200)
        
        assert roi.e_min == 100
        assert roi.e_max == 200
        assert roi.label is None
        assert roi.method == "manual"
    
    def test_create_roi_with_label(self):
        """Test creating ROI with label."""
        roi = EnergyROI(e_min=100, e_max=200, label="K-40 Peak")
        
        assert roi.label == "K-40 Peak"
    
    def test_create_roi_with_method(self):
        """Test creating ROI with method."""
        roi = EnergyROI(e_min=100, e_max=200, method="Censored Energy Windows")
        
        assert roi.method == "Censored Energy Windows"
    
    def test_create_roi_with_metadata(self):
        """Test creating ROI with metadata."""
        metadata = {'peak_fwhm': 5.0, 'confidence': 0.95}
        roi = EnergyROI(e_min=100, e_max=200, metadata=metadata)
        
        assert roi.metadata['peak_fwhm'] == 5.0
        assert roi.metadata['confidence'] == 0.95
    
    def test_invalid_roi_raises(self):
        """Test that invalid ROI raises error."""
        with pytest.raises(ValueError, match="e_min.*must be < e_max"):
            EnergyROI(e_min=200, e_max=100)
        
        with pytest.raises(ValueError):
            EnergyROI(e_min=100, e_max=100)


class TestEnergyROIProperties:
    """Test EnergyROI properties and methods."""
    
    def test_roi_width(self):
        """Test ROI width calculation."""
        roi = EnergyROI(e_min=100, e_max=200)
        assert roi.width == 100
    
    def test_roi_center(self):
        """Test ROI center calculation."""
        roi = EnergyROI(e_min=100, e_max=200)
        assert roi.center == 150
    
    def test_contains_energy(self):
        """Test checking if energy is in ROI."""
        roi = EnergyROI(e_min=100, e_max=200)
        
        assert roi.contains(100)  # Inclusive
        assert roi.contains(150)
        assert roi.contains(200)  # Inclusive
        assert not roi.contains(99)
        assert not roi.contains(201)
    
    def test_overlaps_detection(self):
        """Test overlap detection between ROIs."""
        roi1 = EnergyROI(e_min=100, e_max=200)
        roi2 = EnergyROI(e_min=150, e_max=250)  # Overlaps
        roi3 = EnergyROI(e_min=300, e_max=400)  # No overlap
        
        assert roi1.overlaps(roi2)
        assert roi2.overlaps(roi1)  # Symmetric
        assert not roi1.overlaps(roi3)
        assert not roi3.overlaps(roi1)
    
    def test_roi_repr(self):
        """Test ROI string representation."""
        roi = EnergyROI(e_min=100, e_max=200, label="Test", method="manual")
        repr_str = repr(roi)
        
        assert "Test" in repr_str
        assert "100" in repr_str
        assert "200" in repr_str
        assert "manual" in repr_str


class TestEnergyROISerialization:
    """Test ROI serialization."""
    
    def test_to_dict(self):
        """Test converting ROI to dictionary."""
        roi = EnergyROI(
            e_min=100,
            e_max=200,
            label="Test ROI",
            method="manual",
            metadata={'key': 'value'}
        )
        
        d = roi.to_dict()
        
        assert d['e_min'] == 100
        assert d['e_max'] == 200
        assert d['label'] == "Test ROI"
        assert d['method'] == "manual"
        assert d['metadata']['key'] == 'value'
    
    def test_from_dict(self):
        """Test creating ROI from dictionary."""
        d = {
            'e_min': 100,
            'e_max': 200,
            'label': "Test ROI",
            'method': "manual",
            'metadata': {'key': 'value'}
        }
        
        roi = EnergyROI.from_dict(d)
        
        assert roi.e_min == 100
        assert roi.e_max == 200
        assert roi.label == "Test ROI"
        assert roi.method == "manual"
        assert roi.metadata['key'] == 'value'
    
    def test_roundtrip(self):
        """Test to_dict -> from_dict roundtrip."""
        original = EnergyROI(
            e_min=100,
            e_max=200,
            label="Test",
            method="auto",
            metadata={'a': 1, 'b': 2}
        )
        
        restored = EnergyROI.from_dict(original.to_dict())
        
        assert restored.e_min == original.e_min
        assert restored.e_max == original.e_max
        assert restored.label == original.label
        assert restored.method == original.method
        assert restored.metadata == original.metadata


class TestROISpectrumIntegration:
    """Test integrating spectra over ROIs."""
    
    def test_integrate_single_roi(self):
        """Test integrating spectrum over a single ROI."""
        # Create calibrated spectrum
        energy_edges = np.linspace(0, 1000, 1001)
        counts = np.ones(1000) * 100  # Uniform counts
        spec = Spectrum(counts, energy_edges=energy_edges)
        
        # ROI from 100 to 200 keV (should capture ~100 bins * 100 counts)
        roi = EnergyROI(e_min=100, e_max=200)
        integrated = roi.integrate_spectrum(spec)
        
        assert integrated == pytest.approx(10000, rel=0.1)
    
    def test_integrate_multiple_rois(self):
        """Test integrating spectrum over multiple ROIs."""
        energy_edges = np.linspace(0, 1000, 1001)
        counts = np.ones(1000) * 100
        spec = Spectrum(counts, energy_edges=energy_edges)
        
        rois = [
            EnergyROI(e_min=100, e_max=200, label="ROI 1"),
            EnergyROI(e_min=500, e_max=600, label="ROI 2")
        ]
        
        results = [roi.integrate_spectrum(spec) for roi in rois]
        
        # Both ROIs should have similar counts (uniform spectrum)
        assert results[0] == pytest.approx(results[1], rel=0.01)


class TestRebinSpectrumROIs:
    """Test rebinning spectra using ROIs."""
    
    def test_rebin_basic(self):
        """Test basic spectrum rebinning with ROIs."""
        energy_edges = np.linspace(0, 1000, 1001)
        counts = np.ones(1000) * 100
        spec = Spectrum(counts, energy_edges=energy_edges)
        
        rois = [
            EnergyROI(e_min=100, e_max=200, label="Low"),
            EnergyROI(e_min=500, e_max=600, label="Mid"),
            EnergyROI(e_min=900, e_max=1000, label="High")
        ]
        
        rebinned = rebin_spectrum_rois(spec, rois)
        
        assert len(rebinned) == 3
        assert all(rebinned > 0)
    
    def test_rebin_with_labels(self):
        """Test rebinning with label return."""
        energy_edges = np.linspace(0, 1000, 1001)
        counts = np.ones(1000) * 100
        spec = Spectrum(counts, energy_edges=energy_edges)
        
        rois = [
            EnergyROI(e_min=100, e_max=200, label="K-40"),
            EnergyROI(e_min=500, e_max=600, label="Cs-137")
        ]
        
        rebinned, labels = rebin_spectrum_rois(spec, rois, return_labels=True)
        
        assert len(rebinned) == 2
        assert labels == ["K-40", "Cs-137"]
    
    def test_rebin_unlabeled_rois(self):
        """Test rebinning with unlabeled ROIs generates default labels."""
        energy_edges = np.linspace(0, 1000, 1001)
        counts = np.ones(1000) * 100
        spec = Spectrum(counts, energy_edges=energy_edges)
        
        rois = [
            EnergyROI(e_min=100, e_max=200),  # No label
            EnergyROI(e_min=500, e_max=600)   # No label
        ]
        
        rebinned, labels = rebin_spectrum_rois(spec, rois, return_labels=True)
        
        assert labels == ["ROI_0", "ROI_1"]
    
    def test_rebin_requires_calibrated_spectrum(self):
        """Test that rebinning requires calibrated spectrum."""
        # Uncalibrated spectrum
        counts = np.ones(1000) * 100
        spec = Spectrum(counts)  # No energy_edges
        
        rois = [EnergyROI(e_min=100, e_max=200)]
        
        with pytest.raises(ValueError, match="must be energy-calibrated"):
            rebin_spectrum_rois(spec, rois)
    
    def test_rebin_requires_rois(self):
        """Test that at least one ROI is required."""
        energy_edges = np.linspace(0, 1000, 1001)
        counts = np.ones(1000) * 100
        spec = Spectrum(counts, energy_edges=energy_edges)
        
        with pytest.raises(ValueError, match="Must provide at least one ROI"):
            rebin_spectrum_rois(spec, [])


class TestROIOverlapping:
    """Test overlapping ROI functionality."""
    
    def test_overlapping_rois_allowed(self):
        """Test that overlapping ROIs are allowed."""
        energy_edges = np.linspace(0, 1000, 1001)
        counts = np.ones(1000) * 100
        spec = Spectrum(counts, energy_edges=energy_edges)
        
        # Overlapping ROIs
        rois = [
            EnergyROI(e_min=100, e_max=200, label="A"),
            EnergyROI(e_min=150, e_max=250, label="B")  # Overlaps with A
        ]
        
        # Should work without error
        rebinned = rebin_spectrum_rois(spec, rois)
        assert len(rebinned) == 2
        
        # Overlapping region counts twice
        assert rebinned[0] > 0
        assert rebinned[1] > 0
    
    def test_check_overlaps_function(self):
        """Test checking for overlaps."""
        rois = [
            EnergyROI(e_min=100, e_max=200, label="A"),
            EnergyROI(e_min=150, e_max=250, label="B"),
            EnergyROI(e_min=300, e_max=400, label="C")
        ]
        
        overlaps = check_roi_overlaps(rois)
        
        assert len(overlaps) == 1
        assert overlaps[0] == (0, 1)  # A and B overlap


class TestROINonConsecutive:
    """Test non-consecutive ROI functionality."""
    
    def test_non_consecutive_rois(self):
        """Test that ROIs don't need to be consecutive."""
        energy_edges = np.linspace(0, 1000, 1001)
        counts = np.ones(1000) * 100
        spec = Spectrum(counts, energy_edges=energy_edges)
        
        # Non-consecutive ROIs with gaps
        rois = [
            EnergyROI(e_min=100, e_max=150, label="Peak 1"),
            EnergyROI(e_min=400, e_max=450, label="Peak 2"),  # Gap
            EnergyROI(e_min=800, e_max=850, label="Peak 3")   # Gap
        ]
        
        rebinned = rebin_spectrum_rois(spec, rois)
        
        assert len(rebinned) == 3
        # All should have similar counts (uniform spectrum)
        assert rebinned[0] == pytest.approx(rebinned[1], rel=0.1)
        assert rebinned[1] == pytest.approx(rebinned[2], rel=0.1)


class TestRebinTimeSeriesROIs:
    """Test rebinning time series using ROIs."""
    
    def test_rebin_time_series_basic(self):
        """Test basic time series rebinning."""
        # Create time series
        counts = np.random.poisson(100, size=(20, 1000))
        energy_edges = np.linspace(0, 1000, 1001)
        ts = SpectralTimeSeries.from_array(counts, energy_edges=energy_edges)
        
        rois = [
            EnergyROI(e_min=100, e_max=200, label="ROI 1"),
            EnergyROI(e_min=500, e_max=600, label="ROI 2")
        ]
        
        rebinned = rebin_time_series_rois(ts, rois)
        
        assert rebinned.shape == (20, 2)  # 20 spectra, 2 ROIs
        assert np.all(rebinned > 0)
    
    def test_rebin_time_series_with_labels(self):
        """Test time series rebinning with labels."""
        counts = np.random.poisson(100, size=(10, 1000))
        energy_edges = np.linspace(0, 1000, 1001)
        ts = SpectralTimeSeries.from_array(counts, energy_edges=energy_edges)
        
        rois = [
            EnergyROI(e_min=100, e_max=200, label="K-40"),
            EnergyROI(e_min=500, e_max=600, label="Cs-137"),
            EnergyROI(e_min=800, e_max=900, label="Co-60")
        ]
        
        rebinned, labels = rebin_time_series_rois(ts, rois, return_labels=True)
        
        assert rebinned.shape == (10, 3)
        assert labels == ["K-40", "Cs-137", "Co-60"]
    
    def test_time_series_roi_time_evolution(self):
        """Test analyzing time evolution of ROIs."""
        # Create time series with time-varying counts
        n_spectra = 50
        energy_edges = np.linspace(0, 1000, 1001)
        
        # Simulate increasing counts over time in specific energy range
        counts = np.random.poisson(100, size=(n_spectra, 1000))
        # Add increasing trend in 500-600 keV range
        for i in range(n_spectra):
            counts[i, 500:600] += i * 10
        
        ts = SpectralTimeSeries.from_array(counts, energy_edges=energy_edges)
        
        rois = [
            EnergyROI(e_min=100, e_max=200, label="Stable"),
            EnergyROI(e_min=500, e_max=600, label="Increasing")
        ]
        
        rebinned = rebin_time_series_rois(ts, rois)
        
        # Check that "Increasing" ROI shows trend
        stable_roi = rebinned[:, 0]
        increasing_roi = rebinned[:, 1]
        
        assert np.mean(increasing_roi) > np.mean(stable_roi)


class TestCreateROICollection:
    """Test ROI collection creation utilities."""
    
    def test_create_from_tuples(self):
        """Test creating ROIs from simple tuples."""
        roi_defs = [
            (100, 200, "Low"),
            (500, 600, "Mid"),
            (900, 1000, "High")
        ]
        
        rois = create_roi_collection(roi_defs)
        
        assert len(rois) == 3
        assert rois[0].label == "Low"
        assert rois[1].label == "Mid"
        assert rois[2].label == "High"
        assert all(roi.method == "manual" for roi in rois)
    
    def test_create_without_labels(self):
        """Test creating ROIs without labels."""
        roi_defs = [
            (100, 200),
            (500, 600)
        ]
        
        rois = create_roi_collection(roi_defs)
        
        assert len(rois) == 2
        assert rois[0].label is None
        assert rois[1].label is None
    
    def test_create_with_method(self):
        """Test creating ROIs with shared method."""
        roi_defs = [
            (100, 200, "Peak 1"),
            (500, 600, "Peak 2")
        ]
        
        rois = create_roi_collection(roi_defs, method="peak_search")
        
        assert all(roi.method == "peak_search" for roi in rois)
    
    def test_create_with_shared_metadata(self):
        """Test creating ROIs with shared metadata."""
        roi_defs = [
            (100, 200, "A"),
            (500, 600, "B")
        ]
        
        shared_meta = {'algorithm': 'auto', 'confidence': 0.95}
        rois = create_roi_collection(roi_defs, shared_metadata=shared_meta)
        
        assert all(roi.metadata['algorithm'] == 'auto' for roi in rois)
        assert all(roi.metadata['confidence'] == 0.95 for roi in rois)
    
    def test_invalid_definition_raises(self):
        """Test that invalid definitions raise error."""
        roi_defs = [
            (100, 200, "A"),
            (500,)  # Invalid - only one value
        ]
        
        with pytest.raises(ValueError, match="must be"):
            create_roi_collection(roi_defs)


class TestROIRealWorldScenarios:
    """Test realistic ROI use cases."""
    
    def test_gamma_spectroscopy_analysis(self):
        """Test typical gamma spectroscopy workflow."""
        # Simulate Cs-137 + K-40 spectrum
        energy_edges = np.linspace(0, 1500, 1501)
        counts = np.ones(1500) * 50  # Flat background
        
        # Add Cs-137 peak at 661.7 keV
        cs137_idx = int(661.7 * 1500 / 1500)
        counts[cs137_idx-5:cs137_idx+5] += 500  # Strong peak
        
        # Add K-40 peak at 1460.8 keV
        k40_idx = int(1460.8 * 1500 / 1500)
        counts[k40_idx-5:k40_idx+5] += 300  # Medium peak
        
        spec = Spectrum(counts, energy_edges=energy_edges)
        
        # Define ROIs for peaks (same width for fair comparison)
        rois = [
            EnergyROI(e_min=650, e_max=675, label="Cs-137 (661.7 keV)"),
            EnergyROI(e_min=1450, e_max=1475, label="K-40 (1460.8 keV)"),
            EnergyROI(e_min=100, e_max=125, label="Background")  # Same 25 keV width
        ]
        
        counts_roi, labels = rebin_spectrum_rois(spec, rois, return_labels=True)
        
        # Peak ROIs should have more counts than background (same width)
        assert counts_roi[0] > counts_roi[2]  # Cs-137 > Background
        assert counts_roi[1] > counts_roi[2]  # K-40 > Background
    
    def test_censored_energy_windows(self):
        """Test Censored Energy Windows method for background estimation."""
        energy_edges = np.linspace(0, 2000, 2001)
        counts = np.random.poisson(100, size=2000)
        spec = Spectrum(counts, energy_edges=energy_edges)
        
        # Define censored windows (avoiding known peaks)
        censored_rois = create_roi_collection(
            [
                (100, 500, "Window 1"),
                (700, 1200, "Window 2"),
                (1500, 1900, "Window 3")
            ],
            method="Censored Energy Windows",
            shared_metadata={'purpose': 'background_estimation'}
        )
        
        # Check method is stored
        assert all(roi.method == "Censored Energy Windows" for roi in censored_rois)
        assert all(roi.metadata['purpose'] == 'background_estimation' for roi in censored_rois)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

