"""
Tests for the Spectra base class.
"""

import pytest
import numpy as np
from gammaflow import Spectrum, Spectra


class TestSpectraCreation:
    """Test creating Spectra collections."""
    
    def test_create_from_list(self):
        """Test creating Spectra from list of Spectrum objects."""
        spectra_list = [Spectrum(np.random.poisson(100, 64)) for _ in range(10)]
        spectra = Spectra(spectra_list)
        
        assert spectra.n_spectra == 10
        assert spectra.n_bins == 64
        assert spectra.counts.shape == (10, 64)
    
    def test_create_empty_list_raises(self):
        """Test that empty list raises error."""
        with pytest.raises(ValueError, match="empty list"):
            Spectra([])
    
    def test_validate_same_bins(self):
        """Test validation that all spectra have same number of bins."""
        spectra_list = [
            Spectrum(np.random.poisson(100, 64)),
            Spectrum(np.random.poisson(100, 32)),  # Different size!
        ]
        with pytest.raises(Exception):  # Should raise IncompatibleBinningError
            Spectra(spectra_list)
    
    def test_shared_calibration_mode(self):
        """Test shared calibration mode."""
        edges = np.linspace(0, 3000, 65)
        spectra_list = [
            Spectrum(np.random.poisson(100, 64), energy_edges=edges)
            for _ in range(10)
        ]
        spectra = Spectra(spectra_list, shared_calibration=True)
        
        assert spectra.uses_shared_calibration is True
        assert spectra.is_calibrated is True
        assert len(spectra.energy_edges) == 65
    
    def test_independent_calibration_mode(self):
        """Test independent calibration mode."""
        spectra_list = [Spectrum(np.random.poisson(100, 64)) for _ in range(10)]
        spectra = Spectra(spectra_list, shared_calibration=False)
        
        assert spectra.uses_shared_calibration is False


class TestSpectraProperties:
    """Test Spectra properties."""
    
    def test_counts_array(self):
        """Test counts array access."""
        spectra_list = [Spectrum(np.ones(64) * i) for i in range(5)]
        spectra = Spectra(spectra_list)
        
        assert spectra.counts.shape == (5, 64)
        assert np.allclose(spectra.counts[0], 0)
        assert np.allclose(spectra.counts[4], 4)
    
    def test_energy_edges_shared(self):
        """Test energy_edges property with shared calibration."""
        edges = np.linspace(0, 3000, 65)
        spectra_list = [
            Spectrum(np.random.poisson(100, 64), energy_edges=edges)
            for _ in range(10)
        ]
        spectra = Spectra(spectra_list, shared_calibration=True)
        
        assert spectra.energy_edges.shape == (65,)
        assert np.allclose(spectra.energy_edges, edges)
    
    def test_energy_centers(self):
        """Test energy_centers property."""
        edges = np.linspace(0, 3000, 65)
        spectra_list = [
            Spectrum(np.random.poisson(100, 64), energy_edges=edges)
            for _ in range(3)
        ]
        spectra = Spectra(spectra_list)
        
        centers = spectra.energy_centers
        expected = (edges[:-1] + edges[1:]) / 2
        assert np.allclose(centers, expected)


class TestSpectraStatistics:
    """Test statistical operations on Spectra."""
    
    def test_mean_spectrum(self):
        """Test computing mean spectrum."""
        counts_array = np.array([
            [10, 20, 30],
            [20, 30, 40],
            [30, 40, 50],
        ])
        spectra_list = [Spectrum(counts_array[i]) for i in range(3)]
        spectra = Spectra(spectra_list)
        
        mean_spec = spectra.mean_spectrum()
        expected = np.array([20, 30, 40])
        
        assert isinstance(mean_spec, Spectrum)
        assert np.allclose(mean_spec.counts, expected)
    
    def test_median_spectrum(self):
        """Test computing median spectrum."""
        counts_array = np.array([
            [10, 20, 30],
            [20, 30, 40],
            [100, 200, 300],  # Outlier
        ])
        spectra_list = [Spectrum(counts_array[i]) for i in range(3)]
        spectra = Spectra(spectra_list)
        
        median_spec = spectra.median_spectrum()
        expected = np.array([20, 30, 40])
        
        assert np.allclose(median_spec.counts, expected)
    
    def test_std_spectrum(self):
        """Test computing std spectrum."""
        counts_array = np.array([
            [10, 20, 30],
            [20, 30, 40],
            [30, 40, 50],
        ])
        spectra_list = [Spectrum(counts_array[i]) for i in range(3)]
        spectra = Spectra(spectra_list)
        
        std_spec = spectra.std_spectrum()
        expected = np.std(counts_array, axis=0)
        
        assert np.allclose(std_spec.counts, expected)
    
    def test_sum_spectrum(self):
        """Test computing sum spectrum."""
        counts_array = np.array([
            [10, 20, 30],
            [20, 30, 40],
            [30, 40, 50],
        ])
        spectra_list = [Spectrum(counts_array[i]) for i in range(3)]
        spectra = Spectra(spectra_list)
        
        sum_spec = spectra.sum_spectrum()
        expected = np.array([60, 90, 120])
        
        assert np.allclose(sum_spec.counts, expected)


class TestSpectraVectorizedOperations:
    """Test vectorized operations."""
    
    def test_apply_vectorized(self):
        """Test applying vectorized function."""
        spectra_list = [Spectrum(np.ones(64) * (i+1)) for i in range(5)]
        spectra = Spectra(spectra_list)
        
        # Double all counts
        doubled = spectra.apply_vectorized(lambda counts: counts * 2)
        
        assert doubled.n_spectra == 5
        assert np.allclose(doubled.counts[0], 2)
        assert np.allclose(doubled.counts[4], 10)
    
    def test_apply_vectorized_normalize(self):
        """Test normalizing each spectrum."""
        spectra_list = [Spectrum(np.random.poisson(100, 64)) for _ in range(5)]
        spectra = Spectra(spectra_list)
        
        normalized = spectra.apply_vectorized(
            lambda counts: counts / counts.sum(axis=1, keepdims=True)
        )
        
        # Each spectrum should sum to 1
        for i in range(5):
            assert np.isclose(np.sum(normalized.counts[i]), 1.0)


class TestSpectraIndexing:
    """Test indexing and iteration."""
    
    def test_getitem_int(self):
        """Test getting single spectrum by index."""
        spectra_list = [Spectrum(np.ones(64) * i) for i in range(5)]
        spectra = Spectra(spectra_list)
        
        spec = spectra[2]
        assert isinstance(spec, Spectrum)
        assert np.allclose(spec.counts, 2)
    
    def test_getitem_slice(self):
        """Test slicing spectra."""
        spectra_list = [Spectrum(np.ones(64) * i) for i in range(10)]
        spectra = Spectra(spectra_list)
        
        subset = spectra[2:5]
        assert isinstance(subset, Spectra)
        assert subset.n_spectra == 3
        assert np.allclose(subset[0].counts, 2)
    
    def test_len(self):
        """Test __len__."""
        spectra_list = [Spectrum(np.ones(64)) for _ in range(7)]
        spectra = Spectra(spectra_list)
        
        assert len(spectra) == 7
    
    def test_iter(self):
        """Test iteration."""
        spectra_list = [Spectrum(np.ones(64) * i) for i in range(5)]
        spectra = Spectra(spectra_list)
        
        for i, spec in enumerate(spectra):
            assert isinstance(spec, Spectrum)
            assert np.allclose(spec.counts, i)


class TestSpectraMemoryViews:
    """Test memory view behavior."""
    
    def test_shared_memory_changes_propagate(self):
        """Test that changes to counts array affect individual spectra."""
        spectra_list = [Spectrum(np.ones(64)) for _ in range(5)]
        spectra = Spectra(spectra_list, shared_calibration=True)
        
        # Modify via array
        spectra.counts[2, 10] = 999.0
        
        # Check via spectrum object
        spec = spectra[2]
        assert spec.counts[10] == 999.0
    
    def test_numpy_array_protocol(self):
        """Test __array__ protocol."""
        spectra_list = [Spectrum(np.ones(64) * i) for i in range(5)]
        spectra = Spectra(spectra_list)
        
        arr = np.array(spectra)
        assert arr.shape == (5, 64)
        assert np.allclose(arr[0], 0)
        assert np.allclose(arr[4], 4)


class TestSpectraRepr:
    """Test string representation."""
    
    def test_repr(self):
        """Test __repr__."""
        spectra_list = [Spectrum(np.ones(64)) for _ in range(10)]
        spectra = Spectra(spectra_list)
        
        repr_str = repr(spectra)
        assert "Spectra" in repr_str
        assert "n_spectra=10" in repr_str
        assert "n_bins=64" in repr_str
        assert "uncalibrated" in repr_str

