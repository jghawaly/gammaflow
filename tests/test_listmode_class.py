"""
Tests for ListMode class.
"""

import numpy as np
import pytest

from gammaflow.core.listmode import ListMode
from gammaflow.core.time_series import SpectralTimeSeries


class TestListModeCreation:
    """Test ListMode creation and properties."""
    
    def test_basic_creation(self):
        """Test basic ListMode creation."""
        time_deltas = np.array([0.1, 0.2, 0.15, 0.25])
        energies = np.array([100, 200, 150, 300])
        
        lm = ListMode(time_deltas, energies)
        
        assert lm.n_events == 4
        np.testing.assert_array_equal(lm.time_deltas, time_deltas)
        np.testing.assert_array_equal(lm.energies, energies)
    
    def test_with_metadata(self):
        """Test ListMode with metadata."""
        time_deltas = np.array([0.1, 0.2])
        energies = np.array([100, 200])
        metadata = {'detector': 'HPGe', 'run_id': 42}
        
        lm = ListMode(time_deltas, energies, metadata=metadata)
        
        assert lm.metadata['detector'] == 'HPGe'
        assert lm.metadata['run_id'] == 42
    
    def test_mismatched_lengths_raises(self):
        """Test that mismatched lengths raise error."""
        time_deltas = np.array([0.1, 0.2, 0.3])
        energies = np.array([100, 200])  # Wrong length
        
        with pytest.raises(ValueError, match="same length"):
            ListMode(time_deltas, energies)
    
    def test_empty_listmode(self):
        """Test empty ListMode."""
        lm = ListMode(np.array([]), np.array([]))
        
        assert lm.n_events == 0
        assert lm.total_time == 0.0
        assert lm.mean_rate == 0.0
        assert lm.energy_range == (0.0, 0.0)


class TestListModeProperties:
    """Test ListMode computed properties."""
    
    def test_absolute_times(self):
        """Test absolute times computation."""
        time_deltas = np.array([0.1, 0.2, 0.15, 0.25])
        energies = np.array([100, 200, 150, 300])
        
        lm = ListMode(time_deltas, energies)
        
        expected_times = np.cumsum(time_deltas)
        np.testing.assert_array_equal(lm.absolute_times, expected_times)
    
    def test_total_time(self):
        """Test total time calculation."""
        time_deltas = np.array([0.1, 0.2, 0.15, 0.25])
        energies = np.array([100, 200, 150, 300])
        
        lm = ListMode(time_deltas, energies)
        
        assert lm.total_time == pytest.approx(0.7)
    
    def test_mean_rate(self):
        """Test mean rate calculation."""
        time_deltas = np.full(100, 0.01)  # 100 events over 1 second
        energies = np.random.uniform(0, 1000, 100)
        
        lm = ListMode(time_deltas, energies)
        
        assert lm.mean_rate == pytest.approx(100.0)
    
    def test_energy_range(self):
        """Test energy range property."""
        time_deltas = np.array([0.1, 0.2, 0.15])
        energies = np.array([100, 500, 250])
        
        lm = ListMode(time_deltas, energies)
        
        assert lm.energy_range == (100.0, 500.0)


class TestListModeFiltering:
    """Test ListMode filtering methods."""
    
    def test_filter_energy(self):
        """Test filtering by energy."""
        time_deltas = np.array([0.1, 0.2, 0.15, 0.25, 0.3])
        energies = np.array([100, 200, 300, 400, 500])
        
        lm = ListMode(time_deltas, energies)
        
        # Filter to 200-400 keV
        filtered = lm.filter_energy(e_min=200, e_max=400)
        
        assert filtered.n_events == 2  # 200 and 300
        np.testing.assert_array_equal(filtered.energies, np.array([200, 300]))
    
    def test_filter_energy_min_only(self):
        """Test filtering with only minimum energy."""
        time_deltas = np.array([0.1, 0.2, 0.15, 0.25])
        energies = np.array([100, 200, 300, 400])
        
        lm = ListMode(time_deltas, energies)
        filtered = lm.filter_energy(e_min=250)
        
        assert filtered.n_events == 2
        np.testing.assert_array_equal(filtered.energies, np.array([300, 400]))
    
    def test_filter_energy_max_only(self):
        """Test filtering with only maximum energy."""
        time_deltas = np.array([0.1, 0.2, 0.15, 0.25])
        energies = np.array([100, 200, 300, 400])
        
        lm = ListMode(time_deltas, energies)
        filtered = lm.filter_energy(e_max=250)
        
        assert filtered.n_events == 2
        np.testing.assert_array_equal(filtered.energies, np.array([100, 200]))
    
    def test_slice_time(self):
        """Test slicing by time."""
        time_deltas = np.array([1.0, 1.0, 1.0, 1.0])  # Events at 1, 2, 3, 4 seconds
        energies = np.array([100, 200, 300, 400])
        
        lm = ListMode(time_deltas, energies)
        
        # Get events from 1.5 to 3.5 seconds
        sliced = lm.slice_time(t_min=1.5, t_max=3.5)
        
        assert sliced.n_events == 2  # Events at 2 and 3 seconds
        np.testing.assert_array_equal(sliced.energies, np.array([200, 300]))
    
    def test_slice_time_recomputes_deltas(self):
        """Test that slicing recomputes time deltas correctly."""
        time_deltas = np.array([1.0, 1.0, 1.0, 1.0])
        energies = np.array([100, 200, 300, 400])
        
        lm = ListMode(time_deltas, energies)
        sliced = lm.slice_time(t_min=1.5, t_max=3.5)
        
        # Time deltas should be recomputed
        assert len(sliced.time_deltas) == sliced.n_events


class TestListModeCopy:
    """Test ListMode copy method."""
    
    def test_copy(self):
        """Test that copy creates independent copy."""
        time_deltas = np.array([0.1, 0.2, 0.15])
        energies = np.array([100, 200, 300])
        metadata = {'test': 'value'}
        
        lm = ListMode(time_deltas, energies, metadata=metadata)
        lm_copy = lm.copy()
        
        # Should be equal
        np.testing.assert_array_equal(lm_copy.time_deltas, lm.time_deltas)
        np.testing.assert_array_equal(lm_copy.energies, lm.energies)
        assert lm_copy.metadata == lm.metadata
        
        # But independent
        lm_copy._time_deltas[0] = 999
        assert lm.time_deltas[0] != 999


class TestListModeStringRepresentation:
    """Test ListMode string methods."""
    
    def test_repr(self):
        """Test __repr__."""
        time_deltas = np.full(100, 0.01)
        energies = np.random.uniform(0, 1000, 100)
        
        lm = ListMode(time_deltas, energies)
        repr_str = repr(lm)
        
        assert 'ListMode' in repr_str
        assert 'n_events=100' in repr_str
        assert 'duration=' in repr_str
        assert 'rate=' in repr_str
    
    def test_len(self):
        """Test __len__."""
        time_deltas = np.array([0.1, 0.2, 0.15])
        energies = np.array([100, 200, 300])
        
        lm = ListMode(time_deltas, energies)
        
        assert len(lm) == 3


class TestListModeToSpectralTimeSeries:
    """Test integration with SpectralTimeSeries."""
    
    def test_from_listmode_object(self):
        """Test creating SpectralTimeSeries from ListMode object."""
        time_deltas = np.random.exponential(0.01, size=10000)
        energies = np.random.uniform(0, 1000, size=10000)
        
        lm = ListMode(time_deltas, energies)
        
        # Create time series from ListMode object
        ts = SpectralTimeSeries.from_list_mode(
            lm,
            integration_time=10.0,
            energy_bins=256
        )
        
        assert ts.n_bins == 256
        assert ts.n_spectra > 0
    
    def test_both_calling_conventions_equivalent(self):
        """Test that both calling conventions give same result."""
        np.random.seed(42)
        time_deltas = np.random.exponential(0.01, size=5000)
        energies = np.random.uniform(0, 1000, size=5000)
        
        # Method 1: Arrays
        ts1 = SpectralTimeSeries.from_list_mode(
            time_deltas, energies,
            integration_time=10.0,
            energy_bins=128
        )
        
        # Method 2: ListMode object
        lm = ListMode(time_deltas, energies)
        ts2 = SpectralTimeSeries.from_list_mode(
            lm,
            integration_time=10.0,
            energy_bins=128
        )
        
        # Should be equivalent
        assert ts1.n_spectra == ts2.n_spectra
        assert ts1.n_bins == ts2.n_bins
        np.testing.assert_array_almost_equal(ts1.counts, ts2.counts)
    
    def test_missing_integration_time_raises(self):
        """Test that missing integration_time raises error."""
        lm = ListMode(np.array([0.1, 0.2]), np.array([100, 200]))
        
        with pytest.raises(ValueError, match="integration_time is required"):
            SpectralTimeSeries.from_list_mode(lm)
    
    def test_missing_energies_with_arrays_raises(self):
        """Test that missing energies with arrays raises error."""
        time_deltas = np.array([0.1, 0.2, 0.3])
        
        with pytest.raises(ValueError, match="energies is required"):
            SpectralTimeSeries.from_list_mode(
                time_deltas,
                integration_time=10.0
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

