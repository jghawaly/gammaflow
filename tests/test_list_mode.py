"""
Tests for list mode functionality.
"""

import numpy as np
import pytest

from gammaflow.core.time_series import SpectralTimeSeries


class TestFromListMode:
    """Test SpectralTimeSeries.from_list_mode() method."""
    
    def test_basic_list_mode(self):
        """Test basic list mode conversion."""
        # Generate synthetic list mode data
        # 1000 events over ~100 seconds at ~10 Hz
        time_deltas = np.random.exponential(0.1, size=1000)
        energies = np.random.uniform(0, 3000, size=1000)
        
        # Create time series with 10-second non-overlapping windows
        ts = SpectralTimeSeries.from_list_mode(
            time_deltas, energies,
            integration_time=10.0,
            energy_bins=512
        )
        
        # Should have ~10 spectra
        assert ts.n_spectra >= 8  # Allow for some variance
        assert ts.n_bins == 512
        assert ts.uses_shared_calibration
        
        # Each spectrum should have real_time = 10.0
        for spec in ts:
            assert spec.real_time == 10.0
            assert spec.live_time is None  # List mode doesn't have dead time
    
    def test_overlapping_windows(self):
        """Test overlapping windows (stride < integration_time)."""
        time_deltas = np.random.exponential(0.01, size=10000)  # ~100 Hz
        energies = np.random.uniform(0, 1000, size=10000)
        
        # 10-second windows, 1-second stride (90% overlap)
        ts = SpectralTimeSeries.from_list_mode(
            time_deltas, energies,
            integration_time=10.0,
            stride_time=1.0,
            energy_bins=256
        )
        
        # Should have many more spectra due to overlap
        assert ts.n_spectra > 50  # Depends on total time
        
        # Verify timestamps are spaced by stride
        if ts.n_spectra > 1:
            timestamp_diffs = np.diff(ts.timestamps)
            assert np.allclose(timestamp_diffs, 1.0, atol=0.01)
    
    def test_custom_energy_range(self):
        """Test with custom energy range."""
        time_deltas = np.random.exponential(0.1, size=1000)
        energies = np.random.uniform(0, 5000, size=1000)
        
        # Specify energy range
        ts = SpectralTimeSeries.from_list_mode(
            time_deltas, energies,
            integration_time=10.0,
            energy_bins=1024,
            energy_range=(0, 3000)  # Only 0-3000 keV
        )
        
        # Check energy range
        assert ts.energy_edges[0] == pytest.approx(0)
        assert ts.energy_edges[-1] == pytest.approx(3000)
        assert ts.n_bins == 1024
    
    def test_explicit_energy_bins(self):
        """Test with explicit bin edges."""
        time_deltas = np.random.exponential(0.1, size=1000)
        energies = np.random.uniform(0, 1000, size=1000)
        
        # Custom bin edges
        edges = np.linspace(0, 1000, 129)  # 128 bins
        
        ts = SpectralTimeSeries.from_list_mode(
            time_deltas, energies,
            integration_time=10.0,
            energy_bins=edges
        )
        
        assert ts.n_bins == 128
        np.testing.assert_array_almost_equal(ts.energy_edges, edges)
    
    def test_metadata_stored(self):
        """Test that metadata is properly stored."""
        time_deltas = np.random.exponential(0.1, size=100)
        energies = np.random.uniform(0, 1000, size=100)
        
        ts = SpectralTimeSeries.from_list_mode(
            time_deltas, energies,
            integration_time=10.0,
            energy_bins=128
        )
        
        # Check metadata
        for spec in ts:
            assert 'window_start' in spec.metadata
            assert 'window_end' in spec.metadata
            assert 'n_events' in spec.metadata
            assert spec.metadata['window_end'] - spec.metadata['window_start'] == pytest.approx(10.0)
            assert isinstance(spec.metadata['n_events'], int)
    
    def test_event_counting(self):
        """Test that events are correctly counted in windows."""
        # Create deterministic data
        # Events spaced exactly 0.1s apart
        n_events = 300
        time_deltas = np.full(n_events, 0.1)  # Exactly 0.1s between events
        energies = np.random.uniform(0, 1000, size=n_events)
        
        # 10-second windows should each contain ~100 events
        # (10 seconds / 0.1 seconds per event = 100 events per window)
        ts = SpectralTimeSeries.from_list_mode(
            time_deltas, energies,
            integration_time=10.0,
            energy_bins=128
        )
        
        # Should have 3-4 windows depending on floating point (30s total / 10s stride)
        # np.arange(0, 30.0, 10.0) can be [0, 10, 20] or [0, 10, 20, 30] due to FP
        assert 3 <= ts.n_spectra <= 4
        
        # Most windows should have ~100 events
        event_counts = [spec.metadata['n_events'] for spec in ts]
        # At least one window should be full
        assert max(event_counts) >= 95
    
    def test_empty_list_mode(self):
        """Test with empty arrays."""
        time_deltas = np.array([])
        energies = np.array([])
        
        ts = SpectralTimeSeries.from_list_mode(
            time_deltas, energies,
            integration_time=10.0,
            energy_bins=128
        )
        
        assert ts.n_spectra == 0
    
    def test_mismatched_lengths_raises(self):
        """Test that mismatched array lengths raise error."""
        time_deltas = np.random.exponential(0.1, size=100)
        energies = np.random.uniform(0, 1000, size=99)  # Wrong length
        
        with pytest.raises(ValueError, match="same length"):
            SpectralTimeSeries.from_list_mode(
                time_deltas, energies,
                integration_time=10.0
            )
    
    def test_negative_integration_time_raises(self):
        """Test that negative integration_time raises error."""
        time_deltas = np.random.exponential(0.1, size=100)
        energies = np.random.uniform(0, 1000, size=100)
        
        with pytest.raises(ValueError, match="must be positive"):
            SpectralTimeSeries.from_list_mode(
                time_deltas, energies,
                integration_time=-1.0
            )
    
    def test_negative_stride_raises(self):
        """Test that negative stride_time raises error."""
        time_deltas = np.random.exponential(0.1, size=100)
        energies = np.random.uniform(0, 1000, size=100)
        
        with pytest.raises(ValueError, match="must be positive"):
            SpectralTimeSeries.from_list_mode(
                time_deltas, energies,
                integration_time=10.0,
                stride_time=-1.0
            )
    
    def test_timestamps_are_window_centers(self):
        """Test that timestamps are at window centers."""
        time_deltas = np.random.exponential(0.1, size=1000)
        energies = np.random.uniform(0, 1000, size=1000)
        
        ts = SpectralTimeSeries.from_list_mode(
            time_deltas, energies,
            integration_time=10.0,
            energy_bins=128
        )
        
        # Timestamp should be center of window
        for spec in ts:
            expected_timestamp = spec.metadata['window_start'] + 5.0
            assert spec.timestamp == pytest.approx(expected_timestamp)
    
    def test_large_stride_time(self):
        """Test with stride_time > integration_time (gaps between windows)."""
        time_deltas = np.random.exponential(0.01, size=10000)
        energies = np.random.uniform(0, 1000, size=10000)
        
        # 5-second windows, 10-second stride (gaps)
        ts = SpectralTimeSeries.from_list_mode(
            time_deltas, energies,
            integration_time=5.0,
            stride_time=10.0,
            energy_bins=128
        )
        
        # Windows should be spaced 10 seconds apart
        if ts.n_spectra > 1:
            timestamp_diffs = np.diff(ts.timestamps)
            assert np.allclose(timestamp_diffs, 10.0, atol=0.01)


class TestListModeIntegration:
    """Integration tests for list mode functionality."""
    
    def test_list_mode_to_vectorized_operations(self):
        """Test that list mode output works with vectorized operations."""
        time_deltas = np.random.exponential(0.01, size=10000)
        energies = np.random.uniform(0, 1000, size=10000)
        
        ts = SpectralTimeSeries.from_list_mode(
            time_deltas, energies,
            integration_time=10.0,
            energy_bins=256
        )
        
        # Should be able to do numpy operations
        total_counts = np.sum(ts.counts, axis=1)
        assert len(total_counts) == ts.n_spectra
        
        # Background subtraction (in-place on the array)
        background = np.mean(ts.counts, axis=0)
        ts.counts[:] -= background  # Use [:] for in-place modification
        
        # Should work
        assert ts.counts.shape[0] == ts.n_spectra
    
    def test_list_mode_with_energy_slicing(self):
        """Test energy slicing on list mode data."""
        time_deltas = np.random.exponential(0.01, size=5000)
        energies = np.random.uniform(0, 1000, size=5000)
        
        ts = SpectralTimeSeries.from_list_mode(
            time_deltas, energies,
            integration_time=10.0,
            energy_bins=512,
            energy_range=(0, 1000)
        )
        
        # Slice each spectrum
        for spec in ts:
            roi = spec.slice_energy(e_min=200, e_max=400)
            # Check that slicing worked
            assert roi.energy_edges[0] >= 199  # Allow small tolerance
            assert roi.energy_edges[-1] <= 401  # Allow small tolerance
            assert roi.n_bins < spec.n_bins  # Should have fewer bins


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

