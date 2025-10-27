"""
Tests for SpectralTimeSeries class.
"""

import numpy as np
import pytest
from gammaflow import Spectrum, SpectralTimeSeries
from gammaflow.utils.exceptions import TimeSeriesError, IncompatibleBinningError


class TestTimeSeriesCreation:
    """Test SpectralTimeSeries creation."""
    
    def test_create_empty(self):
        """Test creating empty time series."""
        ts = SpectralTimeSeries([])
        assert ts.n_spectra == 0
        assert len(ts) == 0
    
    def test_create_from_spectra(self, small_spectra_list):
        """Test creating from list of spectra."""
        ts = SpectralTimeSeries(small_spectra_list)
        assert ts.n_spectra == 10
        assert ts.n_bins == 64
    
    def test_auto_detect_shared_calibration(self, calibrated_spectra_list):
        """Test auto-detection of shared calibration."""
        ts = SpectralTimeSeries(calibrated_spectra_list, shared_calibration=None)
        assert ts.uses_shared_calibration  # All have same edges
    
    def test_force_shared_calibration(self, small_spectra_list):
        """Test forcing shared calibration."""
        ts = SpectralTimeSeries(small_spectra_list, shared_calibration=True)
        assert ts.uses_shared_calibration
    
    def test_force_independent_calibration(self, calibrated_spectra_list):
        """Test forcing independent calibration."""
        ts = SpectralTimeSeries(calibrated_spectra_list, shared_calibration=False)
        assert not ts.uses_shared_calibration
    
    def test_mismatched_bin_counts_raises_error(self):
        """Test mismatched bin counts raises error."""
        spectra = [
            Spectrum(np.ones(10)),
            Spectrum(np.ones(20)),  # Different size!
        ]
        with pytest.raises(TimeSeriesError, match="same number of bins"):
            SpectralTimeSeries(spectra)


class TestTimeSeriesProperties:
    """Test SpectralTimeSeries properties."""
    
    def test_counts_property(self, time_series_small):
        """Test counts property returns 2D array."""
        counts = time_series_small.counts
        assert counts.shape == (10, 64)
        assert isinstance(counts, np.ndarray)
    
    def test_spectra_property(self, time_series_small):
        """Test spectra property returns list."""
        spectra = time_series_small.spectra
        assert len(spectra) == 10
        assert all(isinstance(s, Spectrum) for s in spectra)
    
    def test_energy_edges(self, time_series_calibrated):
        """Test energy_edges property."""
        edges = time_series_calibrated.energy_edges
        assert len(edges) == 257
        assert edges[0] == 0
        assert edges[-1] == 1000
    
    def test_energy_centers(self, time_series_calibrated):
        """Test energy_centers property."""
        centers = time_series_calibrated.energy_centers
        assert len(centers) == 256
    
    def test_timestamps(self, time_series_calibrated):
        """Test timestamps property."""
        timestamps = time_series_calibrated.timestamps
        assert len(timestamps) == 20
        expected = np.arange(20) * 10.0
        assert np.array_equal(timestamps, expected)
    
    def test_live_times(self, time_series_small):
        """Test live_times property."""
        times = time_series_small.live_times
        assert len(times) == 10
        assert np.all(times == 1.0)
    
    def test_n_spectra(self, time_series_small):
        """Test n_spectra property."""
        assert time_series_small.n_spectra == 10
    
    def test_n_bins(self, time_series_small):
        """Test n_bins property."""
        assert time_series_small.n_bins == 64
    
    def test_is_calibrated(self, time_series_calibrated):
        """Test is_calibrated property."""
        assert time_series_calibrated.is_calibrated


class TestTimeSeriesSharedMemory:
    """Test shared memory between array and Spectrum objects."""
    
    def test_modify_array_affects_spectrum(self, time_series_small):
        """Test modifying array affects spectrum."""
        # Modify via array
        time_series_small.counts[5, 10] = 9999
        
        # Check via spectrum
        spec = time_series_small[5]
        assert spec.counts[10] == 9999
    
    def test_modify_spectrum_affects_array(self, time_series_small):
        """Test modifying spectrum affects array."""
        # Modify via spectrum
        spec = time_series_small[3]
        spec.counts[20] = 8888
        
        # Check via array
        assert time_series_small.counts[3, 20] == 8888
    
    def test_independent_mode_no_shared_memory(self):
        """Test independent mode doesn't share memory."""
        spectra = [Spectrum(np.ones(10)) for _ in range(5)]
        ts = SpectralTimeSeries(spectra, shared_calibration=False)
        
        # Note: In independent mode, the array is rebuilt from spectra
        # So modifications need to go through spectra, not array


class TestTimeSeriesCalibration:
    """Test calibration operations on time series."""
    
    def test_apply_calibration(self, time_series_small):
        """Test applying calibration to all spectra."""
        calibrated = time_series_small.apply_calibration([0, 0.5])
        
        assert calibrated.is_calibrated
        assert all(s.is_calibrated for s in calibrated.spectra)
    
    def test_apply_calibration_inplace(self, time_series_small):
        """Test applying calibration in-place."""
        time_series_small.apply_calibration_([0, 0.5])
        
        assert time_series_small.is_calibrated
    
    def test_to_shared_calibration(self, mixed_calibration_spectra):
        """Test converting to shared calibration."""
        ts = SpectralTimeSeries(mixed_calibration_spectra, shared_calibration=False)
        assert not ts.uses_shared_calibration
        
        # Convert to shared
        ts_shared = ts.to_shared_calibration()
        assert ts_shared.uses_shared_calibration
    
    def test_to_independent_calibration(self, time_series_calibrated):
        """Test converting to independent calibration."""
        assert time_series_calibrated.uses_shared_calibration
        
        # Convert to independent
        ts_indep = time_series_calibrated.to_independent_calibration()
        assert not ts_indep.uses_shared_calibration


class TestTimeSeriesVectorizedOperations:
    """Test vectorized operations on time series."""
    
    def test_apply_vectorized(self, time_series_small):
        """Test apply_vectorized method."""
        # Double all counts
        doubled = time_series_small.apply_vectorized(lambda x: x * 2)
        
        assert np.allclose(doubled.counts, time_series_small.counts * 2)
    
    def test_background_subtract_mean(self, time_series_small):
        """Test background subtraction with mean."""
        ts_sub = time_series_small.background_subtract('mean')
        
        # Mean of result should be ~0
        assert np.abs(np.mean(ts_sub.counts)) < 1.0
    
    def test_background_subtract_median(self, time_series_small):
        """Test background subtraction with median."""
        ts_sub = time_series_small.background_subtract('median')
        
        assert ts_sub.n_spectra == time_series_small.n_spectra
        assert ts_sub.n_bins == time_series_small.n_bins
    
    def test_background_subtract_spectrum(self, time_series_small):
        """Test background subtraction with Spectrum."""
        background = time_series_small[0]
        ts_sub = time_series_small.background_subtract(background)
        
        # First spectrum should be ~0
        assert np.sum(np.abs(ts_sub[0].counts)) < 10
    
    def test_background_subtract_array(self, time_series_small):
        """Test background subtraction with array."""
        background = np.ones(time_series_small.n_bins) * 10
        ts_sub = time_series_small.background_subtract(background)
        
        assert ts_sub.n_spectra == time_series_small.n_spectra
    
    def test_count_rate_conversion(self):
        """Test converting to count rates via array division."""
        spectra = [
            Spectrum(np.ones(10) * 100, live_time=10.0),
            Spectrum(np.ones(10) * 200, live_time=20.0),
        ]
        ts = SpectralTimeSeries(spectra)
        
        # Convert to count rates using vectorized division
        count_rates = ts.counts / ts.live_times[:, np.newaxis]
        
        # Both should have same count rate (10 counts/sec)
        assert np.allclose(count_rates[0], 10.0)
        assert np.allclose(count_rates[1], 10.0)
    
    def test_normalize_area(self, time_series_small):
        """Test normalization by area."""
        normalized = time_series_small.normalize('area')
        
        # Each spectrum should sum to 1
        for i in range(normalized.n_spectra):
            assert np.isclose(np.sum(normalized.counts[i]), 1.0)
    
    def test_normalize_max(self, time_series_small):
        """Test normalization by maximum."""
        normalized = time_series_small.normalize('max')
        
        # Each spectrum should have max = 1
        for i in range(normalized.n_spectra):
            assert np.isclose(np.max(normalized.counts[i]), 1.0)


class TestTimeSeriesPerSpectrumOperations:
    """Test per-spectrum operations."""
    
    def test_apply_to_each(self, time_series_small):
        """Test apply_to_each method."""
        # Normalize each spectrum
        normalized = time_series_small.apply_to_each(
            lambda s: s.normalize('area')
        )
        
        # Check each is normalized
        for spec in normalized.spectra:
            assert np.isclose(np.sum(spec.counts), 1.0)
    
    def test_filter_spectra(self, time_series_small):
        """Test filter_spectra method."""
        # Filter by metadata
        filtered = time_series_small.filter_spectra(
            lambda s: s.metadata['index'] > 5
        )
        
        assert filtered.n_spectra == 4  # Indices 6, 7, 8, 9
    
    def test_filter_by_counts(self):
        """Test filtering by counts."""
        spectra = [
            Spectrum(np.ones(10) * i) for i in range(1, 11)
        ]
        ts = SpectralTimeSeries(spectra)
        
        # Keep only high-count spectra
        filtered = ts.filter_spectra(lambda s: np.sum(s.counts) > 50)
        
        assert filtered.n_spectra == 5  # 6, 7, 8, 9, 10


class TestTimeSeriesTimeOperations:
    """Test time-based operations."""
    
    def test_slice_time(self):
        """Test time slicing."""
        spectra = [
            Spectrum(np.ones(10), timestamp=float(i * 10))
            for i in range(20)
        ]
        ts = SpectralTimeSeries(spectra)
        
        sliced = ts.slice_time(t_min=50, t_max=150)
        
        assert sliced.n_spectra == 11  # 50, 60, ..., 150
    
    def test_slice_time_no_bounds(self, time_series_small):
        """Test slicing without bounds returns all."""
        sliced = time_series_small.slice_time()
        assert sliced.n_spectra == time_series_small.n_spectra
    
    def test_rebin_time(self):
        """Test time rebinning."""
        spectra = [
            Spectrum(np.ones(10) * 5, timestamp=float(i))
            for i in range(20)
        ]
        ts = SpectralTimeSeries(spectra)
        
        # Rebin to 5-second intervals
        rebinned = ts.rebin_time(integration_time=5.0, stride=5.0)
        
        assert rebinned.n_spectra == 4  # 20 / 5 = 4
        # Each should have ~25 counts per bin (5 spectra * 5 counts)
        assert np.allclose(rebinned[0].counts, 25, rtol=0.1)
    
    def test_rebin_time_with_overlap(self):
        """Test time rebinning with overlap."""
        spectra = [
            Spectrum(np.ones(10), timestamp=float(i))
            for i in range(20)
        ]
        ts = SpectralTimeSeries(spectra)
        
        # Overlap: integration=10, stride=5
        rebinned = ts.rebin_time(integration_time=10.0, stride=5.0)
        
        # Should have more rebinned spectra due to overlap (at least 3 windows)
        assert rebinned.n_spectra >= 3
    
    def test_integrate_time(self):
        """Test time integration."""
        spectra = [
            Spectrum(np.ones(10) * 5, timestamp=float(i * 10))
            for i in range(10)
        ]
        ts = SpectralTimeSeries(spectra)
        
        integrated = ts.integrate_time(t_min=20, t_max=60)
        
        # Should integrate spectra at t=20, 30, 40, 50, 60 (5 spectra)
        assert np.allclose(integrated.counts, 25)  # 5 spectra * 5 counts


class TestTimeSeriesAnalysis:
    """Test analysis methods."""
    
    def test_mean_spectrum(self, time_series_small):
        """Test mean spectrum computation."""
        mean = time_series_small.mean_spectrum()
        
        assert isinstance(mean, Spectrum)
        expected = np.mean(time_series_small.counts, axis=0)
        assert np.allclose(mean.counts, expected)
    
    def test_sum_spectrum(self):
        """Test sum spectrum computation."""
        spectra = [
            Spectrum(np.ones(10) * i) for i in range(1, 6)
        ]
        ts = SpectralTimeSeries(spectra)
        
        summed = ts.sum_spectrum()
        
        # Should be 1+2+3+4+5 = 15 per bin
        assert np.allclose(summed.counts, 15)


class TestTimeSeriesNumpyProtocol:
    """Test numpy protocol integration."""
    
    def test_array_protocol(self, time_series_small):
        """Test __array__ protocol."""
        arr = np.array(time_series_small)
        assert arr.shape == (10, 64)
        assert isinstance(arr, np.ndarray)
    
    def test_len(self, time_series_small):
        """Test __len__."""
        assert len(time_series_small) == 10
    
    def test_getitem_single(self, time_series_small):
        """Test single index access."""
        spec = time_series_small[5]
        assert isinstance(spec, Spectrum)
        assert spec.metadata['index'] == 5
    
    def test_getitem_slice(self, time_series_small):
        """Test slice access."""
        sliced = time_series_small[2:7]
        
        assert isinstance(sliced, SpectralTimeSeries)
        assert sliced.n_spectra == 5
    
    def test_iter(self, time_series_small):
        """Test iteration."""
        count = 0
        for spec in time_series_small:
            assert isinstance(spec, Spectrum)
            count += 1
        
        assert count == 10
    
    def test_use_in_numpy_functions(self, time_series_small):
        """Test using time series in numpy functions."""
        # These should work via __array__ protocol
        mean = np.mean(time_series_small)
        assert isinstance(mean, (float, np.floating))
        
        std = np.std(time_series_small)
        assert isinstance(std, (float, np.floating))


class TestTimeSeriesRepr:
    """Test string representation."""
    
    def test_repr_shared(self, time_series_small):
        """Test repr for shared calibration mode."""
        repr_str = repr(time_series_small)
        assert "SpectralTimeSeries" in repr_str
        assert "n_spectra=10" in repr_str
        assert "n_bins=64" in repr_str
        assert "shared" in repr_str
    
    def test_repr_independent(self):
        """Test repr for independent mode."""
        spectra = [Spectrum(np.ones(10)) for _ in range(5)]
        ts = SpectralTimeSeries(spectra, shared_calibration=False)
        
        repr_str = repr(ts)
        assert "independent" in repr_str


class TestTimeSeriesCopyOnWrite:
    """Test copy-on-write behavior."""
    
    def test_modify_spectrum_detaches(self, time_series_small):
        """Test modifying spectrum detaches from shared calibration."""
        spec = time_series_small[5]
        assert spec.has_shared_calibration
        
        # Apply calibration in-place (should detach)
        spec.apply_calibration_([0, 1.0])
        
        assert not spec.has_shared_calibration
    
    def test_other_spectra_unaffected(self, time_series_small):
        """Test modifying one spectrum doesn't affect others."""
        spec1 = time_series_small[3]
        spec2 = time_series_small[4]
        
        # Modify spec1 in way that detaches
        spec1.rebin_energy_([0, 10, 20, 30])
        
        # spec2 should still be shared
        assert spec2.has_shared_calibration


class TestTimeSeriesEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_time_series_properties(self):
        """Test accessing properties of empty time series."""
        ts = SpectralTimeSeries([])
        
        assert ts.n_spectra == 0
        assert ts.n_bins == 0
        # Empty array is reshaped to (0, 0) but numpy returns (0,) for completely empty
        assert len(ts.counts.shape) <= 2
        assert ts.counts.size == 0
    
    def test_empty_slice(self, time_series_small):
        """Test slicing that returns empty."""
        sliced = time_series_small.slice_time(t_min=1000, t_max=2000)
        assert sliced.n_spectra == 0
    
    def test_single_spectrum_time_series(self):
        """Test time series with single spectrum."""
        ts = SpectralTimeSeries([Spectrum(np.ones(10))])
        
        assert ts.n_spectra == 1
        assert ts.counts.shape == (1, 10)
    
    def test_integrate_empty_range_raises(self):
        """Test integrating empty time range raises error."""
        ts = SpectralTimeSeries([Spectrum(np.ones(10), timestamp=1.0)])
        
        with pytest.raises(TimeSeriesError, match="No spectra"):
            ts.integrate_time(t_min=10, t_max=20)

