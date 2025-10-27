"""
Tests for scalar time parameters in SpectralTimeSeries.
"""

import numpy as np
import pytest

from gammaflow.core.spectrum import Spectrum
from gammaflow.core.time_series import SpectralTimeSeries


class TestScalarTimesInTimeSeries:
    """Test scalar real_time and live_time parameters in SpectralTimeSeries."""
    
    def test_scalar_real_time_in_init(self):
        """Test providing scalar real_time to __init__."""
        # Create spectra without time info
        spectra = [
            Spectrum(np.array([100, 200, 150])),
            Spectrum(np.array([150, 250, 200])),
            Spectrum(np.array([120, 220, 170])),
        ]
        
        # Create time series with scalar real_time
        ts = SpectralTimeSeries(spectra, real_time=10.0)
        
        # All spectra should have real_time=10.0
        assert ts.n_spectra == 3
        for spec in ts:
            assert spec.real_time == 10.0
        
        # Array should also have all 10.0
        assert all(rt == 10.0 for rt in ts.real_times)
    
    def test_scalar_real_time_preserves_live_time(self):
        """Test that scalar real_time preserves individual live_times when they differ from real_time."""
        # Create spectra with explicit dead time (live_time != real_time)
        spectra = [
            Spectrum(np.array([100, 200, 150]), live_time=9.2, real_time=10.0),
            Spectrum(np.array([150, 250, 200]), live_time=9.5, real_time=10.0),
        ]
        
        # Apply different scalar real_time
        ts = SpectralTimeSeries(spectra, real_time=12.0)
        
        # real_time should be updated, live_time should be preserved
        assert ts[0].real_time == 12.0
        assert ts[1].real_time == 12.0
        assert ts[0].live_time == 9.2  # Preserved (explicit dead time)
        assert ts[1].live_time == 9.5  # Preserved (explicit dead time)
    
    def test_scalar_real_time_overrides_existing(self):
        """Test that scalar real_time overrides existing spectrum times."""
        # Create spectra with different real_times
        spectra = [
            Spectrum(np.array([100, 200, 150]), real_time=5.0),
            Spectrum(np.array([150, 250, 200]), real_time=7.0),
            Spectrum(np.array([120, 220, 170]), real_time=9.0),
        ]
        
        # Create time series with uniform real_time
        ts = SpectralTimeSeries(spectra, real_time=10.0)
        
        # All should now have real_time=10.0 (overridden)
        for spec in ts:
            assert spec.real_time == 10.0
        
        assert all(rt == 10.0 for rt in ts.real_times)
    
    def test_scalar_time_preserves_other_properties(self):
        """Test that applying scalar time preserves other spectrum properties."""
        spectra = [
            Spectrum(
                np.array([100, 200, 150]),
                energy_edges=np.array([0, 1, 2, 3]),
                timestamp=1000.0,
                metadata={'detector': 'A'}
            ),
            Spectrum(
                np.array([150, 250, 200]),
                energy_edges=np.array([0, 1, 2, 3]),
                timestamp=2000.0,
                metadata={'detector': 'B'}
            ),
        ]
        
        # Apply scalar real_time
        ts = SpectralTimeSeries(spectra, real_time=10.0)
        
        # Check that other properties are preserved
        assert ts[0].timestamp == 1000.0
        assert ts[1].timestamp == 2000.0
        assert ts[0].metadata['detector'] == 'A'
        assert ts[1].metadata['detector'] == 'B'
        assert ts[0].is_calibrated
        np.testing.assert_array_equal(
            ts[0].energy_edges, 
            np.array([0, 1, 2, 3])
        )
    
    def test_count_rate_with_scalar_real_time(self):
        """Test count rate calculation with scalar real_time."""
        counts = np.array([100, 200, 150])
        spectra = [
            Spectrum(counts),
            Spectrum(counts * 2),
        ]
        
        # Apply scalar real_time
        ts = SpectralTimeSeries(spectra, real_time=10.0)
        
        # Count rates should use the real_time
        expected_rate_1 = counts / 10.0
        expected_rate_2 = (counts * 2) / 10.0
        
        np.testing.assert_array_almost_equal(ts[0].count_rate, expected_rate_1)
        np.testing.assert_array_almost_equal(ts[1].count_rate, expected_rate_2)
    
    def test_vectorized_operations_with_scalar_time(self):
        """Test vectorized operations work correctly with scalar times."""
        spectra = [
            Spectrum(np.ones(10) * 100),
            Spectrum(np.ones(10) * 200),
            Spectrum(np.ones(10) * 300),
        ]
        
        # Apply scalar real_time
        ts = SpectralTimeSeries(spectra, real_time=10.0)
        
        # Vectorized count rate calculation
        count_rates = ts.counts / ts.real_times[:, np.newaxis]
        
        # Should match individual count rates
        for i, spec in enumerate(ts):
            np.testing.assert_array_almost_equal(
                count_rates[i], 
                spec.count_rate
            )
    
    def test_empty_spectra_list_with_real_time(self):
        """Test that empty spectra list works with real_time parameter."""
        ts = SpectralTimeSeries([], real_time=10.0)
        assert ts.n_spectra == 0
    
    def test_comparison_with_from_array(self):
        """Test that __init__ with scalar time matches from_array behavior."""
        counts_2d = np.random.poisson(100, size=(5, 20))
        
        # Method 1: from_array with scalar real_time
        ts1 = SpectralTimeSeries.from_array(counts_2d, real_times=10.0)
        
        # Method 2: Create spectra then apply scalar real_time
        spectra = [Spectrum(counts_2d[i]) for i in range(5)]
        ts2 = SpectralTimeSeries(spectra, real_time=10.0)
        
        # Both should have same times
        np.testing.assert_array_equal(ts1.real_times, ts2.real_times)
        for i in range(5):
            assert ts1[i].real_time == ts2[i].real_time
    
    def test_real_time_override_preserves_live_time(self):
        """Test that overriding real_time preserves individual live_times."""
        # Create spectra with different live_times (realistic due to dead time)
        spectra = [
            Spectrum(np.array([100, 200]), live_time=9.0, real_time=10.0),
            Spectrum(np.array([150, 250]), live_time=8.5, real_time=10.0),
        ]
        
        # Override real_time
        ts = SpectralTimeSeries(spectra, real_time=12.0)
        
        # real_time should be overridden, but live_time preserved
        assert ts[0].real_time == 12.0
        assert ts[1].real_time == 12.0
        assert ts[0].live_time == 9.0  # Preserved (varies per spectrum)
        assert ts[1].live_time == 8.5  # Preserved (varies per spectrum)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

