"""
Tests for optional live_time functionality.
"""

import numpy as np
import pytest

from gammaflow.core.spectrum import Spectrum
from gammaflow.core.time_series import SpectralTimeSeries


class TestOptionalLiveTime:
    """Test cases for optional live_time parameter."""
    
    def test_spectrum_with_no_times(self):
        """Test Spectrum with neither live_time nor real_time."""
        counts = np.array([100, 200, 150])
        spec = Spectrum(counts)
        
        # Should default to 1.0 for both
        assert spec.live_time == 1.0
        assert spec.real_time == 1.0
        assert spec.dead_time_fraction == 0.0
        
        # Count rate should equal counts
        np.testing.assert_array_equal(spec.count_rate, counts)
    
    def test_spectrum_with_only_live_time(self):
        """Test Spectrum with only live_time provided."""
        counts = np.array([100, 200, 150])
        spec = Spectrum(counts, live_time=10.0)
        
        assert spec.live_time == 10.0
        assert spec.real_time == 10.0  # Should equal live_time
        assert spec.dead_time_fraction == 0.0
        
        # Count rate uses live_time
        np.testing.assert_array_almost_equal(spec.count_rate, counts / 10.0)
    
    def test_spectrum_with_only_real_time(self):
        """Test Spectrum with only real_time provided."""
        counts = np.array([100, 200, 150])
        spec = Spectrum(counts, real_time=10.0)
        
        assert spec.live_time is None
        assert spec.real_time == 10.0
        assert spec.dead_time_fraction == 0.0  # No live_time, so no dead time
        
        # Count rate falls back to real_time
        np.testing.assert_array_almost_equal(spec.count_rate, counts / 10.0)
    
    def test_spectrum_with_both_times(self):
        """Test Spectrum with both live_time and real_time."""
        counts = np.array([100, 200, 150])
        spec = Spectrum(counts, live_time=9.5, real_time=10.0)
        
        assert spec.live_time == 9.5
        assert spec.real_time == 10.0
        assert spec.dead_time_fraction == pytest.approx(0.05)
        
        # Count rate uses live_time (preferred)
        np.testing.assert_array_almost_equal(spec.count_rate, counts / 9.5)
    
    def test_time_series_from_array_no_times(self):
        """Test SpectralTimeSeries.from_array with no time information."""
        counts = np.random.poisson(100, size=(10, 50))
        ts = SpectralTimeSeries.from_array(counts)
        
        # Should default to 1.0 for all
        assert len(ts.live_times) == 10
        assert len(ts.real_times) == 10
        assert all(lt == 1.0 for lt in ts.live_times)
        assert all(rt == 1.0 for rt in ts.real_times)
    
    def test_time_series_from_array_only_real_times(self):
        """Test SpectralTimeSeries.from_array with only real_times."""
        counts = np.random.poisson(100, size=(10, 50))
        real_times = np.linspace(5.0, 15.0, 10)
        
        ts = SpectralTimeSeries.from_array(counts, real_times=real_times)
        
        # live_times should be None
        assert all(lt is None for lt in ts.live_times)
        # real_times should be as provided
        np.testing.assert_array_almost_equal(
            ts.real_times.astype(float), real_times
        )
        
        # Count rate should use real_times
        for i, spec in enumerate(ts):
            expected_rate = counts[i] / real_times[i]
            np.testing.assert_array_almost_equal(spec.count_rate, expected_rate)
    
    def test_time_series_from_array_only_live_times(self):
        """Test SpectralTimeSeries.from_array with only live_times."""
        counts = np.random.poisson(100, size=(10, 50))
        live_times = np.linspace(5.0, 15.0, 10)
        
        ts = SpectralTimeSeries.from_array(counts, live_times=live_times)
        
        # Both should be set to live_times
        np.testing.assert_array_almost_equal(
            ts.live_times.astype(float), live_times
        )
        np.testing.assert_array_almost_equal(ts.real_times, live_times)
        
        # Count rate should use live_times
        for i, spec in enumerate(ts):
            expected_rate = counts[i] / live_times[i]
            np.testing.assert_array_almost_equal(spec.count_rate, expected_rate)
    
    def test_time_series_from_array_both_times(self):
        """Test SpectralTimeSeries.from_array with both time arrays."""
        counts = np.random.poisson(100, size=(10, 50))
        live_times = np.linspace(9.0, 9.5, 10)
        real_times = np.ones(10) * 10.0
        
        ts = SpectralTimeSeries.from_array(
            counts, live_times=live_times, real_times=real_times
        )
        
        # Both should be as provided
        np.testing.assert_array_almost_equal(
            ts.live_times.astype(float), live_times
        )
        np.testing.assert_array_almost_equal(ts.real_times, real_times)
        
        # Count rate should use live_times (preferred)
        for i, spec in enumerate(ts):
            expected_rate = counts[i] / live_times[i]
            np.testing.assert_array_almost_equal(spec.count_rate, expected_rate)
            # Dead time should be calculable
            assert spec.dead_time_fraction > 0
    
    def test_time_series_from_array_scalar_real_time(self):
        """Test SpectralTimeSeries.from_array with scalar real_time."""
        counts = np.random.poisson(100, size=(10, 50))
        
        ts = SpectralTimeSeries.from_array(counts, real_times=10.0)
        
        # live_times should be None
        assert all(lt is None for lt in ts.live_times)
        # real_times should all be 10.0
        assert all(rt == 10.0 for rt in ts.real_times)
        
        # Count rate should use real_time
        for spec in ts:
            assert spec.real_time == 10.0
            assert spec.live_time is None
    
    def test_spectrum_arithmetic_preserves_time_info(self):
        """Test that arithmetic operations preserve time information."""
        counts = np.array([100, 200, 150])
        spec1 = Spectrum(counts, real_time=10.0)
        spec2 = Spectrum(counts * 2, real_time=10.0)
        
        # Addition - times are summed (physically correct for combined acquisition)
        result = spec1 + spec2
        assert result.live_time is None  # None + None = None
        assert result.real_time == 20.0  # 10.0 + 10.0
        
        # Scalar multiplication
        result = spec1 * 2.0
        assert result.live_time is None
        assert result.real_time == 10.0
    
    def test_spectrum_copy_preserves_time_info(self):
        """Test that copy preserves optional live_time."""
        counts = np.array([100, 200, 150])
        spec = Spectrum(counts, real_time=10.0)
        
        # Shallow copy
        copy = spec.copy(deep=False)
        assert copy.live_time is None
        assert copy.real_time == 10.0
        
        # Deep copy
        copy = spec.copy(deep=True)
        assert copy.live_time is None
        assert copy.real_time == 10.0


class TestCountRateFallback:
    """Test count rate calculation with optional live_time."""
    
    def test_count_rate_prefers_live_time(self):
        """Test that count_rate prefers live_time when available."""
        counts = np.array([100, 200, 150])
        spec = Spectrum(counts, live_time=5.0, real_time=10.0)
        
        # Should use live_time (5.0), not real_time (10.0)
        expected = counts / 5.0
        np.testing.assert_array_almost_equal(spec.count_rate, expected)
    
    def test_count_rate_falls_back_to_real_time(self):
        """Test that count_rate uses real_time when live_time is None."""
        counts = np.array([100, 200, 150])
        spec = Spectrum(counts, real_time=10.0)
        
        # Should use real_time
        expected = counts / 10.0
        np.testing.assert_array_almost_equal(spec.count_rate, expected)
    
    def test_dead_time_with_no_live_time(self):
        """Test that dead_time_fraction is 0 when live_time is None."""
        counts = np.array([100, 200, 150])
        spec = Spectrum(counts, real_time=10.0)
        
        # No live_time means no dead time correction possible
        assert spec.dead_time_fraction == 0.0
    
    def test_dead_time_with_live_time(self):
        """Test dead_time_fraction calculation when live_time is available."""
        counts = np.array([100, 200, 150])
        spec = Spectrum(counts, live_time=9.0, real_time=10.0)
        
        # Dead time = (10 - 9) / 10 = 0.1
        assert spec.dead_time_fraction == pytest.approx(0.1)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

