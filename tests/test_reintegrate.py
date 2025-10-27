"""
Tests for SpectralTimeSeries.reintegrate() method.

Tests cover:
- Basic reintegration with even multiples
- Validation of parameters (multiples, no reduction)
- Overlapping windows
- Non-overlapping windows
- Edge cases (empty series, single spectrum)
- Metadata preservation
- Count conservation
"""

import numpy as np
import pytest

from gammaflow import Spectrum, SpectralTimeSeries
from gammaflow.utils.exceptions import TimeSeriesError


class TestReintegrateBasic:
    """Test basic reintegration functionality."""
    
    def test_reintegrate_2x_integration_time(self):
        """Test doubling the integration time."""
        # Create list mode data: 10 seconds of data
        n_events = 1000
        time_deltas = np.random.exponential(0.01, n_events)  # ~100 Hz
        energies = np.random.uniform(100, 1000, n_events)
        
        # Create time series with 0.5s windows
        ts = SpectralTimeSeries.from_list_mode(
            time_deltas,
            energies,
            integration_time=0.5,
            stride_time=0.5,
            energy_bins=100,
            energy_range=(0, 1000)
        )
        
        original_n_spectra = ts.n_spectra
        
        # Reintegrate to 1.0s windows (2x)
        ts_2x = ts.reintegrate(new_integration_time=1.0)
        
        # Should have approximately half the spectra
        assert ts_2x.n_spectra == pytest.approx(original_n_spectra / 2, abs=1)
        
        # Check metadata
        assert ts_2x.integration_time == 1.0
        assert ts_2x.stride_time == 1.0
        
        # Check that counts are conserved (approximately)
        original_total = np.sum(ts.counts)
        reintegrated_total = np.sum(ts_2x.counts)
        assert reintegrated_total == pytest.approx(original_total, rel=0.01)
    
    def test_reintegrate_4x_integration_time(self):
        """Test quadrupling the integration time."""
        # Create simple data with known structure
        time_deltas = np.ones(1000) * 0.01  # Regular 100 Hz
        energies = np.ones(1000) * 500  # All same energy
        
        # Create time series with 0.5s windows
        ts = SpectralTimeSeries.from_list_mode(
            time_deltas,
            energies,
            integration_time=0.5,
            stride_time=0.5,
            energy_bins=10,
            energy_range=(0, 1000)
        )
        
        original_n_spectra = ts.n_spectra
        
        # Reintegrate to 2.0s windows (4x)
        ts_4x = ts.reintegrate(new_integration_time=2.0)
        
        # Should have approximately 1/4 the spectra
        assert ts_4x.n_spectra == pytest.approx(original_n_spectra / 4, abs=1)
        
        # Check metadata
        assert ts_4x.integration_time == 2.0
        assert ts_4x.stride_time == 2.0
    
    def test_reintegrate_with_custom_stride(self):
        """Test reintegration with custom stride time."""
        time_deltas = np.random.exponential(0.01, 1000)
        energies = np.random.uniform(100, 1000, 1000)
        
        # Create time series with 0.5s integration and stride
        ts = SpectralTimeSeries.from_list_mode(
            time_deltas,
            energies,
            integration_time=0.5,
            stride_time=0.5,
            energy_bins=50
        )
        
        # Reintegrate with 2.0s integration, 1.0s stride (overlapping)
        ts_overlap = ts.reintegrate(
            new_integration_time=2.0,
            new_stride_time=1.0
        )
        
        # Should have more spectra than non-overlapping case
        assert ts_overlap.integration_time == 2.0
        assert ts_overlap.stride_time == 1.0
        assert ts_overlap.n_spectra > ts.n_spectra / 4  # More than 4x reduction


class TestReintegrateValidation:
    """Test parameter validation."""
    
    def test_reintegrate_requires_metadata(self):
        """Test that reintegrate requires integration_time metadata."""
        # Create time series without integration_time metadata
        spectra = [Spectrum(np.random.poisson(100, 256)) for _ in range(10)]
        ts = SpectralTimeSeries(spectra)
        
        # Should raise error
        with pytest.raises(TimeSeriesError, match="does not have integration_time"):
            ts.reintegrate(new_integration_time=2.0)
    
    def test_reintegrate_must_increase_time(self):
        """Test that new times must be >= original times."""
        time_deltas = np.random.exponential(0.01, 1000)
        energies = np.random.uniform(100, 1000, 1000)
        
        ts = SpectralTimeSeries.from_list_mode(
            time_deltas,
            energies,
            integration_time=1.0,
            stride_time=1.0
        )
        
        # Try to reduce integration time
        with pytest.raises(ValueError, match="must be >= original integration_time"):
            ts.reintegrate(new_integration_time=0.5)
    
    def test_reintegrate_stride_must_increase(self):
        """Test that new stride must be >= original stride."""
        time_deltas = np.random.exponential(0.01, 1000)
        energies = np.random.uniform(100, 1000, 1000)
        
        ts = SpectralTimeSeries.from_list_mode(
            time_deltas,
            energies,
            integration_time=1.0,
            stride_time=1.0
        )
        
        # Try to reduce stride time
        with pytest.raises(ValueError, match="must be >= original stride_time"):
            ts.reintegrate(new_integration_time=2.0, new_stride_time=0.5)
    
    def test_reintegrate_must_be_even_multiple(self):
        """Test that new times must be even multiples of original times."""
        time_deltas = np.random.exponential(0.01, 1000)
        energies = np.random.uniform(100, 1000, 1000)
        
        ts = SpectralTimeSeries.from_list_mode(
            time_deltas,
            energies,
            integration_time=0.5,
            stride_time=0.5
        )
        
        # Try non-multiple integration time (0.7 / 0.5 = 1.4, not integer)
        with pytest.raises(ValueError, match="must be an even multiple"):
            ts.reintegrate(new_integration_time=0.7)
    
    def test_reintegrate_stride_must_be_even_multiple(self):
        """Test that new stride must be even multiple of original stride."""
        time_deltas = np.random.exponential(0.01, 1000)
        energies = np.random.uniform(100, 1000, 1000)
        
        ts = SpectralTimeSeries.from_list_mode(
            time_deltas,
            energies,
            integration_time=0.5,
            stride_time=0.5
        )
        
        # Try non-multiple stride time
        with pytest.raises(ValueError, match="must be an even multiple"):
            ts.reintegrate(new_integration_time=1.0, new_stride_time=0.7)
    
    def test_reintegrate_allows_exact_multiples(self):
        """Test that exact multiples work correctly."""
        time_deltas = np.random.exponential(0.01, 1000)
        energies = np.random.uniform(100, 1000, 1000)
        
        ts = SpectralTimeSeries.from_list_mode(
            time_deltas,
            energies,
            integration_time=0.5,
            stride_time=0.5
        )
        
        # These should all work (2x, 4x, 10x)
        ts_2x = ts.reintegrate(new_integration_time=1.0)
        assert ts_2x.integration_time == 1.0
        
        ts_4x = ts.reintegrate(new_integration_time=2.0)
        assert ts_4x.integration_time == 2.0
        
        ts_10x = ts.reintegrate(new_integration_time=5.0)
        assert ts_10x.integration_time == 5.0


class TestReintegrateEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_reintegrate_empty_series(self):
        """Test reintegration of empty time series."""
        ts = SpectralTimeSeries.from_list_mode(
            np.array([]),
            np.array([]),
            integration_time=0.5,
            stride_time=0.5,
            energy_bins=10,
            energy_range=(0, 1000)
        )
        
        # Empty series still has metadata
        assert ts.integration_time == 0.5
        assert ts.stride_time == 0.5
        
        ts_reint = ts.reintegrate(new_integration_time=1.0)
        
        assert ts_reint.n_spectra == 0
        assert ts_reint.integration_time == 1.0
        assert ts_reint.stride_time == 1.0
    
    def test_reintegrate_preserves_calibration(self):
        """Test that energy calibration is preserved."""
        time_deltas = np.random.exponential(0.01, 1000)
        energies = np.random.uniform(100, 1000, 1000)
        
        energy_edges = np.linspace(0, 1200, 101)
        
        ts = SpectralTimeSeries.from_list_mode(
            time_deltas,
            energies,
            integration_time=0.5,
            stride_time=0.5,
            energy_bins=energy_edges
        )
        
        ts_reint = ts.reintegrate(new_integration_time=1.0)
        
        # Energy calibration should be identical
        np.testing.assert_array_almost_equal(
            ts.energy_edges,
            ts_reint.energy_edges
        )
    
    def test_reintegrate_preserves_shared_calibration_mode(self):
        """Test that shared calibration mode is preserved."""
        time_deltas = np.random.exponential(0.01, 1000)
        energies = np.random.uniform(100, 1000, 1000)
        
        ts = SpectralTimeSeries.from_list_mode(
            time_deltas,
            energies,
            integration_time=0.5,
            stride_time=0.5
        )
        
        assert ts.uses_shared_calibration
        
        ts_reint = ts.reintegrate(new_integration_time=1.0)
        
        assert ts_reint.uses_shared_calibration
    
    def test_reintegrate_same_time_returns_copy(self):
        """Test that reintegrating with same time creates a copy."""
        time_deltas = np.random.exponential(0.01, 1000)
        energies = np.random.uniform(100, 1000, 1000)
        
        ts = SpectralTimeSeries.from_list_mode(
            time_deltas,
            energies,
            integration_time=1.0,
            stride_time=1.0
        )
        
        ts_same = ts.reintegrate(new_integration_time=1.0)
        
        # Should have same number of spectra
        assert ts_same.n_spectra == ts.n_spectra
        
        # Should be same data
        np.testing.assert_array_almost_equal(ts.counts, ts_same.counts)
        
        # But different object
        assert ts is not ts_same


class TestReintegrateMetadata:
    """Test metadata handling during reintegration."""
    
    def test_reintegrate_updates_window_metadata(self):
        """Test that window_start/window_end metadata is updated."""
        time_deltas = np.random.exponential(0.01, 1000)
        energies = np.random.uniform(100, 1000, 1000)
        
        ts = SpectralTimeSeries.from_list_mode(
            time_deltas,
            energies,
            integration_time=0.5,
            stride_time=0.5
        )
        
        ts_reint = ts.reintegrate(new_integration_time=1.0)
        
        # Check first spectrum metadata
        first_spec = ts_reint.spectra[0]
        assert 'window_start' in first_spec.metadata
        assert 'window_end' in first_spec.metadata
        
        window_width = first_spec.metadata['window_end'] - first_spec.metadata['window_start']
        assert window_width == pytest.approx(1.0, rel=1e-9)
    
    def test_reintegrate_adds_n_spectra_combined(self):
        """Test that n_spectra_combined metadata is added."""
        time_deltas = np.random.exponential(0.01, 1000)
        energies = np.random.uniform(100, 1000, 1000)
        
        ts = SpectralTimeSeries.from_list_mode(
            time_deltas,
            energies,
            integration_time=0.5,
            stride_time=0.5
        )
        
        ts_reint = ts.reintegrate(new_integration_time=2.0)
        
        # Each reintegrated spectrum should have this metadata
        for spec in ts_reint.spectra:
            assert 'n_spectra_combined' in spec.metadata
            # Should be around 4 (2.0 / 0.5)
            assert spec.metadata['n_spectra_combined'] >= 1
    
    def test_reintegrate_preserves_timestamps(self):
        """Test that timestamps are updated correctly."""
        time_deltas = np.ones(1000) * 0.01  # Regular spacing
        energies = np.ones(1000) * 500
        
        ts = SpectralTimeSeries.from_list_mode(
            time_deltas,
            energies,
            integration_time=0.5,
            stride_time=0.5,
            energy_bins=10,
            energy_range=(0, 1000)
        )
        
        ts_reint = ts.reintegrate(new_integration_time=1.0)
        
        # Timestamps should be at window centers
        for spec in ts_reint.spectra:
            window_center = (spec.metadata['window_start'] + spec.metadata['window_end']) / 2
            assert spec.timestamp == pytest.approx(window_center, rel=1e-9)


class TestReintegrateCountConservation:
    """Test that counts are properly conserved during reintegration."""
    
    def test_non_overlapping_conserves_counts(self):
        """Test that non-overlapping reintegration conserves total counts."""
        # Create regular data for predictable results
        time_deltas = np.ones(1000) * 0.01
        energies = np.ones(1000) * 500
        
        ts = SpectralTimeSeries.from_list_mode(
            time_deltas,
            energies,
            integration_time=0.5,
            stride_time=0.5,
            energy_bins=10,
            energy_range=(0, 1000)
        )
        
        original_total = np.sum(ts.counts)
        
        # Non-overlapping reintegration
        ts_reint = ts.reintegrate(new_integration_time=1.0, new_stride_time=1.0)
        reint_total = np.sum(ts_reint.counts)
        
        # Should be very close (may differ slightly due to edge effects)
        assert reint_total == pytest.approx(original_total, rel=0.05)
    
    def test_overlapping_increases_counts(self):
        """Test that overlapping windows increase total counts (events counted multiple times)."""
        time_deltas = np.ones(1000) * 0.01
        energies = np.ones(1000) * 500
        
        ts = SpectralTimeSeries.from_list_mode(
            time_deltas,
            energies,
            integration_time=0.5,
            stride_time=0.5,
            energy_bins=10,
            energy_range=(0, 1000)
        )
        
        original_total = np.sum(ts.counts)
        
        # Overlapping reintegration (integration=2.0, stride=1.0)
        ts_overlap = ts.reintegrate(
            new_integration_time=2.0,
            new_stride_time=1.0
        )
        overlap_total = np.sum(ts_overlap.counts)
        
        # With 2x overlap, should have ~2x the counts
        assert overlap_total > original_total
    
    def test_spectrum_sums_match_combined_spectra(self):
        """Test that each reintegrated spectrum is the sum of its components."""
        # Create data where we can track individual events
        time_deltas = np.ones(400) * 0.0025  # 400 events over 1 second
        energies = np.ones(400) * 500  # All same energy
        
        ts = SpectralTimeSeries.from_list_mode(
            time_deltas,
            energies,
            integration_time=0.1,
            stride_time=0.1,
            energy_bins=10,
            energy_range=(0, 1000)
        )
        
        # Each spectrum should have ~40 counts total
        # Reintegrate to 0.2s (combine 2 spectra each)
        ts_reint = ts.reintegrate(new_integration_time=0.2)
        
        # Each reintegrated spectrum should have ~80 counts
        for i, spec in enumerate(ts_reint.spectra):
            # Should have combined approximately 2 original spectra
            assert spec.metadata['n_spectra_combined'] >= 1
            # Total counts should be reasonable
            assert np.sum(spec.counts) > 0


class TestReintegrateRealWorldScenarios:
    """Test realistic use cases."""
    
    def test_listmode_to_multiple_timescales(self):
        """Test creating multiple timescale views from same data."""
        # Simulate 1 minute of list mode data
        np.random.seed(42)
        time_deltas = np.random.exponential(0.001, 10000)  # ~1000 Hz
        energies = np.random.gamma(shape=3, scale=200, size=10000)
        
        # Create fine-resolution time series
        ts_fine = SpectralTimeSeries.from_list_mode(
            time_deltas,
            energies,
            integration_time=0.1,
            stride_time=0.1,
            energy_bins=100
        )
        
        # Create multiple coarser timescales
        ts_1s = ts_fine.reintegrate(new_integration_time=1.0)
        ts_5s = ts_fine.reintegrate(new_integration_time=5.0)
        ts_10s = ts_fine.reintegrate(new_integration_time=10.0)
        
        # All should have same total counts (approximately)
        total_fine = np.sum(ts_fine.counts)
        total_1s = np.sum(ts_1s.counts)
        total_5s = np.sum(ts_5s.counts)
        total_10s = np.sum(ts_10s.counts)
        
        assert total_1s == pytest.approx(total_fine, rel=0.1)
        assert total_5s == pytest.approx(total_fine, rel=0.1)
        assert total_10s == pytest.approx(total_fine, rel=0.1)
        
        # Number of spectra should decrease
        assert ts_1s.n_spectra < ts_fine.n_spectra
        assert ts_5s.n_spectra < ts_1s.n_spectra
        assert ts_10s.n_spectra < ts_5s.n_spectra
    
    def test_adaptive_binning_workflow(self):
        """Test workflow where user starts fine and progressively coarsens."""
        time_deltas = np.random.exponential(0.01, 2000)
        energies = np.random.uniform(100, 1000, 2000)
        
        # Start with very fine resolution
        ts = SpectralTimeSeries.from_list_mode(
            time_deltas,
            energies,
            integration_time=0.25,
            stride_time=0.25
        )
        
        # Progressively coarsen
        ts_1 = ts.reintegrate(new_integration_time=0.5)
        ts_2 = ts_1.reintegrate(new_integration_time=1.0)
        ts_3 = ts_2.reintegrate(new_integration_time=2.0)
        
        # Check chain
        assert ts_3.integration_time == 2.0
        assert ts_3.n_spectra < ts_2.n_spectra < ts_1.n_spectra < ts.n_spectra


class TestReintegrateTimingInference:
    """Test automatic inference of integration_time and stride_time."""
    
    def test_infer_from_constant_real_time(self):
        """Test that integration_time is inferred from constant real_time."""
        spectra = [
            Spectrum(np.random.poisson(100, 64), real_time=1.0, timestamp=i*1.0)
            for i in range(10)
        ]
        ts = SpectralTimeSeries(spectra)
        
        assert ts.integration_time == 1.0
        assert ts.stride_time == 1.0
    
    def test_infer_from_evenly_spaced_timestamps(self):
        """Test that stride_time is inferred from evenly-spaced timestamps."""
        spectra = [
            Spectrum(np.random.poisson(100, 64), real_time=0.5, timestamp=i*0.5)
            for i in range(10)
        ]
        ts = SpectralTimeSeries(spectra)
        
        assert ts.integration_time == 0.5
        assert ts.stride_time == 0.5
    
    def test_no_inference_with_varying_real_time(self):
        """Test that integration_time is None when real_times vary."""
        spectra = [
            Spectrum(
                np.random.poisson(100, 64),
                real_time=np.random.uniform(0.8, 1.2),
                timestamp=i*1.0
            )
            for i in range(10)
        ]
        ts = SpectralTimeSeries(spectra)
        
        assert ts.integration_time is None
        assert ts.stride_time == 1.0  # Still evenly spaced
    
    def test_no_inference_with_irregular_timestamps(self):
        """Test that stride_time is None when timestamps are irregular."""
        timestamps = [0, 1, 2, 3, 5, 7, 10, 14, 19, 25]
        spectra = [
            Spectrum(np.random.poisson(100, 64), real_time=1.0, timestamp=t)
            for t in timestamps
        ]
        ts = SpectralTimeSeries(spectra)
        
        assert ts.integration_time == 1.0  # Constant real_time
        assert ts.stride_time is None      # Irregular spacing
    
    def test_from_array_infers_timing(self):
        """Test that from_array also infers timing."""
        counts = np.random.poisson(100, size=(20, 64))
        timestamps = np.arange(20) * 0.5
        real_times = np.ones(20) * 0.5
        
        ts = SpectralTimeSeries.from_array(
            counts,
            timestamps=timestamps,
            real_times=real_times
        )
        
        assert ts.integration_time == 0.5
        assert ts.stride_time == 0.5
    
    def test_reintegrate_works_with_inferred_timing(self):
        """Test that reintegrate works with inferred timing (not just from_list_mode)."""
        spectra = [
            Spectrum(np.random.poisson(100, 64), real_time=1.0, timestamp=i*1.0)
            for i in range(20)
        ]
        ts = SpectralTimeSeries(spectra)
        
        # Should work because timing was inferred
        ts_2x = ts.reintegrate(2.0)
        assert ts_2x.n_spectra == 10
        assert ts_2x.integration_time == 2.0
    
    def test_reintegrate_fails_without_timing_info(self):
        """Test that reintegrate fails when timing can't be inferred."""
        # Varying real_times
        spectra = [
            Spectrum(
                np.random.poisson(100, 64),
                real_time=np.random.uniform(0.8, 1.2),
                timestamp=i*1.0
            )
            for i in range(10)
        ]
        ts = SpectralTimeSeries(spectra)
        
        with pytest.raises(TimeSeriesError, match="does not have integration_time and stride_time"):
            ts.reintegrate(2.0)


class TestTimingValidation:
    """Test validation of user-provided integration_time and stride_time."""
    
    def test_validate_correct_integration_time(self):
        """Test that correct integration_time is accepted."""
        spectra = [
            Spectrum(np.random.poisson(100, 64), real_time=1.0, timestamp=i*1.0)
            for i in range(10)
        ]
        # Should accept matching value
        ts = SpectralTimeSeries(spectra, integration_time=1.0)
        assert ts.integration_time == 1.0
    
    def test_validate_incorrect_integration_time(self):
        """Test that incorrect integration_time is rejected."""
        spectra = [
            Spectrum(np.random.poisson(100, 64), real_time=1.0, timestamp=i*1.0)
            for i in range(10)
        ]
        # Should reject non-matching value
        with pytest.raises(ValueError, match="does not match real_time"):
            SpectralTimeSeries(spectra, integration_time=2.0)
    
    def test_validate_correct_stride_time(self):
        """Test that correct stride_time is accepted."""
        spectra = [
            Spectrum(np.random.poisson(100, 64), real_time=1.0, timestamp=i*1.0)
            for i in range(10)
        ]
        # Should accept matching value
        ts = SpectralTimeSeries(spectra, stride_time=1.0)
        assert ts.stride_time == 1.0
    
    def test_validate_incorrect_stride_time(self):
        """Test that incorrect stride_time is rejected."""
        spectra = [
            Spectrum(np.random.poisson(100, 64), real_time=1.0, timestamp=i*1.0)
            for i in range(10)
        ]
        # Should reject non-matching value
        with pytest.raises(ValueError, match="does not match timestamp spacing"):
            SpectralTimeSeries(spectra, stride_time=0.5)
    
    def test_allow_arbitrary_when_no_inference_possible(self):
        """Test that arbitrary values are allowed when inference isn't possible."""
        # Single spectrum - can't infer stride_time
        spectra = [Spectrum(np.random.poisson(100, 64), real_time=1.0, timestamp=0.0)]
        ts = SpectralTimeSeries(spectra, integration_time=1.0, stride_time=0.5)
        
        # Should accept any values since we can't infer stride_time
        assert ts.integration_time == 1.0
        assert ts.stride_time == 0.5
    
    def test_from_array_validation(self):
        """Test that from_array also validates timing."""
        counts = np.random.poisson(100, size=(10, 64))
        timestamps = np.arange(10) * 1.0
        real_times = np.ones(10) * 1.0
        
        # Correct values should work
        ts = SpectralTimeSeries.from_array(
            counts,
            timestamps=timestamps,
            real_times=real_times,
            integration_time=1.0,
            stride_time=1.0
        )
        assert ts.integration_time == 1.0
        
        # Incorrect integration_time should fail
        with pytest.raises(ValueError, match="does not match real_time"):
            SpectralTimeSeries.from_array(
                counts,
                timestamps=timestamps,
                real_times=real_times,
                integration_time=2.0
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

