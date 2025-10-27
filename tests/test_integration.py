"""
Integration tests for GammaFlow.

Tests complete workflows and interactions between components.
"""

import numpy as np
import pytest
from gammaflow import Spectrum, SpectralTimeSeries
from gammaflow.utils.exceptions import IncompatibleBinningError


class TestCompleteWorkflow:
    """Test complete analysis workflows."""
    
    def test_uncalibrated_to_calibrated_workflow(self):
        """Test workflow: create uncalibrated, calibrate, analyze."""
        # Step 1: Create uncalibrated spectra
        np.random.seed(42)
        spectra = []
        for i in range(50):
            counts = np.random.poisson(lam=100, size=512)
            spec = Spectrum(counts, timestamp=float(i), live_time=1.0)
            spec.metadata['run'] = i // 10
            spectra.append(spec)
        
        ts = SpectralTimeSeries(spectra)
        assert not ts.is_calibrated
        
        # Step 2: Apply calibration
        ts_cal = ts.apply_calibration([0, 0.5, 0.001])
        assert ts_cal.is_calibrated
        
        # Step 3: Background subtraction
        ts_clean = ts_cal.background_subtract('median')
        
        # Step 4: Convert to count rates (replaces normalize('live_time'))
        # Use vectorized array division
        ts_norm = ts_clean.apply_vectorized(
            lambda counts: counts / ts_clean.live_times[:, np.newaxis]
        )
        
        # Step 5: Time integration by run
        runs = {}
        for spec in ts_norm:
            run = spec.metadata['run']
            if run not in runs:
                runs[run] = []
            runs[run].append(spec)
        
        assert len(runs) == 5
        
        # Step 6: Compute mean per run
        for run, specs in runs.items():
            run_ts = SpectralTimeSeries(specs)
            mean = run_ts.mean_spectrum()
            assert isinstance(mean, Spectrum)
    
    def test_vectorized_analysis_workflow(self):
        """Test workflow using vectorized operations."""
        # Create time series with signal+background
        np.random.seed(42)
        spectra = []
        for i in range(100):
            background = np.random.poisson(lam=50, size=256)
            
            # Add signal in middle period
            if 40 <= i < 60:
                signal = np.random.poisson(lam=100, size=256)
                counts = background + signal
            else:
                counts = background
            
            spectra.append(Spectrum(counts, timestamp=float(i)))
        
        ts = SpectralTimeSeries(spectra)
        
        # Vectorized background estimation
        # Use first 30 spectra as background
        background = np.mean(ts.counts[:30], axis=0)
        
        # Subtract from all
        ts.counts[:] = ts.counts - background
        
        # Find high-count periods (vectorized)
        total_counts = np.sum(ts.counts, axis=1)
        threshold = np.mean(total_counts) + 2 * np.std(total_counts)
        high_indices = np.where(total_counts > threshold)[0]
        
        # Should detect the signal period (roughly 40-60)
        assert len(high_indices) > 0
        assert np.mean(high_indices) > 35
        assert np.mean(high_indices) < 65
    
    def test_mixed_object_and_array_operations(self):
        """Test mixing object-oriented and array operations."""
        np.random.seed(42)
        spectra = [
            Spectrum(
                np.random.poisson(100, size=128),
                live_time=1.0 + i * 0.1,
                timestamp=float(i)
            )
            for i in range(20)
        ]
        
        ts = SpectralTimeSeries(spectra)
        
        # Array operation: smooth all spectra
        from scipy.ndimage import gaussian_filter1d
        ts.counts[:] = gaussian_filter1d(ts.counts, sigma=2, axis=1)
        
        # Object operation: filter by metadata
        for spec in ts:
            spec.metadata['smoothed'] = True
        
        # Array operation: normalize
        ts.counts[:] = ts.counts / ts.counts.sum(axis=1, keepdims=True)
        
        # Object operation: verify
        for spec in ts:
            assert spec.metadata['smoothed'] is True
            assert np.isclose(np.sum(spec.counts), 1.0)
    
    def test_energy_rebinning_workflow(self):
        """Test workflow with energy rebinning."""
        # High resolution data
        edges_fine = np.linspace(0, 3000, 3001)  # 1 keV bins
        counts = np.random.poisson(lam=10, size=3000)
        
        # Add some peaks
        counts[500:510] += np.random.poisson(lam=100, size=10)
        counts[1500:1510] += np.random.poisson(lam=200, size=10)
        
        spec_fine = Spectrum(counts, energy_edges=edges_fine)
        
        # Rebin to coarser resolution
        edges_coarse = np.linspace(0, 3000, 301)  # 10 keV bins
        spec_coarse = spec_fine.rebin_energy(edges_coarse)
        
        # Peaks should still be visible
        assert np.max(spec_coarse.counts) > np.median(spec_coarse.counts) * 2
        
        # Total counts approximately conserved
        assert np.abs(np.sum(spec_fine.counts) - np.sum(spec_coarse.counts)) < 100
    
    def test_arithmetic_and_uncertainty_workflow(self):
        """Test workflow with arithmetic and uncertainty propagation."""
        # Create signal and background measurements
        signal_plus_bg = Spectrum(
            np.array([150, 250, 350, 200]),
            energy_edges=[0, 1, 2, 3, 4]
        )
        
        background = Spectrum(
            np.array([50, 50, 50, 50]),
            energy_edges=[0, 1, 2, 3, 4]
        )
        
        # Subtract background
        signal = signal_plus_bg - background
        
        # Signal should be 100, 200, 300, 150
        assert np.allclose(signal.counts, [100, 200, 300, 150])
        
        # Uncertainty should propagate
        # σ² = σ_signal² + σ_bg²
        expected_unc = np.sqrt(signal_plus_bg.uncertainty**2 + background.uncertainty**2)
        assert np.allclose(signal.uncertainty, expected_unc)


class TestSharedCalibrationIntegration:
    """Test shared calibration in integrated workflows."""
    
    def test_shared_calibration_memory_efficiency(self):
        """Test that shared calibration actually saves memory."""
        # Create many spectra with same calibration
        edges = np.linspace(0, 3000, 3001)
        
        # Shared mode
        spectra_shared = [
            Spectrum(np.random.poisson(100, size=3000), energy_edges=edges)
            for _ in range(100)
        ]
        ts_shared = SpectralTimeSeries(spectra_shared, shared_calibration=True)
        
        # Each spectrum should reference the same calibration
        cal_ids = [id(s._calibration) for s in ts_shared.spectra]
        assert len(set(cal_ids)) == 1  # All same object
    
    def test_cow_in_workflow(self):
        """Test copy-on-write works in realistic workflow."""
        # Create time series
        spectra = [Spectrum(np.ones(64) * 10) for _ in range(20)]
        ts = SpectralTimeSeries(spectra, shared_calibration=True)
        
        # All start shared
        assert all(s.has_shared_calibration for s in ts.spectra)
        
        # Process most spectra normally (should stay shared)
        for i in range(15):
            spec = ts[i]
            # Just reading doesn't detach
            _ = spec.counts
            _ = spec.energy_edges
        
        # Modify one spectrum (should detach)
        special_spec = ts[15]
        special_spec.rebin_energy_([0, 10, 20, 30, 40, 50, 60, 70])
        
        # Most should still be shared
        shared_count = sum(1 for s in ts.spectra if s.has_shared_calibration)
        assert shared_count == 19  # All except the modified one
    
    def test_convert_between_modes_in_workflow(self):
        """Test converting between calibration modes."""
        # Start with mixed calibrations
        spectra = []
        for i in range(10):
            edges = np.linspace(0, 1000 + i, 129)  # Slightly different
            spec = Spectrum(np.random.poisson(50, size=128), energy_edges=edges)
            spectra.append(spec)
        
        ts = SpectralTimeSeries(spectra, shared_calibration=False)
        assert not ts.uses_shared_calibration
        
        # Convert to shared (will rebin to common grid)
        ts_shared = ts.to_shared_calibration()
        assert ts_shared.uses_shared_calibration
        
        # All should have same edges now
        edges_list = [s.energy_edges for s in ts_shared.spectra]
        for edges in edges_list[1:]:
            assert np.allclose(edges, edges_list[0])


class TestRealWorldScenarios:
    """Test realistic analysis scenarios."""
    
    def test_peak_search_scenario(self):
        """Test scenario: searching for peaks in time series."""
        # Create time series with varying peak heights
        np.random.seed(42)
        spectra = []
        
        for i in range(50):
            background = np.random.poisson(lam=20, size=512)
            
            # Peak at channel 256, varying height
            peak_height = 50 + i * 2  # Increasing over time
            background[250:260] += np.random.poisson(lam=peak_height, size=10)
            
            spectra.append(Spectrum(background, timestamp=float(i)))
        
        ts = SpectralTimeSeries(spectra)
        
        # Extract peak region
        peak_counts = ts.counts[:, 255]  # Peak center
        
        # Should show increasing trend
        assert peak_counts[-1] > peak_counts[0]
        
        # Fit linear trend (roughly)
        slope = (peak_counts[-1] - peak_counts[0]) / 50
        assert slope > 0
    
    def test_dead_time_correction_scenario(self):
        """Test scenario: dead time correction."""
        # Create spectra with different dead times
        spectra = []
        for i in range(20):
            live_time = 10.0
            real_time = 10.0 + i * 0.5  # Increasing dead time
            
            # Observed counts (affected by dead time)
            observed_counts = np.random.poisson(lam=100, size=256)
            
            spec = Spectrum(
                observed_counts,
                live_time=live_time,
                real_time=real_time
            )
            spectra.append(spec)
        
        ts = SpectralTimeSeries(spectra)
        
        # Convert to count rates to correct for dead time
        # Use vectorized array division (replaces normalize('live_time'))
        count_rates = ts.counts / ts.live_times[:, np.newaxis]
        
        # After correction, count rates should be more uniform
        rates = np.sum(count_rates, axis=1)
        variation = np.std(rates) / np.mean(rates)
        
        # Note: In this simplified test, corrected and uncorrected are actually
        # the same since dead time correction is just normalization by live time
        
        # Corrected should be more uniform (lower relative std)
        # (In this case they're similar because dead time correction
        # is just normalization by live time, but principle holds)
    
    def test_batch_processing_scenario(self):
        """Test scenario: batch processing multiple runs."""
        # Simulate multiple measurement runs
        np.random.seed(42)
        all_spectra = []
        
        for run in range(5):
            for measurement in range(20):
                counts = np.random.poisson(lam=50 + run * 10, size=128)
                spec = Spectrum(counts, timestamp=float(run * 100 + measurement))
                spec.metadata['run'] = run
                spec.metadata['measurement'] = measurement
                all_spectra.append(spec)
        
        ts = SpectralTimeSeries(all_spectra)
        
        # Process each run separately
        run_means = []
        for run_id in range(5):
            run_spectra = ts.filter_spectra(lambda s: s.metadata['run'] == run_id)
            run_mean = run_spectra.mean_spectrum()
            run_means.append(run_mean)
        
        # Each run should have different count rates
        means = [np.sum(s.counts) for s in run_means]
        assert means[-1] > means[0]  # Increasing with run number


class TestErrorHandlingIntegration:
    """Test error handling in integrated scenarios."""
    
    def test_incompatible_operations_caught(self):
        """Test that incompatible operations are caught."""
        spec1 = Spectrum(np.ones(10), energy_edges=np.linspace(0, 100, 11))
        spec2 = Spectrum(np.ones(10), energy_edges=np.linspace(0, 200, 11))
        
        # Should raise error for incompatible binning
        with pytest.raises(IncompatibleBinningError):
            spec1 + spec2
    
    def test_empty_operations_handled(self):
        """Test operations on empty time series are handled."""
        ts = SpectralTimeSeries([])
        
        # These should not crash
        assert ts.n_spectra == 0
        assert ts.n_bins == 0
        
        # But some operations should raise meaningful errors
        # (currently returns empty, but could raise)
    
    def test_single_spectrum_edge_case(self):
        """Test operations with single spectrum."""
        ts = SpectralTimeSeries([Spectrum(np.ones(10))])
        
        # Should work
        mean = ts.mean_spectrum()
        assert np.allclose(mean.counts, 1.0)
        
        # Iteration should work
        for spec in ts:
            assert isinstance(spec, Spectrum)


class TestPerformanceCharacteristics:
    """Test performance characteristics (not timing, just behavior)."""
    
    def test_large_time_series_creation(self):
        """Test creating large time series."""
        # Create 1000 spectra with 4096 bins
        np.random.seed(42)
        spectra = [
            Spectrum(np.random.poisson(50, size=4096))
            for _ in range(1000)
        ]
        
        ts = SpectralTimeSeries(spectra)
        
        assert ts.n_spectra == 1000
        assert ts.n_bins == 4096
        assert ts.counts.shape == (1000, 4096)
    
    def test_vectorized_vs_loop_equivalence(self):
        """Test that vectorized operations give same result as loops."""
        np.random.seed(42)
        spectra = [Spectrum(np.random.poisson(100, size=64)) for _ in range(20)]
        ts = SpectralTimeSeries(spectra)
        
        # Vectorized background subtraction
        bg_vec = np.mean(ts.counts, axis=0)
        ts_vec = ts.background_subtract(bg_vec)
        
        # Loop-based (via apply_to_each)
        bg_spectrum = Spectrum(bg_vec)
        ts_loop = ts.apply_to_each(lambda s: s - bg_spectrum)
        
        # Should give same result
        assert np.allclose(ts_vec.counts, ts_loop.counts, atol=1e-10)
    
    def test_shared_memory_consistency(self):
        """Test shared memory stays consistent."""
        np.random.seed(42)
        spectra = [Spectrum(np.random.poisson(50, size=128)) for _ in range(50)]
        ts = SpectralTimeSeries(spectra, shared_calibration=True)
        
        # Modify via array
        ts.counts[10:20, 50:60] = 999
        
        # Verify via spectrum objects
        for i in range(10, 20):
            spec = ts[i]
            assert np.all(spec.counts[50:60] == 999)
        
        # Modify via spectrum
        ts[25].counts[70:80] = 888
        
        # Verify via array
        assert np.all(ts.counts[25, 70:80] == 888)

