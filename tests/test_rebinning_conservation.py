"""
Tests for verifying physical correctness of rebinning operations.

This test suite specifically validates that histogram rebinning properly
conserves counts, which is a fundamental physical requirement.
"""

import numpy as np
import pytest
from gammaflow import Spectrum


class TestRebinningConservation:
    """Test count conservation during rebinning operations."""
    
    def test_rebin_coarser_bins(self):
        """Test rebinning to coarser bins conserves total counts."""
        # Create spectrum with 1024 bins
        counts = np.random.poisson(100, 1024)
        old_edges = np.linspace(0, 3000, 1025)
        spectrum = Spectrum(counts, energy_edges=old_edges)
        
        # Rebin to 512 bins (2x coarser)
        new_edges = np.linspace(0, 3000, 513)
        rebinned = spectrum.rebin_energy(new_edges, method='histogram')
        
        # Verify count conservation
        assert np.isclose(rebinned.counts.sum(), spectrum.counts.sum()), \
            f"Counts not conserved: {rebinned.counts.sum()} != {spectrum.counts.sum()}"
    
    def test_rebin_finer_bins(self):
        """Test rebinning to finer bins conserves total counts."""
        # Create spectrum with 128 bins
        counts = np.random.poisson(100, 128)
        old_edges = np.linspace(0, 3000, 129)
        spectrum = Spectrum(counts, energy_edges=old_edges)
        
        # Rebin to 256 bins (2x finer)
        new_edges = np.linspace(0, 3000, 257)
        rebinned = spectrum.rebin_energy(new_edges, method='histogram')
        
        # Verify count conservation
        assert np.isclose(rebinned.counts.sum(), spectrum.counts.sum()), \
            f"Counts not conserved: {rebinned.counts.sum()} != {spectrum.counts.sum()}"
    
    def test_rebin_misaligned_bins(self):
        """Test rebinning with non-aligned bins conserves counts."""
        # Create spectrum with regular bins
        counts = np.random.poisson(100, 100)
        old_edges = np.linspace(0, 1000, 101)  # 10 keV bins
        spectrum = Spectrum(counts, energy_edges=old_edges)
        
        # Rebin to misaligned bins (offset by 5 keV)
        new_edges = np.linspace(5, 995, 50)  # 20 keV bins, offset
        rebinned = spectrum.rebin_energy(new_edges, method='histogram')
        
        # Verify approximate count conservation (may lose counts at edges)
        # Should be within 1% due to edge effects
        ratio = rebinned.counts.sum() / spectrum.counts.sum()
        assert 0.99 < ratio < 1.01, \
            f"Counts not conserved within 1%: ratio={ratio}"
    
    def test_rebin_nested_bins_exact(self):
        """Test that perfectly nested bins give exact conservation."""
        # Create spectrum
        counts = np.array([100, 200, 300, 400], dtype=float)
        old_edges = np.array([0, 10, 20, 30, 40], dtype=float)
        spectrum = Spectrum(counts, energy_edges=old_edges)
        
        # Rebin to nested bins (combine 2 bins into 1)
        new_edges = np.array([0, 20, 40], dtype=float)
        rebinned = spectrum.rebin_energy(new_edges, method='histogram')
        
        # Should have exact conservation
        assert rebinned.counts.sum() == spectrum.counts.sum()
        assert rebinned.counts[0] == 300  # 100 + 200
        assert rebinned.counts[1] == 700  # 300 + 400
    
    def test_rebin_fractional_overlap(self):
        """Test rebinning with fractional bin overlap."""
        # Create simple spectrum: 4 bins, 100 counts each
        counts = np.array([100, 100, 100, 100], dtype=float)
        old_edges = np.array([0, 10, 20, 30, 40], dtype=float)
        spectrum = Spectrum(counts, energy_edges=old_edges)
        
        # New bins that split old bins in half
        # New bin [5, 15] should get 50 from [0,10] and 50 from [10,20]
        new_edges = np.array([5, 15, 25, 35], dtype=float)
        rebinned = spectrum.rebin_energy(new_edges, method='histogram')
        
        # Total counts should be conserved
        # But we lose the first 5 keV and last 5 keV
        expected_total = 300  # 3 full old bins worth
        assert np.isclose(rebinned.counts.sum(), expected_total), \
            f"Expected {expected_total}, got {rebinned.counts.sum()}"
        
        # Each new bin should have 100 counts (50 + 50 from adjacent old bins)
        assert np.allclose(rebinned.counts, [100, 100, 100]), \
            f"Expected [100, 100, 100], got {rebinned.counts}"
    
    def test_rebin_single_old_bin_split(self):
        """Test splitting a single old bin into multiple new bins."""
        # One bin with 100 counts
        counts = np.array([100], dtype=float)
        old_edges = np.array([0, 10], dtype=float)
        spectrum = Spectrum(counts, energy_edges=old_edges)
        
        # Split into 4 equal new bins
        new_edges = np.array([0, 2.5, 5.0, 7.5, 10.0], dtype=float)
        rebinned = spectrum.rebin_energy(new_edges, method='histogram')
        
        # Total counts conserved
        assert np.isclose(rebinned.counts.sum(), 100)
        
        # Each new bin should have 25 counts (uniform distribution)
        assert np.allclose(rebinned.counts, [25, 25, 25, 25]), \
            f"Expected [25, 25, 25, 25], got {rebinned.counts}"
    
    def test_rebin_multiple_old_bins_to_one(self):
        """Test combining multiple old bins into one new bin."""
        # 5 bins with different counts
        counts = np.array([10, 20, 30, 40, 50], dtype=float)
        old_edges = np.array([0, 10, 20, 30, 40, 50], dtype=float)
        spectrum = Spectrum(counts, energy_edges=old_edges)
        
        # Combine all into one bin
        new_edges = np.array([0, 50], dtype=float)
        rebinned = spectrum.rebin_energy(new_edges, method='histogram')
        
        # Should sum all counts
        assert rebinned.counts[0] == 150  # 10+20+30+40+50
    
    def test_rebin_partial_overlap_edges(self):
        """Test rebinning where new range doesn't fully cover old range."""
        # 4 bins
        counts = np.array([100, 200, 300, 400], dtype=float)
        old_edges = np.array([0, 10, 20, 30, 40], dtype=float)
        spectrum = Spectrum(counts, energy_edges=old_edges)
        
        # New bins only cover [10, 30]
        new_edges = np.array([10, 20, 30], dtype=float)
        rebinned = spectrum.rebin_energy(new_edges, method='histogram')
        
        # Should only have counts from bins [10-20] and [20-30]
        assert np.isclose(rebinned.counts.sum(), 500)  # 200 + 300
        assert rebinned.counts[0] == 200
        assert rebinned.counts[1] == 300
    
    def test_rebin_complex_overlap(self):
        """Test complex overlapping pattern."""
        # 3 bins: [0-10], [10-20], [20-30]
        counts = np.array([300, 600, 900], dtype=float)
        old_edges = np.array([0, 10, 20, 30], dtype=float)
        spectrum = Spectrum(counts, energy_edges=old_edges)
        
        # New bins with complex overlap: [5-15], [15-25]
        new_edges = np.array([5, 15, 25], dtype=float)
        rebinned = spectrum.rebin_energy(new_edges, method='histogram')
        
        # First new bin [5-15]:
        #   - Gets 50% of [0-10] = 150
        #   - Gets 50% of [10-20] = 300
        #   - Total: 450
        # Second new bin [15-25]:
        #   - Gets 50% of [10-20] = 300
        #   - Gets 50% of [20-30] = 450
        #   - Total: 750
        
        assert np.isclose(rebinned.counts[0], 450), \
            f"First bin: expected 450, got {rebinned.counts[0]}"
        assert np.isclose(rebinned.counts[1], 750), \
            f"Second bin: expected 750, got {rebinned.counts[1]}"
        
        # Total should be 1200 (missing first and last 5 keV)
        assert np.isclose(rebinned.counts.sum(), 1200)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

