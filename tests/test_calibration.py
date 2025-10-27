"""
Tests for EnergyCalibration class.
"""

import numpy as np
import pytest
from gammaflow.core.calibration import EnergyCalibration
from gammaflow.utils.exceptions import CalibrationError


class TestEnergyCalibrationCreation:
    """Test EnergyCalibration creation and validation."""
    
    def test_create_with_edges(self, simple_edges):
        """Test creating calibration with edges."""
        cal = EnergyCalibration(simple_edges)
        assert cal.edges is not None
        assert len(cal.edges) == 5
        assert np.array_equal(cal.edges, simple_edges)
    
    def test_create_without_edges(self):
        """Test creating uncalibrated calibration."""
        cal = EnergyCalibration(None)
        assert cal.edges is None
        assert cal.n_bins is None
    
    def test_create_shared(self, simple_edges):
        """Test creating shared calibration."""
        cal = EnergyCalibration(simple_edges, is_shared=True)
        assert cal._is_shared_flag is True
        assert cal._ref_count == 0  # Not shared yet (no refs)
    
    def test_validate_non_monotonic(self):
        """Test validation catches non-monotonic edges."""
        bad_edges = np.array([0, 1, 3, 2, 4])  # Not monotonic!
        with pytest.raises(CalibrationError, match="monotonically increasing"):
            EnergyCalibration(bad_edges)
    
    def test_validate_too_few_edges(self):
        """Test validation catches too few edges."""
        bad_edges = np.array([0])
        with pytest.raises(CalibrationError, match="at least 2 elements"):
            EnergyCalibration(bad_edges)
    
    def test_validate_decreasing_edges(self):
        """Test validation catches decreasing edges."""
        bad_edges = np.array([4, 3, 2, 1, 0])
        with pytest.raises(CalibrationError, match="monotonically increasing"):
            EnergyCalibration(bad_edges)


class TestEnergyCalibrationProperties:
    """Test EnergyCalibration properties."""
    
    def test_edges_property(self, simple_edges):
        """Test edges property getter."""
        cal = EnergyCalibration(simple_edges)
        assert np.array_equal(cal.edges, simple_edges)
    
    def test_edges_setter(self, simple_edges):
        """Test edges property setter."""
        cal = EnergyCalibration(None)
        cal.edges = simple_edges
        assert np.array_equal(cal.edges, simple_edges)
    
    def test_n_bins_calibrated(self, simple_edges):
        """Test n_bins for calibrated."""
        cal = EnergyCalibration(simple_edges)
        assert cal.n_bins == 4
    
    def test_n_bins_uncalibrated(self):
        """Test n_bins for uncalibrated."""
        cal = EnergyCalibration(None)
        assert cal.n_bins is None
    
    def test_is_shared_with_refs(self, simple_edges):
        """Test is_shared property with references."""
        cal = EnergyCalibration(simple_edges, is_shared=True)
        assert cal.is_shared is False  # No refs yet
        
        cal._ref_count = 2
        assert cal.is_shared is True  # Now shared
    
    def test_is_shared_not_flagged(self, simple_edges):
        """Test is_shared when not flagged as shared."""
        cal = EnergyCalibration(simple_edges, is_shared=False)
        cal._ref_count = 10
        assert cal.is_shared is False  # Not flagged as shared


class TestEnergyCalibrationCopy:
    """Test EnergyCalibration copy and detach."""
    
    def test_copy_creates_independent(self, simple_edges):
        """Test copy creates independent calibration."""
        cal1 = EnergyCalibration(simple_edges)
        cal1._ref_count = 5
        
        cal2 = cal1.copy()
        assert cal2._ref_count == 0
        assert cal2._is_shared_flag is False
        assert np.array_equal(cal2.edges, cal1.edges)
        
        # Modify copy doesn't affect original
        cal2.edges = np.array([0, 10, 20, 30, 40])
        assert not np.array_equal(cal2.edges, cal1.edges)
    
    def test_copy_uncalibrated(self):
        """Test copying uncalibrated calibration."""
        cal1 = EnergyCalibration(None)
        cal2 = cal1.copy()
        assert cal2.edges is None
    
    def test_detach_when_shared(self, simple_edges):
        """Test detach when calibration is shared."""
        cal1 = EnergyCalibration(simple_edges, is_shared=True)
        cal1._ref_count = 3
        
        cal2 = cal1.detach()
        assert cal1._ref_count == 2
        assert cal2._ref_count == 0
        assert cal2 is not cal1
    
    def test_detach_when_not_shared(self, simple_edges):
        """Test detach when not shared returns self."""
        cal1 = EnergyCalibration(simple_edges, is_shared=False)
        cal2 = cal1.detach()
        assert cal2 is cal1  # Same object


class TestEnergyCalibrationRefCounting:
    """Test reference counting."""
    
    def test_increment_ref(self, simple_edges):
        """Test incrementing reference count."""
        cal = EnergyCalibration(simple_edges)
        assert cal._ref_count == 0
        
        cal.increment_ref()
        assert cal._ref_count == 1
        
        cal.increment_ref()
        assert cal._ref_count == 2
    
    def test_decrement_ref(self, simple_edges):
        """Test decrementing reference count."""
        cal = EnergyCalibration(simple_edges)
        cal._ref_count = 5
        
        cal.decrement_ref()
        assert cal._ref_count == 4
    
    def test_decrement_ref_floor(self, simple_edges):
        """Test decrement doesn't go below zero."""
        cal = EnergyCalibration(simple_edges)
        cal._ref_count = 0
        
        cal.decrement_ref()
        assert cal._ref_count == 0  # Doesn't go negative


class TestEnergyCalibrationFromCoefficients:
    """Test creating calibration from coefficients."""
    
    def test_linear_calibration(self):
        """Test linear calibration from coefficients."""
        cal = EnergyCalibration.from_coefficients(
            n_channels=10,
            coefficients=[0, 0.5],
            model='polynomial'
        )
        expected = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        assert np.allclose(cal.edges, expected)
    
    def test_quadratic_calibration(self):
        """Test quadratic calibration from coefficients."""
        cal = EnergyCalibration.from_coefficients(
            n_channels=4,
            coefficients=[0, 1, 0.1],
            model='polynomial'
        )
        # E = 0 + 1*ch + 0.1*ch^2
        # ch=0: 0, ch=1: 1.1, ch=2: 2.4, ch=3: 3.9, ch=4: 5.6
        expected = np.array([0, 1.1, 2.4, 3.9, 5.6])
        assert np.allclose(cal.edges, expected)
    
    def test_linear_model_shortcut(self):
        """Test 'linear' model."""
        cal = EnergyCalibration.from_coefficients(
            n_channels=5,
            coefficients=[10, 2],
            model='linear'
        )
        expected = np.array([10, 12, 14, 16, 18, 20])
        assert np.allclose(cal.edges, expected)
    
    def test_invalid_model(self):
        """Test invalid calibration model raises error."""
        with pytest.raises(CalibrationError, match="Unknown calibration model"):
            EnergyCalibration.from_coefficients(
                n_channels=10,
                coefficients=[0, 1],
                model='invalid_model'
            )
    
    def test_higher_order_polynomial(self):
        """Test higher order polynomial."""
        cal = EnergyCalibration.from_coefficients(
            n_channels=3,
            coefficients=[1, 2, 0.5, 0.1],  # Cubic
            model='polynomial'
        )
        # E = 1 + 2*ch + 0.5*ch^2 + 0.1*ch^3
        expected = np.array([
            1,                           # ch=0
            1 + 2 + 0.5 + 0.1,          # ch=1: 3.6
            1 + 4 + 2.0 + 0.8,          # ch=2: 7.8
            1 + 6 + 4.5 + 2.7           # ch=3: 14.2
        ])
        assert np.allclose(cal.edges, expected)


class TestEnergyCalibrationRepr:
    """Test string representation."""
    
    def test_repr_calibrated(self, simple_edges):
        """Test repr for calibrated."""
        cal = EnergyCalibration(simple_edges)
        repr_str = repr(cal)
        assert "EnergyCalibration" in repr_str
        assert "n_bins=4" in repr_str
        assert "0.00" in repr_str
        assert "4.00" in repr_str
    
    def test_repr_uncalibrated(self):
        """Test repr for uncalibrated."""
        cal = EnergyCalibration(None)
        repr_str = repr(cal)
        assert "uncalibrated" in repr_str
    
    def test_repr_shows_shared_status(self, simple_edges):
        """Test repr shows shared status."""
        cal = EnergyCalibration(simple_edges, is_shared=True)
        cal._ref_count = 2
        repr_str = repr(cal)
        assert "shared=True" in repr_str

