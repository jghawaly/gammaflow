"""
Energy calibration handling for spectra.

This module provides the EnergyCalibration class which supports shared
calibration across multiple spectra with copy-on-write semantics.
"""

from typing import Optional, List, Callable
import numpy as np
import numpy.typing as npt

from gammaflow.utils.exceptions import CalibrationError


class EnergyCalibration:
    """
    Energy calibration container with support for shared references.
    
    This class wraps energy bin edges and can be shared across multiple
    Spectrum objects for memory efficiency. It tracks reference counts
    and supports copy-on-write semantics.
    
    Parameters
    ----------
    edges : np.ndarray or None
        Energy bin edges. If None, spectrum is uncalibrated.
    is_shared : bool, optional
        Whether this calibration is intended to be shared across spectra.
        Default is False.
    
    Attributes
    ----------
    edges : np.ndarray or None
        The energy bin edges array.
    is_shared : bool
        Whether calibration is actually shared (ref_count > 1).
    """
    
    def __init__(
        self,
        edges: Optional[npt.ArrayLike] = None,
        is_shared: bool = False
    ):
        if edges is not None:
            self._edges = np.asarray(edges, dtype=float)
            self._validate_edges()
        else:
            self._edges = None
        
        self._is_shared_flag = is_shared
        self._ref_count = 0
    
    def _validate_edges(self):
        """Validate that energy edges are monotonically increasing."""
        if self._edges is None:
            return
        
        if len(self._edges) < 2:
            raise CalibrationError("Energy edges must have at least 2 elements")
        
        if not np.all(np.diff(self._edges) > 0):
            raise CalibrationError("Energy edges must be monotonically increasing")
    
    @property
    def edges(self) -> Optional[np.ndarray]:
        """Get energy bin edges."""
        return self._edges
    
    @edges.setter
    def edges(self, value: Optional[npt.ArrayLike]):
        """Set energy bin edges."""
        if value is not None:
            self._edges = np.asarray(value, dtype=float)
            self._validate_edges()
        else:
            self._edges = None
    
    @property
    def is_shared(self) -> bool:
        """Check if this calibration is actually shared (ref_count > 1)."""
        return self._is_shared_flag and self._ref_count > 1
    
    @property
    def n_bins(self) -> Optional[int]:
        """Get number of energy bins (None if uncalibrated)."""
        if self._edges is None:
            return None
        return len(self._edges) - 1
    
    def copy(self) -> 'EnergyCalibration':
        """
        Create an independent copy of this calibration.
        
        Returns
        -------
        EnergyCalibration
            New calibration with copied edges and ref_count=0.
        """
        return EnergyCalibration(
            edges=self._edges.copy() if self._edges is not None else None,
            is_shared=False
        )
    
    def detach(self) -> 'EnergyCalibration':
        """
        Detach from shared calibration.
        
        If this calibration is shared (ref_count > 1), creates a new
        independent copy and decrements the reference count. Otherwise,
        returns self.
        
        Returns
        -------
        EnergyCalibration
            Independent calibration (either self or a new copy).
        """
        if not self.is_shared:
            return self  # Already independent
        
        self._ref_count -= 1
        return self.copy()
    
    def increment_ref(self):
        """Increment reference count."""
        self._ref_count += 1
    
    def decrement_ref(self):
        """Decrement reference count."""
        self._ref_count = max(0, self._ref_count - 1)
    
    @staticmethod
    def from_coefficients(
        n_channels: int,
        coefficients: List[float],
        model: str = 'polynomial'
    ) -> 'EnergyCalibration':
        """
        Create calibration from calibration coefficients.
        
        Parameters
        ----------
        n_channels : int
            Number of channels (bins).
        coefficients : list of float
            Calibration coefficients.
        model : str, optional
            Calibration model. Options:
            - 'polynomial': E = a0 + a1*ch + a2*ch^2 + ...
            - 'linear': E = a0 + a1*ch (shortcut, uses first 2 coeffs)
            Default is 'polynomial'.
        
        Returns
        -------
        EnergyCalibration
            New calibration with computed edges.
        
        Examples
        --------
        >>> # Linear calibration: E = 0 + 0.5*channel
        >>> cal = EnergyCalibration.from_coefficients(1024, [0, 0.5])
        
        >>> # Quadratic: E = 0 + 0.5*ch + 0.001*ch^2
        >>> cal = EnergyCalibration.from_coefficients(1024, [0, 0.5, 0.001])
        """
        channel_edges = np.arange(n_channels + 1, dtype=float)
        
        if model in ('polynomial', 'linear'):
            # E = a0 + a1*ch + a2*ch^2 + ...
            energy_edges = np.zeros(n_channels + 1)
            for i, coeff in enumerate(coefficients):
                energy_edges += coeff * (channel_edges ** i)
        else:
            raise CalibrationError(f"Unknown calibration model: {model}")
        
        return EnergyCalibration(energy_edges, is_shared=False)
    
    def __repr__(self) -> str:
        if self._edges is None:
            return "EnergyCalibration(uncalibrated)"
        return (
            f"EnergyCalibration(n_bins={self.n_bins}, "
            f"range=[{self._edges[0]:.2f}, {self._edges[-1]:.2f}], "
            f"shared={self.is_shared})"
        )

