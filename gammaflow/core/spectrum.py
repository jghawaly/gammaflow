"""
Spectrum class for representing gamma ray spectra.

This module provides the core Spectrum class with support for:
- Optional energy calibration
- Arithmetic operations
- Energy rebinning and slicing
- Uncertainty propagation
- Copy-on-write for shared calibrations
"""

from typing import Optional, Union, Dict, Any, Tuple, List
from datetime import datetime
import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d

from gammaflow.core.calibration import EnergyCalibration
from gammaflow.utils.exceptions import SpectrumError, IncompatibleBinningError, CalibrationError


class Spectrum:
    """
    A gamma ray spectrum with optional energy calibration.
    
    The Spectrum class represents a single gamma ray spectrum with counts
    per energy bin. Energy calibration is optional - without it, the spectrum
    uses channel indices.
    
    Parameters
    ----------
    counts : array-like
        Counts per bin. Shape: (n_bins,)
    energy_edges : array-like or None, optional
        Energy bin edges. Shape: (n_bins + 1,). If None, spectrum is
        uncalibrated (uses channel indices). Default is None.
    uncertainty : array-like or None, optional
        Uncertainty per bin. If None, Poisson statistics are assumed.
        Shape: (n_bins,). Default is None.
    timestamp : float or datetime or None, optional
        Acquisition timestamp. Default is None.
    live_time : float or None, optional
        Actual counting time in seconds (dead time excluded). If None,
        uses real_time for count rate calculations. Default is None.
    real_time : float or None, optional
        Elapsed time in seconds (dead time included). If None and live_time
        is provided, set equal to live_time. If both are None, defaults to 1.0.
        Default is None.
    energy_unit : str, optional
        Energy unit ('keV', 'MeV', 'eV', 'channel'). Default is 'keV' if
        calibrated, 'channel' if not.
    metadata : dict or None, optional
        Additional metadata. Default is empty dict.
    _calibration : EnergyCalibration or None, optional
        Internal parameter for sharing calibration. Default is None.
    _is_view : bool, optional
        Internal parameter indicating if counts array is a view. Default is False.
    
    Attributes
    ----------
    counts : np.ndarray
        Counts per bin.
    energy_edges : np.ndarray
        Energy bin edges (or channel edges if uncalibrated).
    energy_centers : np.ndarray
        Energy bin centers (computed property).
    energy_widths : np.ndarray
        Energy bin widths (computed property).
    uncertainty : np.ndarray
        Uncertainty per bin (Poisson if not specified).
    is_calibrated : bool
        Whether spectrum has energy calibration.
    has_shared_calibration : bool
        Whether calibration is shared with other spectra.
    is_view : bool
        Whether counts array is a view into a larger array.
    
    Examples
    --------
    >>> # Create uncalibrated spectrum
    >>> counts = np.array([100, 200, 150, 300])
    >>> spec = Spectrum(counts)
    >>> spec.is_calibrated
    False
    >>> spec.energy_centers
    array([0.5, 1.5, 2.5, 3.5])
    
    >>> # Create calibrated spectrum
    >>> spec = Spectrum(counts, energy_edges=[0, 1, 2, 3, 4])
    >>> spec.is_calibrated
    True
    
    >>> # Apply calibration
    >>> calibrated = spec.apply_calibration([0, 0.5])  # E = 0 + 0.5*ch
    """
    
    def __init__(
        self,
        counts: npt.ArrayLike,
        energy_edges: Optional[npt.ArrayLike] = None,
        uncertainty: Optional[npt.ArrayLike] = None,
        timestamp: Optional[Union[float, datetime]] = None,
        live_time: Optional[float] = None,
        real_time: Optional[float] = None,
        energy_unit: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        _calibration: Optional[EnergyCalibration] = None,
        _is_view: bool = False,
    ):
        # Store counts
        if _is_view:
            # Store as view (shares memory with parent array)
            assert isinstance(counts, np.ndarray), "View must be numpy array"
            self._counts = counts
            self._is_view = True
        else:
            # Store as independent copy
            self._counts = np.asarray(counts, dtype=float)
            self._is_view = False
        
        # Handle calibration
        if _calibration is not None:
            # Using shared calibration object (from TimeSeries)
            self._calibration = _calibration
            self._calibration.increment_ref()
        else:
            # Create new independent calibration
            edges_array = np.asarray(energy_edges) if energy_edges is not None else None
            self._calibration = EnergyCalibration(edges_array, is_shared=False)
        
        # Store uncertainty
        if uncertainty is not None:
            self._uncertainty = np.asarray(uncertainty, dtype=float)
        else:
            self._uncertainty = None
        
        # Store time information
        self._timestamp = timestamp
        
        # Handle live_time and real_time
        if live_time is not None and real_time is not None:
            # Both provided
            self._live_time = float(live_time)
            self._real_time = float(real_time)
        elif live_time is not None:
            # Only live_time provided
            self._live_time = float(live_time)
            self._real_time = float(live_time)  # real_time defaults to live_time
        elif real_time is not None:
            # Only real_time provided
            self._live_time = None
            self._real_time = float(real_time)
        else:
            # Neither provided - default to 1.0
            self._live_time = 1.0
            self._real_time = 1.0
        
        # Energy unit
        if energy_unit is not None:
            self._energy_unit = energy_unit
        else:
            self._energy_unit = 'keV' if self.is_calibrated else 'channel'
        
        # Metadata
        self._metadata = metadata.copy() if metadata is not None else {}
        
        # Validate
        self._validate()
    
    def _validate(self):
        """Validate spectrum data."""
        # Check shapes
        if self._calibration.edges is not None:
            if len(self._calibration.edges) != len(self._counts) + 1:
                raise SpectrumError(
                    f"Energy edges length ({len(self._calibration.edges)}) must be "
                    f"counts length + 1 ({len(self._counts) + 1})"
                )
        
        # Check uncertainty shape
        if self._uncertainty is not None:
            if self._uncertainty.shape != self._counts.shape:
                raise SpectrumError(
                    f"Uncertainty shape {self._uncertainty.shape} must match "
                    f"counts shape {self._counts.shape}"
                )
        
        # Check times are positive
        if self._live_time is not None and self._live_time < 0:
            raise SpectrumError("Live time must be non-negative")
        if self._real_time < 0:
            raise SpectrumError("Real time must be non-negative")
        if self._live_time is not None and self._real_time < self._live_time:
            raise SpectrumError("Real time must be >= live time")
    
    # ========================================
    # Properties
    # ========================================
    
    @property
    def counts(self) -> np.ndarray:
        """Get counts array."""
        return self._counts
    
    @property
    def energy_edges(self) -> np.ndarray:
        """Get energy edges (or channel edges if uncalibrated)."""
        if self._calibration.edges is None:
            # Uncalibrated: return channel indices
            return np.arange(len(self._counts) + 1, dtype=float)
        return self._calibration.edges
    
    @property
    def energy_centers(self) -> np.ndarray:
        """Get energy bin centers (computed from edges)."""
        edges = self.energy_edges
        return (edges[:-1] + edges[1:]) / 2
    
    @property
    def energy_widths(self) -> np.ndarray:
        """Get energy bin widths."""
        return np.diff(self.energy_edges)
    
    @property
    def uncertainty(self) -> np.ndarray:
        """
        Get uncertainty per bin.
        
        If not explicitly set, returns Poisson uncertainty (sqrt(counts)).
        """
        if self._uncertainty is None:
            return np.sqrt(np.maximum(self._counts, 0))
        return self._uncertainty
    
    @property
    def count_rate(self) -> np.ndarray:
        """
        Get count rate (counts per second).
        
        Uses live_time if available, otherwise falls back to real_time.
        """
        # Use live_time if available, otherwise real_time
        time = self._live_time if self._live_time is not None else self._real_time
        
        if time > 0:
            return self._counts / time
        return self._counts
    
    @property
    def count_density(self) -> np.ndarray:
        """Get count density (counts per energy unit)."""
        return self._counts / self.energy_widths
    
    @property
    def timestamp(self) -> Optional[Union[float, datetime]]:
        """Get acquisition timestamp."""
        return self._timestamp
    
    @property
    def live_time(self) -> Optional[float]:
        """Get live time (actual counting time). May be None if not provided."""
        return self._live_time
    
    @property
    def real_time(self) -> float:
        """Get real time (elapsed time including dead time)."""
        return self._real_time
    
    @property
    def dead_time_fraction(self) -> float:
        """
        Get dead time fraction.
        
        Returns 0.0 if live_time is not available.
        """
        if self._live_time is None or self._real_time == 0:
            return 0.0
        return 1.0 - (self._live_time / self._real_time)
    
    @property
    def energy_unit(self) -> str:
        """Get energy unit."""
        return self._energy_unit
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata dictionary."""
        return self._metadata
    
    @property
    def is_calibrated(self) -> bool:
        """Check if spectrum has energy calibration."""
        return self._calibration.edges is not None
    
    @property
    def has_shared_calibration(self) -> bool:
        """Check if this spectrum uses shared calibration."""
        return self._calibration.is_shared
    
    @property
    def is_view(self) -> bool:
        """Check if counts array is a view into a larger array."""
        return self._is_view
    
    @property
    def n_bins(self) -> int:
        """Get number of bins."""
        return len(self._counts)
    
    # ========================================
    # Copy-on-Write Methods
    # ========================================
    
    def _ensure_independent_calibration(self):
        """
        Detach from shared calibration (copy-on-write).
        
        Called internally before operations that modify energy edges.
        """
        if self._calibration.is_shared:
            self._calibration = self._calibration.detach()
    
    def _ensure_independent_data(self):
        """
        Make counts array independent (copy-on-write for views).
        
        Called before operations that would break view semantics.
        """
        if self._is_view:
            self._counts = self._counts.copy()
            self._is_view = False
    
    def detach(self) -> 'Spectrum':
        """
        Detach from shared calibration and array views.
        
        Makes this spectrum fully independent. Useful before multiple
        in-place modifications.
        
        Returns
        -------
        Spectrum
            Self (for chaining).
        """
        self._ensure_independent_calibration()
        self._ensure_independent_data()
        return self
    
    def copy(self, deep: bool = True) -> 'Spectrum':
        """
        Create a copy of this spectrum.
        
        Parameters
        ----------
        deep : bool, optional
            If True, create independent copy (new calibration, new data).
            If False, share calibration (reference same edges). Default is True.
        
        Returns
        -------
        Spectrum
            Copy of this spectrum.
        """
        if deep or not self.is_calibrated:
            # Deep copy: independent calibration and data
            return Spectrum(
                counts=self._counts.copy(),
                energy_edges=self.energy_edges.copy() if self.is_calibrated else None,
                uncertainty=self._uncertainty.copy() if self._uncertainty is not None else None,
                timestamp=self._timestamp,
                live_time=self._live_time,
                real_time=self._real_time,
                energy_unit=self._energy_unit,
                metadata=self._metadata.copy(),
            )
        else:
            # Shallow copy: share calibration
            return Spectrum(
                counts=self._counts.copy(),
                _calibration=self._calibration,  # Share reference
                uncertainty=self._uncertainty.copy() if self._uncertainty is not None else None,
                timestamp=self._timestamp,
                live_time=self._live_time,
                real_time=self._real_time,
                energy_unit=self._energy_unit,
                metadata=self._metadata.copy(),
            )
    
    # ========================================
    # Arithmetic Operations
    # ========================================
    
    def _check_compatibility(self, other: 'Spectrum'):
        """Check if two spectra are compatible for arithmetic operations."""
        if not np.allclose(self.energy_edges, other.energy_edges):
            raise IncompatibleBinningError(
                "Spectra must have identical energy binning for arithmetic operations"
            )
    
    def __add__(self, other: Union['Spectrum', float]) -> 'Spectrum':
        """Add two spectra or add scalar to counts."""
        if isinstance(other, Spectrum):
            self._check_compatibility(other)
            new_counts = self._counts + other._counts
            # Propagate uncertainty: σ² = σ₁² + σ₂²
            new_uncertainty = np.sqrt(self.uncertainty**2 + other.uncertainty**2)
            
            # Handle optional live_time: only add if both have it
            if self._live_time is not None and other._live_time is not None:
                combined_live_time = self._live_time + other._live_time
            else:
                combined_live_time = None
            
            return Spectrum(
                counts=new_counts,
                energy_edges=self.energy_edges if self.is_calibrated else None,
                uncertainty=new_uncertainty,
                live_time=combined_live_time,
                real_time=self._real_time + other._real_time,
                energy_unit=self._energy_unit,
            )
        else:
            # Scalar addition
            return Spectrum(
                counts=self._counts + other,
                energy_edges=self.energy_edges if self.is_calibrated else None,
                uncertainty=self.uncertainty.copy(),
                timestamp=self._timestamp,
                live_time=self._live_time,
                real_time=self._real_time,
                energy_unit=self._energy_unit,
                metadata=self._metadata.copy(),
            )
    
    def __radd__(self, other: float) -> 'Spectrum':
        """Right addition (scalar + spectrum)."""
        return self.__add__(other)
    
    def __sub__(self, other: Union['Spectrum', float]) -> 'Spectrum':
        """Subtract spectrum or scalar from this spectrum."""
        if isinstance(other, Spectrum):
            self._check_compatibility(other)
            new_counts = self._counts - other._counts
            new_uncertainty = np.sqrt(self.uncertainty**2 + other.uncertainty**2)
            return Spectrum(
                counts=new_counts,
                energy_edges=self.energy_edges if self.is_calibrated else None,
                uncertainty=new_uncertainty,
                live_time=self._live_time,
                real_time=self._real_time,
                energy_unit=self._energy_unit,
            )
        else:
            return Spectrum(
                counts=self._counts - other,
                energy_edges=self.energy_edges if self.is_calibrated else None,
                uncertainty=self.uncertainty.copy(),
                timestamp=self._timestamp,
                live_time=self._live_time,
                real_time=self._real_time,
                energy_unit=self._energy_unit,
                metadata=self._metadata.copy(),
            )
    
    def __rsub__(self, other: float) -> 'Spectrum':
        """Right subtraction (scalar - spectrum)."""
        return Spectrum(
            counts=other - self._counts,
            energy_edges=self.energy_edges if self.is_calibrated else None,
            uncertainty=self.uncertainty.copy(),
            timestamp=self._timestamp,
            live_time=self._live_time,
            real_time=self._real_time,
            energy_unit=self._energy_unit,
            metadata=self._metadata.copy(),
        )
    
    def __mul__(self, other: float) -> 'Spectrum':
        """Multiply spectrum by scalar (for scaling)."""
        if isinstance(other, Spectrum):
            raise TypeError(
                "Multiplying two spectra is not supported. "
                "Use scalar multiplication for scaling."
            )
        
        # Scalar multiplication
        return Spectrum(
            counts=self._counts * other,
            energy_edges=self.energy_edges if self.is_calibrated else None,
            uncertainty=self.uncertainty * abs(other),
            timestamp=self._timestamp,
            live_time=self._live_time,
            real_time=self._real_time,
            energy_unit=self._energy_unit,
            metadata=self._metadata.copy(),
        )
    
    def __rmul__(self, other: float) -> 'Spectrum':
        """Right multiplication (scalar * spectrum)."""
        return self.__mul__(other)
    
    def __truediv__(self, other: float) -> 'Spectrum':
        """Divide spectrum by scalar (for scaling)."""
        if isinstance(other, Spectrum):
            raise TypeError(
                "Dividing two spectra is not supported. "
                "Use scalar division for scaling."
            )
        
        return Spectrum(
            counts=self._counts / other,
            energy_edges=self.energy_edges if self.is_calibrated else None,
            uncertainty=self.uncertainty / abs(other),
            timestamp=self._timestamp,
            live_time=self._live_time,
            real_time=self._real_time,
            energy_unit=self._energy_unit,
            metadata=self._metadata.copy(),
        )
    
    # ========================================
    # Numpy Interface
    # ========================================
    
    def __array__(self, dtype=None) -> np.ndarray:
        """Numpy array interface - returns counts."""
        return np.asarray(self._counts, dtype=dtype)
    
    def __len__(self) -> int:
        """Length is number of bins."""
        return len(self._counts)
    
    def __getitem__(self, key: Union[int, slice]) -> Union[float, 'Spectrum']:
        """
        Index or slice spectrum.
        
        Single index returns counts value.
        Slice returns new Spectrum with sliced data.
        """
        if isinstance(key, (int, np.integer)):
            return self._counts[key]
        elif isinstance(key, slice):
            # Return new spectrum with sliced data
            edges = self.energy_edges
            sliced_edges = edges[key.start:key.stop+1 if key.stop is not None else None]
            return Spectrum(
                counts=self._counts[key],
                energy_edges=sliced_edges if self.is_calibrated else None,
                uncertainty=self.uncertainty[key] if self._uncertainty is not None else None,
                timestamp=self._timestamp,
                live_time=self._live_time,
                real_time=self._real_time,
                energy_unit=self._energy_unit,
                metadata=self._metadata.copy(),
            )
        else:
            raise TypeError(f"Invalid index type: {type(key)}")
    
    # ========================================
    # Calibration Methods
    # ========================================
    
    @staticmethod
    def from_channels(
        counts: npt.ArrayLike,
        calibration_coefficients: List[float],
        model: str = 'polynomial',
        **kwargs
    ) -> 'Spectrum':
        """
        Create calibrated spectrum from channel data.
        
        Parameters
        ----------
        counts : array-like
            Counts per channel.
        calibration_coefficients : list of float
            Calibration coefficients.
        model : str, optional
            Calibration model ('polynomial' or 'linear'). Default is 'polynomial'.
        **kwargs
            Additional arguments passed to Spectrum constructor.
        
        Returns
        -------
        Spectrum
            Calibrated spectrum.
        """
        counts_array = np.asarray(counts, dtype=float)
        n_channels = len(counts_array)
        
        calibration = EnergyCalibration.from_coefficients(
            n_channels, calibration_coefficients, model
        )
        
        return Spectrum(
            counts=counts_array,
            energy_edges=calibration.edges,
            **kwargs
        )
    
    def apply_calibration(
        self,
        coefficients: List[float],
        model: str = 'polynomial'
    ) -> 'Spectrum':
        """
        Apply energy calibration (returns new spectrum).
        
        Parameters
        ----------
        coefficients : list of float
            Calibration coefficients.
        model : str, optional
            Calibration model. Default is 'polynomial'.
        
        Returns
        -------
        Spectrum
            New calibrated spectrum.
        """
        calibration = EnergyCalibration.from_coefficients(
            self.n_bins, coefficients, model
        )
        
        return Spectrum(
            counts=self._counts.copy(),
            energy_edges=calibration.edges,
            uncertainty=self.uncertainty.copy() if self._uncertainty is not None else None,
            timestamp=self._timestamp,
            live_time=self._live_time,
            real_time=self._real_time,
            metadata=self._metadata.copy(),
        )
    
    def apply_calibration_(
        self,
        coefficients: List[float],
        model: str = 'polynomial'
    ) -> 'Spectrum':
        """
        Apply energy calibration in-place.
        
        Automatically detaches from shared calibration.
        
        Parameters
        ----------
        coefficients : list of float
            Calibration coefficients.
        model : str, optional
            Calibration model. Default is 'polynomial'.
        
        Returns
        -------
        Spectrum
            Self (for chaining).
        """
        self._ensure_independent_calibration()
        
        calibration = EnergyCalibration.from_coefficients(
            self.n_bins, coefficients, model
        )
        self._calibration.edges = calibration.edges
        self._energy_unit = 'keV'
        
        return self
    
    def to_channels(self) -> 'Spectrum':
        """
        Remove calibration (returns new uncalibrated spectrum).
        
        Returns
        -------
        Spectrum
            Uncalibrated spectrum.
        """
        return Spectrum(
            counts=self._counts.copy(),
            energy_edges=None,
            uncertainty=self.uncertainty.copy() if self._uncertainty is not None else None,
            timestamp=self._timestamp,
            live_time=self._live_time,
            real_time=self._real_time,
            energy_unit='channel',
            metadata=self._metadata.copy(),
        )
    
    # ========================================
    # Energy Operations
    # ========================================
    
    def slice_energy(
        self,
        e_min: Optional[float] = None,
        e_max: Optional[float] = None
    ) -> 'Spectrum':
        """
        Extract energy slice (returns new spectrum).
        
        Parameters
        ----------
        e_min : float or None, optional
            Minimum energy. If None, use minimum of spectrum.
        e_max : float or None, optional
            Maximum energy. If None, use maximum of spectrum.
        
        Returns
        -------
        Spectrum
            Sliced spectrum.
        """
        edges = self.energy_edges
        
        # Find bin indices
        if e_min is None:
            idx_min = 0
        else:
            idx_min = np.searchsorted(edges, e_min, side='left')
        
        if e_max is None:
            idx_max = len(self._counts)
        else:
            idx_max = np.searchsorted(edges, e_max, side='right')
        
        # Slice data
        sliced_edges = edges[idx_min:idx_max+1]
        
        return Spectrum(
            counts=self._counts[idx_min:idx_max],
            energy_edges=sliced_edges if self.is_calibrated else None,
            uncertainty=self.uncertainty[idx_min:idx_max] if self._uncertainty is not None else None,
            timestamp=self._timestamp,
            live_time=self._live_time,
            real_time=self._real_time,
            energy_unit=self._energy_unit,
            metadata=self._metadata.copy(),
        )
    
    def rebin_energy(
        self,
        new_edges: npt.ArrayLike,
        method: str = 'histogram'
    ) -> 'Spectrum':
        """
        Rebin spectrum to new energy edges (returns new spectrum).
        
        Parameters
        ----------
        new_edges : array-like
            New energy bin edges.
        method : str, optional
            Rebinning method:
            - 'histogram': Conservative rebinning (exact for nested bins)
            - 'linear': Linear interpolation of count density
            Default is 'histogram'.
        
        Returns
        -------
        Spectrum
            Rebinned spectrum.
        """
        new_edges = np.asarray(new_edges, dtype=float)
        old_edges = self.energy_edges
        old_counts = self._counts
        
        if method == 'histogram':
            # Histogram rebinning (conserves counts for nested bins)
            new_counts, _ = np.histogram(
                old_edges[:-1],  # Use left edges as sample points
                bins=new_edges,
                weights=old_counts
            )
            # Need to handle overlaps properly
            # This is approximate - for exact rebinning, need more sophisticated approach
        elif method == 'linear':
            # Interpolate count density
            old_centers = self.energy_centers
            old_density = self.count_density
            
            # Interpolate density at new bin centers
            new_centers = (new_edges[:-1] + new_edges[1:]) / 2
            new_widths = np.diff(new_edges)
            
            interp_func = interp1d(
                old_centers, old_density,
                kind='linear', bounds_error=False, fill_value=0
            )
            new_density = interp_func(new_centers)
            new_counts = new_density * new_widths
        else:
            raise ValueError(f"Unknown rebinning method: {method}")
        
        return Spectrum(
            counts=new_counts,
            energy_edges=new_edges,
            timestamp=self._timestamp,
            live_time=self._live_time,
            real_time=self._real_time,
            energy_unit=self._energy_unit,
            metadata=self._metadata.copy(),
        )
    
    def rebin_energy_(
        self,
        new_edges: npt.ArrayLike,
        method: str = 'histogram'
    ) -> 'Spectrum':
        """
        Rebin spectrum in-place.
        
        Automatically detaches from shared calibration and array views.
        
        Parameters
        ----------
        new_edges : array-like
            New energy bin edges.
        method : str, optional
            Rebinning method. Default is 'histogram'.
        
        Returns
        -------
        Spectrum
            Self (for chaining).
        """
        # Must detach before rebinning (changes array shape)
        self.detach()
        
        rebinned = self.rebin_energy(new_edges, method)
        self._counts = rebinned._counts
        self._calibration.edges = rebinned.energy_edges
        self._uncertainty = None  # Reset uncertainty after rebinning
        
        return self
    
    # ========================================
    # Analysis Methods
    # ========================================
    
    def integrate(
        self,
        e_min: Optional[float] = None,
        e_max: Optional[float] = None
    ) -> float:
        """
        Integrate counts in energy range.
        
        Parameters
        ----------
        e_min : float or None, optional
            Minimum energy. If None, use minimum of spectrum.
        e_max : float or None, optional
            Maximum energy. If None, use maximum of spectrum.
        
        Returns
        -------
        float
            Total counts in range.
        """
        if e_min is None and e_max is None:
            return np.sum(self._counts)
        
        sliced = self.slice_energy(e_min, e_max)
        return np.sum(sliced._counts)
    
    def normalize(
        self,
        method: str = 'area',
        value: float = 1.0
    ) -> 'Spectrum':
        """
        Normalize spectrum (returns new spectrum).
        
        Parameters
        ----------
        method : str, optional
            Normalization method:
            - 'area': Normalize total counts to value
            - 'peak': Normalize maximum to value
            Default is 'area'.
        value : float, optional
            Target normalization value. Default is 1.0.
        
        Returns
        -------
        Spectrum
            Normalized spectrum.
        
        Notes
        -----
        For count rate (counts per second), use the `count_rate` property instead.
        """
        if method == 'area':
            total = np.sum(self._counts)
            if total > 0:
                factor = value / total
            else:
                factor = 1.0
        elif method == 'peak':
            peak = np.max(self._counts)
            if peak > 0:
                factor = value / peak
            else:
                factor = 1.0
        else:
            raise ValueError(
                f"Unknown normalization method: {method}. "
                f"Valid options: 'area', 'peak'. "
                f"For count rate, use the count_rate property."
            )
        
        return self * factor
    
    
    # ========================================
    # String Representation
    # ========================================
    
    def __repr__(self) -> str:
        """String representation."""
        cal_str = "calibrated" if self.is_calibrated else "uncalibrated"
        if self.is_calibrated:
            e_range = f"[{self.energy_edges[0]:.2f}, {self.energy_edges[-1]:.2f}] {self._energy_unit}"
        else:
            e_range = f"channels [0, {self.n_bins}]"
        
        return (
            f"Spectrum(n_bins={self.n_bins}, {cal_str}, "
            f"range={e_range}, total_counts={np.sum(self._counts):.0f})"
        )

