"""
Base class for collections of Spectrum objects.

This module provides the Spectra class, which handles collections of Spectrum
objects that are not necessarily in time order. SpectralTimeSeries inherits from
this class and adds time-specific functionality.
"""

from typing import List, Optional, Dict, Any, Callable, Union
import numpy as np
import numpy.typing as npt

from gammaflow.core.spectrum import Spectrum
from gammaflow.core.calibration import EnergyCalibration
from gammaflow.utils.exceptions import IncompatibleBinningError, CalibrationError


class Spectra:
    """
    Base class for a collection of Spectrum objects (not necessarily time-ordered).
    
    This class provides efficient storage and operations on multiple spectra using
    NumPy arrays while maintaining individual Spectrum objects for metadata access.
    Supports both shared and independent energy calibrations.
    
    Parameters
    ----------
    spectra : list of Spectrum
        List of Spectrum objects to combine. All must have the same number of bins.
    shared_calibration : bool, optional
        If True, all spectra share the same energy calibration (memory efficient).
        If False, each spectrum maintains its own calibration. Default is True.
        
    Attributes
    ----------
    n_spectra : int
        Number of spectra in the collection.
    n_bins : int
        Number of energy bins per spectrum.
    counts : np.ndarray
        2D array of counts with shape (n_spectra, n_bins).
    energy_edges : np.ndarray
        Energy bin edges. Shape is (n_bins+1,) if shared calibration,
        or (n_spectra, n_bins+1) if independent calibrations.
    is_calibrated : bool
        True if spectra have energy calibration.
    uses_shared_calibration : bool
        True if all spectra share the same calibration.
        
    Examples
    --------
    >>> # Create collection of background spectra
    >>> background_spectra = [Spectrum(np.random.poisson(100, 512)) for _ in range(50)]
    >>> spectra_collection = Spectra(background_spectra)
    >>> 
    >>> # Compute statistics
    >>> mean_spec = spectra_collection.mean_spectrum()
    >>> std_spec = spectra_collection.std_spectrum()
    >>> 
    >>> # Access individual spectra
    >>> spec = spectra_collection[0]
    >>> 
    >>> # Vectorized operations
    >>> total_counts = np.sum(spectra_collection.counts, axis=1)
    """
    
    def __init__(
        self,
        spectra: List[Spectrum],
        shared_calibration: bool = True,
    ):
        if len(spectra) == 0:
            raise ValueError("Cannot create Spectra from empty list")
            
        self._validate_spectra(spectra)
        self._n_spectra = len(spectra)
        self._n_bins = spectra[0].n_bins
        self._shared_calibration = shared_calibration
        
        # Create 2D counts array
        self._counts_array = np.zeros((self._n_spectra, self._n_bins))
        for i, spec in enumerate(spectra):
            self._counts_array[i] = spec.counts
            
        # Setup calibration and create spectrum objects with views
        if shared_calibration:
            self._setup_shared_calibration(spectra)
        else:
            self._setup_independent_calibrations(spectra)
            
    def _validate_spectra(self, spectra: List[Spectrum]) -> None:
        """Validate that all spectra have compatible binning."""
        n_bins = spectra[0].n_bins
        for i, spec in enumerate(spectra[1:], 1):
            if spec.n_bins != n_bins:
                raise IncompatibleBinningError(
                    f"Spectrum {i} has {spec.n_bins} bins, but first spectrum has {n_bins}"
                )
                
        # If calibrated, check that all have same energy range
        if spectra[0].is_calibrated:
            edges_0 = spectra[0].energy_edges
            for i, spec in enumerate(spectra[1:], 1):
                if not spec.is_calibrated:
                    raise CalibrationError(
                        f"Spectrum {i} is uncalibrated, but first spectrum is calibrated"
                    )
                if not np.allclose(spec.energy_edges, edges_0, rtol=1e-9):
                    raise IncompatibleBinningError(
                        f"Spectrum {i} has different energy calibration than first spectrum"
                    )
                    
    def _setup_shared_calibration(self, spectra: List[Spectrum]) -> None:
        """Setup shared calibration mode."""
        # Create shared EnergyCalibration object (even for uncalibrated spectra)
        if spectra[0].is_calibrated:
            edges = spectra[0].energy_edges.copy()
        else:
            edges = None  # Uncalibrated, but still shared
        
        self._calibration = EnergyCalibration(edges, is_shared=True)
            
        # Create Spectrum objects with views into counts array
        self._spectra = []
        for i, original_spec in enumerate(spectra):
            # Create view spectrum
            spec = Spectrum(
                counts=self._counts_array[i],  # View into array
                _calibration=self._calibration,  # Shared calibration
                uncertainty=original_spec.uncertainty.copy() if original_spec.uncertainty is not None else None,
                live_time=original_spec.live_time,
                real_time=original_spec.real_time,
                timestamp=original_spec.timestamp,
                _is_view=True,
            )
            # Increment reference count if calibration is shared
            if self._calibration is not None:
                self._calibration.increment_ref()
            # Copy metadata
            spec.metadata.update(original_spec.metadata)
            self._spectra.append(spec)
            
    def _setup_independent_calibrations(self, spectra: List[Spectrum]) -> None:
        """Setup independent calibrations mode."""
        self._calibration = None
        
        # Create deep copies of spectra with views into counts array
        self._spectra = []
        for i, original_spec in enumerate(spectra):
            spec = Spectrum(
                counts=self._counts_array[i],  # View into array
                energy_edges=original_spec.energy_edges.copy() if original_spec.is_calibrated else None,
                uncertainty=original_spec.uncertainty.copy() if original_spec.uncertainty is not None else None,
                live_time=original_spec.live_time,
                real_time=original_spec.real_time,
                timestamp=original_spec.timestamp,
                _is_view=True,
            )
            # Copy metadata
            spec.metadata.update(original_spec.metadata)
            self._spectra.append(spec)
            
    @property
    def n_spectra(self) -> int:
        """Number of spectra in the collection."""
        return self._n_spectra
        
    @property
    def n_bins(self) -> int:
        """Number of energy bins per spectrum."""
        return self._n_bins
        
    @property
    def counts(self) -> np.ndarray:
        """
        2D array of counts.
        
        Returns
        -------
        np.ndarray
            Shape (n_spectra, n_bins). Changes to this array will affect
            the individual Spectrum objects.
        """
        return self._counts_array
        
    @property
    def spectra(self) -> List[Spectrum]:
        """List of individual Spectrum objects."""
        return self._spectra
        
    @property
    def is_calibrated(self) -> bool:
        """True if spectra have energy calibration."""
        return self._spectra[0].is_calibrated
        
    @property
    def uses_shared_calibration(self) -> bool:
        """True if all spectra share the same calibration."""
        return self._shared_calibration
        
    @property
    def energy_edges(self) -> np.ndarray:
        """
        Energy bin edges.
        
        Returns
        -------
        np.ndarray
            If shared calibration: shape (n_bins+1,)
            If independent: shape (n_spectra, n_bins+1)
        """
        if self._shared_calibration and self.is_calibrated:
            return self._calibration.edges
        elif self._shared_calibration:
            # Uncalibrated, return channel indices
            return np.arange(self._n_bins + 1)
        else:
            # Independent calibrations
            return np.array([spec.energy_edges for spec in self._spectra])
            
    @property
    def energy_centers(self) -> np.ndarray:
        """
        Energy bin centers.
        
        Returns
        -------
        np.ndarray
            If shared calibration: shape (n_bins,)
            If independent: shape (n_spectra, n_bins)
        """
        if self._shared_calibration:
            return self._spectra[0].energy_centers
        else:
            return np.array([spec.energy_centers for spec in self._spectra])
            
    def mean_spectrum(self) -> Spectrum:
        """
        Compute mean spectrum across the collection.
        
        Returns
        -------
        Spectrum
            Mean of all spectra.
        """
        mean_counts = np.mean(self._counts_array, axis=0)
        
        # Average uncertainties if present
        uncertainties = [s.uncertainty for s in self._spectra if s.uncertainty is not None]
        if len(uncertainties) == len(self._spectra):
            # Uncertainty of mean = sqrt(sum(σ²) / n²) = sqrt(sum(σ²)) / n
            mean_uncertainty = np.sqrt(np.sum([u**2 for u in uncertainties], axis=0)) / self._n_spectra
        else:
            mean_uncertainty = None
            
        return Spectrum(
            counts=mean_counts,
            energy_edges=self.energy_edges if self._shared_calibration else self._spectra[0].energy_edges,
            uncertainty=mean_uncertainty,
            metadata={'source': 'mean', 'n_spectra': self._n_spectra}
        )
        
    def median_spectrum(self) -> Spectrum:
        """
        Compute median spectrum across the collection.
        
        Returns
        -------
        Spectrum
            Median of all spectra.
        """
        median_counts = np.median(self._counts_array, axis=0)
        
        return Spectrum(
            counts=median_counts,
            energy_edges=self.energy_edges if self._shared_calibration else self._spectra[0].energy_edges,
            metadata={'source': 'median', 'n_spectra': self._n_spectra}
        )
        
    def std_spectrum(self) -> Spectrum:
        """
        Compute standard deviation spectrum across the collection.
        
        Returns
        -------
        Spectrum
            Standard deviation of all spectra.
        """
        std_counts = np.std(self._counts_array, axis=0)
        
        return Spectrum(
            counts=std_counts,
            energy_edges=self.energy_edges if self._shared_calibration else self._spectra[0].energy_edges,
            metadata={'source': 'std', 'n_spectra': self._n_spectra}
        )
        
    def sum_spectrum(self) -> Spectrum:
        """
        Compute sum of all spectra in the collection.
        
        Returns
        -------
        Spectrum
            Sum of all spectra.
        """
        sum_counts = np.sum(self._counts_array, axis=0)
        
        # Sum uncertainties in quadrature if present
        uncertainties = [s.uncertainty for s in self._spectra if s.uncertainty is not None]
        if len(uncertainties) == len(self._spectra):
            sum_uncertainty = np.sqrt(np.sum([u**2 for u in uncertainties], axis=0))
        else:
            sum_uncertainty = None
            
        return Spectrum(
            counts=sum_counts,
            energy_edges=self.energy_edges if self._shared_calibration else self._spectra[0].energy_edges,
            uncertainty=sum_uncertainty,
            metadata={'source': 'sum', 'n_spectra': self._n_spectra}
        )
        
    def apply_vectorized(self, func: Callable[[np.ndarray], np.ndarray]) -> 'Spectra':
        """
        Apply a vectorized function to the counts array.
        
        Parameters
        ----------
        func : callable
            Function that takes a 2D array (n_spectra, n_bins) and returns
            a 2D array of the same shape.
            
        Returns
        -------
        Spectra
            New Spectra object with transformed counts.
            
        Examples
        --------
        >>> # Normalize each spectrum to unit sum
        >>> normalized = spectra.apply_vectorized(
        ...     lambda counts: counts / counts.sum(axis=1, keepdims=True)
        ... )
        """
        new_counts = func(self._counts_array)
        
        if new_counts.shape != self._counts_array.shape:
            raise ValueError(
                f"Function output shape {new_counts.shape} doesn't match "
                f"input shape {self._counts_array.shape}"
            )
            
        # Create new spectra with transformed counts
        new_spectra = []
        for i, spec in enumerate(self._spectra):
            new_spec = Spectrum(
                counts=new_counts[i],
                energy_edges=spec.energy_edges if spec.is_calibrated else None,
                uncertainty=spec.uncertainty.copy() if spec.uncertainty is not None else None,
                live_time=spec.live_time,
                real_time=spec.real_time,
                timestamp=spec.timestamp,
            )
            new_spec.metadata.update(spec.metadata)
            new_spectra.append(new_spec)
            
        return Spectra(new_spectra, shared_calibration=self._shared_calibration)
        
    def __getitem__(self, idx: Union[int, slice]) -> Union[Spectrum, 'Spectra']:
        """
        Get spectrum by index or slice.
        
        Parameters
        ----------
        idx : int or slice
            Index or slice of spectra to retrieve.
            
        Returns
        -------
        Spectrum or Spectra
            Single spectrum if int index, Spectra if slice.
        """
        if isinstance(idx, int):
            return self._spectra[idx]
        elif isinstance(idx, slice):
            return Spectra(self._spectra[idx], shared_calibration=self._shared_calibration)
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")
            
    def __len__(self) -> int:
        """Number of spectra in the collection."""
        return self._n_spectra
        
    def __iter__(self):
        """Iterate over individual spectra."""
        return iter(self._spectra)
        
    def __array__(self) -> np.ndarray:
        """NumPy array protocol - returns counts array."""
        return self._counts_array
        
    def __repr__(self) -> str:
        calib = "calibrated" if self.is_calibrated else "uncalibrated"
        mode = "shared" if self._shared_calibration else "independent"
        return (
            f"Spectra(n_spectra={self._n_spectra}, n_bins={self._n_bins}, "
            f"{calib}, calibration_mode={mode})"
        )

