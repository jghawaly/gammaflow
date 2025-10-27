"""
SpectralTimeSeries class for time series of gamma ray spectra.

This module provides efficient storage and manipulation of time series
of spectra with support for:
- Shared calibration (memory efficient)
- Independent calibration (flexible)
- Numpy array integration (vectorized operations)
- True Spectrum objects (individual metadata)
"""

from typing import Optional, List, Callable, Union, Any, Dict, TYPE_CHECKING
import numpy as np
import numpy.typing as npt

from gammaflow.core.spectrum import Spectrum
from gammaflow.core.calibration import EnergyCalibration
from gammaflow.utils.exceptions import TimeSeriesError, IncompatibleBinningError

if TYPE_CHECKING:
    from gammaflow.core.listmode import ListMode


class SpectralTimeSeries:
    """
    Time series of gamma ray spectra.
    
    Efficiently stores and manipulates multiple spectra with support for
    both shared and independent energy calibrations. Provides numpy-style
    vectorized operations while maintaining true Spectrum objects with
    individual metadata.
    
    Parameters
    ----------
    spectra : list of Spectrum or None, optional
        List of Spectrum objects. Default is empty list.
    shared_calibration : bool or None, optional
        Calibration storage mode:
        - True: Force shared calibration (rebin if needed)
        - False: Keep individual calibrations
        - None: Auto-detect (use shared if all compatible)
        Default is None.
    real_time : float or None, optional
        If provided, sets the real_time (clock time) for all spectra in the series.
        Useful when all spectra have the same acquisition duration. Default is None.
        Note: live_time typically varies between spectra due to dead time effects,
        so it cannot be set as a scalar parameter. Use individual Spectrum objects
        or from_array() with an array of live_times if needed.
    
    See Also
    --------
    from_array : Create from 2D numpy array
    from_channels : Create from channel data with calibration
    
    Attributes
    ----------
    counts : np.ndarray
        2D array of counts with shape (n_times, n_bins). Direct access
        for vectorized operations.
    spectra : list of Spectrum
        List of Spectrum objects with individual metadata.
    energy_edges : np.ndarray
        Energy bin edges (shared if using shared calibration).
    uses_shared_calibration : bool
        Whether series uses shared calibration.
    n_spectra : int
        Number of spectra in series.
    n_bins : int
        Number of energy bins.
    
    Examples
    --------
    >>> # Create time series
    >>> spectra = [Spectrum(np.random.poisson(100, 1024)) for _ in range(100)]
    >>> ts = SpectralTimeSeries(spectra)
    
    >>> # Apply common real_time to all spectra
    >>> ts = SpectralTimeSeries(spectra, real_time=10.0)
    
    >>> # Numpy-style operations
    >>> background = np.mean(ts.counts, axis=0)
    >>> ts.counts -= background
    
    >>> # Object-oriented access
    >>> for spec in ts:
    ...     spec.metadata['processed'] = True
    
    >>> # Vectorized analysis
    >>> total_counts = np.sum(ts.counts, axis=1)
    """
    
    def __init__(
        self,
        spectra: Optional[List[Spectrum]] = None,
        shared_calibration: Optional[bool] = None,
        real_time: Optional[float] = None,
        integration_time: Optional[float] = None,
        stride_time: Optional[float] = None,
    ):
        if spectra is None:
            spectra = []
        
        # If real_time provided, update all spectra
        if real_time is not None:
            spectra = self._apply_real_time_to_spectra(spectra, real_time)
        
        if len(spectra) == 0:
            # Empty time series
            self._spectra = []
            self._counts_array = np.array([]).reshape(0, 0)
            self._timestamps = np.array([])
            self._live_times = np.array([])
            self._real_times = np.array([])
            self._shared_calibration = None
            self._use_shared = False
            # Store integration and stride times for empty series
            self._integration_time = integration_time
            self._stride_time = stride_time
            return
        
        # Validate all spectra have same number of bins
        n_bins = spectra[0].n_bins
        for i, spec in enumerate(spectra[1:], 1):
            if spec.n_bins != n_bins:
                raise TimeSeriesError(
                    f"All spectra must have same number of bins. "
                    f"Spectrum 0 has {n_bins}, spectrum {i} has {spec.n_bins}"
                )
        
        # Determine calibration mode
        if shared_calibration is None:
            # Auto-detect: use shared if all have same edges
            self._use_shared = self._can_use_shared_calibration(spectra)
        else:
            self._use_shared = shared_calibration
        
        # Setup storage based on calibration mode
        if self._use_shared:
            self._setup_shared_calibration(spectra)
        else:
            self._setup_independent_calibrations(spectra)
        
        # Infer or validate integration_time and stride_time
        self._integration_time, self._stride_time = self._infer_and_validate_timing(
            spectra, integration_time, stride_time
        )
    
    def _apply_real_time_to_spectra(
        self,
        spectra: List[Spectrum],
        real_time: float
    ) -> List[Spectrum]:
        """
        Create new Spectrum objects with updated real_time.
        
        Live time handling:
        - If spectrum has explicit live_time != real_time: preserve it (has dead time)
        - If spectrum has live_time == real_time: set to None (no dead time info, use new real_time)
        - If spectrum has live_time=None: keep as None
        
        Parameters
        ----------
        spectra : list of Spectrum
            Original spectra.
        real_time : float
            Real time (clock time) to apply to all spectra.
        
        Returns
        -------
        list of Spectrum
            New spectra with updated real_time.
        """
        updated_spectra = []
        for spec in spectra:
            # Determine live_time: preserve only if it differs from original real_time
            # (indicating explicit dead time correction)
            if spec.live_time is not None and spec.live_time != spec.real_time:
                # Has explicit dead time info - preserve it
                new_live_time = spec.live_time
            else:
                # No explicit dead time (live_time=None or live_time==real_time)
                # Set to None so count_rate uses the new real_time
                new_live_time = None
            
            # Create new spectrum with updated real_time
            new_spec = Spectrum(
                counts=spec.counts.copy(),
                energy_edges=spec.energy_edges if spec.is_calibrated else None,
                uncertainty=spec.uncertainty.copy(),
                timestamp=spec.timestamp,
                live_time=new_live_time,
                real_time=real_time,
                energy_unit=spec.energy_unit,
                metadata=spec.metadata.copy(),
            )
            updated_spectra.append(new_spec)
        
        return updated_spectra
    
    def _infer_and_validate_timing(
        self,
        spectra: List[Spectrum],
        provided_integration_time: Optional[float],
        provided_stride_time: Optional[float]
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Infer or validate integration_time and stride_time from spectra.
        
        Logic:
        - integration_time: Infer from real_time if all constant, validate if provided
        - stride_time: Infer from timestamp spacing if evenly spaced, validate if provided
        
        Parameters
        ----------
        spectra : list of Spectrum
            Spectra to analyze
        provided_integration_time : float or None
            User-provided integration time (to validate)
        provided_stride_time : float or None
            User-provided stride time (to validate)
        
        Returns
        -------
        tuple of (float or None, float or None)
            Validated/inferred (integration_time, stride_time)
        
        Raises
        ------
        ValueError
            If provided values don't match data
        """
        if len(spectra) == 0:
            return provided_integration_time, provided_stride_time
        
        # Infer integration_time from real_time values
        real_times = self._real_times
        inferred_integration_time = None
        
        if len(real_times) > 0:
            # Check if all real_times are the same
            unique_real_times = np.unique(real_times)
            if len(unique_real_times) == 1:
                inferred_integration_time = float(unique_real_times[0])
            elif len(unique_real_times) > 1:
                # Check if they're approximately the same (within tolerance)
                if np.allclose(real_times, real_times[0], rtol=1e-9):
                    inferred_integration_time = float(real_times[0])
        
        # Validate or use provided integration_time
        if provided_integration_time is not None:
            if inferred_integration_time is not None:
                # Check if provided matches inferred
                if not np.isclose(provided_integration_time, inferred_integration_time, rtol=1e-9):
                    raise ValueError(
                        f"Provided integration_time ({provided_integration_time}) does not "
                        f"match real_time values (all {inferred_integration_time}). "
                        f"integration_time should equal real_time for constant acquisition times."
                    )
            integration_time = provided_integration_time
        else:
            integration_time = inferred_integration_time
        
        # Infer stride_time from timestamp spacing
        timestamps = self._timestamps
        inferred_stride_time = None
        
        if len(timestamps) > 1:
            # Check if timestamps are numeric (not None)
            if timestamps.dtype == object or timestamps[0] is None:
                # Can't infer from None timestamps
                pass
            else:
                # Calculate time differences
                time_diffs = np.diff(timestamps)
                
                # Check if evenly spaced (all differences approximately equal)
                if np.allclose(time_diffs, time_diffs[0], rtol=1e-6):
                    inferred_stride_time = float(time_diffs[0])
        
        # Validate or use provided stride_time
        if provided_stride_time is not None:
            if inferred_stride_time is not None:
                # Check if provided matches inferred
                if not np.isclose(provided_stride_time, inferred_stride_time, rtol=1e-6):
                    raise ValueError(
                        f"Provided stride_time ({provided_stride_time}) does not match "
                        f"timestamp spacing (evenly spaced at {inferred_stride_time}). "
                        f"stride_time should equal the time between consecutive spectra."
                    )
            stride_time = provided_stride_time
        else:
            stride_time = inferred_stride_time
        
        return integration_time, stride_time
    
    def _can_use_shared_calibration(self, spectra: List[Spectrum]) -> bool:
        """Check if all spectra have compatible energy edges."""
        if not spectra:
            return True
        
        first_edges = spectra[0].energy_edges
        first_calibrated = spectra[0].is_calibrated
        
        for spectrum in spectra[1:]:
            # Check if calibration state matches
            if spectrum.is_calibrated != first_calibrated:
                return False
            
            # Check if edges match
            if not np.allclose(spectrum.energy_edges, first_edges):
                return False
        
        return True
    
    def _setup_shared_calibration(self, spectra: List[Spectrum]):
        """Setup with shared energy calibration (memory efficient)."""
        # Create shared calibration object
        if spectra[0].is_calibrated:
            edges = spectra[0].energy_edges.copy()
        else:
            edges = None
        
        self._shared_calibration = EnergyCalibration(edges, is_shared=True)
        
        # Extract data into arrays
        self._counts_array = np.array([s._counts for s in spectra], dtype=float)
        self._timestamps = np.array([s._timestamp for s in spectra])
        # Handle optional live_times: use float if all are present, object if any None
        live_times_list = [s._live_time for s in spectra]
        if any(lt is None for lt in live_times_list):
            self._live_times = np.array(live_times_list, dtype=object)
        else:
            self._live_times = np.array(live_times_list, dtype=float)
        self._real_times = np.array([s._real_time for s in spectra], dtype=float)
        
        # Create Spectrum objects with VIEWS into the array
        self._spectra = []
        for i, spec in enumerate(spectra):
            new_spec = Spectrum(
                counts=self._counts_array[i],  # VIEW into array row
                _calibration=self._shared_calibration,  # Shared reference
                uncertainty=spec._uncertainty.copy() if spec._uncertainty is not None else None,
                timestamp=spec._timestamp,
                live_time=spec._live_time,
                real_time=spec._real_time,
                energy_unit=spec._energy_unit,
                metadata=spec._metadata.copy(),
                _is_view=True,  # Flag as view
            )
            self._spectra.append(new_spec)
    
    def _setup_independent_calibrations(self, spectra: List[Spectrum]):
        """Setup with independent calibrations per spectrum."""
        self._shared_calibration = None
        
        # Make deep copies
        self._spectra = [s.copy(deep=True) for s in spectra]
        
        # Still create arrays for convenience (but these are separate from spectrum data)
        self._counts_array = np.array([s._counts for s in self._spectra], dtype=float)
        self._timestamps = np.array([s._timestamp for s in self._spectra])
        # Handle optional live_times: use float if all are present, object if any None
        live_times_list = [s._live_time for s in self._spectra]
        if any(lt is None for lt in live_times_list):
            self._live_times = np.array(live_times_list, dtype=object)
        else:
            self._live_times = np.array(live_times_list, dtype=float)
        self._real_times = np.array([s._real_time for s in self._spectra], dtype=float)
        
        # Note: In independent mode, modifying _counts_array won't affect spectra
        # Need to keep them in sync or rebuild array when needed
    
    # ========================================
    # Factory Methods
    # ========================================
    
    @classmethod
    def from_list_mode(
        cls,
        time_deltas: Union[npt.ArrayLike, 'ListMode'],
        energies: Optional[npt.ArrayLike] = None,
        integration_time: float = None,
        stride_time: Optional[float] = None,
        energy_bins: Optional[Union[int, npt.ArrayLike]] = None,
        energy_range: Optional[tuple] = None,
    ) -> 'SpectralTimeSeries':
        """
        Create SpectralTimeSeries from list mode data.
        
        List mode data consists of individual events with time deltas (time since
        last event) and energies (or pulse heights). This method bins the events
        into spectra using a rolling time window.
        
        Can be called in two ways:
        1. With arrays: from_list_mode(time_deltas, energies, ...)
        2. With ListMode object: from_list_mode(listmode_obj, integration_time=...)
        
        Parameters
        ----------
        time_deltas : array-like or ListMode
            Either:
            - Time since previous event for each event (in seconds). Shape: (n_events,)
            - A ListMode object containing time_deltas and energies
        energies : array-like or None, optional
            Energy or pulse height for each event. Shape: (n_events,)
            Required if time_deltas is an array. Ignored if time_deltas is ListMode.
        integration_time : float
            Width of the rolling time window (in seconds). Each spectrum accumulates
            events within this window.
        stride_time : float or None, optional
            How far to move the window for each spectrum (in seconds). If None,
            defaults to integration_time (non-overlapping windows). If less than
            integration_time, windows will overlap. Default is None.
        energy_bins : int or array-like or None, optional
            Energy binning specification:
            - If int: number of bins (range determined by energy_range or data)
            - If array: bin edges to use
            - If None: auto-create 1024 bins from data range
            Default is None.
        energy_range : tuple or None, optional
            (min_energy, max_energy) for binning. Only used if energy_bins is int
            or None. If None, uses full data range. Default is None.
        
        Returns
        -------
        SpectralTimeSeries
            Time series with spectra from rolling windows.
        
        Notes
        -----
        - Time deltas are converted to absolute times via cumulative sum
        - Each spectrum's timestamp is the center of its time window
        - real_time for each spectrum equals integration_time
        - live_time is set to None (list mode doesn't have dead time correction)
        - Overlapping windows (stride_time < integration_time) are allowed
        
        Examples
        --------
        >>> # Method 1: From arrays
        >>> time_deltas = np.random.exponential(0.001, size=100000)  # 1000 Hz
        >>> energies = np.random.uniform(0, 3000, size=100000)  # keV
        >>> ts = SpectralTimeSeries.from_list_mode(
        ...     time_deltas, energies,
        ...     integration_time=1.0,  # 1 second windows
        ...     energy_bins=512
        ... )
        
        >>> # Method 2: From ListMode object
        >>> from gammaflow.core.listmode import ListMode
        >>> lm = ListMode(time_deltas, energies)
        >>> ts = SpectralTimeSeries.from_list_mode(
        ...     lm,
        ...     integration_time=1.0,
        ...     energy_bins=512
        ... )
        
        >>> # Overlapping windows for temporal analysis
        >>> ts = SpectralTimeSeries.from_list_mode(
        ...     time_deltas, energies,
        ...     integration_time=10.0,  # 10 second windows
        ...     stride_time=1.0,        # Move by 1 second (90% overlap)
        ...     energy_bins=1024
        ... )
        
        >>> # With specific energy range
        >>> ts = SpectralTimeSeries.from_list_mode(
        ...     time_deltas, energies,
        ...     integration_time=5.0,
        ...     energy_bins=2048,
        ...     energy_range=(0, 3000)  # 0-3000 keV
        ... )
        """
        # Import here to avoid circular import
        from gammaflow.core.listmode import ListMode
        
        # Handle two calling conventions
        if isinstance(time_deltas, ListMode):
            # Called with ListMode object
            listmode = time_deltas
            time_deltas_arr = listmode.time_deltas
            energies_arr = listmode.energies
            if integration_time is None:
                raise ValueError(
                    "integration_time is required when calling with ListMode object"
                )
        else:
            # Called with arrays
            time_deltas_arr = np.asarray(time_deltas, dtype=float)
            if energies is None:
                raise ValueError(
                    "energies is required when calling with arrays"
                )
            energies_arr = np.asarray(energies, dtype=float)
            if integration_time is None:
                raise ValueError("integration_time is required")
            
            # Validate inputs
            if len(time_deltas_arr) != len(energies_arr):
                raise ValueError(
                    f"time_deltas and energies must have same length. "
                    f"Got {len(time_deltas_arr)} and {len(energies_arr)}"
                )
        
        if integration_time <= 0:
            raise ValueError(f"integration_time must be positive, got {integration_time}")
        
        # Set stride_time default
        if stride_time is None:
            stride_time = integration_time
        elif stride_time <= 0:
            raise ValueError(f"stride_time must be positive, got {stride_time}")
        
        # Handle empty data
        if len(time_deltas_arr) == 0:
            # Early return for empty data, but preserve integration/stride metadata
            return cls(
                [],
                shared_calibration=True,
                integration_time=integration_time,
                stride_time=stride_time
            )
        
        # Convert time deltas to absolute times
        absolute_times = np.cumsum(time_deltas_arr)
        total_time = absolute_times[-1]
        
        # Determine energy bins
        if energy_bins is None:
            # Default: 1024 bins
            n_bins = 1024
            e_min, e_max = energy_range if energy_range else (energies_arr.min(), energies_arr.max())
            edges = np.linspace(e_min, e_max, n_bins + 1)
        elif isinstance(energy_bins, int):
            # Specified number of bins
            n_bins = energy_bins
            e_min, e_max = energy_range if energy_range else (energies_arr.min(), energies_arr.max())
            edges = np.linspace(e_min, e_max, n_bins + 1)
        else:
            # Explicit bin edges provided
            edges = np.asarray(energy_bins, dtype=float)
            n_bins = len(edges) - 1
        
        # Create rolling windows
        window_starts = np.arange(0, total_time, stride_time)
        n_windows = len(window_starts)
        
        if n_windows == 0:
            # No data - return empty time series
            return cls([], shared_calibration=True)
        
        # Create spectra for each window
        spectra = []
        for i, t_start in enumerate(window_starts):
            t_end = t_start + integration_time
            
            # Find events in this window
            mask = (absolute_times >= t_start) & (absolute_times < t_end)
            window_energies = energies_arr[mask]
            
            # Histogram energies
            counts, _ = np.histogram(window_energies, bins=edges)
            
            # Create spectrum
            timestamp = t_start + integration_time / 2  # Center of window
            spec = Spectrum(
                counts=counts,
                energy_edges=edges,
                timestamp=timestamp,
                live_time=None,  # List mode doesn't have dead time info
                real_time=integration_time,
                metadata={
                    'window_start': t_start,
                    'window_end': t_end,
                    'n_events': int(np.sum(mask)),
                }
            )
            spectra.append(spec)
        
        return cls(
            spectra,
            shared_calibration=True,
            integration_time=integration_time,
            stride_time=stride_time
        )
    
    @classmethod
    def from_array(
        cls,
        counts: npt.ArrayLike,
        energy_edges: Optional[npt.ArrayLike] = None,
        timestamps: Optional[npt.ArrayLike] = None,
        live_times: Optional[Union[float, npt.ArrayLike]] = None,
        real_times: Optional[Union[float, npt.ArrayLike]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        integration_time: Optional[float] = None,
        stride_time: Optional[float] = None,
    ) -> 'SpectralTimeSeries':
        """
        Create SpectralTimeSeries from 2D numpy array.
        
        Parameters
        ----------
        counts : array-like
            2D array of counts with shape (n_spectra, n_bins).
        energy_edges : array-like or None, optional
            Energy bin edges. Shape: (n_bins + 1,). If None, spectra are
            uncalibrated. Default is None.
        timestamps : array-like or None, optional
            Timestamps for each spectrum. Shape: (n_spectra,). If None,
            uses sequential indices. Default is None.
        live_times : float or array-like or None, optional
            Live time(s). If float, same for all spectra. If array,
            shape: (n_spectra,). If None, will use real_times for count
            rate calculations. Default is None.
        real_times : float or array-like or None, optional
            Real time(s). If float, same for all spectra. If array,
            shape: (n_spectra,). If None and live_times provided, set equal
            to live_times. If both None, defaults to 1.0 for all.
            Default is None.
        metadata : list of dict or None, optional
            List of metadata dicts, one per spectrum. If None, empty
            metadata for all. Default is None.
        integration_time : float or None, optional
            Integration time (time window width) for each spectrum. If provided,
            must match real_time values (if constant). Default is None (infer from data).
        stride_time : float or None, optional
            Stride time (time between consecutive spectra). If provided,
            must match timestamp spacing (if evenly spaced). Default is None (infer from data).
        
        Returns
        -------
        SpectralTimeSeries
            New time series with shared calibration.
        
        Examples
        --------
        >>> # Simple case - just counts
        >>> counts = np.random.poisson(100, size=(50, 1024))
        >>> ts = SpectralTimeSeries.from_array(counts)
        
        >>> # With calibration
        >>> edges = np.linspace(0, 3000, 1025)
        >>> ts = SpectralTimeSeries.from_array(counts, energy_edges=edges)
        
        >>> # With timing information
        >>> timestamps = np.arange(50) * 10.0  # Every 10 seconds
        >>> live_times = np.ones(50) * 9.5     # 9.5 seconds live time
        >>> ts = SpectralTimeSeries.from_array(
        ...     counts, 
        ...     energy_edges=edges,
        ...     timestamps=timestamps,
        ...     live_times=live_times
        ... )
        """
        counts_array = np.asarray(counts, dtype=float)
        
        if counts_array.ndim != 2:
            raise ValueError(
                f"Counts array must be 2D, got shape {counts_array.shape}"
            )
        
        n_spectra, n_bins = counts_array.shape
        
        # Handle timestamps
        if timestamps is None:
            timestamps_array = np.arange(n_spectra, dtype=float)
        else:
            timestamps_array = np.asarray(timestamps, dtype=float)
            if timestamps_array.shape != (n_spectra,):
                raise ValueError(
                    f"Timestamps must have shape ({n_spectra},), "
                    f"got {timestamps_array.shape}"
                )
        
        # Handle live times and real times together
        # Case 1: Both provided
        if live_times is not None and real_times is not None:
            live_times_array = (
                np.full(n_spectra, live_times, dtype=float)
                if np.isscalar(live_times)
                else np.asarray(live_times, dtype=float)
            )
            real_times_array = (
                np.full(n_spectra, real_times, dtype=float)
                if np.isscalar(real_times)
                else np.asarray(real_times, dtype=float)
            )
        # Case 2: Only live_times provided
        elif live_times is not None:
            live_times_array = (
                np.full(n_spectra, live_times, dtype=float)
                if np.isscalar(live_times)
                else np.asarray(live_times, dtype=float)
            )
            real_times_array = live_times_array.copy()
        # Case 3: Only real_times provided
        elif real_times is not None:
            live_times_array = np.full(n_spectra, None, dtype=object)  # Use None
            real_times_array = (
                np.full(n_spectra, real_times, dtype=float)
                if np.isscalar(real_times)
                else np.asarray(real_times, dtype=float)
            )
        # Case 4: Neither provided - default to 1.0
        else:
            live_times_array = np.ones(n_spectra, dtype=float)
            real_times_array = np.ones(n_spectra, dtype=float)
        
        # Validate shapes if they're arrays
        if live_times is not None and not np.isscalar(live_times):
            if live_times_array.shape != (n_spectra,):
                raise ValueError(
                    f"Live times must have shape ({n_spectra},), "
                    f"got {live_times_array.shape}"
                )
        
        if real_times is not None and not np.isscalar(real_times):
            if real_times_array.shape != (n_spectra,):
                raise ValueError(
                    f"Real times must have shape ({n_spectra},), "
                    f"got {real_times_array.shape}"
                )
        
        # Handle metadata
        if metadata is None:
            metadata_list = [{} for _ in range(n_spectra)]
        else:
            if len(metadata) != n_spectra:
                raise ValueError(
                    f"Metadata must have length {n_spectra}, got {len(metadata)}"
                )
            metadata_list = metadata
        
        # Create Spectrum objects
        spectra = []
        for i in range(n_spectra):
            spec = Spectrum(
                counts=counts_array[i],
                energy_edges=energy_edges,
                timestamp=timestamps_array[i],
                live_time=live_times_array[i],
                real_time=real_times_array[i],
                metadata=metadata_list[i].copy() if metadata_list[i] else {}
            )
            spectra.append(spec)
        
        # Create time series with shared calibration
        return cls(
            spectra,
            shared_calibration=True,
            integration_time=integration_time,
            stride_time=stride_time
        )
    
    # ========================================
    # Properties
    # ========================================
    
    @property
    def counts(self) -> np.ndarray:
        """
        Get counts as 2D numpy array (n_spectra, n_bins).
        
        Direct access for vectorized operations. In shared calibration mode,
        modifications to this array affect individual Spectrum objects.
        """
        if self._use_shared:
            # Array is live - modifications affect spectra
            return self._counts_array
        else:
            # Rebuild array from spectra (in case they were modified)
            self._counts_array = np.array([s._counts for s in self._spectra], dtype=float)
            return self._counts_array
    
    @property
    def spectra(self) -> List[Spectrum]:
        """Get list of Spectrum objects."""
        return self._spectra
    
    @property
    def energy_edges(self) -> np.ndarray:
        """
        Get energy edges.
        
        If shared calibration: return the shared edges.
        If independent: return edges from first spectrum (with warning if not all same).
        """
        if len(self._spectra) == 0:
            raise TimeSeriesError("No spectra in time series")
        
        if self._use_shared:
            if self._shared_calibration.edges is None:
                return np.arange(self.n_bins + 1, dtype=float)
            return self._shared_calibration.edges
        else:
            # Return first spectrum's edges
            return self._spectra[0].energy_edges
    
    @property
    def energy_centers(self) -> np.ndarray:
        """Get energy bin centers."""
        edges = self.energy_edges
        return (edges[:-1] + edges[1:]) / 2
    
    @property
    def timestamps(self) -> np.ndarray:
        """Get array of timestamps."""
        return self._timestamps
    
    @property
    def live_times(self) -> np.ndarray:
        """Get array of live times."""
        return self._live_times
    
    @property
    def real_times(self) -> np.ndarray:
        """Get array of real times."""
        return self._real_times
    
    @property
    def uses_shared_calibration(self) -> bool:
        """Check if this time series uses shared calibration."""
        return self._use_shared
    
    @property
    def n_spectra(self) -> int:
        """Get number of spectra."""
        return len(self._spectra)
    
    @property
    def integration_time(self) -> Optional[float]:
        """
        Get integration time (time window width) if available.
        
        Returns None if time series was not created with explicit integration time
        (e.g., from list mode).
        """
        return self._integration_time
    
    @property
    def stride_time(self) -> Optional[float]:
        """
        Get stride time (time between window starts) if available.
        
        Returns None if time series was not created with explicit stride time
        (e.g., from list mode).
        """
        return self._stride_time
    
    @property
    def n_bins(self) -> int:
        """Get number of energy bins."""
        if len(self._spectra) == 0:
            return 0
        return self._spectra[0].n_bins
    
    @property
    def is_calibrated(self) -> bool:
        """Check if spectra are energy calibrated."""
        if len(self._spectra) == 0:
            return False
        return self._spectra[0].is_calibrated
    
    # ========================================
    # Numpy Protocol
    # ========================================
    
    def __array__(self, dtype=None) -> np.ndarray:
        """
        Numpy array interface - returns counts array.
        
        Allows np.array(time_series) and use in numpy functions.
        """
        return np.asarray(self.counts, dtype=dtype)
    
    def __len__(self) -> int:
        """Length is number of spectra."""
        return len(self._spectra)
    
    def __getitem__(self, key: Union[int, slice]) -> Union[Spectrum, 'SpectralTimeSeries']:
        """
        Index or slice time series.
        
        Single index returns Spectrum object.
        Slice returns new SpectralTimeSeries.
        """
        if isinstance(key, (int, np.integer)):
            return self._spectra[key]
        elif isinstance(key, slice):
            return SpectralTimeSeries(
                spectra=self._spectra[key],
                shared_calibration=self._use_shared
            )
        else:
            raise TypeError(f"Invalid index type: {type(key)}")
    
    def __iter__(self):
        """Iterate over Spectrum objects."""
        return iter(self._spectra)
    
    # ========================================
    # Calibration Methods
    # ========================================
    
    def apply_calibration(
        self,
        coefficients: List[float],
        model: str = 'polynomial'
    ) -> 'SpectralTimeSeries':
        """
        Apply calibration to all spectra (returns new time series).
        
        Parameters
        ----------
        coefficients : list of float
            Calibration coefficients.
        model : str, optional
            Calibration model. Default is 'polynomial'.
        
        Returns
        -------
        SpectralTimeSeries
            New time series with calibrated spectra.
        """
        if self._use_shared:
            # Efficient: calibrate once, share result
            calibration = EnergyCalibration.from_coefficients(
                self.n_bins, coefficients, model
            )
            
            calibrated_spectra = [
                Spectrum(
                    counts=self._counts_array[i].copy(),
                    energy_edges=calibration.edges,
                    timestamp=self._timestamps[i],
                    live_time=self._live_times[i],
                    real_time=self._real_times[i],
                    metadata=self._spectra[i].metadata.copy(),
                )
                for i in range(len(self._spectra))
            ]
        else:
            # Apply to each independently
            calibrated_spectra = [
                s.apply_calibration(coefficients, model)
                for s in self._spectra
            ]
        
        return SpectralTimeSeries(
            spectra=calibrated_spectra,
            shared_calibration=self._use_shared
        )
    
    def apply_calibration_(
        self,
        coefficients: List[float],
        model: str = 'polynomial'
    ) -> 'SpectralTimeSeries':
        """
        Apply calibration in-place.
        
        If shared calibration: updates shared edges (affects all spectra).
        If independent: updates each spectrum's edges.
        
        Parameters
        ----------
        coefficients : list of float
            Calibration coefficients.
        model : str, optional
            Calibration model. Default is 'polynomial'.
        
        Returns
        -------
        SpectralTimeSeries
            Self (for chaining).
        """
        if self._use_shared:
            # Update shared calibration
            calibration = EnergyCalibration.from_coefficients(
                self.n_bins, coefficients, model
            )
            self._shared_calibration.edges = calibration.edges
        else:
            # Update each independently
            for spectrum in self._spectra:
                spectrum.apply_calibration_(coefficients, model)
        
        return self
    
    def to_shared_calibration(
        self,
        target_edges: Optional[np.ndarray] = None
    ) -> 'SpectralTimeSeries':
        """
        Convert to shared calibration mode (returns new time series).
        
        If spectra have different edges, rebins to common grid.
        
        Parameters
        ----------
        target_edges : np.ndarray or None, optional
            Target energy edges. If None, use edges from first spectrum.
        
        Returns
        -------
        SpectralTimeSeries
            New time series with shared calibration.
        """
        if self._use_shared:
            return self  # Already using shared calibration
        
        if target_edges is None:
            target_edges = self._spectra[0].energy_edges
        
        # Check if rebinning is needed
        needs_rebin = False
        for spec in self._spectra:
            if not np.allclose(spec.energy_edges, target_edges):
                needs_rebin = True
                break
        
        if needs_rebin:
            # Rebin all to common grid
            rebinned = [s.rebin_energy(target_edges) for s in self._spectra]
        else:
            rebinned = self._spectra
        
        return SpectralTimeSeries(
            spectra=rebinned,
            shared_calibration=True
        )
    
    def to_independent_calibration(self) -> 'SpectralTimeSeries':
        """
        Convert to independent calibration mode (returns new time series).
        
        Each spectrum gets its own copy of calibration.
        
        Returns
        -------
        SpectralTimeSeries
            New time series with independent calibrations.
        """
        if not self._use_shared:
            return self  # Already independent
        
        # Create deep copies
        independent_spectra = [s.copy(deep=True) for s in self._spectra]
        
        return SpectralTimeSeries(
            spectra=independent_spectra,
            shared_calibration=False
        )
    
    # ========================================
    # Vectorized Operations
    # ========================================
    
    def apply_vectorized(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        preserve_metadata: bool = True
    ) -> 'SpectralTimeSeries':
        """
        Apply vectorized function to counts array (returns new time series).
        
        Function should operate on 2D array of shape (n_spectra, n_bins).
        
        Parameters
        ----------
        func : callable
            Function that takes 2D array and returns 2D array.
        preserve_metadata : bool, optional
            Whether to preserve per-spectrum metadata. Default is True.
        
        Returns
        -------
        SpectralTimeSeries
            New time series with transformed counts.
        
        Examples
        --------
        >>> # Background subtraction
        >>> ts_sub = ts.apply_vectorized(lambda x: x - x.mean(axis=0))
        
        >>> # Normalization
        >>> ts_norm = ts.apply_vectorized(lambda x: x / x.sum(axis=1, keepdims=True))
        """
        new_counts = func(self.counts)
        
        # Create new spectra with updated counts
        new_spectra = []
        for i in range(len(self._spectra)):
            spec = self._spectra[i]
            new_spec = Spectrum(
                counts=new_counts[i],
                energy_edges=self.energy_edges if self.is_calibrated else None,
                timestamp=spec._timestamp,
                live_time=spec._live_time,
                real_time=spec._real_time,
                energy_unit=spec._energy_unit,
                metadata=spec._metadata.copy() if preserve_metadata else {},
            )
            new_spectra.append(new_spec)
        
        return SpectralTimeSeries(new_spectra, shared_calibration=self._use_shared)
    
    def background_subtract(
        self,
        background: Union[np.ndarray, Spectrum, str] = 'mean',
    ) -> 'SpectralTimeSeries':
        """
        Subtract background from all spectra (returns new time series).
        
        Parameters
        ----------
        background : np.ndarray, Spectrum, or str
            Background to subtract:
            - np.ndarray: Use provided array
            - Spectrum: Use spectrum's counts
            - 'mean': Use mean of all spectra
            - 'median': Use median of all spectra
        
        Returns
        -------
        SpectralTimeSeries
            Background-subtracted time series.
        """
        if isinstance(background, Spectrum):
            bg = background._counts
        elif isinstance(background, np.ndarray):
            bg = background
        elif background == 'mean':
            bg = np.mean(self.counts, axis=0)
        elif background == 'median':
            bg = np.median(self.counts, axis=0)
        else:
            raise ValueError(f"Unknown background specification: {background}")
        
        return self.apply_vectorized(lambda counts: counts - bg)
    
    def normalize(
        self,
        method: str = 'area',
        axis: Optional[int] = None
    ) -> 'SpectralTimeSeries':
        """
        Normalize all spectra (returns new time series).
        
        Parameters
        ----------
        method : str, optional
            Normalization method:
            - 'area': Normalize each spectrum to unit area
            - 'max': Normalize each spectrum to unit maximum
            Default is 'area'.
        axis : int or None, optional
            Axis for normalization (for 'area' and 'max'). Default is None.
        
        Returns
        -------
        SpectralTimeSeries
            Normalized time series.
        
        Notes
        -----
        For count rates, use vectorized division:
        ```
        # Convert all spectra to count rates
        rates = time_series.counts / time_series.live_times[:, np.newaxis]
        ```
        Or use per-spectrum count_rate property:
        ```
        for spec in time_series:
            rate = spec.count_rate
        ```
        """
        if method == 'area':
            areas = np.sum(self.counts, axis=1, keepdims=True)
            return self.apply_vectorized(lambda counts: counts / areas)
        elif method == 'max':
            maxima = np.max(self.counts, axis=1, keepdims=True)
            return self.apply_vectorized(lambda counts: counts / maxima)
        else:
            raise ValueError(
                f"Unknown normalization method: {method}. "
                f"Valid options: 'area', 'max'. "
                f"For count rates, use array division: counts / live_times[:, np.newaxis]"
            )
    
    # ========================================
    # Per-Spectrum Operations
    # ========================================
    
    def apply_to_each(
        self,
        func: Callable[[Spectrum], Spectrum],
        parallel: bool = False
    ) -> 'SpectralTimeSeries':
        """
        Apply function to each Spectrum object individually (returns new time series).
        
        Preserves per-spectrum metadata and behavior.
        
        Parameters
        ----------
        func : callable
            Function that takes Spectrum and returns Spectrum.
        parallel : bool, optional
            Whether to use parallel processing. Default is False.
        
        Returns
        -------
        SpectralTimeSeries
            New time series with transformed spectra.
        
        Examples
        --------
        >>> # Smooth each spectrum
        >>> ts_smooth = ts.apply_to_each(lambda s: s.normalize('area'))
        """
        if parallel:
            # Could use multiprocessing here
            from multiprocessing import Pool
            with Pool() as pool:
                new_spectra = pool.map(func, self._spectra)
        else:
            new_spectra = [func(spec) for spec in self._spectra]
        
        return SpectralTimeSeries(new_spectra, shared_calibration=self._use_shared)
    
    def filter_spectra(
        self,
        condition: Callable[[Spectrum], bool]
    ) -> 'SpectralTimeSeries':
        """
        Filter spectra based on condition (returns new time series).
        
        Parameters
        ----------
        condition : callable
            Function that takes Spectrum and returns bool.
        
        Returns
        -------
        SpectralTimeSeries
            Filtered time series.
        
        Examples
        --------
        >>> # Keep only high-quality spectra
        >>> ts_good = ts.filter_spectra(lambda s: s.live_time > 100)
        """
        filtered = [spec for spec in self._spectra if condition(spec)]
        return SpectralTimeSeries(filtered, shared_calibration=self._use_shared)
    
    # ========================================
    # Time Operations
    # ========================================
    
    def slice_time(
        self,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None
    ) -> 'SpectralTimeSeries':
        """
        Extract time slice (returns new time series).
        
        Parameters
        ----------
        t_min : float or None, optional
            Minimum timestamp.
        t_max : float or None, optional
            Maximum timestamp.
        
        Returns
        -------
        SpectralTimeSeries
            Sliced time series.
        """
        if t_min is None and t_max is None:
            return self
        
        mask = np.ones(len(self._spectra), dtype=bool)
        
        if t_min is not None:
            mask &= self._timestamps >= t_min
        
        if t_max is not None:
            mask &= self._timestamps <= t_max
        
        filtered_spectra = [spec for i, spec in enumerate(self._spectra) if mask[i]]
        return SpectralTimeSeries(filtered_spectra, shared_calibration=self._use_shared)
    
    def rebin_time(
        self,
        integration_time: float,
        stride: Optional[float] = None,
        allow_overlap: bool = True
    ) -> 'SpectralTimeSeries':
        """
        Rebin spectra in time (returns new time series).
        
        Parameters
        ----------
        integration_time : float
            Time window for integration.
        stride : float or None, optional
            Step between windows. If None, use integration_time (no overlap).
        allow_overlap : bool, optional
            Whether to allow overlapping windows. Default is True.
        
        Returns
        -------
        SpectralTimeSeries
            Rebinned time series.
        """
        if stride is None:
            stride = integration_time
        
        if not allow_overlap and stride < integration_time:
            raise ValueError("Stride must be >= integration_time when overlap not allowed")
        
        # Sort by timestamp
        sorted_indices = np.argsort(self._timestamps)
        sorted_times = self._timestamps[sorted_indices]
        
        # Create time windows
        t_start = sorted_times[0]
        t_end = sorted_times[-1]
        window_starts = np.arange(t_start, t_end, stride)
        
        rebinned_spectra = []
        for t_win in window_starts:
            # Find spectra in this window
            mask = (sorted_times >= t_win) & (sorted_times < t_win + integration_time)
            indices = sorted_indices[mask]
            
            if len(indices) == 0:
                continue
            
            # Sum spectra in window
            summed = self._spectra[indices[0]].copy()
            for idx in indices[1:]:
                summed = summed + self._spectra[idx]
            
            # Update timestamp to window center
            summed._timestamp = t_win + integration_time / 2
            rebinned_spectra.append(summed)
        
        return SpectralTimeSeries(rebinned_spectra, shared_calibration=self._use_shared)
    
    def reintegrate(
        self,
        new_integration_time: float,
        new_stride_time: Optional[float] = None
    ) -> 'SpectralTimeSeries':
        """
        Reintegrate time series with larger integration/stride times.
        
        This method allows combining spectra from a time series (typically created
        from list mode) into larger time windows. The new integration and stride
        times must be even multiples of the original values and cannot be smaller.
        
        Parameters
        ----------
        new_integration_time : float
            New integration time (time window width). Must be an even multiple
            of the original integration_time and >= original value.
        new_stride_time : float or None, optional
            New stride time (time between window starts). Must be an even multiple
            of the original stride_time and >= original value. If None, uses
            new_integration_time (non-overlapping windows). Default is None.
        
        Returns
        -------
        SpectralTimeSeries
            Reintegrated time series with larger time windows.
        
        Raises
        ------
        TimeSeriesError
            If time series was not created with integration/stride times, or if
            new times are not valid multiples of original times.
        
        Examples
        --------
        >>> # Create time series from list mode with 0.5s windows
        >>> ts = SpectralTimeSeries.from_list_mode(
        ...     time_deltas, energies,
        ...     integration_time=0.5,
        ...     stride_time=0.5
        ... )
        >>> print(ts.n_spectra)
        100
        
        >>> # Reintegrate to 2.0s windows (4x larger)
        >>> ts_2s = ts.reintegrate(new_integration_time=2.0)
        >>> print(ts_2s.n_spectra)
        25
        
        >>> # Reintegrate with overlapping windows
        >>> ts_overlap = ts.reintegrate(
        ...     new_integration_time=2.0,
        ...     new_stride_time=1.0
        ... )
        
        Notes
        -----
        - This method is designed for time series created from list mode data
        - The original integration/stride times must be stored in the time series
        - New times must be even multiples: new_time / old_time must be an integer
        - Cannot reduce integration or stride time (only increase)
        - Spectra are combined by summing counts and times
        """
        # Validate that time series has integration/stride metadata
        if self._integration_time is None or self._stride_time is None:
            raise TimeSeriesError(
                "Time series does not have integration_time and stride_time information. "
                "Reintegration requires either:\n"
                "  1. Time series created from list mode (from_list_mode())\n"
                "  2. Constant real_time values (for integration_time)\n"
                "  3. Evenly-spaced timestamps (for stride_time)\n"
                f"Current state: integration_time={self._integration_time}, "
                f"stride_time={self._stride_time}"
            )
        
        # Set default stride time
        if new_stride_time is None:
            new_stride_time = new_integration_time
        
        # Validate new times are >= original times
        if new_integration_time < self._integration_time:
            raise ValueError(
                f"new_integration_time ({new_integration_time}) must be >= "
                f"original integration_time ({self._integration_time})"
            )
        if new_stride_time < self._stride_time:
            raise ValueError(
                f"new_stride_time ({new_stride_time}) must be >= "
                f"original stride_time ({self._stride_time})"
            )
        
        # Validate new times are even multiples of original times
        integration_factor = new_integration_time / self._integration_time
        stride_factor = new_stride_time / self._stride_time
        
        # Check if factors are close to integers (allow small floating point error)
        if not np.isclose(integration_factor, round(integration_factor), rtol=1e-9):
            raise ValueError(
                f"new_integration_time ({new_integration_time}) must be an even "
                f"multiple of original integration_time ({self._integration_time}). "
                f"Factor: {integration_factor}"
            )
        if not np.isclose(stride_factor, round(stride_factor), rtol=1e-9):
            raise ValueError(
                f"new_stride_time ({new_stride_time}) must be an even "
                f"multiple of original stride_time ({self._stride_time}). "
                f"Factor: {stride_factor}"
            )
        
        integration_factor = int(round(integration_factor))
        stride_factor = int(round(stride_factor))
        
        if len(self._spectra) == 0:
            return SpectralTimeSeries(
                [],
                shared_calibration=self._use_shared,
                integration_time=new_integration_time,
                stride_time=new_stride_time
            )
        
        # Extract window_start times from metadata
        window_starts = []
        for spec in self._spectra:
            if 'window_start' in spec.metadata:
                window_starts.append(spec.metadata['window_start'])
            else:
                # Fallback: infer from timestamp (center) and integration time
                window_starts.append(spec.timestamp - self._integration_time / 2)
        window_starts = np.array(window_starts)
        
        # Group spectra into new windows
        # New windows start at multiples of new_stride_time from the first window
        first_window_start = window_starts[0]
        last_window_start = window_starts[-1]
        
        # Calculate new window starts
        new_window_starts = np.arange(
            first_window_start,
            last_window_start + new_integration_time,
            new_stride_time
        )
        
        reintegrated_spectra = []
        for new_start in new_window_starts:
            new_end = new_start + new_integration_time
            
            # Find spectra whose windows overlap with this new window
            # A spectrum window [ws, we) overlaps with [new_start, new_end) if:
            # ws < new_end AND we > new_start
            spectrum_ends = window_starts + self._integration_time
            overlap_mask = (window_starts < new_end) & (spectrum_ends > new_start)
            
            overlapping_indices = np.where(overlap_mask)[0]
            
            if len(overlapping_indices) == 0:
                continue
            
            # Sum overlapping spectra
            summed = self._spectra[overlapping_indices[0]].copy()
            for idx in overlapping_indices[1:]:
                summed = summed + self._spectra[idx]
            
            # Update metadata and timing
            summed._timestamp = new_start + new_integration_time / 2
            summed._real_time = new_integration_time  # Set to exact new integration time
            summed._metadata['window_start'] = new_start
            summed._metadata['window_end'] = new_end
            summed._metadata['n_spectra_combined'] = len(overlapping_indices)
            
            reintegrated_spectra.append(summed)
        
        return SpectralTimeSeries(
            reintegrated_spectra,
            shared_calibration=self._use_shared,
            integration_time=new_integration_time,
            stride_time=new_stride_time
        )
    
    def integrate_time(
        self,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None
    ) -> Spectrum:
        """
        Integrate spectra over time range (returns single spectrum).
        
        Parameters
        ----------
        t_min : float or None, optional
            Minimum timestamp.
        t_max : float or None, optional
            Maximum timestamp.
        
        Returns
        -------
        Spectrum
            Integrated spectrum.
        """
        sliced = self.slice_time(t_min, t_max)
        
        if len(sliced) == 0:
            raise TimeSeriesError("No spectra in specified time range")
        
        # Sum all spectra
        integrated = sliced[0].copy()
        for spec in sliced[1:]:
            integrated = integrated + spec
        
        return integrated
    
    # ========================================
    # Analysis Methods
    # ========================================
    
    def mean_spectrum(self) -> Spectrum:
        """
        Compute mean spectrum (returns single spectrum).
        
        Returns
        -------
        Spectrum
            Mean spectrum.
        """
        mean_counts = np.mean(self.counts, axis=0)
        mean_live_time = np.mean(self._live_times)
        mean_real_time = np.mean(self._real_times)
        
        return Spectrum(
            counts=mean_counts,
            energy_edges=self.energy_edges if self.is_calibrated else None,
            live_time=mean_live_time,
            real_time=mean_real_time,
        )
    
    def sum_spectrum(self) -> Spectrum:
        """
        Compute sum of all spectra (returns single spectrum).
        
        Returns
        -------
        Spectrum
            Summed spectrum.
        """
        return self.integrate_time()
    
    # ========================================
    # String Representation
    # ========================================
    
    def __repr__(self) -> str:
        """String representation."""
        cal_mode = "shared" if self._use_shared else "independent"
        cal_str = "calibrated" if self.is_calibrated else "uncalibrated"
        
        return (
            f"SpectralTimeSeries(n_spectra={self.n_spectra}, "
            f"n_bins={self.n_bins}, {cal_str}, "
            f"calibration_mode={cal_mode})"
        )

