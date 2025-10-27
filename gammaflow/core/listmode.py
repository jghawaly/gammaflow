"""
ListMode class for event-by-event gamma ray data.

This module provides the ListMode class for storing and manipulating
individual gamma ray detection events with their timing and energy information.
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
import numpy.typing as npt


class ListMode:
    """
    Container for list mode (event-by-event) gamma ray data.
    
    List mode data stores individual detection events with time deltas
    (time since previous event) and energies (or pulse heights).
    
    Parameters
    ----------
    time_deltas : array-like
        Time since previous event for each event (in seconds). Shape: (n_events,)
    energies : array-like
        Energy or pulse height for each event. Shape: (n_events,)
    metadata : dict or None, optional
        Additional metadata (e.g., detector info, acquisition settings).
        Default is empty dict.
    
    Attributes
    ----------
    time_deltas : np.ndarray
        Time since previous event (seconds).
    energies : np.ndarray
        Event energies or pulse heights.
    metadata : dict
        Additional metadata.
    n_events : int
        Number of events.
    total_time : float
        Total acquisition time (seconds).
    absolute_times : np.ndarray
        Absolute time of each event (computed on access).
    mean_rate : float
        Mean event rate (Hz).
    
    Examples
    --------
    >>> # Create list mode data
    >>> import numpy as np
    >>> time_deltas = np.random.exponential(0.001, size=10000)  # ~1000 Hz
    >>> energies = np.random.gamma(2, 500, size=10000)
    >>> lm = ListMode(time_deltas, energies)
    >>> print(f"Events: {lm.n_events}, Duration: {lm.total_time:.1f}s")
    
    >>> # Access properties
    >>> print(f"Mean rate: {lm.mean_rate:.1f} Hz")
    >>> print(f"Energy range: {lm.energy_range}")
    
    >>> # Filter events by energy
    >>> lm_filtered = lm.filter_energy(e_min=200, e_max=800)
    
    >>> # Slice by time
    >>> lm_slice = lm.slice_time(t_min=10.0, t_max=20.0)
    """
    
    def __init__(
        self,
        time_deltas: npt.ArrayLike,
        energies: npt.ArrayLike,
        metadata: Optional[Dict[str, Any]] = None
    ):
        # Convert to numpy arrays
        self._time_deltas = np.asarray(time_deltas, dtype=float)
        self._energies = np.asarray(energies, dtype=float)
        
        # Validate
        if len(self._time_deltas) != len(self._energies):
            raise ValueError(
                f"time_deltas and energies must have same length. "
                f"Got {len(self._time_deltas)} and {len(self._energies)}"
            )
        
        # Store metadata
        self._metadata = metadata.copy() if metadata is not None else {}
        
        # Cache for absolute times (computed on first access)
        self._absolute_times = None
    
    # ========================================
    # Properties
    # ========================================
    
    @property
    def time_deltas(self) -> np.ndarray:
        """Get time deltas (time since previous event)."""
        return self._time_deltas
    
    @property
    def energies(self) -> np.ndarray:
        """Get event energies."""
        return self._energies
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata dictionary."""
        return self._metadata
    
    @property
    def n_events(self) -> int:
        """Get number of events."""
        return len(self._energies)
    
    @property
    def absolute_times(self) -> np.ndarray:
        """
        Get absolute times of events.
        
        Computed via cumulative sum of time deltas. Cached after first access.
        """
        if self._absolute_times is None:
            self._absolute_times = np.cumsum(self._time_deltas)
        return self._absolute_times
    
    @property
    def total_time(self) -> float:
        """Get total acquisition time (seconds)."""
        if len(self._time_deltas) == 0:
            return 0.0
        return self.absolute_times[-1]
    
    @property
    def mean_rate(self) -> float:
        """Get mean event rate (Hz)."""
        if self.total_time > 0:
            return self.n_events / self.total_time
        return 0.0
    
    @property
    def energy_range(self) -> Tuple[float, float]:
        """Get (min, max) energy range."""
        if self.n_events == 0:
            return (0.0, 0.0)
        return (float(self._energies.min()), float(self._energies.max()))
    
    # ========================================
    # String Representation
    # ========================================
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ListMode(n_events={self.n_events}, "
            f"duration={self.total_time:.2f}s, "
            f"rate={self.mean_rate:.1f}Hz)"
        )
    
    def __len__(self) -> int:
        """Number of events."""
        return self.n_events
    
    # ========================================
    # Filtering and Slicing
    # ========================================
    
    def filter_energy(
        self,
        e_min: Optional[float] = None,
        e_max: Optional[float] = None
    ) -> 'ListMode':
        """
        Filter events by energy range.
        
        Parameters
        ----------
        e_min : float or None, optional
            Minimum energy (inclusive). If None, no lower limit.
        e_max : float or None, optional
            Maximum energy (exclusive). If None, no upper limit.
        
        Returns
        -------
        ListMode
            New ListMode with filtered events.
        
        Examples
        --------
        >>> lm_roi = lm.filter_energy(e_min=200, e_max=800)
        """
        mask = np.ones(self.n_events, dtype=bool)
        
        if e_min is not None:
            mask &= self._energies >= e_min
        
        if e_max is not None:
            mask &= self._energies < e_max
        
        return ListMode(
            time_deltas=self._time_deltas[mask],
            energies=self._energies[mask],
            metadata=self._metadata
        )
    
    def slice_time(
        self,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None
    ) -> 'ListMode':
        """
        Extract events in time range.
        
        Parameters
        ----------
        t_min : float or None, optional
            Minimum time (inclusive).
        t_max : float or None, optional
            Maximum time (exclusive).
        
        Returns
        -------
        ListMode
            New ListMode with events in time range.
        
        Notes
        -----
        Time deltas are adjusted so the first event in the slice has
        delta relative to t_min.
        
        Examples
        --------
        >>> lm_slice = lm.slice_time(t_min=10.0, t_max=20.0)
        """
        abs_times = self.absolute_times
        mask = np.ones(self.n_events, dtype=bool)
        
        if t_min is not None:
            mask &= abs_times >= t_min
        
        if t_max is not None:
            mask &= abs_times < t_max
        
        # Extract events
        filtered_times = abs_times[mask]
        filtered_energies = self._energies[mask]
        
        # Recompute time deltas
        if len(filtered_times) > 0:
            # First event: delta from t_min (or 0 if t_min is None)
            start_time = t_min if t_min is not None else 0.0
            new_deltas = np.diff(np.concatenate([[start_time], filtered_times]))
        else:
            new_deltas = np.array([])
        
        return ListMode(
            time_deltas=new_deltas,
            energies=filtered_energies,
            metadata=self._metadata
        )
    
    def copy(self) -> 'ListMode':
        """
        Create a copy of this ListMode.
        
        Returns
        -------
        ListMode
            New ListMode with copied data.
        """
        return ListMode(
            time_deltas=self._time_deltas.copy(),
            energies=self._energies.copy(),
            metadata=self._metadata.copy()
        )
    
    # ========================================
    # I/O Methods (placeholders for future)
    # ========================================
    
    @classmethod
    def from_file(cls, filename: str) -> 'ListMode':
        """
        Load list mode data from file.
        
        Parameters
        ----------
        filename : str
            Path to file.
        
        Returns
        -------
        ListMode
            Loaded list mode data.
        
        Notes
        -----
        Not yet implemented. Placeholder for future file I/O support.
        """
        raise NotImplementedError("File I/O not yet implemented")
    
    def to_file(self, filename: str) -> None:
        """
        Save list mode data to file.
        
        Parameters
        ----------
        filename : str
            Path to save file.
        
        Notes
        -----
        Not yet implemented. Placeholder for future file I/O support.
        """
        raise NotImplementedError("File I/O not yet implemented")

