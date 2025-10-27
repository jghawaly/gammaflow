"""
Region of Interest (ROI) operations for spectra.

This module provides tools for defining energy ROIs and rebinning spectra
based on these regions. ROIs can overlap, be non-consecutive, and carry
metadata about their creation method and purpose.
"""

from typing import Optional, List, Dict, Any, Union
import numpy as np
import numpy.typing as npt

from gammaflow.core.spectrum import Spectrum
from gammaflow.core.time_series import SpectralTimeSeries


class EnergyROI:
    """
    Represents a Region of Interest (ROI) in energy space.
    
    An ROI defines an energy window [e_min, e_max] with associated metadata
    such as a label and creation method. ROIs are used to rebin spectra into
    integrated counts over specific energy ranges.
    
    Parameters
    ----------
    e_min : float
        Minimum energy (inclusive).
    e_max : float
        Maximum energy (inclusive).
    label : str or None, optional
        Human-readable label for this ROI (e.g., "K-40 Peak", "Background").
        Default is None.
    method : str or None, optional
        Method used to create this ROI (e.g., "manual", "Censored Energy Windows",
        "peak_search", "automatic"). Default is "manual".
    metadata : dict or None, optional
        Additional metadata about this ROI. Default is empty dict.
    
    Attributes
    ----------
    e_min : float
        Minimum energy (inclusive).
    e_max : float
        Maximum energy (inclusive).
    label : str or None
        Human-readable label.
    method : str
        Creation method.
    metadata : dict
        Additional metadata.
    
    Examples
    --------
    >>> # Define a K-40 peak ROI
    >>> k40_roi = EnergyROI(
    ...     e_min=1450, e_max=1470,
    ...     label="K-40 Peak",
    ...     method="manual"
    ... )
    
    >>> # Check if energy is in ROI
    >>> k40_roi.contains(1460)
    True
    
    >>> # Get width
    >>> k40_roi.width
    20.0
    """
    
    def __init__(
        self,
        e_min: float,
        e_max: float,
        label: Optional[str] = None,
        method: str = "manual",
        metadata: Optional[Dict[str, Any]] = None
    ):
        if e_min >= e_max:
            raise ValueError(f"e_min ({e_min}) must be < e_max ({e_max})")
        
        self.e_min = float(e_min)
        self.e_max = float(e_max)
        self.label = label
        self.method = method
        self.metadata = metadata.copy() if metadata is not None else {}
    
    @property
    def width(self) -> float:
        """Get energy width of ROI."""
        return self.e_max - self.e_min
    
    @property
    def center(self) -> float:
        """Get center energy of ROI."""
        return (self.e_min + self.e_max) / 2
    
    def contains(self, energy: float) -> bool:
        """
        Check if energy is within this ROI.
        
        Parameters
        ----------
        energy : float
            Energy to check.
        
        Returns
        -------
        bool
            True if energy is in [e_min, e_max].
        """
        return self.e_min <= energy <= self.e_max
    
    def overlaps(self, other: 'EnergyROI') -> bool:
        """
        Check if this ROI overlaps with another.
        
        Parameters
        ----------
        other : EnergyROI
            Another ROI to check.
        
        Returns
        -------
        bool
            True if ROIs overlap.
        """
        return not (self.e_max < other.e_min or self.e_min > other.e_max)
    
    def integrate_spectrum(self, spectrum: Spectrum) -> float:
        """
        Integrate spectrum counts over this ROI.
        
        Parameters
        ----------
        spectrum : Spectrum
            Spectrum to integrate.
        
        Returns
        -------
        float
            Total counts in this energy range.
        """
        return spectrum.integrate(e_min=self.e_min, e_max=self.e_max)
    
    def __repr__(self) -> str:
        label_str = f"'{self.label}'" if self.label else "unlabeled"
        return (
            f"EnergyROI({label_str}, "
            f"e_min={self.e_min:.1f}, e_max={self.e_max:.1f}, "
            f"method='{self.method}')"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert ROI to dictionary representation.
        
        Returns
        -------
        dict
            Dictionary with all ROI attributes.
        """
        return {
            'e_min': self.e_min,
            'e_max': self.e_max,
            'label': self.label,
            'method': self.method,
            'metadata': self.metadata.copy()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnergyROI':
        """
        Create ROI from dictionary representation.
        
        Parameters
        ----------
        data : dict
            Dictionary with ROI attributes.
        
        Returns
        -------
        EnergyROI
            New ROI instance.
        """
        return cls(
            e_min=data['e_min'],
            e_max=data['e_max'],
            label=data.get('label'),
            method=data.get('method', 'manual'),
            metadata=data.get('metadata')
        )


def rebin_spectrum_rois(
    spectrum: Spectrum,
    rois: List[EnergyROI],
    return_labels: bool = False
) -> Union[np.ndarray, tuple]:
    """
    Rebin spectrum using ROIs (integrate over each ROI).
    
    This rebins a spectrum by integrating counts over each defined ROI,
    returning a 1D array where each element is the integrated counts for
    one ROI. ROIs can overlap and be non-consecutive.
    
    Parameters
    ----------
    spectrum : Spectrum
        Spectrum to rebin.
    rois : list of EnergyROI
        List of ROIs to use for rebinning.
    return_labels : bool, optional
        If True, also return ROI labels. Default is False.
    
    Returns
    -------
    counts : np.ndarray
        1D array of integrated counts, one per ROI. Shape: (n_rois,)
    labels : list of str (optional)
        ROI labels (if return_labels=True).
    
    Examples
    --------
    >>> rois = [
    ...     EnergyROI(100, 200, label="Low Energy"),
    ...     EnergyROI(500, 600, label="Mid Energy"),
    ...     EnergyROI(1400, 1500, label="K-40 Peak")
    ... ]
    >>> counts = rebin_spectrum_rois(spectrum, rois)
    >>> print(counts)
    [1234.0, 5678.0, 9012.0]
    
    >>> # With labels
    >>> counts, labels = rebin_spectrum_rois(spectrum, rois, return_labels=True)
    """
    if not spectrum.is_calibrated:
        raise ValueError("Spectrum must be energy-calibrated to use ROI rebinning")
    
    if len(rois) == 0:
        raise ValueError("Must provide at least one ROI")
    
    # Integrate over each ROI
    counts = np.array([roi.integrate_spectrum(spectrum) for roi in rois])
    
    if return_labels:
        labels = [roi.label if roi.label is not None else f"ROI_{i}" for i, roi in enumerate(rois)]
        return counts, labels
    
    return counts


def rebin_time_series_rois(
    time_series: SpectralTimeSeries,
    rois: List[EnergyROI],
    return_labels: bool = False
) -> Union[np.ndarray, tuple]:
    """
    Rebin time series using ROIs (integrate over each ROI for each spectrum).
    
    This rebins an entire time series by integrating counts over each defined
    ROI for every spectrum, returning a 2D array where rows are time points
    and columns are ROIs.
    
    Parameters
    ----------
    time_series : SpectralTimeSeries
        Time series to rebin.
    rois : list of EnergyROI
        List of ROIs to use for rebinning.
    return_labels : bool, optional
        If True, also return ROI labels. Default is False.
    
    Returns
    -------
    counts : np.ndarray
        2D array of integrated counts. Shape: (n_spectra, n_rois).
        Each row is a time point, each column is an ROI.
    labels : list of str (optional)
        ROI labels (if return_labels=True).
    
    Examples
    --------
    >>> rois = [
    ...     EnergyROI(100, 200, label="Background"),
    ...     EnergyROI(1400, 1500, label="K-40 Peak")
    ... ]
    >>> counts = rebin_time_series_rois(time_series, rois)
    >>> print(counts.shape)
    (100, 2)  # 100 time points, 2 ROIs
    
    >>> # Analyze time evolution of K-40 peak
    >>> k40_counts = counts[:, 1]  # Second ROI
    >>> plt.plot(time_series.timestamps, k40_counts)
    """
    if len(rois) == 0:
        raise ValueError("Must provide at least one ROI")
    
    # Vectorized approach: integrate each ROI for all spectra
    n_spectra = time_series.n_spectra
    n_rois = len(rois)
    counts = np.zeros((n_spectra, n_rois))
    
    for i, roi in enumerate(rois):
        for j, spec in enumerate(time_series.spectra):
            counts[j, i] = roi.integrate_spectrum(spec)
    
    if return_labels:
        labels = [roi.label if roi.label is not None else f"ROI_{i}" for i, roi in enumerate(rois)]
        return counts, labels
    
    return counts


def create_roi_collection(
    roi_definitions: List[tuple],
    method: str = "manual",
    shared_metadata: Optional[Dict[str, Any]] = None
) -> List[EnergyROI]:
    """
    Create a collection of ROIs from simple definitions.
    
    Convenience function to create multiple ROIs at once from a list of
    (e_min, e_max, label) tuples.
    
    Parameters
    ----------
    roi_definitions : list of tuple
        List of ROI definitions. Each tuple should be:
        - (e_min, e_max) or
        - (e_min, e_max, label)
    method : str, optional
        Method string to apply to all ROIs. Default is "manual".
    shared_metadata : dict or None, optional
        Metadata to apply to all ROIs. Default is None.
    
    Returns
    -------
    list of EnergyROI
        List of created ROIs.
    
    Examples
    --------
    >>> # Simple definitions
    >>> rois = create_roi_collection([
    ...     (100, 200, "Background 1"),
    ...     (500, 600, "Cs-137 Peak"),
    ...     (1400, 1500, "K-40 Peak")
    ... ])
    
    >>> # From peak search results
    >>> peak_energies = [661.7, 1460.8]  # keV
    >>> roi_width = 20  # keV
    >>> roi_defs = [(e - roi_width/2, e + roi_width/2, f"Peak at {e:.1f}")
    ...             for e in peak_energies]
    >>> rois = create_roi_collection(roi_defs, method="peak_search")
    """
    rois = []
    for i, roi_def in enumerate(roi_definitions):
        if len(roi_def) == 2:
            e_min, e_max = roi_def
            label = None
        elif len(roi_def) == 3:
            e_min, e_max, label = roi_def
        else:
            raise ValueError(
                f"ROI definition {i} must be (e_min, e_max) or "
                f"(e_min, e_max, label), got {len(roi_def)} elements"
            )
        
        rois.append(EnergyROI(
            e_min=e_min,
            e_max=e_max,
            label=label,
            method=method,
            metadata=shared_metadata.copy() if shared_metadata else {}
        ))
    
    return rois


def check_roi_overlaps(rois: List[EnergyROI]) -> List[tuple]:
    """
    Check for overlaps between ROIs.
    
    Parameters
    ----------
    rois : list of EnergyROI
        List of ROIs to check.
    
    Returns
    -------
    list of tuple
        List of (i, j) index pairs where ROIs overlap.
        Empty list if no overlaps.
    
    Examples
    --------
    >>> rois = [
    ...     EnergyROI(100, 200, "A"),
    ...     EnergyROI(150, 250, "B"),  # Overlaps with A
    ...     EnergyROI(300, 400, "C")
    ... ]
    >>> overlaps = check_roi_overlaps(rois)
    >>> print(overlaps)
    [(0, 1)]  # ROI 0 and 1 overlap
    """
    overlaps = []
    for i in range(len(rois)):
        for j in range(i + 1, len(rois)):
            if rois[i].overlaps(rois[j]):
                overlaps.append((i, j))
    return overlaps


def print_roi_summary(rois: List[EnergyROI], check_overlaps: bool = True):
    """
    Print a formatted summary of ROIs.
    
    Parameters
    ----------
    rois : list of EnergyROI
        ROIs to summarize.
    check_overlaps : bool, optional
        Whether to check and report overlaps. Default is True.
    
    Examples
    --------
    >>> rois = [
    ...     EnergyROI(100, 200, "Background", method="manual"),
    ...     EnergyROI(500, 700, "Cs-137", method="peak_search"),
    ...     EnergyROI(1400, 1500, "K-40", method="manual")
    ... ]
    >>> print_roi_summary(rois)
    """
    print(f"\nROI Summary: {len(rois)} ROIs defined")
    print("=" * 70)
    
    for i, roi in enumerate(rois):
        label_str = f"'{roi.label}'" if roi.label else "unlabeled"
        print(f"{i:2d}. {label_str:20s} [{roi.e_min:8.2f}, {roi.e_max:8.2f}] keV "
              f"(width={roi.width:6.2f}, method='{roi.method}')")
    
    if check_overlaps:
        overlaps = check_roi_overlaps(rois)
        if overlaps:
            print(f"\nWarning: {len(overlaps)} overlap(s) detected:")
            for i, j in overlaps:
                print(f"  ROI {i} ('{rois[i].label}') overlaps with "
                      f"ROI {j} ('{rois[j].label}')")
        else:
            print("\nNo overlaps detected.")
    
    print("=" * 70)

