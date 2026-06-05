"""
Dataset loaders for standard gamma-ray spectroscopy datasets.

Provides high-level interfaces for loading and working with common
benchmark datasets used in radiation detection research.

Available datasets:
- ``APLStarterKitDataset`` — APL Starter Kit dataset (pre-binned spectral data)
- ``TopCoderDataset`` — TopCoder Urban Radiation Search challenge data
- ``RADAIDataset`` — RADAI synthetic dataset (HDF5 list-mode, requires h5py)
"""

from gammaflow.datasets.apl_starter_kit import APLStarterKitDataset
from gammaflow.datasets.topcoder import TopCoderDataset

__all__ = ["APLStarterKitDataset", "TopCoderDataset"]

# RADAI requires h5py — import conditionally
try:
    from gammaflow.datasets.radai import RADAIDataset, SourceEncounter, SourceWindow

    __all__.extend(["RADAIDataset", "SourceEncounter", "SourceWindow"])
except ImportError:
    pass
