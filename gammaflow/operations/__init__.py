"""Operations for spectrum and time series manipulation."""

from gammaflow.operations.roi import (
    EnergyROI,
    rebin_spectrum_rois,
    rebin_time_series_rois,
    create_roi_collection,
    check_roi_overlaps,
    print_roi_summary
)

__all__ = [
    'EnergyROI',
    'rebin_spectrum_rois',
    'rebin_time_series_rois',
    'create_roi_collection',
    'check_roi_overlaps',
    'print_roi_summary'
]

