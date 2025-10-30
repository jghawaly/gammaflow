"""
Visualization tools for gamma-ray spectroscopy.

This module provides publication-quality plotting functions for spectra,
time series, and spectral evolution using matplotlib and seaborn.
"""

from typing import Optional, Tuple, Union, Literal
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm, Normalize

from gammaflow.core.spectrum import Spectrum
from gammaflow.core.time_series import SpectralTimeSeries


# Set seaborn style for publication-quality plots
sns.set_style("whitegrid")
sns.set_context("paper")


def _get_time_axis(time_series: SpectralTimeSeries) -> np.ndarray:
    """Helper to extract time axis from time series."""
    if time_series.timestamps is not None:
        return time_series.timestamps
    elif time_series.real_times is not None:
        return np.cumsum(time_series.real_times)
    else:
        # Extract from individual spectra
        times = np.array([s.real_time for s in time_series.spectra])
        return np.cumsum(times)


def _get_normalization_times(time_series: SpectralTimeSeries) -> np.ndarray:
    """Helper to extract normalization times (prefer live_time)."""
    # Try live_times first (prefer if available and contains valid values)
    try:
        if time_series.live_times is not None:
            live_times_array = np.asarray(time_series.live_times)
            # Check if contains valid (non-None) numeric values
            if live_times_array.size > 0 and not (live_times_array[0] is None):
                return live_times_array.astype(float)
    except (TypeError, ValueError):
        pass
    
    # Try real_times
    try:
        if time_series.real_times is not None:
            real_times_array = np.asarray(time_series.real_times)
            if real_times_array.size > 0 and not (real_times_array[0] is None):
                return real_times_array.astype(float)
    except (TypeError, ValueError):
        pass
    
    # Extract times from individual spectra as fallback
    return np.array([
        s.live_time if s.live_time is not None else s.real_time
        for s in time_series.spectra
    ], dtype=float)


def plot_spectrum(
    spectrum: Spectrum,
    mode: Literal['counts', 'count_rate', 'count_density'] = 'counts',
    log_y: bool = True,
    show_uncertainty: bool = True,
    energy_range: Optional[Tuple[float, float]] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot a single spectrum.
    
    Parameters
    ----------
    spectrum : Spectrum
        Spectrum to plot.
    mode : {'counts', 'count_rate', 'count_density'}, optional
        What to plot on y-axis. Default is 'counts'.
    log_y : bool, optional
        Use logarithmic y-axis. Default is True.
    show_uncertainty : bool, optional
        Show uncertainty bands. Default is True.
    energy_range : tuple of float, optional
        (e_min, e_max) for x-axis limits. If None, use full range.
    fig : Figure, optional
        Existing figure to plot on. If None, create new figure.
    ax : Axes, optional
        Existing axes to plot on. If None, create new axes.
    label : str, optional
        Label for legend.
    color : str, optional
        Color for the plot.
    **kwargs
        Additional keyword arguments passed to matplotlib step plot.
        
    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.
        
    Examples
    --------
    >>> fig, ax = plot_spectrum(spectrum, log_y=True)
    >>> plt.show()
    
    >>> # Multiple spectra on same plot
    >>> fig, ax = plot_spectrum(spectrum1, label='Background', color='blue')
    >>> plot_spectrum(spectrum2, fig=fig, ax=ax, label='Source', color='red')
    >>> ax.legend()
    >>> plt.show()
    """
    # Create figure if needed
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get x and y data
    if spectrum.is_calibrated:
        x = spectrum.energy_edges
        x_centers = spectrum.energy_centers
        xlabel = f'Energy ({spectrum.energy_unit})'
    else:
        x = np.arange(spectrum.n_bins + 1)
        x_centers = np.arange(spectrum.n_bins) + 0.5
        xlabel = 'Channel'
    
    # Select y data based on mode
    if mode == 'counts':
        y = spectrum.counts
        ylabel = 'Counts'
        uncertainty = spectrum.uncertainty
    elif mode == 'count_rate':
        y = spectrum.count_rate
        ylabel = r'Count Rate (s$^{-1}$)'
        # Scale uncertainty by time as well
        time = spectrum.live_time if spectrum.live_time is not None else spectrum.real_time
        uncertainty = spectrum.uncertainty / time if time > 0 else spectrum.uncertainty
    elif mode == 'count_density':
        y = spectrum.count_density
        ylabel = f'Count Density ({spectrum.energy_unit}' + r'$^{-1}$)'
        uncertainty = spectrum.uncertainty / spectrum.energy_widths
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Plot with steps (histogram style)
    ax.step(x, np.append(y, y[-1]), where='post', label=label, color=color, **kwargs)
    
    # Add uncertainty bands
    if show_uncertainty and uncertainty is not None:
        lower = y - uncertainty
        upper = y + uncertainty
        ax.fill_between(
            x_centers, lower, upper,
            alpha=0.3, color=color, step='mid', linewidth=0
        )
    
    # Set y-axis scale
    if log_y:
        ax.set_yscale('log')
        # Set reasonable y limits to avoid log(0) issues
        y_min = np.min(y[y > 0]) if np.any(y > 0) else 0.1
        ax.set_ylim(bottom=y_min * 0.5)
    
    # Set x-axis range if specified
    if energy_range is not None:
        ax.set_xlim(energy_range)
    
    # Labels
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    return fig, ax


def plot_count_rate_time_series(
    time_series: SpectralTimeSeries,
    mode: Literal['gross', 'net'] = 'gross',
    background: Optional[Spectrum] = None,
    show_uncertainty: bool = True,
    time_unit: str = 's',
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot total count rate as a function of time.
    
    Parameters
    ----------
    time_series : SpectralTimeSeries
        Time series to plot.
    mode : {'gross', 'net'}, optional
        'gross': plot total counts without background subtraction
        'net': subtract background before plotting (requires background parameter)
        Default is 'gross'.
    background : Spectrum, optional
        Background spectrum to subtract (required if mode='net').
    show_uncertainty : bool, optional
        Show uncertainty bands (Poisson). Default is True.
    time_unit : str, optional
        Unit for time axis. Default is 's'.
    fig : Figure, optional
        Existing figure to plot on.
    ax : Axes, optional
        Existing axes to plot on.
    **kwargs
        Additional keyword arguments passed to matplotlib plot.
        
    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.
        
    Examples
    --------
    >>> fig, ax = plot_count_rate_time_series(time_series)
    >>> plt.show()
    
    >>> # With background subtraction
    >>> background = time_series[:10].sum_spectrum()
    >>> fig, ax = plot_count_rate_time_series(
    ...     time_series, mode='net', background=background
    ... )
    """
    # Create figure if needed
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    
    # Get time axis
    t = _get_time_axis(time_series)
    
    # Calculate total counts for each spectrum
    gross_counts = np.sum(time_series.counts, axis=1)
    
    # Get time for normalization (prefer live_time)
    times = _get_normalization_times(time_series)
    
    # Calculate count rates
    gross_rate = gross_counts / times
    
    if mode == 'net':
        if background is None:
            raise ValueError("background parameter required for mode='net'")
        
        # Subtract background
        bg_counts = np.sum(background.counts)
        bg_time = background.live_time if background.live_time is not None else background.real_time
        bg_rate = bg_counts / bg_time if bg_time > 0 else bg_counts
        
        y = gross_rate - bg_rate
        ylabel = r'Net Count Rate (s$^{-1}$)'
    else:
        y = gross_rate
        ylabel = r'Gross Count Rate (s$^{-1}$)'
    
    # Plot with steps (better for discrete time bins)
    ax.step(t, y, where='post', linewidth=1.5, **kwargs)
    
    # Add uncertainty bands (Poisson statistics)
    if show_uncertainty:
        # Poisson uncertainty on rate: sqrt(counts) / time
        uncertainty = np.sqrt(gross_counts) / times
        ax.fill_between(t, y - uncertainty, y + uncertainty, alpha=0.3, step='post')
    
    # Labels
    ax.set_xlabel(f'Time ({time_unit})', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at zero for net mode
    if mode == 'net':
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    
    fig.tight_layout()
    
    return fig, ax


def plot_waterfall(
    time_series: SpectralTimeSeries,
    mode: Literal['counts', 'count_rate'] = 'count_rate',
    log_scale: bool = True,
    energy_range: Optional[Tuple[float, float]] = None,
    time_range: Optional[Tuple[float, float]] = None,
    cmap: str = 'viridis',
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    colorbar: bool = True,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Create a waterfall plot (2D heatmap of spectral evolution over time).
    
    Time on y-axis, energy on x-axis, color represents intensity.
    
    Parameters
    ----------
    time_series : SpectralTimeSeries
        Time series to plot.
    mode : {'counts', 'count_rate'}, optional
        What to show in color scale. Default is 'count_rate'.
    log_scale : bool, optional
        Use logarithmic color scale. Default is True.
    energy_range : tuple of float, optional
        (e_min, e_max) for x-axis limits.
    time_range : tuple of float, optional
        (t_min, t_max) for y-axis limits.
    cmap : str, optional
        Matplotlib colormap name. Default is 'viridis'.
    fig : Figure, optional
        Existing figure to plot on.
    ax : Axes, optional
        Existing axes to plot on.
    colorbar : bool, optional
        Show colorbar. Default is True.
    **kwargs
        Additional keyword arguments passed to matplotlib imshow/pcolormesh.
        
    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.
        
    Examples
    --------
    >>> fig, ax = plot_waterfall(time_series, log_scale=True)
    >>> plt.show()
    
    >>> # Focus on specific energy range
    >>> fig, ax = plot_waterfall(
    ...     time_series, energy_range=(400, 700), cmap='hot'
    ... )
    """
    # Create figure if needed
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get time axis
    t = _get_time_axis(time_series)
    
    # Get energy axis
    if time_series.is_calibrated:
        energy_edges = time_series.energy_edges
        energy_centers = time_series.energy_centers
        xlabel = f'Energy ({time_series.spectra[0].energy_unit})'
    else:
        energy_edges = np.arange(time_series.n_bins + 1)
        energy_centers = np.arange(time_series.n_bins) + 0.5
        xlabel = 'Channel'
    
    # Prepare data based on mode
    if mode == 'count_rate':
        # Get time for normalization
        times = _get_normalization_times(time_series)
        
        # Normalize each spectrum by its time
        data = time_series.counts / times[:, np.newaxis]
        clabel = r'Count Rate (s$^{-1}$)'
    else:
        data = time_series.counts
        clabel = 'Counts'
    
    # Apply energy range filter if specified
    if energy_range is not None:
        e_min, e_max = energy_range
        if time_series.is_calibrated:
            mask = (energy_centers >= e_min) & (energy_centers <= e_max)
            energy_edges_plot = energy_edges[:-1][mask]
            energy_edges_plot = np.append(energy_edges_plot, energy_edges[1:][mask][-1])
        else:
            idx_min = int(e_min)
            idx_max = int(e_max)
            mask = np.arange(time_series.n_bins)
            mask = (mask >= idx_min) & (mask <= idx_max)
            energy_edges_plot = energy_edges[mask[0]:mask[-1]+2]
        
        data = data[:, mask]
        energy_centers = energy_centers[mask]
    else:
        energy_edges_plot = energy_edges
    
    # Apply time range filter if specified
    if time_range is not None:
        t_min, t_max = time_range
        mask = (t >= t_min) & (t <= t_max)
        data = data[mask, :]
        t = t[mask]
    
    # Set color normalization
    if log_scale:
        # Avoid log(0) issues
        vmin = np.min(data[data > 0]) if np.any(data > 0) else 1e-3
        vmax = np.max(data)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = Normalize(vmin=np.min(data), vmax=np.max(data))
    
    # Create time edges for pcolormesh (needs n+1 points for 'flat' shading)
    # Compute bin edges from bin centers
    if len(t) > 1:
        dt = np.diff(t)
        t_edges = np.zeros(len(t) + 1)
        t_edges[0] = t[0] - dt[0] / 2
        t_edges[1:-1] = t[:-1] + dt / 2
        t_edges[-1] = t[-1] + dt[-1] / 2
    else:
        # Single time point - create edges around it
        t_edges = np.array([t[0] - 0.5, t[0] + 0.5])
    
    # Create mesh for pcolormesh
    T, E = np.meshgrid(t_edges, energy_edges_plot, indexing='ij')
    
    # Plot
    im = ax.pcolormesh(E, T, data, cmap=cmap, norm=norm, shading='flat', **kwargs)
    
    # Colorbar
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, label=clabel)
    
    # Labels
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_title('Spectral Evolution', fontsize=14, fontweight='bold')
    
    fig.tight_layout()
    
    return fig, ax


def plot_roi_time_series(
    time_series: SpectralTimeSeries,
    rois: list,
    mode: Literal['counts', 'count_rate'] = 'count_rate',
    show_uncertainty: bool = True,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Plot time evolution of specific ROIs (energy regions of interest).
    
    Parameters
    ----------
    time_series : SpectralTimeSeries
        Time series to analyze.
    rois : list of EnergyROI
        Regions of interest to plot.
    mode : {'counts', 'count_rate'}, optional
        What to plot on y-axis. Default is 'count_rate'.
    show_uncertainty : bool, optional
        Show uncertainty bands. Default is True.
    fig : Figure, optional
        Existing figure to plot on.
    ax : Axes, optional
        Existing axes to plot on.
    **kwargs
        Additional keyword arguments passed to matplotlib plot.
        
    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.
        
    Examples
    --------
    >>> from gammaflow.operations import EnergyROI
    >>> rois = [
    ...     EnergyROI(655, 668, label="Cs-137"),
    ...     EnergyROI(1450, 1470, label="K-40")
    ... ]
    >>> fig, ax = plot_roi_time_series(time_series, rois)
    >>> plt.show()
    """
    from gammaflow.operations import rebin_time_series_rois
    
    # Create figure if needed
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))
    
    # Get time axis
    t = _get_time_axis(time_series)
    
    # Rebin to ROIs
    roi_counts, labels = rebin_time_series_rois(time_series, rois, return_labels=True)
    
    # Convert to rates if needed
    if mode == 'count_rate':
        # Get time for normalization
        times = _get_normalization_times(time_series)
        
        roi_data = roi_counts / times[:, np.newaxis]
        ylabel = r'Count Rate (s$^{-1}$)'
        uncertainties = np.sqrt(roi_counts) / times[:, np.newaxis]
    else:
        roi_data = roi_counts
        ylabel = 'Counts'
        uncertainties = np.sqrt(roi_counts)
    
    # Plot each ROI with steps (better for discrete time bins)
    for i, label in enumerate(labels):
        line = ax.step(t, roi_data[:, i], where='post', label=label, linewidth=1.5, **kwargs)
        
        if show_uncertainty:
            color = line[0].get_color()
            ax.fill_between(
                t,
                roi_data[:, i] - uncertainties[:, i],
                roi_data[:, i] + uncertainties[:, i],
                alpha=0.3,
                color=color,
                step='post'
            )
    
    # Labels and legend
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    return fig, ax


def plot_spectrum_comparison(
    spectra: list,
    labels: Optional[list] = None,
    mode: Literal['counts', 'count_rate', 'count_density'] = 'counts',
    log_y: bool = True,
    normalize: bool = False,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    **kwargs
) -> Tuple[Figure, Axes]:
    """
    Compare multiple spectra on the same plot.
    
    Parameters
    ----------
    spectra : list of Spectrum
        Spectra to compare.
    labels : list of str, optional
        Labels for each spectrum.
    mode : {'counts', 'count_rate', 'count_density'}, optional
        What to plot on y-axis. Default is 'counts'.
    log_y : bool, optional
        Use logarithmic y-axis. Default is True.
    normalize : bool, optional
        Normalize all spectra to unit area. Default is False.
    fig : Figure, optional
        Existing figure to plot on.
    ax : Axes, optional
        Existing axes to plot on.
    **kwargs
        Additional keyword arguments passed to plot_spectrum.
        
    Returns
    -------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.
        
    Examples
    --------
    >>> fig, ax = plot_spectrum_comparison(
    ...     [background, source],
    ...     labels=['Background', 'Source'],
    ...     log_y=True
    ... )
    >>> plt.show()
    """
    # Create figure if needed
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Default labels
    if labels is None:
        labels = [f'Spectrum {i+1}' for i in range(len(spectra))]
    
    # Color palette
    colors = sns.color_palette("husl", len(spectra))
    
    # Plot each spectrum
    for i, (spec, label) in enumerate(zip(spectra, labels)):
        # Normalize if requested
        if normalize:
            spec_to_plot = spec.normalize(method='area')
        else:
            spec_to_plot = spec
        
        plot_spectrum(
            spec_to_plot,
            mode=mode,
            log_y=log_y,
            show_uncertainty=False,
            fig=fig,
            ax=ax,
            label=label,
            color=colors[i],
            **kwargs
        )
    
    # Add legend
    ax.legend(loc='best')
    
    return fig, ax

