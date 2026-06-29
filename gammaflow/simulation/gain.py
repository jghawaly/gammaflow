"""
Temperature-induced gain shift, offset drift, and resolution degradation.

Scintillator (e.g. NaI(Tl)) light yield and PMT gain vary with temperature,
shifting photopeak positions and broadening peaks. This module models that
forward effect on already-binned spectra:

    E_observed = relative_gain(T) * E_true + offset(T)

with optional temperature-dependent resolution broadening driven by the
relative light yield. Because a :class:`SpectralTimeSeries` stores all frames
on one shared energy grid, the effect is applied by **redistributing counts**
onto that fixed grid (as a fixed-ADC instrument actually records it), not by
re-calibrating each frame.

Sign convention: ``relative_gain`` is the gain relative to the reference
state. Higher temperature typically lowers light yield, so a negative
``alpha_per_C`` gives ``relative_gain < 1`` at higher T, shifting peaks to
lower apparent energy.

Resolution caveat: the reference spectrum already carries the detector's
measured resolution, so this module can only *add* broadening (when a hotter
state degrades resolution); it never sharpens, since that would require
ill-posed deconvolution. When a state would improve resolution the added
width is clamped to zero.

Examples
--------
>>> import numpy as np
>>> from gammaflow.simulation import temperature as temp
>>> from gammaflow.simulation.gain import (
...     LinearGainModel, ResolutionModel, TemperatureResponseModel,
...     apply_temperature_drift,
... )
>>> t = np.asarray(ts.timestamps)
>>> temps = temp.diurnal(t, mean=25.0, amplitude=10.0)
>>> model = TemperatureResponseModel(
...     gain=LinearGainModel(alpha_per_C=-0.003, t_ref=20.0),
...     offset_per_C=0.0,
...     resolution=ResolutionModel(fwhm_ref_frac=0.07, e_ref=662.0),
... )
>>> drifted = apply_temperature_drift(ts, temps, model)
"""

from typing import Callable, Optional, Union

import numpy as np
import numpy.typing as npt

from gammaflow.core.spectrum import Spectrum
from gammaflow.core.time_series import SpectralTimeSeries
from gammaflow.core.listmode import ListMode

__all__ = [
    "GainModel",
    "LinearGainModel",
    "PolynomialGainModel",
    "CallableGainModel",
    "ResolutionModel",
    "TemperatureResponseModel",
    "apply_gain_shift",
    "apply_temperature_drift",
]

_FWHM_PER_SIGMA = 2.3548200450309493  # 2*sqrt(2*ln2)


# ============================================================================
# Gain models: temperature -> relative gain
# ============================================================================
class GainModel:
    """Abstract temperature -> relative-gain model."""

    def relative_gain(self, temperature: npt.ArrayLike) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, temperature: npt.ArrayLike) -> np.ndarray:
        return self.relative_gain(temperature)


class LinearGainModel(GainModel):
    """Linear gain model: ``r(T) = 1 + alpha_per_C * (T - t_ref)``.

    Parameters
    ----------
    alpha_per_C : float
        Fractional gain change per degree C. Default -0.003 (-0.3 %/C), a
        typical NaI(Tl)+PMT system coefficient near room temperature.
    t_ref : float
        Reference temperature (deg C) at which the input spectra were taken,
        where ``r = 1``.
    """

    def __init__(self, alpha_per_C: float = -0.003, t_ref: float = 20.0):
        self.alpha_per_C = float(alpha_per_C)
        self.t_ref = float(t_ref)

    def relative_gain(self, temperature: npt.ArrayLike) -> np.ndarray:
        t = np.asarray(temperature, dtype=float)
        return 1.0 + self.alpha_per_C * (t - self.t_ref)

    def __repr__(self) -> str:
        return f"LinearGainModel(alpha_per_C={self.alpha_per_C}, " f"t_ref={self.t_ref})"


class PolynomialGainModel(GainModel):
    """Polynomial gain model in ``(T - t_ref)``.

    ``r(T) = sum_k coeffs[k] * (T - t_ref)**k`` (coefficients in increasing
    power order). For a well-formed relative gain, ``coeffs[0]`` is typically
    1.0. Use for wide temperature ranges where the light-yield curve bends.

    Parameters
    ----------
    coeffs : array-like
        Polynomial coefficients, increasing power order.
    t_ref : float
        Reference temperature (deg C).
    """

    def __init__(self, coeffs: npt.ArrayLike, t_ref: float = 20.0):
        self.coeffs = np.asarray(coeffs, dtype=float)
        if self.coeffs.ndim != 1 or len(self.coeffs) == 0:
            raise ValueError("coeffs must be a non-empty 1D array")
        self.t_ref = float(t_ref)

    def relative_gain(self, temperature: npt.ArrayLike) -> np.ndarray:
        t = np.asarray(temperature, dtype=float)
        dt = t - self.t_ref
        # Horner evaluation in increasing-power order
        result = np.zeros_like(dt, dtype=float)
        for c in self.coeffs[::-1]:
            result = result * dt + c
        return result

    def __repr__(self) -> str:
        return f"PolynomialGainModel(coeffs={self.coeffs.tolist()}, " f"t_ref={self.t_ref})"


class CallableGainModel(GainModel):
    """Wrap any ``func(temperature) -> relative_gain`` as a GainModel.

    The function must accept a numpy array and return an array of the same
    shape.
    """

    def __init__(self, func: Callable[[np.ndarray], np.ndarray]):
        if not callable(func):
            raise TypeError("func must be callable")
        self.func = func

    def relative_gain(self, temperature: npt.ArrayLike) -> np.ndarray:
        t = np.asarray(temperature, dtype=float)
        return np.asarray(self.func(t), dtype=float)

    def __repr__(self) -> str:
        return "CallableGainModel(func=<callable>)"


# ============================================================================
# Resolution model
# ============================================================================
class ResolutionModel:
    """Energy resolution and its temperature (light-yield) dependence.

    Reference resolution follows the statistical scaling
    ``FWHM(E) = fwhm_ref_frac * sqrt(e_ref * E)`` (so FWHM/E ~ 1/sqrt(E),
    equal to ``fwhm_ref_frac`` at ``e_ref``). The statistical width scales
    with the photoelectron count, i.e. with light yield, so at a relative
    light yield ``ly = LY(T)/LY_ref`` the resolution degrades as
    ``sigma(E)^2 = sigma_ref(E)^2 / ly``. The *added* width injected on top
    of the reference spectrum is

        added_sigma(E) = sigma_ref(E) * sqrt(max(1/ly - 1, 0))

    (zero when ``ly >= 1``; the module never sharpens).

    Parameters
    ----------
    fwhm_ref_frac : float
        Fractional FWHM resolution at ``e_ref`` (e.g. 0.07 = 7 % at 662 keV).
    e_ref : float
        Reference energy (same unit as the spectrum's energy axis).
    """

    def __init__(self, fwhm_ref_frac: float = 0.07, e_ref: float = 662.0):
        if fwhm_ref_frac < 0:
            raise ValueError("fwhm_ref_frac must be >= 0")
        if e_ref <= 0:
            raise ValueError("e_ref must be > 0")
        self.fwhm_ref_frac = float(fwhm_ref_frac)
        self.e_ref = float(e_ref)

    def sigma_ref(self, energy: npt.ArrayLike) -> np.ndarray:
        """Reference Gaussian sigma (energy units) at each energy."""
        e = np.maximum(np.asarray(energy, dtype=float), 0.0)
        fwhm = self.fwhm_ref_frac * np.sqrt(self.e_ref * e)
        return fwhm / _FWHM_PER_SIGMA

    def added_sigma(self, energy: npt.ArrayLike, light_yield_ratio: float) -> np.ndarray:
        """Additional sigma to inject given relative light yield ``ly``."""
        ly = float(light_yield_ratio)
        if ly <= 0:
            raise ValueError("light_yield_ratio must be > 0")
        extra = max(1.0 / ly - 1.0, 0.0)
        if extra == 0.0:
            return np.zeros_like(np.asarray(energy, dtype=float))
        return self.sigma_ref(energy) * np.sqrt(extra)

    def __repr__(self) -> str:
        return f"ResolutionModel(fwhm_ref_frac={self.fwhm_ref_frac}, " f"e_ref={self.e_ref})"


# ============================================================================
# Combined response model
# ============================================================================
class TemperatureResponseModel:
    """Bundle gain, zero-offset drift, and resolution into one response.

    Parameters
    ----------
    gain : GainModel
        Temperature -> relative gain (centroid scaling).
    offset_per_C : float
        Additive zero/offset drift slope (energy units per deg C). The offset
        at temperature T is ``offset_per_C * (T - t_ref_offset)``. Default 0
        (no offset drift) so the simplest model is a pure gain shift.
    t_ref_offset : float
        Reference temperature for the offset term (deg C).
    resolution : ResolutionModel or None
        If given, inject temperature-dependent broadening; if None, no
        broadening is applied.
    light_yield : callable or None
        ``func(T) -> relative light yield`` driving resolution. If None, the
        gain model's relative gain is used as a light-yield proxy (valid when
        the gain shift is light-yield dominated, as for NaI(Tl); it ignores
        any PMT-only gain contribution).
    """

    def __init__(
        self,
        gain: GainModel,
        offset_per_C: float = 0.0,
        t_ref_offset: float = 20.0,
        resolution: Optional[ResolutionModel] = None,
        light_yield: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        if not isinstance(gain, GainModel):
            raise TypeError("gain must be a GainModel")
        self.gain = gain
        self.offset_per_C = float(offset_per_C)
        self.t_ref_offset = float(t_ref_offset)
        self.resolution = resolution
        self.light_yield = light_yield

    def gain_at(self, temperature: float) -> float:
        return float(self.gain.relative_gain(np.asarray([temperature]))[0])

    def offset_at(self, temperature: float) -> float:
        return self.offset_per_C * (float(temperature) - self.t_ref_offset)

    def light_yield_at(self, temperature: float) -> float:
        if self.light_yield is not None:
            return float(np.asarray(self.light_yield(np.asarray([temperature])))[0])
        return self.gain_at(temperature)

    def added_sigma(self, energy: npt.ArrayLike, temperature: float) -> Optional[np.ndarray]:
        if self.resolution is None:
            return None
        return self.resolution.added_sigma(energy, self.light_yield_at(temperature))

    def __repr__(self) -> str:
        return (
            f"TemperatureResponseModel(gain={self.gain!r}, "
            f"offset_per_C={self.offset_per_C}, resolution={self.resolution!r})"
        )


# ============================================================================
# Core kernels (operate on raw count arrays + fixed edges)
# ============================================================================
def _gaussian_rows(centers: np.ndarray, sigma: np.ndarray, n_sigma: float = 4.0):
    """Per-source-bin Gaussian spread weights, normalized over in-grid bins.

    Returns a list of (dest_index_array, weight_array) with weights summing to
    1 over the available destinations (so broadening conserves counts).
    """
    n = len(centers)
    rows = []
    for j in range(n):
        sj = sigma[j]
        if sj <= 0:
            rows.append((np.array([j]), np.array([1.0])))
            continue
        reach = n_sigma * sj
        lo = np.searchsorted(centers, centers[j] - reach, side="left")
        hi = np.searchsorted(centers, centers[j] + reach, side="right")
        idx = np.arange(lo, hi)
        d = (centers[idx] - centers[j]) / sj
        w = np.exp(-0.5 * d * d)
        s = w.sum()
        if s <= 0:
            rows.append((np.array([j]), np.array([1.0])))
        else:
            rows.append((idx, w / s))
    return rows


def _affine_rows(edges: np.ndarray, r: float, b: float):
    """Per-source-bin overlap weights for the affine map E_obs = r*E + b.

    Returns a list of (dest_index_array, weight_array). Weights are overlap
    fractions of the source bin's mapped interval with each output bin; they
    sum to <= 1 (the remainder mapped off-grid is spillover).
    """
    n = len(edges) - 1
    rows = []
    for j in range(n):
        lo_e = r * edges[j] + b
        hi_e = r * edges[j + 1] + b
        if hi_e < lo_e:
            lo_e, hi_e = hi_e, lo_e
        width = hi_e - lo_e
        if width <= 0:
            rows.append((np.array([], dtype=int), np.array([])))
            continue
        # output bins overlapping [lo_e, hi_e]
        i_lo = np.searchsorted(edges, lo_e, side="right") - 1
        i_hi = np.searchsorted(edges, hi_e, side="left")
        i_lo = max(i_lo, 0)
        i_hi = min(i_hi, n)
        if i_hi <= i_lo:
            rows.append((np.array([], dtype=int), np.array([])))
            continue
        idx = np.arange(i_lo, i_hi)
        ov_lo = np.maximum(lo_e, edges[idx])
        ov_hi = np.minimum(hi_e, edges[idx + 1])
        w = np.clip(ov_hi - ov_lo, 0.0, None) / width
        keep = w > 0
        rows.append((idx[keep], w[keep]))
    return rows


def _apply_rows(
    counts: np.ndarray,
    rows,
    n_out: int,
    stochastic: bool,
    rng: Optional[np.random.RandomState],
) -> np.ndarray:
    """Redistribute counts through transfer rows.

    Deterministic: weighted sum. Stochastic: per-source multinomial draw
    (counts rounded to integers; off-grid spillover handled as a discard
    category when row weights sum < 1).
    """
    out = np.zeros(n_out)
    for j, (idx, w) in enumerate(rows):
        cj = counts[j]
        if cj == 0 or len(idx) == 0:
            continue
        if stochastic:
            wsum = w.sum()
            n_draw = int(round(cj))
            if n_draw <= 0:
                continue
            if wsum >= 1.0:
                probs = w / wsum
                draw = rng.multinomial(n_draw, probs)
            else:
                probs = np.append(w, 1.0 - wsum)  # last = off-grid discard
                draw = rng.multinomial(n_draw, probs)[:-1]
            np.add.at(out, idx, draw)
        else:
            np.add.at(out, idx, cj * w)
    return out


def _transform_counts(
    counts: np.ndarray,
    edges: np.ndarray,
    centers: np.ndarray,
    r: float,
    b: float,
    added_sigma: Optional[np.ndarray],
    stochastic: bool,
    rng: Optional[np.random.RandomState],
) -> np.ndarray:
    """Apply (optional) broadening then affine remap to one count vector."""
    work = counts
    if added_sigma is not None and np.any(added_sigma > 0):
        rows = _gaussian_rows(centers, added_sigma)
        work = _apply_rows(work, rows, len(counts), stochastic, rng)
    rows = _affine_rows(edges, r, b)
    return _apply_rows(work, rows, len(counts), stochastic, rng)


# ============================================================================
# List-mode (event-level) helpers
# ============================================================================
def _resolve_event_temperatures(temperatures, event_times, n_events):
    """Map a temperature spec to one temperature per event.

    Accepts a scalar (constant), a per-event array (length n_events), a
    callable ``f(times) -> temps``, or a ``(profile_times, profile_temps)``
    pair which is linearly interpolated (and held flat beyond its ends) to
    the event times.
    """
    if callable(temperatures):
        return np.asarray(temperatures(event_times), dtype=float)
    if np.isscalar(temperatures):
        return np.full(n_events, float(temperatures))
    arr = temperatures
    # (profile_times, profile_temps) pair
    if (
        isinstance(arr, (tuple, list))
        and len(arr) == 2
        and np.ndim(arr[0]) == 1
        and np.ndim(arr[1]) == 1
        and len(arr[0]) == len(arr[1])
        and len(arr[0]) != n_events
    ):
        pt = np.asarray(arr[0], dtype=float)
        pv = np.asarray(arr[1], dtype=float)
        return np.interp(event_times, pt, pv)
    arr = np.asarray(temperatures, dtype=float)
    if arr.shape != (n_events,):
        raise ValueError(
            f"per-event temperatures length {arr.shape} != n_events {n_events}; "
            f"pass a scalar, a length-{n_events} array, a callable, or a "
            f"(profile_times, profile_temps) pair"
        )
    return arr


def _drift_event_energies(energies, r, offset, sigma, rng):
    """Transform event energies: broaden (per-event Gaussian) then affine.

    ``E_obs = r * (E + N(0, sigma)) + offset`` with per-event ``r``, ``offset``,
    and ``sigma`` (broadcastable). Broadening precedes the gain map, matching
    detection physics (resolution smears the deposited energy; gain then maps
    pulse height to recorded energy).
    """
    e = np.asarray(energies, dtype=float)
    if sigma is not None and np.any(sigma > 0):
        e = e + rng.normal(0.0, 1.0, size=e.shape) * sigma
    return r * e + offset


def _rebuild_listmode(lm, new_energies, keep_mask, extra_meta=None):
    """Build a ListMode from transformed energies, recomputing time_deltas for
    surviving events so their absolute times are preserved."""
    abs_times = lm.absolute_times[keep_mask]
    energies = new_energies[keep_mask]
    if len(abs_times) == 0:
        deltas = np.array([], dtype=float)
    else:
        deltas = np.empty(len(abs_times))
        deltas[0] = abs_times[0]
        deltas[1:] = np.diff(abs_times)
    meta = dict(lm.metadata)
    if extra_meta:
        meta.update(extra_meta)
    return ListMode(time_deltas=deltas, energies=energies, metadata=meta)


def _event_sigma(resolution, energies, light_yield_ratio):
    """Per-event added Gaussian sigma (energy units), or None."""
    if resolution is None:
        return None
    ly = np.asarray(light_yield_ratio, dtype=float)
    extra = np.maximum(1.0 / ly - 1.0, 0.0)
    if np.all(extra == 0):
        return None
    return resolution.sigma_ref(energies) * np.sqrt(extra)


def _apply_gain_shift_listmode(
    lm, relative_gain, offset, resolution, light_yield_ratio, seed, e_range
):
    n = lm.n_events
    r = np.broadcast_to(np.asarray(relative_gain, dtype=float), (n,))
    if np.any(r <= 0):
        raise ValueError("relative_gain must be > 0")
    b = np.broadcast_to(np.asarray(offset, dtype=float), (n,))
    rng = np.random.RandomState(seed)
    sigma = _event_sigma(resolution, lm.energies, light_yield_ratio)
    e_obs = _drift_event_energies(lm.energies, r, b, sigma, rng)
    keep = np.isfinite(e_obs)
    if e_range is not None:
        keep &= (e_obs >= e_range[0]) & (e_obs < e_range[1])
    else:
        keep &= e_obs > 0.0
    return _rebuild_listmode(lm, e_obs, keep)


def _apply_temperature_drift_listmode(lm, temperatures, model, seed, return_diagnostics, e_range):
    n = lm.n_events
    times = lm.absolute_times
    temps = _resolve_event_temperatures(temperatures, times, n)
    if temps.shape != (n,):
        raise ValueError(f"resolved temperatures length {temps.shape} != n_events {n}")
    r = model.gain.relative_gain(temps)
    if np.any(r <= 0):
        raise ValueError("relative_gain <= 0 for some events; check the gain model")
    b = model.offset_per_C * (temps - model.t_ref_offset)
    if model.light_yield is not None:
        ly = np.asarray(model.light_yield(temps), dtype=float)
    else:
        ly = r
    rng = np.random.RandomState(seed)
    sigma = _event_sigma(model.resolution, lm.energies, ly)
    e_obs = _drift_event_energies(lm.energies, r, b, sigma, rng)
    keep = np.isfinite(e_obs)
    if e_range is not None:
        keep &= (e_obs >= e_range[0]) & (e_obs < e_range[1])
    else:
        keep &= e_obs > 0.0
    drifted = _rebuild_listmode(lm, e_obs, keep)
    if return_diagnostics:
        diag = {
            "temperature": temps,
            "relative_gain": r,
            "offset": b,
            "light_yield": ly,
            "dropped_fraction": float((~keep).sum()) / n if n > 0 else 0.0,
        }
        return drifted, diag
    return drifted


# ============================================================================
# Public application functions
# ============================================================================
def apply_gain_shift(
    spectrum,
    relative_gain,
    offset: float = 0.0,
    added_sigma: Optional[npt.ArrayLike] = None,
    stochastic: bool = False,
    seed: Optional[int] = None,
    resolution: Optional[ResolutionModel] = None,
    light_yield_ratio: float = 1.0,
    e_range: Optional[tuple] = None,
) -> Spectrum:
    """Apply a single gain shift (and optional offset/broadening).

    Accepts a :class:`Spectrum` (counts redistributed on the fixed grid) or a
    :class:`~gammaflow.core.listmode.ListMode` (exact per-event transform,
    ``E_obs = relative_gain * (E + resolution noise) + offset``). For list-mode
    this is the physically exact model — no binning approximation — so prefer
    it when raw events are available.

    Parameters
    ----------
    spectrum : Spectrum or ListMode
        Calibrated input spectrum, or event list.
    relative_gain : float or array-like
        Gain relative to the reference state (> 0). For ``ListMode`` it may be
        a per-event array.
    offset : float or array-like
        Additive energy offset (energy units); per-event array allowed for
        ``ListMode``.
    added_sigma : array-like or None
        **Spectrum only.** Extra Gaussian sigma per energy bin (energy units),
        applied as broadening before the affine remap. None for no broadening.
    stochastic : bool
        **Spectrum only.** Multinomial (integer-preserving) redistribution if
        True, else deterministic fractional. Assumes integer input counts.
    seed : int or None
        RNG seed (stochastic spectrum path, or list-mode broadening draws).
    resolution : ResolutionModel or None
        **ListMode only.** If given, inject per-event Gaussian broadening using
        ``light_yield_ratio``.
    light_yield_ratio : float
        **ListMode only.** Relative light yield ``LY(T)/LY_ref`` driving the
        broadening (< 1 degrades resolution; >= 1 adds none).
    e_range : (float, float) or None
        **ListMode only.** Drop events whose observed energy falls outside
        ``[e_min, e_max)``; if None, drop only non-positive energies.

    Returns
    -------
    Spectrum or ListMode
        Same type as the input, with the shift applied.
    """
    if isinstance(spectrum, ListMode):
        return _apply_gain_shift_listmode(
            spectrum,
            relative_gain,
            offset,
            resolution,
            light_yield_ratio,
            seed,
            e_range,
        )
    if np.ndim(relative_gain) != 0:
        raise ValueError("array relative_gain is only supported for ListMode")
    relative_gain = float(relative_gain)
    offset = float(offset)
    if not spectrum.is_calibrated:
        raise ValueError("spectrum must be calibrated (have energy edges)")
    if relative_gain <= 0:
        raise ValueError("relative_gain must be > 0")

    edges = np.asarray(spectrum.energy_edges, dtype=float)
    centers = np.asarray(spectrum.energy_centers, dtype=float)
    counts = np.asarray(spectrum.counts, dtype=float)

    sig = None
    if added_sigma is not None:
        sig = np.asarray(added_sigma, dtype=float)
        if sig.shape != counts.shape:
            raise ValueError(f"added_sigma shape {sig.shape} != n_bins {counts.shape}")

    rng = np.random.RandomState(seed) if stochastic else None
    new_counts = _transform_counts(
        counts,
        edges,
        centers,
        float(relative_gain),
        float(offset),
        sig,
        stochastic,
        rng,
    )

    return Spectrum(
        counts=new_counts,
        energy_edges=edges,
        timestamp=spectrum.timestamp,
        live_time=spectrum.live_time,
        real_time=spectrum.real_time,
        energy_unit=spectrum.energy_unit,
        metadata=dict(spectrum.metadata),
    )


def apply_temperature_drift(
    time_series,
    temperatures,
    model: TemperatureResponseModel,
    stochastic: bool = False,
    seed: Optional[int] = None,
    return_diagnostics: bool = False,
    e_range: Optional[tuple] = None,
):
    """Apply temperature-induced drift to a spectral time series or event list.

    For a :class:`SpectralTimeSeries`, each frame's counts are redistributed on
    the shared grid by its temperature's gain/offset/(optional) broadening. For
    a :class:`~gammaflow.core.listmode.ListMode`, each event is transformed
    exactly at its own temperature (interpolated to the event time) — the
    physically exact model; prefer it when raw events are available.

    Parameters
    ----------
    time_series : SpectralTimeSeries or ListMode
        Calibrated input series, or event list.
    temperatures : array-like or callable
        Temperature (deg C). For a series: one value per frame. For
        ``ListMode``: a scalar, a per-event array, a callable ``f(times)``, or
        a ``(profile_times, profile_temps)`` pair interpolated to event times.
    model : TemperatureResponseModel
        Gain/offset/resolution response.
    stochastic : bool
        **Series only.** Multinomial (integer-preserving) redistribution if
        True, else deterministic fractional.
    seed : int or None
        RNG seed (stochastic series path, or list-mode broadening draws).
    return_diagnostics : bool
        If True, also return a per-frame/per-event diagnostics dict.
    e_range : (float, float) or None
        **ListMode only.** Drop events whose observed energy falls outside
        ``[e_min, e_max)``; if None, drop only non-positive energies.

    Returns
    -------
    SpectralTimeSeries or ListMode
        Same type as the input. If ``return_diagnostics``, returns
        ``(result, diag)``.
    """
    if isinstance(time_series, ListMode):
        return _apply_temperature_drift_listmode(
            time_series, temperatures, model, seed, return_diagnostics, e_range
        )
    counts = np.asarray(time_series.counts, dtype=float)
    n_frames, n_bins = counts.shape
    temps = np.asarray(temperatures, dtype=float)
    if temps.shape != (n_frames,):
        raise ValueError(f"temperatures length {temps.shape} != n_spectra {n_frames}")

    edges = np.asarray(time_series.energy_edges, dtype=float)
    if edges is None or len(edges) != n_bins + 1:
        raise ValueError("time_series must be calibrated with a shared grid")
    centers = 0.5 * (edges[:-1] + edges[1:])

    rng = np.random.RandomState(seed) if stochastic else None
    r_all = model.gain.relative_gain(temps)
    out = np.zeros_like(counts)
    diag_r = np.empty(n_frames)
    diag_b = np.empty(n_frames)
    diag_ly = np.empty(n_frames)
    diag_spill = np.empty(n_frames)

    for k in range(n_frames):
        r = float(r_all[k])
        if r <= 0:
            raise ValueError(f"relative_gain {r} <= 0 at frame {k}; check the gain model")
        b = model.offset_at(temps[k])
        sig = model.added_sigma(centers, temps[k])
        out[k] = _transform_counts(counts[k], edges, centers, r, b, sig, stochastic, rng)
        diag_r[k] = r
        diag_b[k] = b
        diag_ly[k] = model.light_yield_at(temps[k])
        in_sum = counts[k].sum()
        diag_spill[k] = (in_sum - out[k].sum()) / in_sum if in_sum > 0 else 0.0

    drifted = SpectralTimeSeries.from_array(
        out,
        energy_edges=edges,
        timestamps=np.asarray(time_series.timestamps, dtype=float),
        live_times=np.asarray(time_series.live_times, dtype=float),
        real_times=np.asarray(time_series.real_times, dtype=float),
    )

    if return_diagnostics:
        diag = {
            "temperature": temps,
            "relative_gain": diag_r,
            "offset": diag_b,
            "light_yield": diag_ly,
            "spillover_fraction": diag_spill,
        }
        return drifted, diag
    return drifted
