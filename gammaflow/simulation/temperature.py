"""
Synthetic temperature time series for gain-drift simulation.

Generators produce a temperature (degrees C) for each sample time, suitable
for driving a :class:`~gammaflow.simulation.gain.TemperatureResponseModel`.
All generators take a ``times`` array (seconds) and return a float ndarray of
the same length, so they compose directly with a
:class:`~gammaflow.core.time_series.SpectralTimeSeries`'s ``timestamps``.

Examples
--------
>>> import numpy as np
>>> from gammaflow.simulation import temperature as temp
>>> t = np.arange(0, 3600.0, 0.5)            # 1 hour at 0.5 s cadence
>>> ambient = temp.diurnal(t, mean=22.0, amplitude=8.0)
>>> crystal = temp.thermal_lag(ambient, t, tau=900.0)   # 15 min lag
"""

from typing import Optional, Union

import numpy as np
import numpy.typing as npt

__all__ = [
    "constant",
    "linear_ramp",
    "diurnal",
    "ornstein_uhlenbeck",
    "thermal_lag",
    "from_array",
]


def _as_times(times: Union[int, npt.ArrayLike]) -> np.ndarray:
    """Coerce an int (sample count) or array-like into a float times array."""
    if np.isscalar(times):
        return np.arange(int(times), dtype=float)
    arr = np.asarray(times, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"times must be 1D, got shape {arr.shape}")
    return arr


def constant(times: Union[int, npt.ArrayLike], value: float) -> np.ndarray:
    """Constant temperature.

    Parameters
    ----------
    times : int or array-like
        Sample times (s), or an integer sample count.
    value : float
        Temperature (deg C).
    """
    t = _as_times(times)
    return np.full(t.shape, float(value))


def linear_ramp(times: Union[int, npt.ArrayLike], start: float, end: float) -> np.ndarray:
    """Linear temperature ramp from ``start`` (at the first time) to ``end``
    (at the last time).

    Parameters
    ----------
    times : int or array-like
        Sample times (s), or an integer sample count.
    start, end : float
        Temperatures (deg C) at the first and last samples.
    """
    t = _as_times(times)
    if len(t) == 1:
        return np.array([float(start)])
    span = t[-1] - t[0]
    if span <= 0:
        return np.full(t.shape, float(start))
    frac = (t - t[0]) / span
    return start + (end - start) * frac


def diurnal(
    times: Union[int, npt.ArrayLike],
    mean: float = 20.0,
    amplitude: float = 8.0,
    period: float = 86400.0,
    phase: float = 0.0,
) -> np.ndarray:
    """Sinusoidal (e.g. day/night) temperature cycle.

    ``T(t) = mean + amplitude * sin(2*pi*t/period + phase)``

    Parameters
    ----------
    times : int or array-like
        Sample times (s), or an integer sample count.
    mean : float
        Mean temperature (deg C).
    amplitude : float
        Peak deviation from the mean (deg C).
    period : float
        Cycle period (s). Default 86400 (24 h).
    phase : float
        Phase offset (radians).
    """
    t = _as_times(times)
    return mean + amplitude * np.sin(2.0 * np.pi * t / period + phase)


def ornstein_uhlenbeck(
    times: Union[int, npt.ArrayLike],
    mean: float = 20.0,
    sigma: float = 3.0,
    tau: float = 1800.0,
    seed: Optional[int] = None,
    initial: Optional[float] = None,
) -> np.ndarray:
    """Mean-reverting (Ornstein-Uhlenbeck) temperature random walk.

    Models stochastic ambient drift with thermal inertia: excursions decay
    back toward ``mean`` with correlation time ``tau`` while being perturbed
    by noise. ``sigma`` is the stationary standard deviation. Uses the exact
    discrete update, valid for non-uniform sample spacing.

    Parameters
    ----------
    times : int or array-like
        Sample times (s), or an integer sample count.
    mean : float
        Long-run mean temperature (deg C).
    sigma : float
        Stationary standard deviation (deg C).
    tau : float
        Correlation (mean-reversion) time (s). Must be > 0.
    seed : int or None
        Seed for reproducibility.
    initial : float or None
        Initial temperature. Defaults to ``mean``.
    """
    if tau <= 0:
        raise ValueError("tau must be > 0")
    t = _as_times(times)
    n = len(t)
    rng = np.random.RandomState(seed)
    out = np.empty(n)
    out[0] = mean if initial is None else float(initial)
    for k in range(1, n):
        dt = t[k] - t[k - 1]
        if dt < 0:
            raise ValueError("times must be non-decreasing")
        decay = np.exp(-dt / tau)
        noise_std = sigma * np.sqrt(max(1.0 - decay * decay, 0.0))
        out[k] = mean + (out[k - 1] - mean) * decay + noise_std * rng.randn()
    return out


def thermal_lag(
    ambient: npt.ArrayLike,
    times: Union[int, npt.ArrayLike],
    tau: float = 900.0,
    initial: Optional[float] = None,
) -> np.ndarray:
    """First-order thermal lag of a detector relative to ambient.

    The crystal/PMT temperature does not track air temperature instantly; it
    follows ``dT_det/dt = (T_ambient - T_det) / tau``. This applies that
    first-order (RC) low-pass exactly under a zero-order hold on ``ambient``
    over each step, valid for non-uniform spacing.

    Parameters
    ----------
    ambient : array-like
        Ambient temperature per sample (deg C).
    times : int or array-like
        Sample times (s), or an integer sample count. Must match ``ambient``.
    tau : float
        Thermal time constant (s). Larger = more sluggish. Must be > 0.
    initial : float or None
        Initial detector temperature. Defaults to ``ambient[0]``.
    """
    if tau <= 0:
        raise ValueError("tau must be > 0")
    amb = np.asarray(ambient, dtype=float)
    t = _as_times(times)
    if amb.shape != t.shape:
        raise ValueError(f"ambient shape {amb.shape} != times shape {t.shape}")
    n = len(t)
    out = np.empty(n)
    out[0] = amb[0] if initial is None else float(initial)
    for k in range(1, n):
        dt = t[k] - t[k - 1]
        if dt < 0:
            raise ValueError("times must be non-decreasing")
        decay = np.exp(-dt / tau)
        # zero-order hold: ambient held at amb[k-1] across the step
        out[k] = amb[k - 1] + (out[k - 1] - amb[k - 1]) * decay
    return out


def from_array(values: npt.ArrayLike) -> np.ndarray:
    """Validate and pass through a user-supplied temperature array (deg C).

    Use this to drive the gain model from a measured temperature log.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"temperature array must be 1D, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("temperature array contains non-finite values")
    return arr
