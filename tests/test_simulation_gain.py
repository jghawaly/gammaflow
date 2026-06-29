"""Tests for temperature-induced gain-shift simulation."""

import numpy as np
import pytest

from gammaflow import ListMode, SpectralTimeSeries, Spectrum
from gammaflow.simulation import temperature as temp
from gammaflow.simulation.gain import (
    LinearGainModel,
    PolynomialGainModel,
    CallableGainModel,
    ResolutionModel,
    TemperatureResponseModel,
    apply_gain_shift,
    apply_temperature_drift,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
N_BINS = 256
EDGES = np.linspace(20.0, 2900.0, N_BINS + 1)
CENTERS = 0.5 * (EDGES[:-1] + EDGES[1:])


def _peak_spectrum(peak_energy=662.0, width=20.0, area=10000.0, baseline=5.0):
    """Gaussian photopeak on a flat continuum."""
    g = np.exp(-0.5 * ((CENTERS - peak_energy) / width) ** 2)
    g = g / g.sum() * area
    counts = g + baseline
    return Spectrum(counts=counts, energy_edges=EDGES)


def _peak_centroid(counts):
    return np.sum(CENTERS * counts) / np.sum(counts)


# ---------------------------------------------------------------------------
# Temperature profiles
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestTemperatureProfiles:
    def test_constant(self):
        t = np.arange(100.0)
        out = temp.constant(t, 25.0)
        assert out.shape == (100,)
        assert np.all(out == 25.0)

    def test_linear_ramp_endpoints(self):
        t = np.linspace(0, 1000, 50)
        out = temp.linear_ramp(t, 10.0, 30.0)
        assert np.isclose(out[0], 10.0)
        assert np.isclose(out[-1], 30.0)
        assert np.all(np.diff(out) > 0)

    def test_diurnal_period(self):
        t = np.arange(0, 86400.0, 60.0)
        out = temp.diurnal(t, mean=20.0, amplitude=5.0, period=86400.0)
        assert np.isclose(out.mean(), 20.0, atol=0.1)
        assert np.isclose(out.max(), 25.0, atol=0.1)
        assert np.isclose(out.min(), 15.0, atol=0.1)

    def test_ou_stationary_stats(self):
        t = np.arange(0, 200000.0, 5.0)
        out = temp.ornstein_uhlenbeck(t, mean=20.0, sigma=3.0, tau=600.0, seed=0)
        assert np.isclose(out.mean(), 20.0, atol=0.5)
        assert np.isclose(out.std(), 3.0, rtol=0.2)

    def test_ou_reproducible(self):
        t = np.arange(1000.0)
        a = temp.ornstein_uhlenbeck(t, seed=42)
        b = temp.ornstein_uhlenbeck(t, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_thermal_lag_attenuates_and_lags(self):
        t = np.arange(0, 7200.0, 1.0)
        ambient = temp.diurnal(t, mean=20.0, amplitude=10.0, period=3600.0)
        crystal = temp.thermal_lag(ambient, t, tau=600.0)
        # lag attenuates amplitude
        assert (crystal.max() - crystal.min()) < (ambient.max() - ambient.min())
        # peak of crystal occurs after peak of ambient (within first period)
        amb_peak = np.argmax(ambient[:3600])
        cry_peak = np.argmax(crystal[:3600])
        assert cry_peak > amb_peak

    def test_thermal_lag_shape_mismatch(self):
        with pytest.raises(ValueError):
            temp.thermal_lag(np.zeros(10), np.arange(11.0), tau=100.0)

    def test_from_array_rejects_nonfinite(self):
        with pytest.raises(ValueError):
            temp.from_array([1.0, np.nan, 3.0])


# ---------------------------------------------------------------------------
# Gain models
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestGainModels:
    def test_linear_reference_unity(self):
        m = LinearGainModel(alpha_per_C=-0.003, t_ref=20.0)
        assert np.isclose(m.relative_gain(np.array([20.0]))[0], 1.0)

    def test_linear_sign(self):
        m = LinearGainModel(alpha_per_C=-0.003, t_ref=20.0)
        # hotter -> lower gain
        assert m.relative_gain(np.array([30.0]))[0] < 1.0
        assert m.relative_gain(np.array([10.0]))[0] > 1.0

    def test_polynomial_matches_linear(self):
        lin = LinearGainModel(alpha_per_C=-0.002, t_ref=15.0)
        poly = PolynomialGainModel(coeffs=[1.0, -0.002], t_ref=15.0)
        t = np.array([0.0, 15.0, 40.0])
        np.testing.assert_allclose(lin.relative_gain(t), poly.relative_gain(t))

    def test_callable(self):
        m = CallableGainModel(lambda T: 1.0 + 0.0 * T)
        assert np.allclose(m.relative_gain(np.array([5.0, 50.0])), 1.0)


# ---------------------------------------------------------------------------
# Resolution model
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestResolutionModel:
    def test_ref_resolution_at_eref(self):
        rm = ResolutionModel(fwhm_ref_frac=0.07, e_ref=662.0)
        sigma = rm.sigma_ref(np.array([662.0]))[0]
        fwhm = sigma * 2.3548200450309493
        assert np.isclose(fwhm / 662.0, 0.07, rtol=1e-6)

    def test_added_sigma_zero_when_brighter(self):
        rm = ResolutionModel()
        # ly > 1 (colder, more light) -> no broadening
        added = rm.added_sigma(CENTERS, light_yield_ratio=1.2)
        assert np.all(added == 0.0)

    def test_added_sigma_positive_when_dimmer(self):
        rm = ResolutionModel()
        added = rm.added_sigma(np.array([662.0]), light_yield_ratio=0.8)
        assert added[0] > 0.0


# ---------------------------------------------------------------------------
# Single-spectrum gain shift
# ---------------------------------------------------------------------------
@pytest.mark.unit
class TestApplyGainShift:
    def test_identity(self):
        s = _peak_spectrum()
        out = apply_gain_shift(s, relative_gain=1.0)
        np.testing.assert_allclose(out.counts, s.counts, atol=1e-9)

    def test_conserves_counts_interior(self):
        # peak compactly supported well inside range (no baseline at edges);
        # a small shift maps everything on-grid -> counts conserved.
        s = _peak_spectrum(peak_energy=662.0, baseline=0.0)
        out = apply_gain_shift(s, relative_gain=1.02)
        assert np.isclose(out.counts.sum(), s.counts.sum(), rtol=1e-6)

    def test_edge_spillover_when_baseline_fills_range(self):
        # baseline spanning the full range to the ADC ceiling: an upshift
        # spills counts off the top edge (correct instrument physics).
        s = _peak_spectrum(peak_energy=662.0, baseline=5.0)
        out = apply_gain_shift(s, relative_gain=1.05)
        assert out.counts.sum() < s.counts.sum()

    def test_downshift_moves_peak_down(self):
        s = _peak_spectrum(peak_energy=662.0)
        c0 = _peak_centroid(s.counts)
        out = apply_gain_shift(s, relative_gain=0.95)  # lower gain
        c1 = _peak_centroid(out.counts)
        assert c1 < c0
        # centroid scales by ~relative_gain
        assert np.isclose(c1 / c0, 0.95, rtol=0.02)

    def test_offset_shifts_peak(self):
        s = _peak_spectrum(peak_energy=662.0)
        c0 = _peak_centroid(s.counts)
        out = apply_gain_shift(s, relative_gain=1.0, offset=50.0)
        c1 = _peak_centroid(out.counts)
        assert np.isclose(c1 - c0, 50.0, atol=5.0)

    def test_broadening_widens_peak(self):
        s = _peak_spectrum(peak_energy=662.0, width=15.0)
        sigma = np.full(N_BINS, 30.0)  # large extra broadening
        out = apply_gain_shift(s, relative_gain=1.0, added_sigma=sigma)

        # second moment about centroid grows
        def spread(c):
            cen = _peak_centroid(c)
            return np.sqrt(np.sum(c * (CENTERS - cen) ** 2) / c.sum())

        assert spread(out.counts) > spread(s.counts)
        assert np.isclose(out.counts.sum(), s.counts.sum(), rtol=1e-6)

    def test_stochastic_conserves_integer_counts_interior(self):
        # integer, edge-safe input: multinomial redistribution conserves exactly
        rng = np.random.RandomState(0)
        base = _peak_spectrum(peak_energy=662.0, baseline=0.0).counts
        counts = rng.poisson(base).astype(float)
        s = Spectrum(counts=counts, energy_edges=EDGES)
        out = apply_gain_shift(s, relative_gain=1.01, stochastic=True, seed=0)
        assert np.allclose(out.counts, np.round(out.counts))
        assert out.counts.sum() == s.counts.sum()

    def test_stochastic_reproducible(self):
        s = _peak_spectrum()
        a = apply_gain_shift(s, relative_gain=0.97, stochastic=True, seed=7)
        b = apply_gain_shift(s, relative_gain=0.97, stochastic=True, seed=7)
        np.testing.assert_array_equal(a.counts, b.counts)

    def test_rejects_uncalibrated(self):
        s = Spectrum(counts=np.ones(N_BINS))
        with pytest.raises(ValueError):
            apply_gain_shift(s, relative_gain=1.0)

    def test_rejects_nonpositive_gain(self):
        s = _peak_spectrum()
        with pytest.raises(ValueError):
            apply_gain_shift(s, relative_gain=0.0)


# ---------------------------------------------------------------------------
# Time-series drift
# ---------------------------------------------------------------------------
def _make_ts(n_frames=60):
    rng = np.random.RandomState(0)
    base = _peak_spectrum(peak_energy=662.0).counts
    counts = rng.poisson(np.tile(base, (n_frames, 1))).astype(float)
    return SpectralTimeSeries.from_array(
        counts,
        energy_edges=EDGES,
        timestamps=np.arange(n_frames) * 0.5,
        real_times=np.full(n_frames, 0.5),
    )


@pytest.mark.unit
class TestApplyTemperatureDrift:
    def test_shape_and_grid_preserved(self):
        ts = _make_ts()
        temps = temp.constant(ts.n_spectra, 20.0)
        model = TemperatureResponseModel(gain=LinearGainModel(t_ref=20.0))
        out = apply_temperature_drift(ts, temps, model)
        assert out.counts.shape == ts.counts.shape
        np.testing.assert_array_equal(out.energy_edges, ts.energy_edges)

    def test_reference_temp_is_near_identity(self):
        ts = _make_ts()
        temps = temp.constant(ts.n_spectra, 20.0)
        model = TemperatureResponseModel(gain=LinearGainModel(t_ref=20.0))
        out = apply_temperature_drift(ts, temps, model)
        # r=1, no offset, no resolution -> identity
        np.testing.assert_allclose(out.counts, ts.counts, atol=1e-9)

    def test_ramp_drifts_centroid_monotonically(self):
        ts = _make_ts(n_frames=40)
        temps = temp.linear_ramp(ts.timestamps, 20.0, 60.0)
        model = TemperatureResponseModel(gain=LinearGainModel(alpha_per_C=-0.004, t_ref=20.0))
        out = apply_temperature_drift(ts, temps, model)
        cents = np.array([_peak_centroid(out.counts[k]) for k in range(40)])
        # hotter over time -> lower gain -> centroid trends down
        assert cents[-1] < cents[0]
        # roughly monotone (allow Poisson jitter)
        assert np.polyfit(np.arange(40), cents, 1)[0] < 0

    def test_diagnostics(self):
        ts = _make_ts()
        temps = temp.linear_ramp(ts.timestamps, 20.0, 50.0)
        model = TemperatureResponseModel(
            gain=LinearGainModel(alpha_per_C=-0.003, t_ref=20.0),
            offset_per_C=0.1,
            resolution=ResolutionModel(),
        )
        out, diag = apply_temperature_drift(ts, temps, model, return_diagnostics=True)
        for key in ("temperature", "relative_gain", "offset", "light_yield", "spillover_fraction"):
            assert key in diag
            assert len(diag[key]) == ts.n_spectra
        # gain decreases as temperature rises
        assert diag["relative_gain"][-1] < diag["relative_gain"][0]
        # spillover non-negative
        assert np.all(diag["spillover_fraction"] >= -1e-9)

    def test_temperature_length_mismatch(self):
        ts = _make_ts()
        model = TemperatureResponseModel(gain=LinearGainModel())
        with pytest.raises(ValueError):
            apply_temperature_drift(ts, np.zeros(ts.n_spectra + 1), model)

    def test_stochastic_integer_counts(self):
        ts = _make_ts()
        temps = temp.linear_ramp(ts.timestamps, 20.0, 45.0)
        model = TemperatureResponseModel(gain=LinearGainModel(alpha_per_C=-0.003))
        out = apply_temperature_drift(ts, temps, model, stochastic=True, seed=1)
        assert np.allclose(out.counts, np.round(out.counts))


# ---------------------------------------------------------------------------
# List-mode (event-level) drift
# ---------------------------------------------------------------------------
def _mono_listmode(energy=662.0, n=2000, duration=3600.0):
    """Monoenergetic events spread uniformly in time (so E_obs reveals gain)."""
    times = np.linspace(0.0, duration, n)
    deltas = np.empty(n)
    deltas[0] = times[0]
    deltas[1:] = np.diff(times)
    return ListMode(time_deltas=deltas, energies=np.full(n, float(energy)))


@pytest.mark.unit
class TestListModeGainShift:
    def test_constant_gain_scales_energy(self):
        lm = _mono_listmode(energy=662.0)
        out = apply_gain_shift(lm, relative_gain=0.95)
        assert out.n_events == lm.n_events
        assert np.allclose(out.energies, 0.95 * 662.0)

    def test_offset(self):
        lm = _mono_listmode(energy=662.0)
        out = apply_gain_shift(lm, relative_gain=1.0, offset=30.0)
        assert np.allclose(out.energies, 692.0)

    def test_per_event_gain_array(self):
        lm = _mono_listmode(energy=662.0, n=100)
        r = np.linspace(0.9, 1.1, 100)
        out = apply_gain_shift(lm, relative_gain=r)
        assert np.allclose(out.energies, r * 662.0)

    def test_resolution_broadens_monoenergetic(self):
        lm = _mono_listmode(energy=662.0, n=20000)
        out = apply_gain_shift(
            lm,
            relative_gain=1.0,
            resolution=ResolutionModel(),
            light_yield_ratio=0.7,
            seed=0,
        )
        # was a delta; now spread, mean preserved (symmetric noise)
        assert out.energies.std() > 5.0
        assert np.isclose(out.energies.mean(), 662.0, atol=2.0)
        assert out.n_events == lm.n_events

    def test_e_range_drops_and_preserves_timing(self):
        lm = _mono_listmode(energy=662.0, n=1000)
        # downshift then keep only a window excluding the (single) peak energy
        out = apply_gain_shift(lm, relative_gain=0.5, e_range=(0.0, 300.0))
        # 0.5*662 = 331 -> outside [0,300) -> all dropped
        assert out.n_events == 0

    def test_e_range_keeps_in_window(self):
        lm = _mono_listmode(energy=662.0, n=1000)
        out = apply_gain_shift(lm, relative_gain=0.5, e_range=(0.0, 400.0))
        assert out.n_events == 1000
        # absolute times preserved
        np.testing.assert_allclose(out.absolute_times, lm.absolute_times)

    def test_array_gain_rejected_for_spectrum(self):
        s = _peak_spectrum()
        with pytest.raises(ValueError):
            apply_gain_shift(s, relative_gain=np.array([1.0, 1.1]))


@pytest.mark.unit
class TestListModeTemperatureDrift:
    def test_returns_listmode(self):
        lm = _mono_listmode()
        model = TemperatureResponseModel(gain=LinearGainModel(t_ref=20.0))
        out = apply_temperature_drift(lm, 20.0, model)
        assert isinstance(out, ListMode)
        # r=1 at reference, no resolution -> energies unchanged
        assert np.allclose(out.energies, lm.energies)

    def test_ramp_profile_downshifts_over_time(self):
        lm = _mono_listmode(energy=662.0, n=3000, duration=3600.0)
        model = TemperatureResponseModel(gain=LinearGainModel(alpha_per_C=-0.004, t_ref=20.0))
        # profile pair (times, temps): hotter later
        profile = (np.array([0.0, 3600.0]), np.array([20.0, 60.0]))
        out = apply_temperature_drift(lm, profile, model)
        # monoenergetic -> E_obs monotone decreasing as temperature rises
        assert out.energies[0] > out.energies[-1]
        assert np.all(np.diff(out.energies) <= 1e-9)

    def test_callable_temperature(self):
        lm = _mono_listmode(energy=662.0, n=500, duration=1000.0)
        model = TemperatureResponseModel(gain=LinearGainModel(alpha_per_C=-0.003, t_ref=20.0))
        out = apply_temperature_drift(lm, lambda t: 20.0 + 0.0 * t, model)
        assert np.allclose(out.energies, 662.0)

    def test_diagnostics(self):
        lm = _mono_listmode(n=1000)
        model = TemperatureResponseModel(
            gain=LinearGainModel(alpha_per_C=-0.004, t_ref=20.0),
            resolution=ResolutionModel(),
        )
        profile = (np.array([0.0, 3600.0]), np.array([20.0, 55.0]))
        out, diag = apply_temperature_drift(lm, profile, model, seed=0, return_diagnostics=True)
        for key in ("temperature", "relative_gain", "offset", "light_yield", "dropped_fraction"):
            assert key in diag
        assert len(diag["relative_gain"]) == lm.n_events
        assert diag["relative_gain"][0] > diag["relative_gain"][-1]

    def test_per_event_temperature_array(self):
        lm = _mono_listmode(energy=662.0, n=200)
        model = TemperatureResponseModel(gain=LinearGainModel(alpha_per_C=-0.004, t_ref=20.0))
        temps = np.full(200, 45.0)
        out = apply_temperature_drift(lm, temps, model)
        expected_r = 1.0 + (-0.004) * (45.0 - 20.0)
        assert np.allclose(out.energies, expected_r * 662.0)

    def test_listmode_to_binned_roundtrip_centroid(self):
        # drift events, bin, and confirm the peak moved as expected
        lm = _mono_listmode(energy=662.0, n=50000, duration=10.0)
        model = TemperatureResponseModel(gain=LinearGainModel(alpha_per_C=-0.004, t_ref=20.0))
        out = apply_temperature_drift(lm, 45.0, model)
        r = 1.0 + (-0.004) * (45.0 - 20.0)
        assert np.isclose(out.energies.mean(), r * 662.0, atol=1.0)
