"""Tests of the Kelvin-Voigt bar (material key tau): the wavenumber and its small-omega-tau expansions, the
force operator, exact composition of the transfer functions, the analytic continuation used by the analysis,
the band limit "auto", and the separation of attenuated records, checked against the truth limited to the
same band."""
from pathlib import Path

import numpy as np
import pytest

from bussac_analysis import (AUTO_MAX_AMPLIFICATION, compare_settings, default_f_max, next_pow2, run_bussac,
                             wavenumber_laplace)
from hopkinson_simulator import (BarMaterial, BarModel, LoveRod, PochhammerChree, SHPBSimulation, Spectral,
                                 WaveEq1D, deep_merge, force_eff, generate, load_config, propagate,
                                 time_derivative, transfer)
from test_bussac import full_acfg

CASES = Path(__file__).resolve().parents[1] / "simulation_cases"
SHPB = load_config(CASES / "SHPB.toml")
PC = BarMaterial(E=2.3e9, rho=1200.0, nu=0.37, radius=8.15e-3, tau=1e-7)
TRAPEZOID = {"preset": "trapezoid", "rise": 30e-6, "fall": 30e-6}


@pytest.mark.parametrize("model", [WaveEq1D, LoveRod])
def test_kelvin_voigt_wavenumber(model):
    m = model(PC)
    w = 2 * np.pi * np.linspace(1.0, 2e6, 20001)        # includes the Love cut-off of the elastic bar (~100 kHz)
    k = m.wavenumber(w)
    assert np.all(k.imag < 0) and np.all(k.real > 0)
    elastic = model(BarMaterial(PC.E, PC.rho, PC.nu, PC.radius))
    assert np.array_equal(model(BarMaterial(PC.E, PC.rho, PC.nu, PC.radius, tau=0.0)).wavenumber(w),
                          elastic.wavenumber(w))


def test_kelvin_voigt_small_omega_tau_expansions():
    """Model A, omega tau = 0.01: attenuation omega^2 tau/(2 c0), phase velocity c0 (1 + 3/8 (omega tau)^2)."""
    x = 0.01
    w = np.array([x / PC.tau])
    k = WaveEq1D(PC).wavenumber(w)[0]
    assert abs(-k.imag / (w[0] ** 2 * PC.tau / (2 * PC.c0)) - 1) < 1e-3
    assert abs(((w[0] / k.real) / PC.c0 - 1) / (3 * x ** 2 / 8) - 1) < 1e-3


def test_kelvin_voigt_force_operator():
    """Model A: A rho (omega/k)^2 eps is the Kelvin-Voigt force A E (eps + tau eps_t)."""
    sp = Spectral(4000, 2e-7, 8)
    t = np.arange(4000) * 2e-7
    eps = -1e-3 * np.exp(-0.5 * ((t - 100e-6) / 5e-6) ** 2)
    m = WaveEq1D(PC)
    rhs = PC.area * PC.E * (eps + PC.tau * time_derivative(sp, eps))
    assert np.max(np.abs(force_eff(sp, m, eps) - rhs)) < 1e-10 * np.max(np.abs(rhs))


def test_kelvin_voigt_pulse_decays():
    sp = Spectral(6000, 2e-7, 8)
    t = np.arange(6000) * 2e-7
    g = -1e-3 * np.exp(-0.5 * ((t - 50e-6) / 3e-6) ** 2)
    for model in (WaveEq1D, LoveRod):
        peaks = [np.abs(propagate(sp, model(PC), g, d)).max() for d in (0.0, 0.3, 0.6, 0.9)]
        assert np.all(np.diff(peaks) < 0), peaks


def test_pochhammer_chree_refuses_tau():
    with pytest.raises(ValueError, match="tau"):
        PochhammerChree(PC)


@pytest.mark.parametrize("model", ["A", "B"])
def test_kelvin_voigt_transfer_functions_compose_exactly(model):
    """H(a) H(b) = H(a + b), including the excluded band, so that records and truth stay consistent."""
    sim = SHPBSimulation(deep_merge(SHPB, {"bar_model": model, "bars": {"material_input": {"tau": 3e-7}}}))
    sp = sim.sp
    for a, b in ((0.3, 1.7), (0.85, 1.15), (2.0, 1.7)):
        Ha = transfer(sim.model1, sp.omega, a, 0.0, sp.max_group_delay, sp.mask_distance)
        Hb = transfer(sim.model1, sp.omega, b, 0.0, sp.max_group_delay, sp.mask_distance)
        Hab = transfer(sim.model1, sp.omega, a + b, 0.0, sp.max_group_delay, sp.mask_distance)
        assert np.max(np.abs(Ha * Hb - Hab)) < 1e-11, (a, b)


@pytest.mark.parametrize("model", [WaveEq1D, LoveRod])
def test_kelvin_voigt_laplace_wavenumber(model):
    """The closed-form continuation k(omega - i eta) agrees with the Taylor expansion of the complex k(omega)."""
    m = model(PC)
    w = 2 * np.pi * np.linspace(0, 60e3, 6001)
    exact = wavenumber_laplace(m, w, 1000.0)
    generic = BarModel.__new__(BarModel); generic.mat = PC; generic.wavenumber = m.wavenumber
    taylor = wavenumber_laplace(generic, w, 1000.0)
    # the difference is the third-order term the Taylor expansion leaves out (1.1e-6 for model B at 60 kHz,
    # 0.6 x the cut-off of the elastic Love bar)
    assert np.max(np.abs(taylor - exact)[5:-5] / np.abs(exact[5:-5])) < 1e-5
    assert np.all(exact.imag[1:] < 0)


def test_auto_band_limit_of_kelvin_voigt_bar():
    for model in (WaveEq1D, LoveRod):
        m = model(PC)
        f = default_f_max(m, x_max=0.5, f_nyquist=2.5e6)
        kappa = -m.wavenumber(np.array([2 * np.pi * f])).imag[0]
        assert abs(np.exp(kappa * 0.5) / AUTO_MAX_AMPLIFICATION - 1) < 1e-2, (model.name, f)
        if model is LoveRod:
            assert f < 0.8 * m.omega_cutoff / (2 * np.pi)
    assert default_f_max(WaveEq1D(BarMaterial(PC.E, PC.rho, PC.nu, PC.radius))) == np.inf


def _band_limited(x, t, eta, f_max):
    """The truth as the separation sees it: Laplace window, band limit f_max, inverse transform."""
    n, tt = t.size, t - t[0]
    nfft = next_pow2(4 * n)
    X = np.fft.rfft(x * np.exp(-eta * tt), nfft)
    X[2 * np.pi * np.fft.rfftfreq(nfft, tt[1]) > 2 * np.pi * f_max] = 0.0
    return np.fft.irfft(X, nfft)[:n] * np.exp(eta * tt)


@pytest.mark.parametrize("model", ["A", "B"])
def test_separation_of_kelvin_voigt_records(model):
    """With the right tau the separated waves equal the truth limited to the same band (to 1e-5); assuming
    an elastic bar is clearly worse against the full truth. Smooth pulse, both bars viscoelastic."""
    cfg = deep_merge(SHPB, {"bar_model": model, "pulse": TRAPEZOID, "bars": {"material_input": {"tau": 3e-7}}})
    case, _ = generate(cfg)
    data = {**case, "channels": {k: case["channels"][k] for k in ("names", "bar", "records", "records_point")}}
    acfg = full_acfg(cfg)
    assert [r["key"] for r in compare_settings(acfg, cfg) if r["status"] == "differs"] == []
    r, tv, _, _ = run_bussac(data, acfg)
    t, itf = case["time"], case["interface"]
    valid = t < tv
    sc = np.abs(itf["bar1"]["wave_in"]).max()
    waves = [(b, w) for b in ("bar1", "bar2") for w in ("wave_in", "wave_out")]
    f_max = r["bar1"]["f_max"]
    assert 30e3 < f_max < 150e3, f_max
    e_band = max(np.abs(r[b][w] - _band_limited(itf[b][w], t, 1000.0, f_max))[valid].max() for b, w in waves) / sc
    assert e_band < 1e-5, e_band
    e_right = max(np.abs(r[b][w] - itf[b][w])[valid].max() for b, w in waves) / sc
    elastic = deep_merge(acfg, {"bars": {"material_input": {"tau": 0.0}, "material_output": {"tau": 0.0}}})
    assert [x["key"] for x in compare_settings(elastic, cfg) if x["status"] == "differs"] == \
        ["bars.material_input.tau", "bars.material_output.tau"]
    r0, *_ = run_bussac(data, elastic)
    e_elastic = max(np.abs(r0[b][w] - itf[b][w])[valid].max() for b, w in waves) / sc
    assert e_elastic > 3 * e_right, (e_elastic, e_right)
