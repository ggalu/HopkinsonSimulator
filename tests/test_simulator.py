"""Tests of hopkinson_simulator.py: closed-form solutions with model A (d'Alembert, direct impact, the
loading end of the tension bar, the specimen of the tension bar as a spring), the Love and Pochhammer-Chree
dispersion relations, and the consistency of the stored waves with the gauge records."""
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.signal import hilbert

from hopkinson_simulator import (BarMaterial, LoveRod, PochhammerChree, SHPBSimulation, Spectral, deep_merge,
                                 force_eff, generate, load_case, load_config, pc_first_mode_table, propagate,
                                 rayleigh_speed_ratio, save_case, time_derivative, transfer)

CASES = Path(__file__).resolve().parents[1] / "simulation_cases"
SHPB = load_config(CASES / "SHPB.toml")
IMPACT = load_config(CASES / "direct_impact.toml")
SHTB = load_config(CASES / "SHTB.toml")
MAT = BarMaterial(**SHPB["bars"]["material"])
ELASTIC = {"sigma0": 1e12, "rate_dependent": False}
TRAPEZOID = {"preset": "trapezoid", "rise": 30e-6, "fall": 30e-6}


def variant(cfg, **over):
    return deep_merge(cfg, over)


# ------------------------------------------------------------------ model A, closed forms
def test_dalembert_model_a_elastic_specimen():
    cfg = variant(SHPB, bar_model="A", pulse=TRAPEZOID, sampling={"duration": 700e-6, "n_sub": 8},
                  specimen={"material": {**ELASTIC, "tension_cutoff": False}})
    case, sim = generate(cfg)
    m1, m2, spc, f = sim.mat1, sim.mat2, sim.specimen, sim.pulse
    c = m1.c0
    Q = spc.As * (m1.c0 / (m1.E * m1.area) + m2.c0 / (m2.E * m2.area))
    ks = spc.mat.E / spc.Ls
    a1 = lambda t: f(t - sim.d_ref / c)
    sol = solve_ivp(lambda t, s: ks * (2 * c * a1(t) - Q * s), (0, sim.t[-1]), [0.0], method="DOP853",
                    rtol=1e-13, atol=1e-6, max_step=0.2e-6, dense_output=True)
    sig = lambda t: np.where(t > 0, sol.sol(np.maximum(t, 0))[0], 0.0)
    t = case["time"]
    for i, (bar, d) in enumerate([(1, 0.30), (1, 0.85), (2, 0.30), (2, 0.85)]):
        if bar == 1:
            ex = f(t - (sim.d_ref - d) / c) + spc.As * sig(t - d / c) / (m1.E * m1.area) - a1(t - d / c)
        else:
            ex = spc.As * sig(t - d / c) / (m2.E * m2.area)
        err = np.max(np.abs(case["channels"]["records_point"][i] - ex)) / np.max(np.abs(ex))
        assert err < 2e-4, (i, err)


def test_direct_impact_model_a_elastic_specimen():
    """Moving input bar on an elastic lumped specimen, before any far-end echo: with P = -v0,
    sigma(t) = -(v0/Q) (1 - exp(-(Es/Ls) Q (t - t_i))); each bar carries N/(E A) delayed by d/c."""
    cfg = variant(IMPACT, bar_model="A", sampling={"duration": 500e-6},
                  specimen={"material": {"sigma0": 1e12, "tension_cutoff": False}})
    case, sim = generate(cfg)
    m1, m2, spc, v0, ti = sim.mat1, sim.mat2, sim.specimen, sim.v0, sim.t_impact
    Q = spc.As * (m1.c0 / (m1.E * m1.area) + m2.c0 / (m2.E * m2.area))
    N = lambda t: -spc.As * v0 / Q * (1 - np.exp(-spc.mat.E / spc.Ls * Q * np.maximum(t - ti, 0.0)))
    t = case["time"]
    for i, (bar, d) in enumerate([(1, 0.13), (1, 0.53), (2, 0.12), (2, 0.50)]):
        m = m1 if bar == 1 else m2
        ex = N(t - d / m.c0) / (m.E * m.area)
        err = np.max(np.abs(case["channels"]["records_point"][i] - ex)) / np.max(np.abs(ex))
        assert err < 2e-3, (i, err)
    v_ex = v0 + m1.c0 * N(t) / (m1.E * m1.area)          # rigid motion + reflected wave
    assert np.max(np.abs(case["interface"]["bar1"]["velocity"] - v_ex)) / v0 < 2e-3


def test_shtb_loading_end_model_a():
    """Striker on a rigid flange, before any echo (model A, a specimen of bar size and very high stiffness,
    so that it transmits the wave unchanged): f(t) = -(v_s/c1) Z_s/(Z1 + Z_s) (1 - exp(-(t - t_i)/tau)),
    tau = m/(Z1 + Z_s); the input-bar gauges see f delayed by (L1 - d)/c1, the output-bar gauges by (L1 + d)/c1."""
    cfg = variant(SHTB, bar_model="A", sampling={"duration": 1000e-6},
                  specimen={"diameter": 0.016, "material": {**ELASTIC, "E": 1e15}})
    case, sim = generate(cfg)
    m1, ms = sim.mat1, sim.mat_s
    Z1, Zs, c = m1.impedance, ms.impedance, m1.c0
    tau = sim.flange_mass / (Z1 + Zs)
    f = lambda t: -sim.v0_striker / c * Zs / (Z1 + Zs) * (1 - np.exp(-np.maximum(t - sim.t_impact, 0.0) / tau))
    t = case["time"]
    assert t[-1] < sim.t_impact + 2 * sim.L_striker / ms.c0         # before the striker echo
    for i, (bar, d) in enumerate([(1, 0.3), (1, 0.85), (2, 0.3), (2, 0.85)]):
        delay = (sim.L1 - d) / c if bar == 1 else (sim.L1 + d) / c
        ex = f(t - delay)
        err = np.max(np.abs(case["channels"]["records_point"][i] - ex)) / np.max(np.abs(ex))
        assert err < 2e-3, (i, err)


def test_shtb_elastic_specimen_is_a_spring():
    """An elastic lumped specimen bonded between two equal bars (model A) is a massless spring of compliance
    C = Ls/(Es As). Before any echo it transmits the incident wave with T = 1/(1 + i omega Z C/2)
    (time factor exp(i omega t)), i.e. (Z C/2) db2/dt + b2 = a1."""
    cfg = variant(SHTB, bar_model="A", sampling={"duration": 1400e-6}, specimen={"material": ELASTIC})
    case, sim = generate(cfg)
    spc, m = sim.specimen, sim.mat1
    tau = m.impedance * spc.Ls / (spc.mat.E * spc.As) / 2
    assert 10e-6 < tau < 30e-6                       # a time constant that the sampling resolves
    t, a1 = case["time"], case["interface"]["bar1"]["wave_in"]
    sol = solve_ivp(lambda s, y: (np.interp(s, t, a1) - y) / tau, (0, t[-1]), [0.0], method="DOP853",
                    rtol=1e-11, atol=1e-14, max_step=0.1e-6, t_eval=t)
    b2 = case["interface"]["bar2"]["wave_out"]
    a2 = case["interface"]["bar2"]["wave_in"]
    assert np.max(np.abs(a2)) < 1e-5 * np.max(np.abs(a1))     # no echo yet (only spectral round-off)
    err = np.max(np.abs(b2 - sol.y[0])) / np.max(np.abs(b2))
    assert err < 2e-3, err
    assert np.max(np.abs(b2 - a1)) / np.max(np.abs(a1)) > 0.1    # the spring changes the wave noticeably


def test_shtb_needs_bonded_specimen():
    with pytest.raises(ValueError, match="tension_cutoff"):
        generate(variant(SHTB, specimen={"material": {"tension_cutoff": True}}))


# ------------------------------------------------------------------ model B
def test_love_wavenumber_satisfies_dispersion_relation():
    love = LoveRod(MAT)
    w = 2 * np.pi * np.linspace(1e3, 0.99 * love.omega_cutoff / (2 * np.pi), 200)
    k = np.real(love.wavenumber(w))
    c = w / k
    assert np.allclose(c ** 2 / MAT.c0 ** 2, 1 / (1 + MAT.nu ** 2 * k ** 2 * love.rg ** 2), rtol=1e-12)
    # evanescent above the cut-off
    assert np.all(np.imag(love.wavenumber(np.array([1.01, 2.0]) * love.omega_cutoff)) < 0)


def test_love_propagated_field_satisfies_love_pde_second_order():
    love = LoveRod(MAT)
    rho, E, nu, rg = MAT.rho, MAT.E, MAT.nu, love.rg
    res = []
    for h, dt in ((2e-3, 4e-7), (1e-3, 2e-7)):
        n = int(round(300e-6 / dt))
        sp = Spectral(n, dt, 8)
        t = np.arange(n) * dt
        g = -1e-3 * np.exp(-0.5 * ((t - 40e-6) / 4e-6) ** 2)
        X = np.array([propagate(sp, love, g, 0.5 + j * h) for j in (-1, 0, 1)])
        d2t = lambda y: (np.roll(y, -1) - 2 * y + np.roll(y, 1)) / dt ** 2
        e_xx = (X[0] - 2 * X[1] + X[2]) / h ** 2
        R = rho * d2t(X[1]) - E * e_xx - rho * nu ** 2 * rg ** 2 * d2t(e_xx)
        R0 = rho * d2t(X[1]) - E * e_xx
        sl = slice(2, -2)
        res.append(np.max(np.abs(R[sl])) / np.max(np.abs(rho * d2t(X[1])[sl])))
        wave_res = np.max(np.abs(R0[sl])) / np.max(np.abs(rho * d2t(X[1])[sl]))
    assert res[1] < 2e-5
    assert 3.5 < res[0] / res[1] < 4.5          # second-order FD convergence
    assert wave_res > 100 * res[1]              # the 1D wave equation is clearly violated


def test_love_tone_burst_phase_and_group_velocity():
    love = LoveRod(MAT)
    dt, n = 1 / 50e6, 60000
    sp, t = Spectral(n, dt, 4), np.arange(n) * 1 / 50e6
    fc = 100e3
    tw = 20 / fc
    x = np.exp(-0.5 * ((t - 1.2 * tw) / (tw / 6)) ** 2) * np.sin(2 * np.pi * fc * (t - 1.2 * tw))
    ha, hb = hilbert(propagate(sp, love, x, 0.85)), hilbert(propagate(sp, love, x, 0.852))
    ip = int(np.argmax(np.abs(ha)))
    c_meas = 2 * np.pi * fc * 2e-3 / np.angle(ha[ip] * np.conj(hb[ip]))
    c_th = np.real(love.phase_velocity(np.array([2 * np.pi * fc])))[0]
    assert abs(c_meas / c_th - 1) < 1e-3
    h1 = np.abs(hilbert(propagate(sp, love, x, 0.3)))
    cg = 0.55 / ((np.argmax(np.abs(ha)) - np.argmax(h1)) * dt)
    k = 2 * np.pi * fc / c_th
    assert abs(cg / (MAT.c0 / (1 + (MAT.nu * love.rg * k) ** 2) ** 1.5) - 1) < 2e-3


def test_force_eff_equals_love_effective_stress():
    love = LoveRod(MAT)
    sp = Spectral(4000, 2e-7, 8)
    t = np.arange(4000) * 2e-7
    eps = propagate(sp, love, -1e-3 * np.exp(-0.5 * ((t - 50e-6) / 3e-6) ** 2), 1.0)
    rhs = MAT.area * (MAT.E * eps + MAT.rho * MAT.nu ** 2 * love.rg ** 2 * time_derivative(sp, eps, 2))
    assert np.max(np.abs(force_eff(sp, love, eps) - rhs)) < 1e-10 * np.max(np.abs(rhs))


# ------------------------------------------------------------------ model C
@pytest.mark.parametrize("a_over_lambda,c_ref", [(0.1, 0.9905), (0.5, 0.68814)])
def test_pochhammer_chree_phase_velocity_published_nu030(a_over_lambda, c_ref):
    """Published first-mode roots for nu = 0.30 (Rigby et al. 2018, solved as in Bancroft 1941)."""
    pc = PochhammerChree(BarMaterial(71.7e9, 2810, 0.30, 0.008))
    c = float(pc.phase_velocity_ratio_vs_a_over_lambda(a_over_lambda))
    assert abs(c - c_ref) <= 0.6 * 10 ** -(len(str(c_ref)) - 2)


def test_pochhammer_chree_limits():
    nu = MAT.nu
    xi, cb, c0b, cRb = pc_first_mode_table(nu)
    small = xi < 0.05
    rl = 1 - nu ** 2 * xi[small] ** 2 / 4       # Rayleigh-Love long-wave expansion
    assert np.max(np.abs(cb[small] / c0b - rl) / (1 - rl)) < 2e-3   # agrees to leading order
    assert abs(cb[-1] / cRb - 1) < 2e-3            # short-wave limit: Rayleigh speed
    assert abs(rayleigh_speed_ratio(0.25) - 0.919402) < 1e-6   # classical value for nu = 1/4


# ------------------------------------------------------------------ consistency and file format
def test_transfer_functions_compose_exactly():
    """Gauge signals and interface waves are forward propagations of the same waves:
    exactness requires H(a) H(b) = H(a + b) for every model, including the excluded band."""
    for model in ("A", "B", "C"):
        sim = SHPBSimulation(variant(SHPB, bar_model=model))
        sp = sim.sp
        for m in (sim.model1, sim.model2):
            for a, b in ((0.3, 1.7), (0.85, 1.15), (2.0, 1.7)):
                Ha = transfer(m, sp.omega, a, 0.0, sp.max_group_delay, sp.mask_distance)
                Hb = transfer(m, sp.omega, b, 0.0, sp.max_group_delay, sp.mask_distance)
                Hab = transfer(m, sp.omega, a + b, 0.0, sp.max_group_delay, sp.mask_distance)
                assert np.max(np.abs(Ha * Hb - Hab)) < 1e-11, (model, a, b)


@pytest.mark.parametrize("cfg,tol", [(SHPB, 1e-4), (SHTB, 3e-4)], ids=["SHPB", "SHTB"])
def test_gauges_are_forward_propagations_of_interface_waves(cfg, tol, tmp_path):
    """Check on the stored (time-truncated) files. Re-propagating a record truncated at the end
    of the window creates small band-limited ringing, hence the tolerance. It is larger for the SHTB,
    whose records still end at 20-30 % of their peak."""
    case, sim = generate(cfg)
    save_case(case, tmp_path / "c.h5")
    c = load_case(tmp_path / "c.h5")
    ch, itf = c["channels"], c["interface"]
    for i in range(4):
        bar, d = int(ch["bar"][i]), float(ch["distance_from_interface"][i])
        m = sim.model1 if bar == 1 else sim.model2
        win, wout = itf[f"bar{bar}"]["wave_in"], itf[f"bar{bar}"]["wave_out"]
        incoming = ch["records_point"][i] - propagate(sim.sp_out, m, wout, d)
        scale = max(np.max(np.abs(win)), np.max(np.abs(wout)))
        err = np.max(np.abs(propagate(sim.sp_out, m, incoming, d) - win)) / scale
        assert err < tol, (i, err)
    assert np.array_equal(c["channels"]["records"], case["channels"]["records"])
    assert c["meta"]["config"]["setup"] == cfg["setup"]
