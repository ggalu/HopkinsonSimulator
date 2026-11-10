"""Bussac wave separation of simulated Hopkinson-bar gauge records, compared with the exact ground truth.

    python bussac_analysis.py analysis_cases/SHPB.toml [more.toml ...] [--out results] [--headless]

Each analysis case names a simulation file written by hopkinson_simulator.py and the settings of the
separation. By default the analysis assumes the bar model, bar material and gauge positions of the
simulation; the case file can override them, to study the effect of a wrong model or a wrong gauge position
without changing the simulation. Output in <out>/<name>/:
    bussac.json     settings, end of the valid window, errors relative to the ground truth
    bussac.png      analysed records, separated waves, interface force and velocity, each against the truth
    separated.txt   time and the separated quantities of both bars

Errors are max and rms over the VALID window t < T - max over both bars of (x_max/c0) - valid_margin: the
outgoing wave at the interface at time t needs gauge data up to x_max/c0 later. SHPB and SHTB: errors are
relative to the peak incident wave, the peak input-bar velocity and the peak input-bar force. Direct impact:
each bar is scaled by its own peaks, because the input-bar strain is much smaller than the output-bar strain;
the known impact velocity v0 is added to the separated input-bar velocity (the rigid-body motion of the
projectile carries no strain).
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

from hopkinson_simulator import (COLORS, INVALID, PM, US, BarMaterial, BarModel, LoveRod, WaveEq1D,
                                 apply_style, deep_merge, gauge_average_factor, load_case, load_config,
                                 make_bar_model, next_pow2, to_json)

ROOT = Path(__file__).resolve().parent

# ============================================================================================================
# Bussac separation (from wavePINN src/wavepinn/bussac.py)
# ============================================================================================================
# Bussac multi-gauge least-squares wave separation (reference method).
#
# Bussac, Collet, Gary, Othman, J. Mech. Phys. Solids 50 (2002) 321-349. Adapted from the
# WaveSeparation/3wave implementation (wave_separation_code/wave_separation.py),
# which was checked and corrected for use with dispersive bar models:
#
# 1. Laplace shift with dispersion: the damped transform evaluates the spectra at the complex
#    frequency omega - i*eta, so the wavenumber must be k(omega - i*eta) (analytic continuation
#    of the dispersion relation, Im k < 0). The original used (omega - i*eta)/c_p(omega), which
#    damps with 1/c_p instead of ~1/c_g and causes an amplitude error ~exp(eta x (1/c_g - 1/c_p))
#    at dispersive frequencies.
# 2. Band limit: components above ``f_max`` are set to zero. Needed for Love above (and close
#    to) its cut-off, where the wave toward the interface carries exp(+kappa x).
# 3. Interface velocity and force with the model-consistent operators (v = -/+ (s/k) eps,
#    N_eff = A rho (s/k)^2 eps), evaluated in the Laplace domain before the inverse transform.
#    The original used c0 for the velocity and A E eps for the force only.
# 4. Optional gauge-length averaging sin(kL/2)/(kL/2) in the gauge model.
#
# Local coordinate per bar: x >= 0 from the interface INTO the bar. "plus" travels away from
# the specimen (= outgoing wave b), "minus" travels toward it (= incoming wave a). Gauge k:
#     E_k(s) = G_k(s) [P(s) exp(-i k(s) x_k) + M(s) exp(+i k(s) x_k)],   s = omega - i eta.
# Strain tension positive. Velocities are returned in the GLOBAL frame (x from the incident
# toward the transmitted bar), using ``outward`` = -1 for the incident bar, +1 for the
# transmitted bar.
#
# Regularisation: eta > 0 (units 1/s) removes the singularity at DC and, for two gauges a
# distance D apart, at f = n c / (2 D): the determinant there is 4 sinh^2(eta D / c) instead of 0.
# eta * T must stay well below ~700 (exp(+eta t) is applied on the way out); the price of a
# large eta is noise amplification by exp(eta t) late in the record.

def wavenumber_laplace(model: BarModel, omega: np.ndarray, eta: float) -> np.ndarray:
    """k(omega - i eta), Im k < 0 (decaying in the direction of travel).

    Exact for Models A and B; for other models (Pochhammer-Chree) a second-order Taylor
    expansion k(w) - i eta k'(w) - eta^2 k''(w)/2 on the (uniform) omega grid.
    """
    omega = np.asarray(omega, dtype=float)
    s = omega - 1j * eta
    m = model.mat
    if isinstance(model, WaveEq1D):
        return s / m.c0
    if isinstance(model, LoveRod):
        # k^2 = s^2 / (c0^2 - nu^2 rg^2 s^2); for omega > 0, eta > 0 the radicand has Im > 0,
        # so the principal root is continuous and gives Im k < 0 (propagating and evanescent)
        return s / np.sqrt(m.c0 ** 2 - (m.nu * model.rg * s) ** 2)
    k = np.real(model.wavenumber(omega))
    dw = omega[1] - omega[0]
    k1 = np.gradient(k, dw)
    k2 = np.gradient(k1, dw)
    return k - 1j * eta * k1 - 0.5 * eta ** 2 * k2


def default_f_max(model: BarModel) -> float:
    """Band limit: 0.8 x Love cut-off for Model B, none otherwise."""
    if isinstance(model, LoveRod):
        return 0.8 * model.omega_cutoff / (2 * np.pi)
    return np.inf


def separate_spectra(t, signals, positions, model: BarModel, eta: float = 1000.0, f_max: float | None = None,
                     gauge_length: float | np.ndarray = 0.0, n_fft: int | None = None) -> dict:
    """Least-squares separation in the Laplace domain (s = omega - i eta) for ONE bar.

    Same arguments as ``separate``. Returns the spectra of the two strain waves at x = 0 (P: plus,
    M: minus; zero outside ``ok``), with omega, s, k(s), the complex phase velocity cs = s/k, the mask
    ``ok`` of the solved frequencies, the normalised determinant and ``inv``: the inverse transform,
    including the factor exp(+eta t), that maps a spectrum to a time history on the grid of ``t``.
    """
    t = np.asarray(t, dtype=float)
    E_t = np.atleast_2d(np.asarray(signals, dtype=float))
    x = np.asarray(positions, dtype=float)
    n = t.size
    if E_t.shape != (x.size, n):
        raise ValueError(f"signals {E_t.shape} do not match positions {x.shape} and t {t.shape}")
    if x.size < 2 or np.any(x <= 0) or np.unique(x).size != x.size:
        raise ValueError("need >= 2 distinct gauge positions > 0")
    if eta <= 0:
        raise ValueError("eta must be > 0")
    tau = t - t[0]
    if eta * tau[-1] > 700:
        raise ValueError(f"eta * record length = {eta * tau[-1]:.0f}: exp(+eta t) overflows")
    dt = tau[1]
    if not np.allclose(np.diff(t), dt, rtol=1e-6):
        raise ValueError("t must be uniformly sampled")
    n_fft = int(n_fft or next_pow2(4 * n))
    f_max = default_f_max(model) if f_max is None else float(f_max)

    omega = 2 * np.pi * np.fft.rfftfreq(n_fft, dt)
    band = omega <= 2 * np.pi * f_max
    win = np.exp(-eta * tau)
    E = np.fft.rfft(E_t * win, n_fft, axis=-1)
    k = wavenumber_laplace(model, omega, eta)
    k = np.where(band, k, 0.0)                      # values outside the band are discarded below
    Lg = np.broadcast_to(np.asarray(gauge_length, dtype=float), x.shape)
    G = np.array([gauge_average_factor(k, L) if L > 0 else np.ones_like(k) for L in Lg])
    a = G * np.exp(-1j * k[None, :] * x[:, None])   # plus-wave coefficients
    b = G * np.exp(+1j * k[None, :] * x[:, None])   # minus-wave coefficients
    h1 = np.sum(np.abs(a) ** 2, axis=0)
    h2 = np.sum(np.abs(b) ** 2, axis=0)
    g = np.sum(np.conj(a) * b, axis=0)
    E1 = np.sum(np.conj(a) * E, axis=0)
    E2 = np.sum(np.conj(b) * E, axis=0)
    det = h1 * h2 - np.abs(g) ** 2
    ok = band & (det > 1e-14 * h1 * h2)
    P = np.zeros_like(E1); M = np.zeros_like(E1)
    P[ok] = (h2[ok] * E1[ok] - g[ok] * E2[ok]) / det[ok]
    M[ok] = (-np.conj(g[ok]) * E1[ok] + h1[ok] * E2[ok]) / det[ok]

    s = omega - 1j * eta
    cs = np.zeros_like(s)
    cs[ok] = s[ok] / k[ok]                          # complex phase velocity; -> c0 for s -> 0
    amp = np.exp(eta * tau)

    def inv(X):
        return np.fft.irfft(X, n_fft)[:n] * amp

    return {"omega": omega, "s": s, "k": k, "cs": cs, "P": P, "M": M, "ok": ok, "inv": inv,
            "det_normalised": np.where(band, det / np.maximum(h1 * h2, 1e-300), 0.0),
            "f_max": f_max, "eta": eta, "model": model}


def separate(t, signals, positions, model: BarModel, eta: float = 1000.0, f_max: float | None = None,
             gauge_length: float | np.ndarray = 0.0, n_fft: int | None = None, outward: int = +1):
    """Separate the gauge records of ONE bar into the two waves at the interface (x = 0).

    Parameters
    ----------
    t : (N,) uniform time [s], starting before the first arrival (bar at rest).
    signals : (n_g, N) strain records; positions : (n_g,) distances from the interface [m].
    model : bar model (A, B or C) used for the separation.
    eta : exponential-window damping [1/s].
    f_max : band limit [Hz]; None -> default_f_max(model).
    gauge_length : gauge length(s) [m]; 0 -> point gauges.
    outward : global direction of the bar's local +x axis (-1 incident bar, +1 transmitted bar).

    Returns dict with time histories at the interface:
        plus, minus (strain waves; plus = away from the specimen),
        strain, velocity (global frame), force_E (A E eps), force_eff (A rho (s/k)^2 eps),
    and diagnostics: freq, det_normalised (1 ideal, 0 singular), f_max.
    """
    sp = separate_spectra(t, signals, positions, model, eta, f_max, gauge_length, n_fft)
    P, M, cs, inv, mat = sp["P"], sp["M"], sp["cs"], sp["inv"], model.mat
    # model-consistent interface quantities in the Laplace domain (s = omega - i eta)
    V_local = cs * (M - P)                          # plus: v = -(s/k) eps, minus: +(s/k) eps
    plus, minus = inv(P), inv(M)
    strain = plus + minus
    return {
        "plus": plus, "minus": minus, "strain": strain,
        "velocity": outward * inv(V_local),
        "force_E": mat.E * mat.area * strain,
        "force_eff": inv(mat.area * mat.rho * cs ** 2 * (P + M)),
        "freq": sp["omega"] / (2 * np.pi),
        "det_normalised": sp["det_normalised"],
        "f_max": sp["f_max"], "eta": eta,
    }


def separate_case(case: dict, bar_models: dict, eta: float = 1000.0, f_max: float | None = None,
                  use_gauge_length: bool = True, records_key: str = "records"):
    """Apply ``separate`` to both bars of a loaded case (io.load_case format).

    bar_models: {1: model_bar1, 2: model_bar2}. Returns {"bar1": ..., "bar2": ...} with the keys
    of ``separate`` plus wave_in (= minus, toward the specimen) and wave_out (= plus).
    """
    ch = case["channels"]
    out = {}
    for bar, outward in ((1, -1), (2, +1)):
        sel = np.flatnonzero(np.asarray(ch["bar"]) == bar)
        r = separate(case["time"], np.asarray(ch[records_key])[sel], np.asarray(ch["distance_from_interface"])[sel],
                     bar_models[bar], eta=eta, f_max=f_max,
                     gauge_length=np.asarray(ch["gauge_length"])[sel] if use_gauge_length else 0.0,
                     outward=outward)
        r["wave_out"], r["wave_in"] = r["plus"], r["minus"]
        out[f"bar{bar}"] = r
    return out




# ============================================================================================================
# Analysis of a simulation file (adapted from wavePINN src/wavepinn/analysis.py)
# ============================================================================================================
SEPARATION_DEFAULTS = {"eta": 1000.0, "f_max": None, "gauge_length_correction": True, "valid_margin": 20e-6}


def assumed_setup(case: dict, acfg: dict):
    """Bar models {1, 2} and channel table that the analysis assumes.

    Defaults: the bar model, materials and gauge positions stored in the simulation file. Overrides from the
    analysis case: [bars] bar_model, material, material_transmitted; [gauges] incident, transmitted."""
    sim_cfg = case["meta"]["config"]
    bars = deep_merge({"bar_model": sim_cfg["bar_model"], "material": sim_cfg["bars"]["material"],
                       "material_transmitted": sim_cfg["bars"].get("material_transmitted",
                                                                   sim_cfg["bars"]["material"])},
                      acfg.get("bars", {}))
    if "material" in acfg.get("bars", {}) and "material_transmitted" not in acfg["bars"] \
            and "material_transmitted" not in sim_cfg["bars"]:
        bars["material_transmitted"] = bars["material"]          # one material for both bars, as in the simulation
    models = {1: make_bar_model(bars["bar_model"], BarMaterial(**bars["material"])),
              2: make_bar_model(bars["bar_model"], BarMaterial(**bars["material_transmitted"]))}
    ch = copy.deepcopy(case["channels"])
    for bar, key in ((1, "incident"), (2, "transmitted")):
        pos = acfg.get("gauges", {}).get(key)
        if pos is None:
            continue
        sel = np.flatnonzero(ch["bar"] == bar)
        if len(pos) != len(sel):
            raise ValueError(f"[gauges] {key}: {len(pos)} positions given, the simulation has {len(sel)} gauges")
        ch["distance_from_interface"][sel] = np.asarray(pos, dtype=float)
    return models, ch, bars


def run_bussac(case: dict, acfg: dict):
    """Separate both bars of a loaded simulation file.

    Returns (result, settings, end of the valid window [s], analysed records, assumed bars)."""
    s = deep_merge(SEPARATION_DEFAULTS, acfg.get("separation", {}))
    models, ch, bars = assumed_setup(case, acfg)
    key = acfg.get("records", "records")
    work = {**case, "channels": ch}
    res = separate_case(work, models, eta=s["eta"], f_max=s["f_max"],
                        use_gauge_length=s["gauge_length_correction"], records_key=key)
    sim_cfg = case["meta"]["config"]
    if sim_cfg.get("setup", "shpb") == "direct_impact":
        res["bar1"]["velocity"] = res["bar1"]["velocity"] + float(sim_cfg["impact"]["velocity"])
    t = case["time"]
    delay = max(np.max(ch["distance_from_interface"][ch["bar"] == b]) / models[b].mat.c0 for b in (1, 2))
    t_valid = t[-1] - delay - s["valid_margin"]
    return res, s, t_valid, ch[key], bars


def metrics(case: dict, res: dict, valid, per_bar: bool = False) -> dict:
    """Errors relative to bar-1 scales (SHPB, SHTB) or to each bar's own scales (per_bar, direct impact)."""
    itf = case["interface"]

    def scales(b):
        w = itf[b]
        return (max(np.abs(w["wave_in"]).max(), np.abs(w["wave_out"]).max()) if per_bar else np.abs(w["wave_in"]).max(),
                np.abs(w["velocity"]).max(), np.abs(w["force_E"]).max())

    out = {}
    for b in ("bar1", "bar2"):
        sw, sv, sf = scales(b if per_bar else "bar1")
        for key, sc in (("wave_in", sw), ("wave_out", sw), ("velocity", sv), ("force_E", sf), ("force_eff", sf)):
            d = (res[b][key] - itf[b][key])[valid]
            out[f"{b}_{key}"] = {"max": float(np.abs(d).max() / sc), "rms": float(np.sqrt(np.mean(d ** 2)) / sc)}
    return out


def summary_line(e: dict) -> str:
    def mx(*keys):
        return max(e[k]["max"] for k in keys)
    return (f"waves max {mx('bar1_wave_in', 'bar1_wave_out', 'bar2_wave_in', 'bar2_wave_out'):.1e}"
            f"  v max {mx('bar1_velocity', 'bar2_velocity'):.1e}"
            f"  N_E max {mx('bar1_force_E', 'bar2_force_E'):.1e}"
            f"  N_eff max {mx('bar1_force_eff', 'bar2_force_eff'):.1e}")


def plot_bussac(case: dict, res: dict, rec, ch: dict, t_valid: float, title: str, path):
    """Bussac result against the truth, saved to ``path``; returns the (open) figure. Left column input bar,
    right column output bar. Row 1: analysed records; row 2: separated waves; row 3: force and velocity."""
    import matplotlib.pyplot as plt
    t = case["time"] * US
    itf = case["interface"]
    setup = case["meta"]["config"].get("setup", "shpb")
    sgn, sense = (1.0, "tension +") if setup == "shtb" else (-1.0, "compression +")
    pre = "" if sgn > 0 else "−"
    fig, ax = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    for col, bar, side in ((0, 1, "input bar"), (1, 2, "output bar")):
        b = f"bar{bar}"
        for j, i in enumerate(np.flatnonzero(ch["bar"] == bar)):
            ax[0, col].plot(t, rec[i] * PM, lw=0.9, color=COLORS[j],
                            label=f"{ch['names'][i]}: {ch['distance_from_interface'][i] * 1e3:.1f} mm (assumed)")
        ax[0, col].set(title=f"A) Analysed gauge records, {side}", ylabel="strain [‰]")
        impact = setup == "direct_impact"
        lab_in, lab_out = (("far-end echo", "impact wave") if bar == 1 and impact else
                           ("incident", "reflected") if bar == 1 else ("far-end echo", "transmitted"))
        for key, lab, c in (("wave_in", lab_in, COLORS[0]), ("wave_out", lab_out, COLORS[1])):
            ax[1, col].plot(t, itf[b][key] * PM, color=c, lw=2.2, alpha=0.35, label=f"{lab}, truth")
            ax[1, col].plot(t, res[b][key] * PM, color=c, ls="--", lw=0.9, label=f"{lab}, Bussac")
        ax[1, col].set(title=f"B) Separated waves at the interface, {side}", ylabel="strain [‰]")
        a2 = ax[2, col]
        a2.plot(t, sgn * itf[b]["force_E"] / 1e3, "k", lw=2.2, alpha=0.3, label=f"{pre}A·E·ε, truth")
        a2.plot(t, sgn * res[b]["force_E"] / 1e3, "k--", lw=0.9, label=f"{pre}A·E·ε, Bussac")
        a2.plot(t, sgn * res[b]["force_eff"] / 1e3, color=COLORS[4], ls=":", lw=0.9, label=f"{pre}A·σ_eff, Bussac")
        a2.set(title=f"C) Interface force ({sense}) and velocity, {side}", xlabel="t [µs]", ylabel="force [kN]")
        a3 = a2.twinx()
        vc = COLORS[2]
        a3.plot(t, itf[b]["velocity"], color=vc, lw=2.2, alpha=0.3, label="v, truth")
        a3.plot(t, res[b]["velocity"], color=vc, ls="--", lw=0.9, label="v, Bussac")
        a3.set_ylabel("v [m/s]", color=vc)
        a3.tick_params(axis="y", colors=vc, labelcolor=vc)
        a3.spines["right"].set_visible(True)
        a3.spines["right"].set_color(vc)
        a3.grid(False)
        a3.legend(fontsize=7, loc="upper right")
        for a in ax[:, col]:
            a.axvspan(t_valid * US, t[-1], color=INVALID, lw=0, zorder=0)
            a.legend(fontsize=7, loc="lower left")
    fig.suptitle(f"{title}: η = {res['bar1']['eta']:.0f} /s, f_max = {res['bar1']['f_max'] / 1e3:.0f} / "
                 f"{res['bar2']['f_max'] / 1e3:.0f} kHz (input / output bar), against the ground truth; "
                 f"grey: outside the valid window")
    fig.tight_layout()
    fig.savefig(path)
    return fig


def write_separated(path, t, res, t_valid):
    cols, names = [t * US], ["t[us]"]
    for b in ("bar1", "bar2"):
        for k, unit in (("wave_in", "-"), ("wave_out", "-"), ("force_E", "N"), ("force_eff", "N"),
                        ("velocity", "m/s")):
            cols.append(res[b][k]); names.append(f"{b}_{k}[{unit}]")
    header = (f"Bussac separation; valid for t < {t_valid * US:.1f} us. Waves: strain, tension +; "
              f"velocity in the global frame (x from the input towards the output bar).\n" + " ".join(names))
    np.savetxt(path, np.column_stack(cols), header=header, fmt="%.9e")


# ============================================================================================================
# Command line
# ============================================================================================================

def run(acfg_path: Path, out_root: Path, show: bool):
    import matplotlib.pyplot as plt
    acfg = load_config(acfg_path)
    sim_path = (ROOT / acfg["simulation"]) if not Path(acfg["simulation"]).is_absolute() else Path(acfg["simulation"])
    if not sim_path.exists():
        sys.exit(f"{acfg['name']}: simulation file {sim_path} not found; run the simulation case first, e.g.\n"
                 f"    python hopkinson_simulator.py simulation_cases/<case>.toml")
    case = load_case(sim_path)
    res, s, t_valid, rec, bars = run_bussac(case, acfg)
    setup = case["meta"]["config"].get("setup", "shpb")
    err = metrics(case, res, case["time"] < t_valid, per_bar=setup == "direct_impact")
    out = out_root / acfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    _, ch, _ = assumed_setup(case, acfg)
    report = {"simulation": str(sim_path), "records": acfg.get("records", "records"),
              "separation": {**s, "f_max_used": [res["bar1"]["f_max"], res["bar2"]["f_max"]]},
              "assumed_bars": bars,
              "assumed_gauges": {"bar": ch["bar"], "distance_from_interface": ch["distance_from_interface"]},
              "t_valid_end_us": t_valid * US, "errors": err}
    (out / "bussac.json").write_text(to_json(report))
    write_separated(out / "separated.txt", case["time"], res, t_valid)
    title = f"{acfg['name']}: Bussac with model {bars['bar_model']} on '{acfg.get('records', 'records')}' of {case['name']}"
    fig = plot_bussac(case, res, rec, ch, t_valid, title, out / "bussac.png")
    if show:
        fig.canvas.manager.set_window_title(f"{acfg['name']}: Bussac analysis")
    else:
        plt.close(fig)
    print(f"{acfg['name']}: valid up to {t_valid * US:.0f} µs; {summary_line(err)} -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="+", type=Path, help="analysis case files (analysis_cases/*.toml)")
    ap.add_argument("--out", type=Path, default=ROOT / "results", help="results root (default: results/)")
    ap.add_argument("--headless", action="store_true", help="save the figures without showing them")
    args = ap.parse_args()
    import matplotlib
    if args.headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    apply_style()
    for p in args.cases:
        run(p, args.out, show=not args.headless)
    if not args.headless:
        plt.show()


if __name__ == "__main__":
    main()
