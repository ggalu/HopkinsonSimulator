"""Bussac wave separation of simulated Hopkinson-bar gauge records, compared with the exact ground truth.

    python bussac_analysis.py analysis_cases/SHPB.toml [more.toml ...] [--out results] [--headless]

Each analysis case names the records file (simulation.csv) and the settings file (simulation.toml) written by
hopkinson_simulator.py, and defines every setting the analysis uses: the setup, the bar model and materials,
the gauge positions and length, and the separation. Nothing is taken from the simulation. The settings file is
read only to compare: on start the analysis settings are printed next to the simulation settings, and every
difference is marked. The analysis then runs with its own values, so that the effect of a wrong model or a
wrong gauge position can be studied on the same records. Output in <out>/<name>/:
    bussac.json     analysis case, comparison with the simulation, end of the valid window, errors
    bussac.png      analysed records, separated waves, interface force and velocity, each against the truth
    equilibrium.png separated forces at the input-bar and output-bar interfaces and their relative imbalance,
                    to judge the force equilibrium of the specimen
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
import tomllib
from pathlib import Path

import numpy as np

from hopkinson_simulator import (COLORS, INVALID, PM, US, BarMaterial, BarModel, LoveRod, WaveEq1D,
                                 BOLD, BOLD_RED, apply_style, format_table, gauge_average_factor, is_terminal,
                                 make_bar_model, next_pow2, read_records_csv, styled, to_json)

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
# Strain tension positive. Velocities are returned in the GLOBAL frame (x from the input
# toward the output bar), using ``outward`` = -1 for the input bar, +1 for the output bar.
#
# Regularisation: eta > 0 (units 1/s) removes the singularity at DC and, for two gauges a
# distance D apart, at f = n c / (2 D): the determinant there is 4 sinh^2(eta D / c) instead of 0.
# eta * T must stay well below ~700 (exp(+eta t) is applied on the way out); the price of a
# large eta is noise amplification by exp(eta t) late in the record.

def wavenumber_laplace(model: BarModel, omega: np.ndarray, eta: float) -> np.ndarray:
    """k(omega - i eta), Im k < 0 (decaying in the direction of travel).

    Exact for Models A and B, elastic or Kelvin-Voigt (1 + i omega tau becomes 1 + i s tau = 1 + eta tau
    + i omega tau, which has no zeros for eta > 0); for other models (Pochhammer-Chree) a second-order
    Taylor expansion k(w) - i eta k'(w) - eta^2 k''(w)/2 on the (uniform) omega grid.
    """
    omega = np.asarray(omega, dtype=float)
    s = omega - 1j * eta
    m = model.mat
    kv = 1 + 1j * s * m.tau                         # Kelvin-Voigt factor; 1 for an elastic bar
    if isinstance(model, WaveEq1D):
        return s / m.c0 if m.tau == 0 else s / (m.c0 * np.sqrt(kv))
    if isinstance(model, LoveRod):
        # k^2 = s^2 / (c0^2 kv - nu^2 rg^2 s^2); for omega > 0, eta > 0 the radicand has Im > 0,
        # so the principal root is continuous and gives Im k < 0 (propagating and evanescent)
        return s / np.sqrt(m.c0 ** 2 * kv - (m.nu * model.rg * s) ** 2)
    k = model.wavenumber(omega)
    dw = omega[1] - omega[0]
    k1 = np.gradient(k, dw)
    k2 = np.gradient(k1, dw)
    return k - 1j * eta * k1 - 0.5 * eta ** 2 * k2


# Kelvin-Voigt bars: the wave leaving the interface is damped by exp(-kappa x) on its way to a gauge at x,
# so its reconstruction at the interface multiplies by exp(+kappa x). "auto" keeps this amplification below
# AUTO_MAX_AMPLIFICATION at the farthest gauge. The value is a choice for noise-free synthetic records, not a
# derived limit: on the example cases the separation stayed usable up to about 1e3-1e4 and became unstable
# above. Measured records, with noise, need a lower value.
AUTO_MAX_AMPLIFICATION = 1.0e3


def default_f_max(model: BarModel, x_max: float = 0.0, f_nyquist: float = np.inf) -> float:
    """Band limit "auto": 0.8 x Love cut-off for Model B, none otherwise; for a Kelvin-Voigt bar in addition
    the frequency at which exp(kappa x_max) reaches AUTO_MAX_AMPLIFICATION."""
    f = 0.8 * model.omega_cutoff / (2 * np.pi) if isinstance(model, LoveRod) else np.inf
    if model.mat.tau > 0 and x_max > 0:
        f_top = min(f, f_nyquist)
        if not np.isfinite(f_top):
            raise ValueError("a Kelvin-Voigt bar needs a finite Nyquist frequency for f_max = \"auto\"")
        w = 2 * np.pi * np.linspace(0.0, f_top, 200001)
        over = np.flatnonzero(-np.imag(model.wavenumber(w)) * x_max > np.log(AUTO_MAX_AMPLIFICATION))
        if over.size:
            f = w[over[0] - 1] / (2 * np.pi)
    return f


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
    f_max = default_f_max(model, float(x.max()), 0.5 / dt) if f_max is None else float(f_max)

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
    f_max : band limit [Hz]; None -> default_f_max (the rule of f_max = "auto").
    gauge_length : gauge length(s) [m]; 0 -> point gauges.
    outward : global direction of the bar's local +x axis (-1 input bar, +1 output bar).

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
    """Apply ``separate`` to both bars of a case (the dict of read_records_csv or generate, with
    distance_from_interface and gauge_length in its channel table).

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
# Analysis case: required settings, comparison with the simulation settings
# ============================================================================================================
# The analysis case defines every setting the analysis uses; nothing is taken from the simulation. The
# simulation settings (simulation.toml) are read only to print the differences.

NUMBER, TEXT, FLAG, POSITIONS = "number", "text", "true/false", "list of numbers"
MATERIAL_KEYS = {"E": NUMBER, "rho": NUMBER, "nu": NUMBER, "radius": NUMBER, "inner_radius": NUMBER, "tau": NUMBER}
ANALYSIS_KEYS = {
    "name": TEXT, "records": TEXT, "settings": TEXT,
    "channels": ("averaged", "point"),
    "setup": ("shpb", "direct_impact", "shtb"),
    "impact": {"velocity": NUMBER},                       # required for setup = "direct_impact" only
    "bars": {"bar_model": ("A", "B", "C"), "material_input": MATERIAL_KEYS, "material_output": MATERIAL_KEYS},
    "gauges": {"input": POSITIONS, "output": POSITIONS, "gauge_length": NUMBER},
    "separation": {"eta": NUMBER, "f_max": "number or \"auto\"", "gauge_length_correction": FLAG,
                   "valid_margin": NUMBER},
}


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _type_ok(v, kind) -> bool:
    if isinstance(kind, tuple):
        return v in kind
    return {NUMBER: _is_number, TEXT: lambda x: isinstance(x, str), FLAG: lambda x: isinstance(x, bool),
            POSITIONS: lambda x: isinstance(x, list) and len(x) > 0 and all(_is_number(y) for y in x),
            "number or \"auto\"": lambda x: _is_number(x) or x == "auto"}[kind](v)


def _kind_text(kind) -> str:
    return " or ".join(f'"{k}"' for k in kind) if isinstance(kind, tuple) else kind


def check_analysis_config(acfg: dict) -> None:
    """Raise ValueError naming every missing, unknown or wrongly typed key of an analysis case."""
    errors = []

    def walk(d, spec, path):
        for k, kind in spec.items():
            key = path + k
            if k == "impact" and not path and acfg.get("setup") != "direct_impact":
                if k in d:
                    errors.append(f"[impact] is only used with setup = \"direct_impact\"")
                continue
            if k not in d:
                errors.append(f"missing: {key} ({'table' if isinstance(kind, dict) else _kind_text(kind)})")
            elif isinstance(kind, dict):
                if isinstance(d[k], dict):
                    walk(d[k], kind, key + ".")
                else:
                    errors.append(f"{key} must be a table")
            elif not _type_ok(d[k], kind):
                errors.append(f"{key} = {d[k]!r}: expected {_kind_text(kind)}")
        for k in d:
            if k not in spec:
                errors.append(f"unknown key: {path + k}")

    walk(acfg, ANALYSIS_KEYS, "")
    if errors:
        raise ValueError("the analysis case is incomplete or wrong:\n    " + "\n    ".join(errors))


def _same(a, b) -> bool:
    if _is_number(a) and _is_number(b):
        return bool(np.isclose(a, b, rtol=1e-12, atol=0.0) or a == b)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def compare_settings(acfg: dict, sim_cfg: dict) -> list[dict]:
    """One row per analysis setting: key, analysis value, simulation value (None: no counterpart) and status
    ("same", "differs" or "analysis only")."""
    rows = []

    def row(key, a, s, counterpart=True):
        status = "analysis only" if not counterpart else ("same" if s is not None and _same(a, s) else "differs")
        rows.append({"key": key, "analysis": a, "simulation": s, "status": status})

    sim_bars, sim_g = sim_cfg.get("bars", {}), sim_cfg.get("gauges", {})
    row("name", acfg["name"], None, False)
    row("records", acfg["records"], None, False)
    row("settings", acfg["settings"], None, False)
    averaging = sim_g.get("averaging", True)
    sim_channels = "averaged" if averaging else "point"   # without averaging both columns hold the centre strain
    row("channels", acfg["channels"], acfg["channels"] if acfg["channels"] == "point" else sim_channels)
    row("setup", acfg["setup"], sim_cfg.get("setup", "shpb"))
    if "impact" in acfg:
        row("impact.velocity", acfg["impact"]["velocity"], sim_cfg.get("impact", {}).get("velocity"))
    row("bars.bar_model", acfg["bars"]["bar_model"], sim_cfg.get("bar_model"))
    for table in ("material_input", "material_output"):
        # settings files of earlier versions: material_output defaulted to the input bar, inner_radius and tau to 0
        sim_mat = sim_bars.get(table, sim_bars.get("material_input", {}))
        mat = acfg["bars"][table]
        for k in MATERIAL_KEYS:
            optional = k in ("inner_radius", "tau")
            row(f"bars.{table}.{k}", mat[k], sim_mat.get(k, 0.0 if optional else None))
    for k in ("input", "output", "gauge_length"):
        row(f"gauges.{k}", acfg["gauges"][k], sim_g.get(k))
    for k, v in acfg["separation"].items():
        row(f"separation.{k}", v, None, False)
    return rows


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.10g}"
    if isinstance(v, list):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)


STATUS_LEGEND = """
  How to read the table: "analysis" is the value this analysis uses (from its case file); "simulation" is the
  value of the same setting in the simulation settings file. The status column says:
    same           both values agree (numbers to a relative 1e-12).
    ≠ DIFFERS      the analysis assumes something other than what was simulated, e.g. a different bar model
                   or gauge position. The analysis still runs, with its own value; this is intended when the
                   effect of a wrong assumption is studied, and an error otherwise.
    analysis only  a setting of the analysis alone (output name, file paths, separation parameters); the
                   simulation has no such setting, so there is nothing to compare.
    —              the simulation settings file does not contain this setting.
"""


def print_comparison(rows: list[dict], title: str, stream=None) -> int:
    """Print the settings as a framed table, with a line between the groups of keys; differences are marked
    (and drawn in bold red on a terminal, where the header is bold). Returns the number of differences."""
    stream = stream or sys.stdout
    color = is_terminal(stream)
    groups, group = [], None
    for r in rows:
        g = r["key"].rsplit(".", 1)[0] if "." in r["key"] else ""
        if g != group:
            groups.append([])
            group = g
        cells = [r["key"], _fmt(r["analysis"]), _fmt(r["simulation"]) if r["status"] != "analysis only" else "",
                 "≠ DIFFERS" if r["status"] == "differs" else r["status"]]
        groups[-1].append([styled(c, BOLD_RED, color and r["status"] == "differs") for c in cells])
    print(styled(title, BOLD, color), file=stream)
    print(format_table(("setting", "analysis", "simulation", "status"), groups, color, wrap=1), file=stream)
    print(STATUS_LEGEND, file=stream)
    n = sum(r["status"] == "differs" for r in rows)
    summary = (f"{n} setting{'s' if n != 1 else ''} differ{'s' if n == 1 else ''} from the simulation; "
               f"the analysis uses its own values" if n else "no differences from the simulation settings")
    print(styled(summary, BOLD_RED, color and n > 0) + "\n", file=stream)
    return n


# ============================================================================================================
# Analysis of a records file (adapted from wavePINN src/wavepinn/analysis.py)
# ============================================================================================================

def assumed_setup(data: dict, acfg: dict):
    """Bar models {1, 2} and channel table of the analysis case: gauge positions and gauge length from
    [gauges], assigned in order to the I* (input bar) and O* (output bar) columns of the records file."""
    b = acfg["bars"]
    models = {1: make_bar_model(b["bar_model"], BarMaterial(**b["material_input"])),
              2: make_bar_model(b["bar_model"], BarMaterial(**b["material_output"]))}
    ch = copy.deepcopy(data["channels"])
    ch["distance_from_interface"] = np.zeros(len(ch["names"]))
    ch["gauge_length"] = np.full(len(ch["names"]), float(acfg["gauges"]["gauge_length"]))
    for bar, key in ((1, "input"), (2, "output")):
        pos = acfg["gauges"][key]
        sel = np.flatnonzero(ch["bar"] == bar)
        if len(pos) != len(sel):
            raise ValueError(f"[gauges] {key}: {len(pos)} positions given, the records file has {len(sel)} "
                             f"gauges on bar {bar} ({', '.join(ch['names'][i] for i in sel)})")
        ch["distance_from_interface"][sel] = np.asarray(pos, dtype=float)
    return models, ch


def run_bussac(data: dict, acfg: dict):
    """Separate both bars of a records file (read_records_csv) with the settings of a checked analysis case.

    Returns (result, end of the valid window [s], analysed records, assumed channel table)."""
    s = acfg["separation"]
    models, ch = assumed_setup(data, acfg)
    key = "records" if acfg["channels"] == "averaged" else "records_point"
    res = separate_case({**data, "channels": ch}, models, eta=float(s["eta"]),
                        f_max=None if s["f_max"] == "auto" else float(s["f_max"]),
                        use_gauge_length=s["gauge_length_correction"], records_key=key)
    if acfg["setup"] == "direct_impact":
        res["bar1"]["velocity"] = res["bar1"]["velocity"] + float(acfg["impact"]["velocity"])
    t = data["time"]
    delay = max(np.max(ch["distance_from_interface"][ch["bar"] == b]) / models[b].mat.c0 for b in (1, 2))
    t_valid = t[-1] - delay - float(s["valid_margin"])
    return res, t_valid, ch[key], ch


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


def plot_bussac(case: dict, res: dict, rec, ch: dict, t_valid: float, title: str, path, setup: str):
    """Bussac result against the truth, saved to ``path``; returns the (open) figure. Left column input bar,
    right column output bar. Row 1: analysed records; row 2: separated waves; row 3: force and velocity."""
    import matplotlib.pyplot as plt
    t = case["time"] * US
    itf = case["interface"]
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


# Force equilibrium of the specimen: the forces at the two bar-specimen interfaces, N1 (input bar) and N2 (output
# bar), both A E eps, the force through which the simulated specimen is coupled to the bars. The imbalance
# (N1 - N2) / Nm with the mean Nm = (N1 + N2)/2 is evaluated only where |Nm| is at least EQUILIBRIUM_MIN_FORCE
# times its peak in the valid window, because the ratio is meaningless while the forces are small (rise, unloading).
EQUILIBRIUM_MIN_FORCE = 0.05


def force_imbalance(n1, n2, valid, min_fraction: float = EQUILIBRIUM_MIN_FORCE):
    """(N1 - N2)/Nm (NaN where not evaluated) and its largest magnitude in the valid window."""
    nm = 0.5 * (n1 + n2)
    peak = np.max(np.abs(nm[valid])) if np.any(valid) else 0.0
    use = valid & (np.abs(nm) >= min_fraction * peak) & (peak > 0)
    r = np.full(nm.shape, np.nan)
    r[use] = (n1[use] - n2[use]) / nm[use]
    return r, float(np.max(np.abs(r[use]))) if np.any(use) else float("nan")


def equilibrium(case: dict, res: dict, valid) -> dict:
    """Largest relative force imbalance of the separated and of the true interface forces."""
    itf = case["interface"]
    _, sep = force_imbalance(res["bar1"]["force_E"], res["bar2"]["force_E"], valid)
    _, true = force_imbalance(itf["bar1"]["force_E"], itf["bar2"]["force_E"], valid)
    return {"force": "A E eps", "min_force_fraction": EQUILIBRIUM_MIN_FORCE,
            "max_relative_imbalance_separated": sep, "max_relative_imbalance_truth": true}


def plot_equilibrium(case: dict, res: dict, t_valid: float, title: str, path, setup: str):
    """Separated interface forces of the input and the output bar against each other, and their relative
    imbalance, to judge the force equilibrium of the specimen; saved to ``path``, returns the (open) figure."""
    import matplotlib.pyplot as plt
    t = case["time"] * US
    itf = case["interface"]
    sgn, sense = (1.0, "tension +") if setup == "shtb" else (-1.0, "compression +")
    valid = case["time"] < t_valid
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for bar, side, c in ((1, "input bar", COLORS[0]), (2, "output bar", COLORS[1])):
        b = f"bar{bar}"
        ax[0].plot(t, sgn * itf[b]["force_E"] / 1e3, color=c, lw=2.2, alpha=0.3, label=f"{side}, truth")
        ax[0].plot(t, sgn * res[b]["force_E"] / 1e3, color=c, lw=1.0, label=f"{side}, Bussac")
    ax[0].set(title=f"A) Interface forces A·E·ε ({sense})", ylabel="force [kN]")
    r_sep, m_sep = force_imbalance(res["bar1"]["force_E"], res["bar2"]["force_E"], valid)
    r_true, m_true = force_imbalance(itf["bar1"]["force_E"], itf["bar2"]["force_E"], valid)
    ax[1].plot(t, 100 * r_true, color="k", lw=2.2, alpha=0.3, label=f"truth (max {100 * m_true:.1f} %)")
    ax[1].plot(t, 100 * r_sep, color=COLORS[4], lw=1.0, label=f"Bussac (max {100 * m_sep:.1f} %)")
    for y in (-5.0, 5.0):
        ax[1].axhline(y, color="k", ls="--", lw=0.8, label="±5 %" if y > 0 else None)
    ax[1].axhline(0.0, color="#888888", lw=0.8)
    ax[1].set(title=f"B) Relative imbalance (N_input − N_output) / mean, where |mean| ≥ "
                    f"{100 * EQUILIBRIUM_MIN_FORCE:.0f} % of its peak", xlabel="t [µs]", ylabel="imbalance [%]")
    finite = r_sep[np.isfinite(r_sep)]
    if finite.size:
        lim = min(max(10.0, 110 * np.max(np.abs(finite))), 100.0)
        ax[1].set_ylim(-lim, lim)
    for a in ax:
        a.axvspan(t_valid * US, t[-1], color=INVALID, lw=0, zorder=0)
        a.legend(fontsize=8, loc="best")
    fig.suptitle(f"{title}: force equilibrium of the specimen; grey: outside the valid window")
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

def _project_path(p: str) -> Path:
    return Path(p) if Path(p).is_absolute() else ROOT / p


def run(acfg_path: Path, out_root: Path, show: bool):
    import matplotlib.pyplot as plt
    acfg = tomllib.loads(Path(acfg_path).read_text())
    try:
        check_analysis_config(acfg)
    except ValueError as e:
        sys.exit(f"{acfg_path}: {e}")
    rec_path, set_path = _project_path(acfg["records"]), _project_path(acfg["settings"])
    for p in (rec_path, set_path):
        if not p.exists():
            sys.exit(f"{acfg['name']}: {p} not found; run the simulation case first, e.g.\n"
                     f"    python hopkinson_simulator.py simulation_cases/<case>.toml")
    rows = compare_settings(acfg, tomllib.loads(set_path.read_text()))
    print_comparison(rows, f"{acfg['name']}: settings of {acfg_path} against the simulation settings {set_path}")
    data = read_records_csv(rec_path)
    try:
        res, t_valid, rec, ch = run_bussac(data, acfg)
    except ValueError as e:
        sys.exit(f"{acfg_path}: {e}")
    setup = acfg["setup"]
    err = metrics(data, res, data["time"] < t_valid, per_bar=setup == "direct_impact")
    out = out_root / acfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    eq = equilibrium(data, res, data["time"] < t_valid)
    report = {"records": str(rec_path), "settings": str(set_path), "analysis_case": acfg,
              "f_max_used": [res["bar1"]["f_max"], res["bar2"]["f_max"]],
              "comparison": rows, "t_valid_end_us": t_valid * US, "errors": err, "equilibrium": eq}
    (out / "bussac.json").write_text(to_json(report))
    write_separated(out / "separated.txt", data["time"], res, t_valid)
    title = (f"{acfg['name']}: Bussac with model {acfg['bars']['bar_model']} on the {acfg['channels']} records "
             f"of {data['name']}")
    figs = {"Bussac analysis": plot_bussac(data, res, rec, ch, t_valid, title, out / "bussac.png", setup),
            "force equilibrium": plot_equilibrium(data, res, t_valid, title, out / "equilibrium.png", setup)}
    for name, fig in figs.items():
        if show:
            fig.canvas.manager.set_window_title(f"{acfg['name']}: {name}")
        else:
            plt.close(fig)
    print(f"{acfg['name']}: valid up to {t_valid * US:.0f} µs; {summary_line(err)}; force imbalance max "
          f"{100 * eq['max_relative_imbalance_separated']:.1f} % (truth "
          f"{100 * eq['max_relative_imbalance_truth']:.1f} %) -> {out}")


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
