"""Makes the figures of the theory report (reports/theory/report.tex) with the code of the two programs, so that
what a figure draws is what the programs compute. Run from any directory:

    python reports/theory/make_figures.py

Figures (Gnuplot 5 line colours):

  figures/dispersion.pdf    phase velocity and group velocity, relative to c0, against frequency for the three
                            bar models, for the aluminium bar of the examples (E = 71.7 GPa, rho = 2810 kg/m^3,
                            nu = 0.33, radius 8 mm); the Love cut-off as a dashed vertical line.
  figures/conditioning.pdf  normalised determinant det / (h1 h2) of the separation against frequency, for a
                            non-dispersive bar, on a logarithmic axis (values below 1e-8 drawn at 1e-8): two
                            gauges at 0.30 m and 0.85 m without and with the Laplace damping eta = 1000 /s, and
                            three gauges at 0.30 m, 0.62 m and 0.85 m without damping (left up to 30 kHz,
                            right up to 300 kHz).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPORT_DIR.parents[1]))
import tomllib                                          # noqa: E402

from bussac_analysis import (check_analysis_config, compare_settings, equilibrium, force_imbalance,  # noqa: E402
                             metrics, run_bussac, wavenumber_laplace)
from hopkinson_simulator import (COLORS, US, PM, BarMaterial, LoveRod, PochhammerChree, WaveEq1D,   # noqa: E402
                                 apply_style, generate, load_config)

ROOT = REPORT_DIR.parents[1]

FIGURES = REPORT_DIR / "figures"
FIGURES.mkdir(exist_ok=True)
AL = BarMaterial(E=71.7e9, rho=2810.0, nu=0.33, radius=8.0e-3)


def figure_dispersion():
    f = np.linspace(1e3, 700e3, 7000)
    w = 2 * np.pi * f
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for model, label, c in ((WaveEq1D(AL), "A: 1D wave equation", COLORS[0]),
                            (LoveRod(AL), "B: Love rod", COLORS[1]),
                            (PochhammerChree(AL), "C: Pochhammer-Chree, first mode", COLORS[2])):
        k = model.wavenumber(w)
        prop = (np.abs(k.imag) < 1e-12) & np.isfinite(k)     # model B: propagating below the cut-off only
        with np.errstate(divide="ignore", invalid="ignore"):
            cp = np.where(prop, w / k.real, np.nan)
            cg = np.where(prop, np.gradient(w, k.real), np.nan)
        ax[0].plot(f / 1e3, cp / AL.c0, color=c, label=label)
        ax[1].plot(f / 1e3, cg / AL.c0, color=c, label=label)
    fc = LoveRod(AL).omega_cutoff / (2 * np.pi) / 1e3
    cR = PochhammerChree(AL).cRbar * PochhammerChree(AL).cT / AL.c0
    for a, name in zip(ax, ("phase velocity $c/c_0$", "group velocity $c_g/c_0$")):
        a.axvline(fc, color="k", ls="--", lw=0.8)
        a.axhline(cR, color="#888888", ls=":", lw=0.8)
        a.set(xlabel="frequency $f$ [kHz]", ylabel=name, xlim=(0, 700), ylim=(0, 1.05))
    ax[0].text(fc + 8, 0.08, "Love cut-off\n431 kHz", fontsize=8)
    ax[0].text(560, cR + 0.03, "Rayleigh speed", fontsize=8, color="#555555")
    ax[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "dispersion.pdf")
    plt.close(fig)


def normalised_det(x, eta, f):
    """det / (h1 h2) of the least-squares separation (no gauge length), as in separate_spectra."""
    k = wavenumber_laplace(WaveEq1D(AL), 2 * np.pi * f, eta) if eta > 0 else 2 * np.pi * f / AL.c0 + 0j
    a = np.exp(-1j * np.outer(x, k))
    b = np.exp(+1j * np.outer(x, k))
    h1, h2 = np.sum(np.abs(a) ** 2, 0), np.sum(np.abs(b) ** 2, 0)
    g = np.sum(np.conj(a) * b, 0)
    return (h1 * h2 - np.abs(g) ** 2) / (h1 * h2)


def figure_conditioning():
    f = np.linspace(0.0, 30e3, 30001)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [3, 2]})
    two, three = np.array([0.30, 0.85]), np.array([0.30, 0.62, 0.85])
    floor = 1e-8                                  # the undamped two-gauge determinant reaches zero
    ax[0].semilogy(f / 1e3, np.maximum(normalised_det(two, 0.0, f), floor), color=COLORS[0],
                   label="two gauges, $\\eta = 0$")
    ax[0].semilogy(f / 1e3, normalised_det(two, 1000.0, f), color=COLORS[1], label="two gauges, $\\eta = 1000$ /s")
    ax[0].semilogy(f / 1e3, np.maximum(normalised_det(three, 0.0, f), floor), color=COLORS[2],
                   label="three gauges, $\\eta = 0$")
    ax[0].set(xlabel="frequency $f$ [kHz]", ylabel="$\\det / (h_1 h_2)$", xlim=(0, 30), ylim=(1e-8, 2))
    ax[0].legend(loc="lower right", fontsize=8, framealpha=0.95)
    fz = np.linspace(0.0, 300e3, 300001)
    ax[1].semilogy(fz / 1e3, np.maximum(normalised_det(three, 0.0, fz), floor), color=COLORS[2],
                   label="three gauges, $\\eta = 0$")
    ax[1].set(xlabel="frequency $f$ [kHz]", ylabel="$\\det / (h_1 h_2)$", xlim=(0, 300), ylim=(1e-8, 2))
    ax[1].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "conditioning.pdf")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ examples
# The worked examples run the simulation case and the analysis case of the project in memory, with the same
# functions as the two programs (generate, run_bussac), but without writing and reading the CSV file; the CSV
# rounds to 12 digits, which changes none of the numbers below in the printed digits.

def run_example(name):
    """Simulation and analysis of the cases simulation_cases/<name>.toml and analysis_cases/<name>.toml."""
    cfg = load_config(ROOT / "simulation_cases" / f"{name}.toml")
    acfg = tomllib.loads((ROOT / "analysis_cases" / f"{name}.toml").read_text())
    check_analysis_config(acfg)
    differ = [r["key"] for r in compare_settings(acfg, cfg) if r["status"] == "differs"]
    if differ:
        raise SystemExit(f"{name}: the analysis case differs from the simulation in {differ}")
    case, _ = generate(cfg)
    data = {**case, "channels": {k: case["channels"][k] for k in ("names", "bar", "records", "records_point")}}
    res, t_valid, rec, ch = run_bussac(data, acfg)
    valid = case["time"] < t_valid
    return {"case": case, "res": res, "t_valid": t_valid, "rec": rec, "ch": ch, "valid": valid,
            "errors": metrics(case, res, valid), "equilibrium": equilibrium(case, res, valid),
            "setup": acfg["setup"]}


def _shade(ax, ex):
    t = ex["case"]["time"] * US
    for a in np.atleast_1d(ax).ravel():
        a.axvspan(ex["t_valid"] * US, t[-1], color="#e4e4e4", lw=0, zorder=0)
        a.set_xlim(t[0], t[-1])


def figure_signals(name, ex):
    """Row 1: simulated gauge records; row 2: separated waves at the interface against the truth."""
    t = ex["case"]["time"] * US
    itf, res, ch = ex["case"]["interface"], ex["res"], ex["ch"]
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    for col, bar, side in ((0, 1, "input bar"), (1, 2, "output bar")):
        for j, i in enumerate(np.flatnonzero(ch["bar"] == bar)):
            ax[0, col].plot(t, ex["rec"][i] * PM, lw=0.9, color=COLORS[j],
                            label=f"{ch['names'][i]} at {ch['distance_from_interface'][i]:.2f} m")
        ax[0, col].set(title=f"(a) gauge records, {side}" if col == 0 else f"(b) gauge records, {side}",
                       ylabel="strain [‰]")
        names = ("incident $a_1$", "reflected $b_1$") if bar == 1 else ("echo $a_2$", "transmitted $b_2$")
        for key, lab, c in (("wave_in", names[0], COLORS[0]), ("wave_out", names[1], COLORS[1])):
            ax[1, col].plot(t, res[f"bar{bar}"][key] * PM, color=c, lw=1.1, label=f"{lab}, separated")
        for key in ("wave_in", "wave_out"):
            ax[1, col].plot(t, itf[f"bar{bar}"][key] * PM, "k--", lw=0.8,
                            label="simulated (truth)" if key == "wave_in" else None)
        ax[1, col].set(title=f"({'c' if col == 0 else 'd'}) waves at the interface, {side}",
                       ylabel="strain [‰]", xlabel="time $t$ [µs]")
        for a in ax[:, col]:
            a.legend(fontsize=7, loc="best")
    _shade(ax, ex)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}_signals.pdf")
    plt.close(fig)


def figure_forces(name, ex):
    """Top: interface forces A E eps of both bars, separated and true; bottom: their relative imbalance."""
    t = ex["case"]["time"] * US
    itf, res, valid = ex["case"]["interface"], ex["res"], ex["valid"]
    sgn = 1.0 if ex["setup"] == "shtb" else -1.0
    fig, ax = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    for bar, side, c in ((1, "input bar, $N_1$", COLORS[0]), (2, "output bar, $N_2$", COLORS[1])):
        ax[0].plot(t, sgn * res[f"bar{bar}"]["force_E"] / 1e3, color=c, lw=1.1, label=f"{side}, separated")
    for bar in (1, 2):
        ax[0].plot(t, sgn * itf[f"bar{bar}"]["force_E"] / 1e3, "k--", lw=0.8,
                   label="simulated (truth)" if bar == 1 else None)
    ax[0].set(title="(a) interface forces " + ("(tension positive)" if sgn > 0 else "(compression positive)"),
              ylabel="force [kN]")
    r_sep, _ = force_imbalance(res["bar1"]["force_E"], res["bar2"]["force_E"], valid)
    r_true, _ = force_imbalance(itf["bar1"]["force_E"], itf["bar2"]["force_E"], valid)
    ax[1].plot(t, 100 * r_sep, color=COLORS[4], lw=1.1, label="separated")
    ax[1].plot(t, 100 * r_true, "k--", lw=0.8, label="simulated (truth)")
    ax[1].set(title="(b) relative imbalance $(N_1 - N_2) / N_m$", ylabel="imbalance [%]",
              xlabel="time $t$ [µs]")
    finite = r_sep[np.isfinite(r_sep)]
    lim = min(max(0.5, 150 * np.max(np.abs(finite))), 100.0) if finite.size else 1.0
    ax[1].set_ylim(-lim, lim)
    for a in ax:
        a.legend(fontsize=8, loc="best")
    _shade(ax, ex)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}_forces.pdf")
    plt.close(fig)


def write_numbers(examples):
    """The numbers the report quotes for the examples, as LaTeX macros in figures/examples.tex."""
    def e(x):
        m, p = f"{x:.1e}".split("e")
        return f"{m} \\cdot 10^{{{int(p)}}}"
    lines = ["% written by make_figures.py; do not edit"]
    for name, ex in examples.items():
        err, eq = ex["errors"], ex["equilibrium"]
        waves = max(err[f"{b}_{w}"]["max"] for b in ("bar1", "bar2") for w in ("wave_in", "wave_out"))
        force = max(err[f"{b}_force_E"]["max"] for b in ("bar1", "bar2"))
        tag = name.lower()
        vals = {"Valid": f"{ex['t_valid'] * US:.0f}", "Fmax": f"{ex['res']['bar1']['f_max'] / 1e3:.0f}",
                "WaveErr": e(waves), "ForceErr": e(force),
                "ImbSep": f"{100 * eq['max_relative_imbalance_separated']:.1f}",
                "ImbTrue": f"{100 * eq['max_relative_imbalance_truth']:.1f}"}
        for k, v in vals.items():
            lines.append(f"\\newcommand{{\\{tag}{k}}}{{{v}}}")
        print(name, vals)
    (FIGURES / "examples.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    apply_style()
    figure_dispersion()
    figure_conditioning()
    examples = {name: run_example(name) for name in ("SHPB", "SHTB")}
    for name, ex in examples.items():
        figure_signals(name, ex)
        figure_forces(name, ex)
    write_numbers(examples)
    print("figures written to", FIGURES)
