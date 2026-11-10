"""Tests of bussac_analysis.py: exact recovery of the interface waves for model A, recovery with the
continued wavenumber for model C, the Laplace wavenumber of the Love model, the tension bar, and the
overrides of the analysis case."""
from pathlib import Path

import numpy as np

from bussac_analysis import compare_settings, run_bussac, separate_case, wavenumber_laplace
from hopkinson_simulator import BarMaterial, BarModel, LoveRod, deep_merge, generate, load_config

CASES = Path(__file__).resolve().parents[1] / "simulation_cases"
SHPB = load_config(CASES / "SHPB.toml")
SHTB = load_config(CASES / "SHTB.toml")
MAT = BarMaterial(**SHPB["bars"]["material_input"])


def _bussac(cfg, records="records", t_end=None, **kw):
    case, sim = generate(cfg)
    r = separate_case(case, {1: sim.model1, 2: sim.model2}, records_key=records, **kw)
    t = case["time"]
    valid = t < (t[-1] - 200e-6 if t_end is None else t_end)    # outgoing wave needs data up to x_max/c after t
    sc = np.abs(case["interface"]["bar1"]["wave_in"]).max()
    err = max(np.abs(r[b][w] - case["interface"][b][w])[valid].max()
              for b in ("bar1", "bar2") for w in ("wave_in", "wave_out")) / sc
    return case, r, err


def test_bussac_recovers_model_a_waves_exactly():
    case, r, err = _bussac(deep_merge(SHPB, {"bar_model": "A"}))
    assert err < 1e-5, err
    v = case["interface"]["bar1"]["velocity"]
    assert np.max(np.abs(r["bar1"]["velocity"] - v)) / np.max(np.abs(v)) < 1e-5


def test_bussac_model_c_with_continued_wavenumber():
    """k(omega - i eta) (not (omega - i eta)/c_p) is what makes dispersive separation exact."""
    _, _, err = _bussac(deep_merge(SHPB, {"bar_model": "C"}), eta=3000.0)
    assert err < 1e-4, err


def test_bussac_shtb_model_a():
    """The SHTB records end at 20-30 % of their peak; the error of the separation then grows towards the end of
    the record. With a 3 ms record the first 1.5 ms are recovered to about 1e-4 (2 ms record: about 4e-4)."""
    case, r, err = _bussac(deep_merge(SHTB, {"bar_model": "A", "sampling": {"duration": 3e-3}}), t_end=1.5e-3)
    assert err < 1e-4, err
    f = case["interface"]["bar2"]["force_E"]
    valid = case["time"] < 1.5e-3
    assert np.max(np.abs(r["bar2"]["force_E"] - f)[valid]) / np.max(np.abs(f)) < 1e-4


def test_love_laplace_wavenumber_taylor_matches_exact():
    love = LoveRod(MAT)
    w = 2 * np.pi * np.linspace(0, 300e3, 3001)
    exact = wavenumber_laplace(love, w, 1000.0)
    generic = BarModel.__new__(BarModel); generic.mat = MAT; generic.wavenumber = love.wavenumber
    taylor = wavenumber_laplace(generic, w, 1000.0)
    assert np.max(np.abs(taylor - exact)[5:-5] / np.abs(exact[5:-5])) < 1e-6
    assert np.all(exact.imag[1:] < 0)


def full_acfg(sim_cfg: dict) -> dict:
    """A complete analysis case with the values of a simulation case."""
    b, g = sim_cfg["bars"], sim_cfg["gauges"]
    acfg = {"name": "test", "records": "simulation.csv", "settings": "simulation.toml", "channels": "averaged",
            "setup": sim_cfg["setup"],
            "bars": {"bar_model": sim_cfg["bar_model"], "material_input": dict(b["material_input"]),
                     "material_output": dict(b.get("material_output", b["material_input"]))},
            "gauges": {k: g[k] for k in ("input", "output", "gauge_length")},
            "separation": {"eta": 1000.0, "f_max": "auto", "gauge_length_correction": True,
                           "valid_margin": 20e-6}}
    if sim_cfg["setup"] == "direct_impact":
        acfg["impact"] = {"velocity": sim_cfg["impact"]["velocity"]}
    return acfg


def test_analysis_case_settings_take_effect(tmp_path):
    """The analysis uses its own settings: with those of the simulation it reports no differences; a wrong
    gauge position or bar model is reported and changes the result."""
    from hopkinson_simulator import read_records_csv, write_records_csv
    cfg = deep_merge(SHPB, {"sampling": {"duration": 700e-6}})
    case, _ = generate(cfg)
    write_records_csv(case, tmp_path / "s.csv")
    data = read_records_csv(tmp_path / "s.csv")
    base = full_acfg(cfg)
    assert not [r for r in compare_settings(base, cfg) if r["status"] == "differs"]
    r0, tv, _, _ = run_bussac(data, base)
    valid = data["time"] < tv
    w = data["interface"]["bar1"]["wave_out"]
    e0 = np.max(np.abs(r0["bar1"]["wave_out"] - w)[valid])
    wrong_gauge = deep_merge(base, {"gauges": {"input": [0.305, 0.85]}})
    wrong_model = deep_merge(base, {"bars": {"bar_model": "A"}})
    for acfg, key in ((wrong_gauge, "gauges.input"), (wrong_model, "bars.bar_model")):
        diff = [r["key"] for r in compare_settings(acfg, cfg) if r["status"] == "differs"]
        assert diff == [key], diff
        r, *_ = run_bussac(data, acfg)
        assert np.max(np.abs(r["bar1"]["wave_out"] - w)[valid]) > 3 * e0
