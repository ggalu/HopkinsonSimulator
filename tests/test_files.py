"""Tests of the output files and of the analysis case: the settings TOML and the records CSV survive a write
and read, the analysis refuses an incomplete case and names the keys, and the comparison with the simulation
settings marks exactly the settings that differ, and the simulator prints every setting of a case."""
import io
import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

from bussac_analysis import check_analysis_config, compare_settings, print_comparison
from hopkinson_simulator import (INTERFACE_KEYS, _setting_text, check_simulation_config, deep_merge, dump_toml,
                                 generate, load_config, print_settings, read_records_csv, simulation_keys,
                                 write_records_csv, write_settings_toml, write_specimen_csv)
from test_bussac import full_acfg

ROOT = Path(__file__).resolve().parents[1]
CASES = {n: load_config(ROOT / "simulation_cases" / f"{n}.toml") for n in ("SHPB", "direct_impact", "SHTB")}
SHORT = {"sampling": {"duration": 300e-6}}


def test_dump_toml_round_trip():
    d = {"a": 1, "b": 2.5e-7, "c": float("inf"), "d": "x \"y\" ü", "e": [0.3, 0.85], "f": True, "g": None,
         "t": {"u": np.float64(3.0), "v": np.array([1, 2]), "w": {"x": -float("inf")}, "y": [[1, 2], [3]]}}
    back = tomllib.loads(dump_toml(d))
    want = {k: v for k, v in d.items() if v is not None}
    want["t"] = {"u": 3.0, "v": [1, 2], "w": {"x": -float("inf")}, "y": [[1, 2], [3]]}
    assert back == want


@pytest.mark.parametrize("name", list(CASES))
def test_output_files_round_trip(name, tmp_path):
    cfg = deep_merge(CASES[name], SHORT)
    case, _ = generate(cfg)
    write_settings_toml(cfg, case["derived"], tmp_path / "s.toml")
    back = tomllib.loads((tmp_path / "s.toml").read_text())
    assert back["derived"]["solve_info"]["iterations"] == case["derived"]["solve_info"]["iterations"]
    del back["derived"]
    assert json.dumps(back, sort_keys=True) == json.dumps(cfg, sort_keys=True)

    write_records_csv(case, tmp_path / "s.csv")
    data = read_records_csv(tmp_path / "s.csv")
    ch = case["channels"]
    assert data["name"] == name and data["channels"]["names"] == ch["names"]
    assert np.array_equal(data["channels"]["bar"], ch["bar"])
    assert abs(data["sampling_rate"] / cfg["sampling"]["rate"] - 1) < 1e-9
    for key in ("records", "records_point"):
        assert np.max(np.abs(data["channels"][key] - ch[key])) < 1e-11 * np.max(np.abs(ch[key]))
    for b in ("bar1", "bar2"):
        for key, _ in INTERFACE_KEYS:
            x = case["interface"][b][key]
            assert np.max(np.abs(data["interface"][b][key] - x)) <= 1e-11 * np.max(np.abs(x)), (b, key)
    # the file reads without the project code
    plain = np.loadtxt(tmp_path / "s.csv", delimiter=",", comments="#", skiprows=7)
    assert plain.shape == (case["time"].size, 1 + 2 * len(ch["names"]) + 2 * len(INTERFACE_KEYS))

    write_specimen_csv(case, tmp_path / "sp.csv")
    header = [ln for ln in (tmp_path / "sp.csv").read_text().splitlines() if not ln.startswith("#")][0]
    assert header.startswith("t[s],stress[Pa],") and header.endswith("source_eps_ref")


def test_analysis_case_complete_and_checked():
    acfg = full_acfg(CASES["SHPB"])
    check_analysis_config(acfg)
    bad = deep_merge(acfg, {"separation": {"etaa": 1.0}})
    del bad["separation"]["valid_margin"]
    del bad["bars"]["material_output"]["nu"]
    with pytest.raises(ValueError) as e:
        check_analysis_config(bad)
    msg = str(e.value)
    for key in ("separation.valid_margin", "separation.etaa", "bars.material_output.nu"):
        assert key in msg, msg
    with pytest.raises(ValueError, match=r"\[impact\] is only used"):
        check_analysis_config({**acfg, "impact": {"velocity": 1.0}})
    no_impact = full_acfg(CASES["direct_impact"])
    del no_impact["impact"]
    with pytest.raises(ValueError, match="missing: impact"):
        check_analysis_config(no_impact)
    with pytest.raises(ValueError, match="f_max"):
        check_analysis_config(deep_merge(acfg, {"separation": {"f_max": "none"}}))
    no_tau = deep_merge(acfg, {})
    del no_tau["bars"]["material_input"]["tau"]
    with pytest.raises(ValueError, match="missing: bars.material_input.tau"):
        check_analysis_config(no_tau)


def test_comparison_marks_differences():
    sim = CASES["direct_impact"]
    acfg = full_acfg(sim)
    rows = compare_settings(acfg, sim)
    assert not [r for r in rows if r["status"] == "differs"]
    assert {r["key"] for r in rows if r["status"] == "analysis only"} >= {"separation.eta", "records"}
    acfg = deep_merge(acfg, {"impact": {"velocity": 6.0}, "gauges": {"gauge_length": 3e-3},
                             "bars": {"material_output": {"E": 2.4e9}}})
    diff = [r["key"] for r in compare_settings(acfg, sim) if r["status"] == "differs"]
    assert diff == ["impact.velocity", "bars.material_output.E", "gauges.gauge_length"], diff
    out = io.StringIO()
    assert print_comparison(compare_settings(acfg, sim), "title", out) == 3
    text = out.getvalue()
    assert sum(row[-1] == "≠ DIFFERS" for row in framed_rows(text)) == 3
    assert "3 settings differ" in text and "How to read the table" in text and "\033[" not in text


ALL_CASES = sorted(p.stem for p in (ROOT / "simulation_cases").glob("*.toml"))


@pytest.mark.parametrize("name", ALL_CASES)
def test_example_cases_are_complete(name):
    check_simulation_config(load_config(ROOT / "simulation_cases" / f"{name}.toml"))


def test_simulation_case_complete_and_checked():
    """A missing, unknown, wrongly typed, misplaced or removed key is refused, and all of them are named at once."""
    bad = deep_merge(CASES["SHPB"], {"sampling": {"n_subs": 3, "n_sub": 2.5}, "colour": "red",
                                     "impact": {"velocity": 5.0}, "pulse": {"preset": "sharp"},
                                     "specimen": {"n_cells": 40, "material": {"Et": 1e9}}})
    del bad["bars"]["material_output"]["tau"]
    with pytest.raises(ValueError) as e:
        check_simulation_config(bad)
    msg = str(e.value)
    for text in ("missing: bars.material_output.tau (number)", "unknown key: sampling.n_subs",
                 "sampling.n_sub = 2.5: expected integer", "unknown key: colour",
                 'impact is used only with setup = "direct_impact"',
                 'specimen.n_cells is used only with specimen.kind = "fd"',
                 "pulse.preset is no longer a setting", "specimen.material.Et is no longer a setting"):
        assert text in msg, (text, msg)
    with pytest.raises(ValueError, match="unknown key: sampling.n_subs"):
        generate(bad)                                    # the simulation itself checks the case


def test_fd_specimen_needs_its_options():
    fd = deep_merge(CASES["SHPB"], {"specimen": {"kind": "fd"}})
    with pytest.raises(ValueError, match="missing: specimen.n_cells"):
        check_simulation_config(fd)
    check_simulation_config(deep_merge(fd, {"specimen": {"n_cells": 40, "cfl": 0.5, "viscosity": 0.3,
                                                         "rate_filter_time": 0.5e-6, "contact_tolerance": 1e-6}}))


def framed_rows(text: str) -> list[list[str]]:
    """The cells of the rows of a framed table (without the header row). A row whose wrapped cells continue on
    the next line, with the first cell empty there, is joined into one row."""
    rows = []
    for ln in text.splitlines():
        if not ln.startswith("│"):
            continue
        cells = [c.strip() for c in ln.strip("│").split("│")]
        if rows and not cells[0]:
            rows[-1] = [(a + ("" if a.endswith("-") else " ") + b).strip() if b else a
                        for a, b in zip(rows[-1], cells)]
        else:
            rows.append(cells)
    return rows[1:]


@pytest.mark.parametrize("name", ALL_CASES)
def test_print_settings_lists_every_key_with_its_comment(name, monkeypatch):
    monkeypatch.setenv("COLUMNS", "100")              # the comments are wrapped to this width
    cfg = load_config(ROOT / "simulation_cases" / f"{name}.toml")
    buf = io.StringIO()
    print_settings(cfg, f"{name}.toml", buf)
    text = buf.getvalue()
    assert "\033[" not in text and max(len(ln) for ln in text.splitlines()) <= 100

    def expected(d, sp, prefix=""):
        rows = [[k, _setting_text(v), sp[k].comment] for k, v in d.items() if not isinstance(v, dict)]
        for k, v in d.items():
            if isinstance(v, dict):
                rows += [[f"[{prefix}{k}]", "", sp[k].comment], *expected(v, sp[k], f"{prefix}{k}.")]
        return rows
    assert framed_rows(text) == expected(cfg, simulation_keys(cfg))
