# HopkinsonSimulator

A simulator for split Hopkinson bar experiments, and a Bussac wave separation that analyses its records.
The simulator produces strain-gauge records together with the exact waves at the bar–specimen interfaces,
so that a wave-separation method can be checked against a known answer. Three setups are covered: the split
Hopkinson pressure bar (SHPB), the direct-impact pressure bar, and the split Hopkinson tension bar (SHTB). Each
example case has two strain gauges on the input bar and two on the output bar.

The two programs are separate. `hopkinson_simulator.py` reads a simulation case and writes a simulation file.
`bussac_analysis.py` reads an analysis case, which names a simulation file and the settings of the separation,
and compares the separated waves with the stored truth.

The code was extracted from the wavePINN project on 2026-11-10. The simulator and the separation of the
pressure setups are unchanged; on the SHPB and direct-impact cases both programs give results identical to
those of wavePINN. The specimen of the tension bar is new (see "SHTB").

## Layout
```
hopkinson_simulator.py   bar models, propagation, pulses, specimens, the three setups, file format, figure
bussac_analysis.py       Bussac separation, errors against the truth, figure
simulation_cases/        SHPB.toml, direct_impact.toml, SHTB.toml
analysis_cases/          SHPB.toml, direct_impact.toml, SHTB.toml
tests/                   pytest tests of both programs
results/                 output (not in git)
```

## Running
The programs need Python 3.11 or later with numpy, scipy, matplotlib and h5py (pytest for the tests). In the
wavePINN conda environment:
```bash
python hopkinson_simulator.py simulation_cases/SHPB.toml          # -> results/SHPB/
python bussac_analysis.py analysis_cases/SHPB.toml                # -> results/SHPB_bussac/
python hopkinson_simulator.py simulation_cases/*.toml --headless  # all cases, figures saved but not shown
python bussac_analysis.py analysis_cases/*.toml --headless
python -m pytest -q                                               # 20 tests, about 50 s
```
Without `--headless` the figures are also shown, and the program ends when all windows are closed.
`--out <dir>` changes the results root.

| program | output in `results/<name>/` |
|---|---|
| simulator | `config.toml` (copy of the case file), `simulation.h5` (records and truth), `waves.png` (gauge records and true interface waves) |
| analysis | `bussac.json` (settings, assumed bars and gauges, end of the valid window, errors), `bussac.png` (records, separated waves, force and velocity against the truth), `separated.txt` (time and the separated quantities of both bars) |

## How the simulator works
The bars are not discretised in space. The simulator solves for the two strain waves at each bar–specimen
interface, as time histories: the incoming wave a, which travels toward the specimen, and the outgoing wave b,
which travels away from it. The interface strain is a + b. Each wave is moved along a bar in the frequency
domain with the transfer function exp(−i k(ω) x), where k(ω) is the dispersion relation of the bar model:
A, the 1D wave equation; B, the Love rod; or C, the first mode of Pochhammer–Chree theory.

A case is solved by a fixed-point iteration. In each pass the incoming waves are formed (from the loading and
from the echoes of earlier outgoing waves at the free far ends), the specimen is solved in the time domain with
these incoming waves, and force equilibrium at each interface gives the outgoing waves, b = N/(E A) − a. Each
pass makes the solution exact for one more round trip in the bars, so the iteration ends after a fixed number of
passes (the change between passes falls below 10⁻⁸).

The gauge records are then computed from the interface waves: each wave is propagated from where it is defined
to the gauge, averaged over the gauge length, and the contributions are added. Interface velocity and forces
come from the same waves with the operators of the bar model: v = ∓(ω/k) ε for each wave, A·E·ε, and
A·σ_eff = A ρ (ω/k)² ε. The stored waves are therefore exactly the waves that produced the gauge records.

One part is not at the level of the dispersive bar models: the coupling between bar and specimen (and, in the
SHTB, between bar and flange). The specimen sees each bar through the relations of the non-dispersive bar,
N = E A (a + b) and v = c0 (∓a ± b), that is through the impedance ρ c0 A. This affects only how realistic the
specimen response is for dispersive bars; it does not affect the consistency between the stored waves and the
gauge records.

Two specimen models are available. The lumped model assumes uniform stress (force equilibrium, no inertia).
The finite-difference model resolves wave propagation in the specimen, for cases without equilibrium. Both use
an elastic–plastic law with linear hardening and an optional logarithmic strain-rate dependence, with
engineering stress and strain. In the pressure setups the specimen faces are in unilateral contact with the
bars by default (`tension_cutoff = true`): a face separates instead of carrying tension.

### The three setups
- **SHPB** (`setup = "shpb"`). The incident strain pulse is prescribed at a source position on the input bar
  (a trapezoid with smooth ramps; the example uses a sharp rise of 4 µs, which excites strong dispersion).
- **Direct impact** (`setup = "direct_impact"`). The input bar is the projectile. It moves at v0 with zero
  strain and hits the specimen at a given time. For the specimen this is the same as a bar at rest that receives
  an extra step wave −v0/(2 c0); the gauges see only the strain waves, and v0 is added to the input-bar velocity.
  The example has a polycarbonate output bar and a polymer-like specimen.
- **SHTB** (`setup = "shtb"`). A tubular striker hits a flange at the far end of the input bar; the specimen is
  bonded between the bars.

### SHTB
The loading end is solved in the time domain. The flange is a rigid mass. The striker is a tube moving at its
initial velocity, with its own strain waves on top of the rigid motion; it can separate from the flange and hit
it again, and its free far end returns its waves with the sign flipped. The incident wave in the input bar is
therefore not prescribed: it is the wave that leaves the loading end, stored as `/source/eps_ref`.

The specimen is either specimen model with `tension_cutoff = false`: its faces are bonded to the bars and carry
tension. The program stops with an error if the case file asks for unilateral contact. The interface waves are
defined at the specimen faces, and gauge positions are distances from the specimen face, as in the pressure
setups. This specimen coupling is new; in wavePINN the two bars of the tension setup were joined by a threaded
coupler. It is checked by two tests with model A: with a very stiff specimen the incident wave from the loading
end arrives unchanged in the output bar, and an elastic specimen transmits the incident wave exactly as a
massless spring of compliance C = Ls/(Es As), with T(ω) = 1/(1 + iωZC/2).

In the example (`simulation_cases/SHTB.toml`) a POM striker of 0.8 m at 5 m/s produces an incident strain of
about 4.3·10⁻⁴ for about 1.1 ms. The specimen (10 mm × Ø4 mm, an annealed aluminium) yields; its strain rate is
about 260 /s, and its plastic strain reaches about 0.43 at the end of the 2 ms record. The specimen dimensions
and the striker material are assumptions. The specimen model has no necking or failure.

## Simulation case file
One self-contained, commented TOML file per case. The tables are: `[bars]` and `[bars.material]` (optional
`[bars.material_transmitted]`), `[gauges]`, the loading (`[source]` and `[pulse]` for the SHPB, `[impact]` for
the direct impact, `[striker]`, `[striker.material]` and `[flange]` for the SHTB), `[specimen]` and
`[specimen.material]`, `[sampling]` and `[coupling]`. Every parameter is commented in the example files. To make a
new case, copy one, rename it and edit it. SI units throughout; strain, stress and force are positive in tension.

## Simulation file (`simulation.h5`, format `hopkinson-1.0`)
Global x points from the input bar towards the output bar; displacements and velocities are positive in +x.
Gauge positions are distances from the respective bar–specimen interface into the bar.

| path | meaning |
|---|---|
| attributes `format_version`, `kind`, `name`, `sampling_rate` | `hopkinson-1.0`, `synthetic`, case name, Hz |
| attribute `metadata_json` | JSON: `config` (the case file as read) and `derived` (bar models, cut-off frequency, solver information, …) |
| `/time` | time [s], uniform |
| `/channels/records` | gauge strain, averaged over the gauge length if `[gauges] averaging = true` |
| `/channels/records_point` | strain at the gauge centre |
| `/channels/names`, `bar`, `distance_from_interface`, `gauge_length` | `I1, I2, T1, T2`; 1 = input, 2 = output bar; [m]; [m] |
| `/source/eps_ref`, `/source/distance_from_interface` | the right-going wave where it is defined: the prescribed pulse (SHPB), zero (direct impact), the wave leaving the loading end (SHTB) |
| `/interface/bar{1,2}/wave_in`, `wave_out` | strain wave arriving at and leaving the interface (bar 1: wave_in travels +x; bar 2: wave_in travels −x) |
| `/interface/bar{1,2}/strain`, `velocity` | wave_in + wave_out; particle velocity (model-consistent; direct impact: bar 1 includes v0) |
| `/interface/bar{1,2}/force_E`, `force_eff` | A·E·ε and A·ρ(ω/k)²ε (model B: exactly A·σ_eff of Love theory; model A: equal to force_E) |
| `/specimen/stress`, `strain`, `strain_rate`, `plastic_strain` | specimen model output (engineering, tension +) |
| `/specimen/force_face{1,2}`, `velocity_face{1,2}`, `contact_open_face{1,2}` | forces and velocities as seen by the specimen model; 1 where a face is separated from its bar |
| `/specimen/contact_force`, `flange_velocity`, `striker_wave_in`, `striker_wave_out`, `contact_open` | SHTB only: the loading end (contact force compression +; 1 where the striker is separated from the flange) |

## Analysis case file
```toml
name = "SHPB_bussac"                       # results/<name>/
simulation = "results/SHPB/simulation.h5"  # relative to the project root
records = "records"                        # or "records_point"
[separation]
eta = 1000.0                               # Laplace damping [1/s]
# f_max = 200.0e3                          # band limit [Hz]; default 0.8 x the Love cut-off
gauge_length_correction = true
valid_margin = 20.0e-6
[bars]                                     # optional: what the analysis assumes
# bar_model = "A"
# material = { E = 71.7e9, rho = 2810.0, nu = 0.33, radius = 8.0e-3 }
[gauges]                                   # optional: assumed gauge positions [m]
# incident = [0.301, 0.85]
```
By default the analysis assumes the bar model, materials and gauge positions of the simulation. The `[bars]`
and `[gauges]` tables override them, so that the effect of a wrong model or a wrong gauge position can be
studied on the same simulation file.

The separation is the frequency-domain least-squares method of Bussac, Collet, Gary and Othman (J. Mech. Phys.
Solids 50, 2002, 321–349). The spectra are evaluated at the complex frequency ω − iη, and the wavenumber is
continued analytically to k(ω − iη); the Laplace damping η removes the singularities of a two-gauge separation
at zero frequency and at the frequencies n c/(2D) of two gauges a distance D apart. The price is that errors
late in the record are amplified by about exp(η t).

The errors in `bussac.json` are the maximum and the rms of separated minus true quantity over the valid
window, t < T − x_max/c0 − valid_margin (the outgoing wave at time t needs gauge data up to x_max/c0 later).
For the SHPB and the SHTB they are relative to the peak incident wave, the peak input-bar velocity and the peak
input-bar force. For the direct impact each bar is scaled by its own peaks, because the input-bar strain is much
smaller than the output-bar strain.

### Results of the example cases
With the default settings on the gauge-averaged records (model B), the largest relative error of the separated
waves within the valid window is 1.1·10⁻² for the SHPB and the direct impact, and 5.9·10⁻³ for the SHTB. The
errors of A·σ_eff are larger (up to 0.30 for the direct impact), because this operator weights the high
frequencies by (ω/k)², where the separation is least accurate. These numbers rest on single runs of
noise-free records and depend on η and f_max.

For the SHTB, the error grows towards the end of the record because the records still end at 20–30 % of their
peak (the incident pulse is long). With model A, where the separation should be exact, the error at 1.5 ms is
about 4·10⁻⁴ with a 2 ms record and about 6·10⁻⁵ with a 3 ms record. A longer record, or a larger η, moves this
error further from the times of interest.

## Tests
`tests/test_simulator.py` compares the simulator with closed-form solutions for model A (d'Alembert solution
with an elastic specimen, direct impact, the loading end of the SHTB, and the bonded elastic specimen of the SHTB
as a spring), checks the Love dispersion relation, the Love equation of motion and the phase and group velocity
of a tone burst, checks the Pochhammer–Chree solver against published values and its limits, and checks that
the gauge records stored in a file are the propagations of the stored interface waves.
`tests/test_bussac.py` checks that the separation is exact for model A (SHPB and SHTB) and for model C with the
continued wavenumber, and that the overrides of the analysis case take effect.

## Left out from wavePINN
Measurement artefacts (gain, offset, drift, noise, spikes, clipping) and the preprocessing that removes them,
the threaded coupler with compliant joints, the bored far end of the output bar, holders and connectors, a
specimen force read from a file, and the identification of setup parameters from experiments.
