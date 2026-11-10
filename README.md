# HopkinsonSimulator

A simulator for split Hopkinson bar experiments, and a Bussac wave separation that analyses its records.
The simulator produces strain-gauge records together with the exact waves at the bar–specimen interfaces,
so that a wave-separation method can be checked against a known answer. Three setups are covered: the split
Hopkinson pressure bar (SHPB), the direct-impact pressure bar, and the split Hopkinson tension bar (SHTB). Each
example case has two strain gauges on the input bar and two on the output bar.

The two programs are separate, and all their files are plain text. `hopkinson_simulator.py` reads a simulation
case and writes the gauge records with the true interface waves (a CSV file) and its settings (a TOML file).
`bussac_analysis.py` reads an analysis case, which defines every setting the analysis uses, separates the
records and compares the separated waves with the stored truth. It reads the simulation settings only to show
where its own settings differ from them.

The code was extracted from the wavePINN project on 2026-11-10. The simulator and the separation of the
pressure setups are unchanged. On the SHPB and direct-impact cases both programs gave results identical to those
of wavePINN while the records were stored in HDF5. The text files store 13 significant digits, which changes the
errors of the analysis by less than 10⁻⁹ of their values. The specimen of the tension bar is new (see "SHTB").

The methods (the spectral propagator of the simulator, the bar models and the Bussac separation), with two worked
examples (SHPB and SHTB), are described in the report [reports/theory/report.tex](reports/theory/report.tex).
Build the PDF with `make` in `reports/theory/` (needs latexmk); `python reports/theory/make_figures.py` remakes
its figures and the numbers it quotes.

## Layout
```
hopkinson_simulator.py   bar models, propagation, pulses, specimens, the three setups, output files, figure
bussac_analysis.py       analysis case check, comparison of settings, Bussac separation, errors, figure
simulation_cases/        SHPB.toml, direct_impact.toml, direct_impact_viscoelastic.toml, SHTB.toml
analysis_cases/          the same four names
tests/                   pytest tests of both programs
reports/theory/          report on the methods (LaTeX) and the script of its figures
reports/template/        LaTeX style of the reports
results/                 output (not in git)
```

## Running
The programs need Python 3.11 or later with numpy, scipy and matplotlib (pytest for the tests). In the
wavePINN conda environment:
```bash
python hopkinson_simulator.py simulation_cases/SHPB.toml          # -> results/SHPB/
python bussac_analysis.py analysis_cases/SHPB.toml                # -> results/SHPB_bussac/
python hopkinson_simulator.py simulation_cases/*.toml --headless  # all cases, figures saved but not shown
python bussac_analysis.py analysis_cases/*.toml --headless
python -m pytest -q                                               # 42 tests, about 50 s
```
Without `--headless` the figures are also shown, and the program ends when all windows are closed.
`--out <dir>` changes the results root.

| program | output in `results/<name>/` |
|---|---|
| simulator | `simulation.toml` (settings), `simulation.csv` (gauge records and interface truth), `specimen.csv` (specimen, source and loading-end histories), `waves.png` (gauge records and true interface waves) |
| analysis | `bussac.json` (analysis case, comparison with the simulation settings, end of the valid window, errors, force imbalance), `bussac.png` (records, separated waves, force and velocity against the truth), `equilibrium.png` (forces at the two interfaces and their imbalance), `separated.txt` (time and the separated quantities of both bars) |

## How the simulator works
The bars are not discretised in space. The simulator solves for the two strain waves at each bar–specimen
interface, as time histories: the incoming wave a, which travels toward the specimen, and the outgoing wave b,
which travels away from it. The interface strain is a + b. Each wave is moved along a bar in the frequency
domain with the transfer function exp(−i k(ω) x), where k(ω) is the dispersion relation of the bar model:
A, the 1D wave equation; B, the Love rod; or C, the first mode of Pochhammer–Chree theory.

### Viscoelastic bars (Kelvin–Voigt)
A bar of model A or B can be viscoelastic: a material table with `tau` > 0 makes it a Kelvin–Voigt solid, a
spring E in parallel with a dashpot of viscosity E τ. The stress is σ = E (ε + τ ε̇), that is the complex
modulus E (1 + iωτ); τ is a retardation time, and the loss tangent is ωτ. In the dispersion relation c0² becomes
c0² (1 + iωτ), so the wavenumber is complex with Im k < 0 and every frequency component decays as it travels.
For ωτ ≪ 1 (model A) the attenuation is ω²τ/(2 c0) and the phase velocity c0 (1 + ⅜ (ωτ)²). With `tau = 0`, the
default, the bar is elastic and the results are those of the elastic models. Model C is elastic only and
refuses `tau` > 0.

Propagation, gauge averaging, interface velocity and force need no change, because they use only k(ω); for
model A, A ρ (ω/k)² ε is exactly the Kelvin–Voigt force A E (ε + τ ε̇). The specimen still sees the bar through
the elastic impedance ρ c0 A, with c0 from E.

The model has one parameter, so it can match a real bar near one frequency only: its loss tangent grows in
proportion to frequency, while that of real polymers varies slowly over many decades. A choice is
τ = tan δ / ω at the main frequency of the pulse. For ωτ ≫ 1 the phase velocity grows without bound, so the bar
has no sharp wave front; these components are strongly damped. The example case
`direct_impact_viscoelastic.toml` gives the polycarbonate output bar τ = 10⁻⁷ s, a placeholder and not a
measured value.

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
One self-contained, commented TOML file per case. The tables are: `[bars]` (with the bar lengths
`length_input` and `length_output`) and `[bars.material_input]` (optional
`[bars.material_output]`, which defaults to the input-bar material), `[gauges]` (with the gauge positions
`input` and `output`), the loading (`[source]` and `[pulse]` for the SHPB, `[impact]` for
the direct impact, `[striker]`, `[striker.material]` and `[flange]` for the SHTB), `[specimen]` and
`[specimen.material]`, `[sampling]` and `[coupling]`. A bar material may contain `tau`, the Kelvin–Voigt
retardation time (see above). Every parameter is commented in the example files. To make a
new case, copy one, rename it and edit it. SI units throughout; strain, stress and force are positive in tension.

## Simulation output
The simulator writes three text files into `results/<name>/`. Global x points from the input bar towards the
output bar; displacements and velocities are positive in +x. Strain, stress and force are positive in tension.
Gauge positions are distances from the respective bar–specimen interface into the bar. SI units throughout.

**`simulation.toml`** holds the case file as it was read (without its comments) and a `[derived]` table of
quantities computed from it: for each bar the model, the material, c0, the radius of gyration r_g and the
cut-off frequency of model B; the specimen; for the SHTB the striker, the flange mass and the nominal incident
strain; round-trip times; and the solver information (number of coupling passes, the change in each pass).

**`simulation.csv`** (format `hopkinson-csv-1.0`) is the file the analysis reads. It begins with six comment
lines that start with `#`: the format name, the case name and sampling rate, the units, the sign convention and
the meaning of the columns. One header row and one row per sample follow, comma-separated, with 13 significant
digits. Units are given in brackets in the header where a column is not a strain. The file can be read without
the project code, for example with `numpy.loadtxt(path, delimiter=",", skiprows=7)`.

| column | meaning |
|---|---|
| `t[s]` | time, uniform |
| `I1, I2, O1, O2` | gauge strain, averaged over the gauge length if `[gauges] averaging = true`; I: input bar, O: output bar, numbered in the order of `[gauges] input` and `output` |
| `I1_point, …, O2_point` | strain at the gauge centre |
| `bar{1,2}_wave_in`, `wave_out` | strain wave arriving at and leaving the interface (bar 1: wave_in travels +x; bar 2: wave_in travels −x) |
| `bar{1,2}_strain`, `velocity[m/s]` | wave_in + wave_out; particle velocity (consistent with the bar model; direct impact: bar 1 includes v0) |
| `bar{1,2}_force_E[N]`, `force_eff[N]` | A·E·ε and A·ρ(ω/k)²ε (model B: exactly A·σ_eff of Love theory; model A: equal to force_E) |

The gauge positions and the gauge length are not in this file. They are settings, and they are in
`simulation.toml` and in the analysis case.

**`specimen.csv`** has the same layout. The analysis does not read it.

| column | meaning |
|---|---|
| `stress[Pa]`, `strain`, `strain_rate[1/s]`, `plastic_strain` | specimen model output (engineering, tension +) |
| `force_face{1,2}[N]`, `velocity_face{1,2}[m/s]`, `contact_open_face{1,2}` | forces and velocities as seen by the specimen model; 1 where a face is separated from its bar |
| `contact_force[N]`, `flange_velocity[m/s]`, `striker_wave_out`, `striker_wave_in`, `contact_open` | SHTB only: the loading end (contact force compression +; 1 where the striker is separated from the flange) |
| `source_eps_ref` | the right-going wave where it is defined: the prescribed pulse (SHPB), zero (direct impact), the wave leaving the loading end (SHTB) |

## Analysis case file
Every key is required; the analysis takes no setting from the simulation. The example files in
`analysis_cases/` carry the values of the matching simulation cases and comment every key.
```toml
name = "SHPB_bussac"                         # results/<name>/
records = "results/SHPB/simulation.csv"      # records file, relative to the project root
settings = "results/SHPB/simulation.toml"    # simulation settings, read only for the comparison
channels = "averaged"                        # "averaged" or "point" (strain at the gauge centres)
setup = "shpb"                               # "shpb", "direct_impact" or "shtb"
# [impact]                                   # required for setup = "direct_impact", refused otherwise
# velocity = 5.0                             # impact velocity v0 [m/s]
[bars]
bar_model = "B"                              # A, B or C
material_input = { E = 71.7e9, rho = 2810.0, nu = 0.33, radius = 8.0e-3 }        # input bar
material_output = { E = 71.7e9, rho = 2810.0, nu = 0.33, radius = 8.0e-3 }       # output bar
                                             # optional: inner_radius [m] (tube) and tau [s] (Kelvin-Voigt); default 0
[gauges]
input = [0.3, 0.85]                          # [m from the interface], in the order I1, I2, …
output = [0.3, 0.85]                         # in the order O1, O2, …
gauge_length = 2.0e-3                        # [m]
[separation]
eta = 1000.0                                 # Laplace damping [1/s]
f_max = "auto"                               # band limit [Hz], inf, or "auto" (see below)
gauge_length_correction = true
valid_margin = 20.0e-6                       # [s]
```
A missing, unknown or wrongly typed key stops the program, with a message that names all such keys. The
number of positions in `[gauges]` must match the number of I and O columns in the records file. The
keys have the same names in the simulation and the analysis case files. A simulation case that still uses the
former names (`length_incident`, `length_transmitted`, `material`, `material_transmitted`, `incident`,
`transmitted`) is refused with a message that names
the new keys.

On start, the analysis prints a table of its settings next to the simulation settings. Each row has the status
"same", "≠ DIFFERS" or "analysis only" (for the file paths and the `[separation]` keys, which have no
counterpart in the simulation). A short explanation of these statuses is printed under the table. Numbers count
as the same when they agree to a relative 10⁻¹². On a terminal the
rows that differ are drawn in bold red. A summary line gives the number of differences. The analysis then runs
with its own values, so that the effect of a wrong model, material or gauge position can be studied on the same
records. The table is also stored in `bussac.json` under `comparison`.

The separation is the frequency-domain least-squares method of Bussac, Collet, Gary and Othman (J. Mech. Phys.
Solids 50, 2002, 321–349). The spectra are evaluated at the complex frequency ω − iη, and the wavenumber is
continued analytically to k(ω − iη); the Laplace damping η removes the singularities of a two-gauge separation
at zero frequency and at the frequencies n c/(2D) of two gauges a distance D apart. The price is that errors
late in the record are amplified by about exp(η t).

The band limit `f_max = "auto"` is 0.8 times the Love cut-off for model B and none for models A and C. For a
Kelvin–Voigt bar it is in addition at most the frequency at which exp(κ x_max) reaches 10³, where κ = −Im k and
x_max is the farthest gauge. The reason is that the wave leaving the interface is damped by exp(−κ x) on its way
to a gauge, so its reconstruction at the interface multiplies by exp(+κ x). This loss of information is a
property of the attenuating bar, not of the method. The value 10³ is a choice for noise-free synthetic records:
on the example cases the separation stayed usable up to an amplification of about 10³–10⁴ and became unstable
above. Measured records, with noise, need a lower band limit. The band limit used is stored in `bussac.json`.

### Force equilibrium
A second figure, `equilibrium.png` (its own window without `--headless`), lets the user judge whether the
specimen was in force equilibrium. Panel A shows the separated forces A·E·ε at the input-bar and the output-bar
interface, with the true forces drawn faintly behind them. A·E·ε is the force through which the simulated
specimen is coupled to the bars. Panel B shows the relative imbalance (N_input − N_output)/N_mean, with
N_mean = (N_input + N_output)/2, for the separated and the true forces, and lines at ±5 % as a guide. The ratio is
shown only where |N_mean| is at least 5 % of its peak in the valid window, because it has no meaning while the
forces are small. Its largest value in the valid window is printed and stored in `bussac.json` under
`equilibrium`. This largest value usually comes from the loading front and the unloading, where one face carries
much more force than the other, so the curve over time is the better guide to whether equilibrium was reached
during the test.

The lumped specimen assumes uniform stress, so its two face forces are equal by construction: for the example
cases the true imbalance is zero, and the separated imbalance shows only the error of the separation (0.1 % for
the SHPB, 3.9 % for the direct impact). The finite-difference specimen (`kind = "fd"`) resolves the stress wave in
the specimen, so a long specimen shows a real imbalance on the rise of the load, which then decays.

The errors in `bussac.json` are the maximum and the rms of separated minus true quantity over the valid
window, t < T − x_max/c0 − valid_margin (the outgoing wave at time t needs gauge data up to x_max/c0 later).
For the SHPB and the SHTB they are relative to the peak incident wave, the peak input-bar velocity and the peak
input-bar force. For the direct impact each bar is scaled by its own peaks, because the input-bar strain is much
smaller than the output-bar strain.

### Results of the example cases
With the settings of the example analysis cases on the gauge-averaged records (model B), the largest relative error of the separated
waves within the valid window is 1.1·10⁻² for the SHPB and the direct impact, and 5.9·10⁻³ for the SHTB. The
errors of A·σ_eff are larger (up to 0.30 for the direct impact), because this operator weights the high
frequencies by (ω/k)², where the separation is least accurate. These numbers rest on single runs of
noise-free records and depend on η and f_max.

For the SHTB, the error grows towards the end of the record because the records still end at 20–30 % of their
peak (the incident pulse is long). With model A, where the separation should be exact, the error at 1.5 ms is
about 4·10⁻⁴ with a 2 ms record and about 6·10⁻⁵ with a 3 ms record. A longer record, or a larger η, moves this
error further from the times of interest.

For the viscoelastic variant of the direct impact (τ = 10⁻⁷ s on the output bar), the band limit "auto" of the
output bar is 66 kHz instead of 83 kHz, and the largest error of the separated output-bar waves is 1.6·10⁻²,
against 1.1·10⁻² for the elastic case. An analysis that ignores the attenuation (τ = 0, with the band limit of
the elastic bar) gives 2.8·10⁻². Against the full truth, a band limit always leaves the error of the missing
high frequencies; against the truth limited to the same band, the separation of Kelvin–Voigt records with the
right τ agrees to 10⁻⁵ or better (test below). These numbers rest on single runs of noise-free records.

## Tests
`tests/test_simulator.py` compares the simulator with closed-form solutions for model A (d'Alembert solution
with an elastic specimen, direct impact, the loading end of the SHTB, and the bonded elastic specimen of the SHTB
as a spring), checks the Love dispersion relation, the Love equation of motion and the phase and group velocity
of a tone burst, checks the Pochhammer–Chree solver against published values and its limits, and checks that
the gauge records stored in a file are the propagations of the stored interface waves.
`tests/test_bussac.py` checks that the separation is exact for model A (SHPB and SHTB) and for model C with the
continued wavenumber, and that the settings of the analysis case take effect and are reported when they differ.
`tests/test_files.py` checks that the settings TOML and the records CSV of all three setups are read back as
written, that an incomplete analysis case is refused with the names of the keys, and that the comparison marks
exactly the settings that differ.
`tests/test_viscoelastic.py` checks the Kelvin–Voigt wavenumber (Im k < 0, the expansions for small ωτ, identity
with the elastic models for τ = 0), the force operator, the decay of a pulse, the exact composition of the
transfer functions, the analytic continuation used by the analysis, the band limit "auto", and the separation
of Kelvin–Voigt records against the truth limited to the same band.

## Left out from wavePINN
Measurement artefacts (gain, offset, drift, noise, spikes, clipping) and the preprocessing that removes them,
the threaded coupler with compliant joints, the bored far end of the output bar, holders and connectors, a
specimen force read from a file, and the identification of setup parameters from experiments.

## License
MIT; see [LICENSE](LICENSE).
