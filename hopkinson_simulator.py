"""Hopkinson bar simulator: synthetic strain-gauge records with exact interface ground truth.

    python hopkinson_simulator.py simulation_cases/SHPB.toml [more.toml ...] [--out results] [--headless]

Each case writes to <out>/<name>/ (name from the case file, default: the file name):
    simulation.toml  the case settings as read, and a [derived] table computed from them
    simulation.csv   gauge records and the true interface waves (the file the analysis reads)
    specimen.csv     specimen histories, the source wave and (SHTB) the loading end
    waves.png        gauge records and true interface waves (left: input bar, right: output bar)

Three setups (``setup`` in the case file):
    "shpb"           split Hopkinson pressure bar; the incident strain pulse is prescribed at a source position
    "direct_impact"  the input bar is the projectile and hits the specimen at a given velocity
    "shtb"           split Hopkinson tension bar; a tubular striker hits a flange at the far end of the input
                     bar, and the specimen is bonded between the two bars

The bars are not discretised in space. The simulator solves for the two strain waves at each bar-specimen
interface, as time histories: a (incoming, travels toward the specimen) and b (outgoing). Waves are moved
along a bar in the frequency domain with exp(-i k(omega) x), with the dispersion relation of model A
(1D wave equation), B (Love rod) or C (Pochhammer-Chree first mode); models A and B can be viscoelastic
(Kelvin-Voigt, material key tau). The specimen is solved in the time
domain; the bars and the specimen are coupled by a fixed-point iteration in which each pass makes the
solution exact for one more round trip in the bars. The gauge records are the propagations of the stored
interface waves to the gauge positions, so the stored waves are the exact reference for wave separation.

The code was extracted from the wavePINN project on 2026-11-10; the sections below name their origin.
SI units throughout. Strain, stress and force are positive in tension; global x points from the input bar
toward the output bar; gauge positions are distances from the bar-specimen interface into each bar.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import tomllib
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import interpolate, optimize, special

ROOT = Path(__file__).resolve().parent

# ============================================================================================================
# Bar models: dispersion relations (from wavePINN src/wavepinn/dispersion.py)
# ============================================================================================================
# Bar models: dispersion relations k(omega) for longitudinal waves in a circular rod.
#
# Conventions
# -----------
# * FFT convention X(omega) = sum x(t) exp(-i omega t)  (numpy.fft.rfft).
# * A wave travelling a distance s >= 0 in its own direction of propagation is
#   multiplied by H = exp(-i k(omega) s).  For propagating components k is real
#   and positive (omega > 0), which is a pure delay s/c(omega).  For evanescent
#   components (Love theory above its cut-off) k = -i*kappa with kappa > 0, so H
#   decays as exp(-kappa s).
# * Only omega >= 0 is used (real signals, rfft).
#
# Kelvin-Voigt bar (models A and B, material key ``tau`` > 0): the stress is
# sigma = E (eps + tau eps_t), i.e. the complex modulus E (1 + i omega tau), a spring E
# in parallel with a dashpot of viscosity E tau. tau is a retardation time; the loss
# tangent is omega tau. In the dispersion relation c0^2 becomes c0^2 (1 + i omega tau), so
# k is complex with Im k < 0 at every frequency and every component decays as it travels.
# For omega tau << 1 (model A): attenuation omega^2 tau / (2 c0), phase velocity
# c0 (1 + 3/8 (omega tau)^2). The propagator, the velocity operator and the force operator
# only use ``wavenumber``, so they need no change; for model A, A rho (omega/k)^2 eps is
# exactly the Kelvin-Voigt force A E (eps + tau eps_t).

@dataclass(frozen=True)
class BarMaterial:
    E: float        # Young's modulus [Pa]
    rho: float      # density [kg/m^3]
    nu: float       # Poisson's ratio [-]
    radius: float   # bar radius a [m] (outer radius of a tube)
    inner_radius: float = 0.0   # inner radius of a tube [m]; 0 for a solid bar
    tau: float = 0.0            # Kelvin-Voigt retardation time [s]; 0 for an elastic bar

    @property
    def c0(self) -> float:
        return float(np.sqrt(self.E / self.rho))

    @property
    def area(self) -> float:
        return float(np.pi * (self.radius ** 2 - self.inner_radius ** 2))

    @property
    def impedance(self) -> float:
        """Mechanical impedance rho*c0*A [N s/m]."""
        return self.rho * self.c0 * self.area


class BarModel:
    """Base class. Subclasses implement ``wavenumber(omega)`` for omega >= 0."""

    name = "base"

    def __init__(self, material: BarMaterial):
        self.mat = material

    # --- to be implemented -------------------------------------------------
    def wavenumber(self, omega: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # --- derived operators (frequency domain) ------------------------------
    def phase_velocity(self, omega: np.ndarray) -> np.ndarray:
        """omega/k; c0 at omega = 0; 0 where k is infinite."""
        omega = np.asarray(omega, dtype=float)
        k = self.wavenumber(omega)
        out = np.full(omega.shape, self.mat.c0, dtype=complex)
        nz = omega != 0
        with np.errstate(divide="ignore", invalid="ignore"):
            out[nz] = omega[nz] / k[nz]
        out[~np.isfinite(out)] = 0.0
        return out

    def describe(self) -> dict:
        return {"model": self.name, **asdict(self.mat), "c0": self.mat.c0}


class WaveEq1D(BarModel):
    """Model A: rho u_tt = E u_xx (non-dispersive); Kelvin-Voigt: rho u_tt = E (u_xx + tau u_xxt)."""

    name = "A"

    def wavenumber(self, omega):
        omega = np.asarray(omega, dtype=float)
        if self.mat.tau == 0:
            return omega / self.mat.c0 + 0j
        return omega / (self.mat.c0 * np.sqrt(1 + 1j * omega * self.mat.tau))


class LoveRod(BarModel):
    """Model B: rho u_tt = d/dx [E u_x + rho nu^2 r_g^2 u_xtt], r_g^2 = a^2/2.

    r_g is the polar radius of gyration of the cross-section, r_g^2 = (a^2 + a_i^2)/2 for a tube
    with inner radius a_i (Love's lateral-inertia term).

    Dispersion: c^2/c0^2 = 1/(1 + nu^2 k^2 r_g^2)  <=>  k^2 = omega^2/(c0^2 - nu^2 r_g^2 omega^2).
    Propagating only below omega_c = c0/(nu r_g); evanescent above.

    Kelvin-Voigt (tau > 0): E -> E (1 + i omega tau), k^2 = omega^2/(c0^2 (1 + i omega tau) - nu^2 r_g^2 omega^2).
    The denominator then has a positive imaginary part for omega > 0, so the principal root is continuous
    (no cut-off singularity) and Im k < 0.
    """

    name = "B"

    @property
    def rg(self) -> float:
        return np.sqrt((self.mat.radius ** 2 + self.mat.inner_radius ** 2) / 2.0)

    @property
    def omega_cutoff(self) -> float:
        return self.mat.c0 / (self.mat.nu * self.rg)

    def wavenumber(self, omega):
        omega = np.asarray(omega, dtype=float)
        if self.mat.tau > 0:
            return omega / np.sqrt(self.mat.c0 ** 2 * (1 + 1j * omega * self.mat.tau)
                                   - (self.mat.nu * self.rg * omega) ** 2)
        den = self.mat.c0 ** 2 - (self.mat.nu * self.rg * omega) ** 2
        k = np.empty(omega.shape, dtype=complex)
        prop = den > 0
        evan = den < 0
        k[prop] = omega[prop] / np.sqrt(den[prop])
        k[evan] = -1j * omega[evan] / np.sqrt(-den[evan])
        k[den == 0] = np.inf
        return k

    def describe(self):
        d = super().describe()
        d.update(r_g=self.rg, f_cutoff=self.omega_cutoff / (2 * np.pi))
        return d


# ---------------------------------------------------------------------------
# Model C: Pochhammer-Chree, first longitudinal mode
# ---------------------------------------------------------------------------
def _alpha_funcs(x2):
    """(x J1(x), J0(x)) for x^2 = x2 (even functions of x), scaled by exp(-|Im x|)."""
    if x2 >= 0:
        x = np.sqrt(x2)
        return x * special.j1(x), special.j0(x)
    g = np.sqrt(-x2)
    return -g * special.i1e(g), special.i0e(g)


def _beta_funcs(x2):
    """(J1(x)/x, J0(x)) for x^2 = x2, scaled by exp(-|Im x|)."""
    if x2 > 0:
        x = np.sqrt(x2)
        return special.j1(x) / x, special.j0(x)
    if x2 < 0:
        g = np.sqrt(-x2)
        return special.i1e(g) / g, special.i0e(g)
    return 0.5, 1.0


def pc_residual(cbar: float, xi: float, nu: float) -> float:
    """Pochhammer-Chree frequency equation (Graff 1975, Eq. 8.2.X) for a traction-free rod.

    (2 alpha/a)(beta^2+k^2) J1(alpha a) J1(beta a) - (beta^2-k^2)^2 J0(alpha a) J1(beta a)
        - 4 k^2 alpha beta J1(alpha a) J0(beta a) = 0,
    alpha^2 = omega^2/cL^2 - k^2, beta^2 = omega^2/cT^2 - k^2.

    Written nondimensionally (a = 1, cT = 1, cbar = c/cT, xi = k a), divided by beta
    so that every term is real for real or imaginary alpha, beta.
    """
    kap2 = (1 - 2 * nu) / (2 * (1 - nu))          # (cT/cL)^2
    a2 = xi ** 2 * (cbar ** 2 * kap2 - 1.0)
    b2 = xi ** 2 * (cbar ** 2 - 1.0)
    aJ1, J0a = _alpha_funcs(a2)
    J1b_b, J0b = _beta_funcs(b2)
    return (2 * (b2 + xi ** 2) * aJ1 * J1b_b
            - (b2 - xi ** 2) ** 2 * J0a * J1b_b
            - 4 * xi ** 2 * aJ1 * J0b)


def rayleigh_speed_ratio(nu: float) -> float:
    """c_R / c_T from the Rayleigh equation."""
    kap2 = (1 - 2 * nu) / (2 * (1 - nu))

    def f(eta):
        return (2 - eta ** 2) ** 2 - 4 * np.sqrt(1 - kap2 * eta ** 2) * np.sqrt(1 - eta ** 2)

    return optimize.brentq(f, 0.5, 0.9999999)


@lru_cache(maxsize=16)
def pc_first_mode_table(nu: float, xi_max: float = 100.0, n: int = 6000):
    """First-mode dispersion curve: arrays xi = k a and cbar = c/cT."""
    c0bar = np.sqrt(2 * (1 + nu))                  # c0/cT
    cRbar = rayleigh_speed_ratio(nu)
    xi = np.unique(np.concatenate([np.geomspace(1e-3, 1.0, 600),
                                   np.linspace(1.0, xi_max, n)]))
    cbar = np.empty_like(xi)
    lo = cRbar * 0.97   # mode 1 dips slightly below c_R (min ~0.996 c_R) before approaching it
    for i, x in enumerate(xi):
        hi = c0bar * (1 + 1e-7)   # c(xi) is not monotonic near c_R, so do not track
        # mode 1 = lowest root above the Rayleigh speed: scan upward from lo
        grid = np.linspace(lo, hi, 120)
        vals = np.array([pc_residual(g, x, nu) for g in grid])
        idx = np.nonzero(np.sign(vals[:-1]) * np.sign(vals[1:]) <= 0)[0]
        if idx.size == 0:
            raise RuntimeError(f"PC root not bracketed at xi={x}")
        j = idx[0]
        cbar[i] = optimize.brentq(pc_residual, grid[j], grid[j + 1], args=(x, nu), xtol=1e-15, rtol=1e-14)
    return xi, cbar, c0bar, cRbar


class PochhammerChree(BarModel):
    """Model C: exact first-mode Pochhammer-Chree dispersion (synthetic data only)."""

    name = "C"

    def __init__(self, material: BarMaterial):
        if material.inner_radius > 0:
            raise ValueError("the Pochhammer-Chree model is implemented for solid bars only")
        if material.tau > 0:
            raise ValueError("the Pochhammer-Chree model is elastic; tau > 0 needs bar model A or B")
        super().__init__(material)
        xi, cbar, self.c0bar, self.cRbar = pc_first_mode_table(float(material.nu))
        self.cT = material.c0 / self.c0bar
        Omega = xi * cbar                       # omega a / cT
        if np.any(np.diff(Omega) <= 0):
            raise RuntimeError("PC first-mode omega(k) not monotonic on the table")
        self._Omega_max = Omega[-1]
        self._xi_of_Omega = interpolate.PchipInterpolator(np.r_[0.0, Omega], np.r_[0.0, xi])

    def wavenumber(self, omega):
        omega = np.asarray(omega, dtype=float)
        Om = omega * self.mat.radius / self.cT
        xi = np.empty_like(Om)
        inside = Om <= self._Omega_max
        xi[inside] = self._xi_of_Omega(Om[inside])
        xi[~inside] = Om[~inside] / self.cRbar   # beyond table: Rayleigh-speed asymptote
        return xi / self.mat.radius + 0j

    def phase_velocity_ratio_vs_a_over_lambda(self, a_over_lambda):
        """c/c0 as a function of a/Lambda (for comparison with tables)."""
        xi_t, cbar, *_ = pc_first_mode_table(float(self.mat.nu))
        xi = 2 * np.pi * np.asarray(a_over_lambda, dtype=float)
        return np.interp(xi, xi_t, cbar) / self.c0bar

    def describe(self):
        d = super().describe()
        d.update(c_T=self.cT, c_R=self.cRbar * self.cT)
        return d


MODELS = {"A": WaveEq1D, "B": LoveRod, "C": PochhammerChree}


def make_bar_model(model: str, material: BarMaterial) -> BarModel:
    return MODELS[model](material)


# ============================================================================================================
# Frequency-domain propagation (from wavePINN src/wavepinn/propagation.py)
# ============================================================================================================
# Frequency-domain propagation of 1D waves along a (possibly dispersive) bar.
#
# All signals are sampled on t_n = n*dt, n = 0..N-1, and are treated as zero
# outside the record. Transforms are zero-padded to ``nfft >= pad_factor*N`` so
# that delayed / dispersed content does not wrap around into the record.

def next_pow2(n: int) -> int:
    return 1 << int(np.ceil(np.log2(max(n, 1))))


class Spectral:
    """Padded real FFT on a fixed time grid."""

    def __init__(self, n: int, dt: float, pad_factor: float = 4.0):
        self.n = int(n)
        self.dt = float(dt)
        self.nfft = next_pow2(int(np.ceil(pad_factor * n)))
        self.omega = 2 * np.pi * np.fft.rfftfreq(self.nfft, dt)
        # components with a larger group delay would wrap around the padded window; they
        # arrive after the end of the record and are excluded (see transfer()).
        self.max_group_delay = 0.8 * (self.nfft - self.n) * dt
        # If set, the excluded band is evaluated for this fixed distance (the longest path in the
        # simulation), so that H(a) H(b) = H(a + b) holds exactly for all propagations.
        self.mask_distance = None

    def fwd(self, x):
        return np.fft.rfft(np.asarray(x, dtype=float), self.nfft, axis=-1)

    def inv(self, X):
        return np.fft.irfft(X, self.nfft, axis=-1)[..., : self.n]

    def upsample(self, x, factor: int):
        """Band-limited (trigonometric) interpolation onto dt/factor; returns N*factor samples."""
        X = self.fwd(x)
        nf = self.nfft * factor
        Xf = np.zeros(nf // 2 + 1, dtype=complex)
        Xf[: X.shape[-1]] = X * factor
        Xf[X.shape[-1] - 1] *= 0.5 if self.nfft % 2 == 0 else 1.0   # split the Nyquist bin
        return np.fft.irfft(Xf, nf)[: self.n * factor]


def gauge_average_factor(k: np.ndarray, gauge_length: float) -> np.ndarray:
    """Spatial average of exp(-i k x) over [-L/2, L/2]: sin(kL/2)/(kL/2), valid for complex k."""
    z = k * gauge_length / 2
    out = np.ones_like(z, dtype=complex)
    m = np.abs(z) > 1e-8
    out[m] = np.sin(z[m]) / z[m]
    return out


def group_delay_per_metre(model: BarModel, omega: np.ndarray) -> np.ndarray:
    """dk/domega = 1/c_g for propagating components (inf where k is not finite)."""
    kr = np.real(model.wavenumber(omega))
    with np.errstate(invalid="ignore"):
        g = np.gradient(kr, omega)
    g[~np.isfinite(g)] = np.inf
    return g


def transfer(model: BarModel, omega: np.ndarray, distance: float, gauge_length: float = 0.0,
             max_group_delay: float | None = None, mask_distance: float | None = None) -> np.ndarray:
    """H(omega) for travelling ``distance`` >= 0 in the direction of propagation.

    With ``gauge_length`` > 0 the result is the strain averaged over a gauge of that
    length centred at the end point. Components whose group delay distance/c_g exceeds
    ``max_group_delay`` are set to zero: they cannot arrive within the (padded) window
    and would otherwise wrap around. This matters only for the Love model close to its
    cut-off, where c_g -> 0. Elastic bars: only propagating components (kappa = 0) are
    excluded. Kelvin-Voigt bars (every kappa > 0): components are excluded unless they are
    damped below exp(-30) over the mask distance.
    """
    if distance < 0:
        raise ValueError("propagation distance must be >= 0 (forward in the direction of travel)")
    k = model.wavenumber(omega)
    H = np.zeros(omega.shape, dtype=complex)
    kappa = -np.imag(k)
    ok = np.isfinite(k) & (kappa * distance < 700.0)
    if max_group_delay is not None and distance > 0:
        dm = distance if mask_distance is None else mask_distance
        weak = (kappa == 0) if model.mat.tau == 0 else (kappa * dm < 30.0)
        ok &= ~(weak & (dm * group_delay_per_metre(model, omega) > max_group_delay))
    H[ok] = np.exp(-1j * k[ok] * distance)
    if gauge_length > 0:
        H[ok] *= gauge_average_factor(k[ok], gauge_length)
    return H


def propagate(sp: Spectral, model: BarModel, x, distance: float, gauge_length: float = 0.0):
    return sp.inv(sp.fwd(x) * transfer(model, sp.omega, distance, gauge_length, sp.max_group_delay, sp.mask_distance))


def wave_velocity(sp: Spectral, model: BarModel, strain_wave, direction: int):
    """Particle velocity of a single wave from its strain: v = -direction * (omega/k) * eps.

    direction = +1 for a wave travelling in +x, -1 for -x (global axis).
    """
    return sp.inv(-direction * model.phase_velocity(sp.omega) * sp.fwd(strain_wave))


def force_eff(sp: Spectral, model: BarModel, strain):
    """'Momentum-flux' axial force A*rho*(omega/k)^2*eps.

    Model A: = A E eps.  Model B (Love): = A (E eps + rho nu^2 r_g^2 eps_tt) = A sigma_eff
    exactly.  Model C: 1D-equivalent force consistent with the PC phase velocity.
    """
    m = model.mat
    c = model.phase_velocity(sp.omega)
    return sp.inv(m.area * m.rho * c ** 2 * sp.fwd(strain))


def time_derivative(sp: Spectral, x, order: int = 1):
    return sp.inv((1j * sp.omega) ** order * sp.fwd(x))


# ============================================================================================================
# Prescribed incident pulses (from wavePINN src/wavepinn/pulses.py)
# ============================================================================================================
# Prescribed incident strain pulses (analytic, so they can serve as exact references).

def _ramp(x, kind: str):
    """Monotone ramp 0 -> 1 on x in [0, 1]."""
    x = np.clip(x, 0.0, 1.0)
    if kind == "linear":
        return x
    if kind == "cosine":                       # C1
        return 0.5 - 0.5 * np.cos(np.pi * x)
    if kind == "smoothstep7":                  # C3: 35x^4 - 84x^5 + 70x^6 - 20x^7
        return x ** 4 * (35 - 84 * x + 70 * x ** 2 - 20 * x ** 3)
    raise ValueError(f"unknown ramp {kind}")


@dataclass
class TrapezoidPulse:
    """Trapezoid with smooth ramps. Times in s, amplitude in strain (signed, tension +).

    rise / fall are the full 0-100 % ramp durations. The pulse starts at t_start.
    """
    amplitude: float = -1.0e-3
    t_start: float = 20e-6
    rise: float = 30e-6
    plateau: float = 150e-6
    fall: float = 30e-6
    ramp: str = "smoothstep7"

    def __call__(self, t):
        t = np.asarray(t, dtype=float)
        t1 = self.t_start + self.rise
        t2 = t1 + self.plateau
        up = _ramp((t - self.t_start) / self.rise, self.ramp)
        down = 1.0 - _ramp((t - t2) / self.fall, self.ramp)
        return self.amplitude * np.where(t < t2, up, down)

    @property
    def duration(self):
        return self.rise + self.plateau + self.fall

    def describe(self):
        return {"type": "trapezoid", **asdict(self)}


PULSE_PRESETS = {
    # pulse-shaped test: smooth 30 us ramps
    "trapezoid": dict(rise=30e-6, plateau=150e-6, fall=30e-6),
    # sharp-rise variant exciting significant dispersion (content up to a few 100 kHz)
    "sharp": dict(rise=4e-6, plateau=150e-6, fall=10e-6),
}


def make_pulse(cfg: dict) -> TrapezoidPulse:
    cfg = dict(cfg)
    preset = cfg.pop("preset", None)
    params = dict(PULSE_PRESETS.get(preset, {}))
    params.update(cfg)
    return TrapezoidPulse(**params)


# ============================================================================================================
# Specimen models (from wavePINN src/wavepinn/specimen.py)
# ============================================================================================================
# Specimen models coupled to the bars at the two interfaces.
#
# Interface coupling (APPROXIMATION, stated explicitly)
# -----------------------------------------------------
# At each bar-specimen interface the bar is represented by the NON-DISPERSIVE
# characteristic relations of Model A, whatever the bar model used for propagation:
#
#     bar 1 (input, x < 0):        N1 = E1 A1 (a1 + b1),   v1 = c1 (-a1 + b1)
#     bar 2 (output, x > Ls):      N2 = E2 A2 (a2 + b2),   v2 = c2 ( a2 - b2)
#
# a1, a2: strain waves arriving at the interface (a1 travels +x, a2 travels -x);
# b1, b2: strain waves leaving the interface. Sign convention: global x from the
# input towards the output bar, velocities positive in +x, strain / stress /
# force positive in tension.
#
# For dispersive bars (Models B, C) the exact interface impedance is frequency
# dependent; using rho c0 A only affects how realistic the specimen response is.
# The stored interface waves a, b are the ground truth, and the gauge signals are
# computed as exact propagations of these waves under the chosen bar model.
#
# Specimen sign convention: strain e = (u2 - u1)/Ls (negative in compression),
# strain rate = (v2 - v1)/Ls (negative in compression).

@dataclass
class SpecimenMaterial:
    """Elastic-plastic, linear isotropic hardening, optional log-rate sensitivity.

    Yield stress (engineering, compression magnitude):
        Y = (sigma0 + H * alpha) * (1 + C * ln(max(|edot| / edot0, 1)))      if rate_dependent
    alpha = accumulated plastic strain, H = plastic hardening modulus
    (tangent modulus E_t = E H / (E + H)).
    tension_cutoff: the specimen-bar contact cannot transmit tension (sigma <= 0);
    a positive trial stress means the faces separate (gap), sigma = 0.
    """
    E: float = 69.0e9
    rho: float = 2710.0
    sigma0: float = 35.0e6
    H: float = 200.0e6
    C: float = 0.015
    edot0: float = 1.0
    rate_dependent: bool = True
    tension_cutoff: bool = True

    def yield_stress(self, alpha, edot_abs):
        Y = self.sigma0 + self.H * alpha
        if self.rate_dependent:
            Y = Y * (1.0 + self.C * np.log(np.maximum(edot_abs / self.edot0, 1.0)))
        return Y


def _coupling_constants(bar1: BarMaterial, bar2: BarMaterial, As: float):
    """edot*Ls = P - Q*sigma with P = 2 (c1 a1 + c2 a2), Q = As (c1/(E1 A1) + c2/(E2 A2))."""
    Q = As * (bar1.c0 / (bar1.E * bar1.area) + bar2.c0 / (bar2.E * bar2.area))
    return Q


class LumpedSpecimen:
    """Homogeneous stress, force equilibrium N1 = N2 = As*sigma, no specimen inertia.

    Kinematics edot = (P - Q sigma)/Ls integrated with the trapezoidal rule (2nd order);
    stress by implicit (backward-Euler) return mapping, solved jointly with the kinematics.
    """

    kind = "lumped"

    def __init__(self, length: float, diameter: float, material: SpecimenMaterial):
        self.Ls = float(length)
        self.Ds = float(diameter)
        self.mat = material
        self.As = np.pi * diameter ** 2 / 4

    def max_dt(self):
        return np.inf

    def describe(self):
        return {"kind": self.kind, "length": self.Ls, "diameter": self.Ds, "area": self.As,
                "material": asdict(self.mat)}

    def solve(self, a1, a2, dt, bar1: BarMaterial, bar2: BarMaterial):
        m = self.mat
        Ls, As, Es = self.Ls, self.As, m.E
        c1, c2 = bar1.c0, bar2.c0
        Q = _coupling_constants(bar1, bar2, As)
        P = 2.0 * (c1 * np.asarray(a1) + c2 * np.asarray(a2))
        n = P.size
        sig = np.zeros(n); e = np.zeros(n); ep = np.zeros(n); alp = np.zeros(n)
        opened = np.zeros(n)
        D = dt * Q / (2 * Ls)
        s_n = e_n = ep_n = al_n = 0.0
        P_n = P[0]
        for i in range(1, n):
            P1 = P[i]
            B = e_n + dt / (2 * Ls) * (P_n - Q * s_n + P1)
            s_tr = Es * (B - ep_n) / (1.0 + Es * D)
            if m.tension_cutoff and s_tr >= 0.0:
                s1, e1, ep1, al1 = 0.0, B, ep_n, al_n
                opened[i] = float(s_tr > 1e-6 * Es)   # diagnostic flag: gap ~ Ls*(e - ep) > 1e-6 Ls
            else:
                e_tr = B - D * s_tr
                Y = m.yield_stress(al_n, abs(e_tr - e_n) / dt)
                if abs(s_tr) <= Y:
                    s1, e1, ep1, al1 = s_tr, e_tr, ep_n, al_n
                else:
                    # compressive plastic flow (tension_cutoff=False & tensile yield: symmetric)
                    sgn = -1.0 if s_tr < 0 else 1.0

                    def g(s):
                        e_new = B - D * s
                        dal = sgn * ((e_new - ep_n) - s / Es)
                        return abs(s) - m.yield_stress(al_n + dal, abs(e_new - e_n) / dt)

                    lo, hi = (s_tr, 0.0) if sgn < 0 else (0.0, s_tr)
                    s1 = optimize.brentq(g, lo, hi, xtol=1e-12 * Y, rtol=1e-14)
                    e1 = B - D * s1
                    dal = sgn * ((e1 - ep_n) - s1 / Es)
                    ep1 = ep_n + sgn * dal
                    al1 = al_n + dal
            sig[i], e[i], ep[i], alp[i] = s1, e1, ep1, al1
            s_n, e_n, ep_n, al_n, P_n = s1, e1, ep1, al1, P1
        N = As * sig
        b1 = N / (bar1.E * bar1.area) - a1
        b2 = N / (bar2.E * bar2.area) - a2
        return {
            "b1": b1, "b2": b2,
            "stress": sig, "strain": e, "strain_rate": (P - Q * sig) / Ls,
            "plastic_strain": ep,
            "force_face1": N, "force_face2": N,
            "velocity_face1": c1 * (-np.asarray(a1) + b1),
            "velocity_face2": c2 * (np.asarray(a2) - b2),
            "contact_open_face1": opened, "contact_open_face2": opened,
        }


class FDSpecimen:
    """Non-equilibrium variant: 1D elastic-plastic specimen bar, explicit staggered FD.

    Nodes i = 0..n carry velocities (lumped masses, half masses at the faces), cells
    carry stress. Face nodes are coupled to the bars through the same non-dispersive
    characteristic relations (treated implicitly, so the face damping is unconditionally
    stable):  N1 = 2 E1 A1 a1 + Z1 v_0,  N2 = 2 E2 A2 a2 - Z2 v_n,  Z = rho c0 A.
    Numerical regularisation (stated, standard in explicit codes): linear artificial viscosity
    q = beta rho c_s dx edot added to the cell stress (beta = ``viscosity``), and the strain
    rate entering the rate-dependent yield stress is low-pass filtered with time constant
    ``rate_filter_time``. Without them the explicit rate-dependent update is unstable /
    chaotic (a 1e-9 input perturbation changed the outputs by ~2 %).
    Cell return mapping is closed-form; the material is symmetric in tension/compression. With ``tension_cutoff`` the faces are in
    unilateral contact: if the coupled face force would be tensile, the face separates
    (bar sees a free end, N = 0) until the tracked gap closes again. A closed face opens only
    if the coupled tensile force exceeds ``contact_tolerance`` * E_bar A_bar (a strain-equivalent
    threshold, default 1e-6). Without it, the ~1e-8 numerical background of the propagated
    incident wave (spectral truncation) toggles the contact thousands of times before the pulse
    arrives, which makes the solution discontinuous in its input (coupling iteration stalls).
    "velocity_face*" are the BAR-side interface velocities (differ from the specimen face
    velocity while a gap is open).
    """

    kind = "fd"

    def __init__(self, length, diameter, material: SpecimenMaterial, n_cells: int = 40, cfl: float = 0.5,
                 viscosity: float = 0.3, rate_filter_time: float = 0.5e-6, contact_tolerance: float = 1e-6):
        self.Ls = float(length)
        self.Ds = float(diameter)
        self.mat = material
        self.As = np.pi * diameter ** 2 / 4
        self.n_cells = int(n_cells)
        self.cfl = float(cfl)
        self.dx = self.Ls / self.n_cells
        self.cs = np.sqrt(material.E / material.rho)
        self.viscosity = float(viscosity)
        self.rate_filter_time = float(rate_filter_time)
        self.contact_tolerance = float(contact_tolerance)

    def max_dt(self):
        # explicit FD with linear viscosity q = beta rho c dx edot: dt <= cfl * dx / (c (sqrt(1+beta^2)+beta))
        b = self.viscosity
        return self.cfl * self.dx / (self.cs * (np.sqrt(1 + b * b) + b))

    def describe(self):
        return {"kind": self.kind, "length": self.Ls, "diameter": self.Ds, "area": self.As,
                "n_cells": self.n_cells, "cfl": self.cfl, "viscosity": self.viscosity,
                "rate_filter_time": self.rate_filter_time,
                "contact_tolerance": self.contact_tolerance, "material": asdict(self.mat)}

    def solve(self, a1, a2, dt, bar1: BarMaterial, bar2: BarMaterial):
        if dt > self.max_dt() * (1 + 1e-12):
            raise ValueError(f"FD specimen needs dt <= {self.max_dt():.3e}, got {dt:.3e}")
        m = self.mat
        nc, dx, As, Es = self.n_cells, self.dx, self.As, m.E
        mass = np.full(nc + 1, m.rho * As * dx)
        mass[0] = mass[-1] = 0.5 * m.rho * As * dx
        Z1, Z2 = bar1.impedance, bar2.impedance
        EA1, EA2 = bar1.E * bar1.area, bar2.E * bar2.area
        c1, c2 = bar1.c0, bar2.c0
        Ntol1, Ntol2 = self.contact_tolerance * EA1, self.contact_tolerance * EA2
        a1 = np.asarray(a1); a2 = np.asarray(a2)
        n = a1.size
        v = np.zeros(nc + 1); sig = np.zeros(nc); e = np.zeros(nc); ep = np.zeros(nc); alp = np.zeros(nc)
        gap1 = gap2 = 0.0          # face separations (>= 0), only with tension_cutoff
        keys = ("N1", "N2", "v0", "vn", "vb1", "vb2", "stress", "strain", "plastic_strain", "open1", "open2")
        out = {k: np.zeros(n) for k in keys}
        rate_C = m.C if m.rate_dependent else 0.0
        qcoef = self.viscosity * m.rho * self.cs * dx
        rate_f = np.zeros(nc)
        q = np.zeros(nc)
        kf = dt / (self.rate_filter_time + dt)
        for i in range(1, n):
            ss = sig + q
            # momentum (symplectic Euler: velocities first, then stresses)
            f = np.zeros(nc + 1)
            f[:-1] += As * ss
            f[1:] -= As * ss
            v_new = v + dt * f / mass
            # face 1: coupled trial; open contact if the coupled force would be tensile
            v0c = (mass[0] * v[0] + dt * (As * ss[0] - 2 * EA1 * a1[i])) / (mass[0] + dt * Z1)
            N1 = 2 * EA1 * a1[i] + Z1 * v0c
            if m.tension_cutoff and (gap1 > 0 or N1 > Ntol1):
                v_new[0] = v[0] + dt * As * ss[0] / mass[0]
                N1, vb1 = 0.0, -2 * c1 * a1[i]
                out["open1"][i] = 1.0
                gap1 = gap1 + dt * (v_new[0] - vb1)      # gap1 = u_specimen_face1 - u_bar1
                if gap1 <= 0:
                    gap1 = 0.0
            else:
                v_new[0], vb1 = v0c, v0c
            vnc = (mass[-1] * v[-1] + dt * (-As * ss[-1] + 2 * EA2 * a2[i])) / (mass[-1] + dt * Z2)
            N2 = 2 * EA2 * a2[i] - Z2 * vnc
            if m.tension_cutoff and (gap2 > 0 or N2 > Ntol2):
                v_new[-1] = v[-1] - dt * As * ss[-1] / mass[-1]
                N2, vb2 = 0.0, 2 * c2 * a2[i]
                out["open2"][i] = 1.0
                gap2 = gap2 + dt * (vb2 - v_new[-1])     # gap2 = u_bar2 - u_specimen_face2
                if gap2 <= 0:
                    gap2 = 0.0
            else:
                v_new[-1], vb2 = vnc, vnc
            v = v_new
            de = dt * np.diff(v) / dx
            e = e + de
            # symmetric elastic-plastic return mapping (material carries tension)
            edot = de / dt
            rate_f += kf * (np.abs(edot) - rate_f)
            q = qcoef * edot
            s_tr = Es * (e - ep)
            Y = m.yield_stress(alp, rate_f)
            rate_fac = 1.0 + rate_C * np.log(np.maximum(rate_f / m.edot0, 1.0))
            dal = np.maximum(np.abs(s_tr) - Y, 0.0) / (Es + m.H * rate_fac)
            ep = ep + np.sign(s_tr) * dal
            alp = alp + dal
            sig = Es * (e - ep)
            out["N1"][i], out["N2"][i] = N1, N2
            out["v0"][i], out["vn"][i] = v[0], v[-1]
            out["vb1"][i], out["vb2"][i] = vb1, vb2
            out["stress"][i] = sig.mean()
            out["strain"][i] = e.mean()
            out["plastic_strain"][i] = ep.mean()
        b1 = out["N1"] / EA1 - a1
        b2 = out["N2"] / EA2 - a2
        return {
            "b1": b1, "b2": b2,
            "stress": out["stress"], "strain": out["strain"],
            "strain_rate": (out["vn"] - out["v0"]) / self.Ls,
            "plastic_strain": out["plastic_strain"],
            "force_face1": out["N1"], "force_face2": out["N2"],
            "velocity_face1": out["vb1"], "velocity_face2": out["vb2"],
            "contact_open_face1": out["open1"], "contact_open_face2": out["open2"],
        }


def make_specimen(cfg: dict):
    cfg = dict(cfg)
    mcfg = dict(cfg.pop("material", {}))
    if "Et" in mcfg:     # tangent modulus (slope of stress vs total strain) -> plastic modulus H
        if "H" in mcfg:
            raise ValueError("give either H or Et for the specimen material, not both")
        Et = mcfg.pop("Et")
        mcfg["H"] = mcfg["E"] * Et / (mcfg["E"] - Et)
    mat = SpecimenMaterial(**mcfg)
    kind = cfg.pop("kind", "lumped")
    if kind == "lumped":
        return LumpedSpecimen(cfg["length"], cfg["diameter"], mat)
    if kind == "fd":
        return FDSpecimen(cfg["length"], cfg["diameter"], mat,
                          n_cells=cfg.get("n_cells", 40), cfl=cfg.get("cfl", 0.5),
                          viscosity=cfg.get("viscosity", 0.3), rate_filter_time=cfg.get("rate_filter_time", 0.5e-6),
                          contact_tolerance=cfg.get("contact_tolerance", 1e-6))
    raise ValueError(kind)


# ============================================================================================================
# SHPB and direct impact (from wavePINN src/wavepinn/generator.py)
# ============================================================================================================
# Method
# ------
# 1. A right-going strain pulse eps_ref(t) is prescribed at distance d_ref from the
#    interface on the input bar.
# 2. It is propagated to the interface with the chosen bar model (A, B or C) in the
#    frequency domain -> incident wave a1 at the interface.
# 3. The specimen (lumped or FD) is solved in the time domain with the
#    non-dispersive interface coupling -> outgoing waves b1, b2 at the interfaces.
# 4. Optional free far ends: waves leaving the interfaces return, after travelling
#    2L, as incoming waves with flipped strain sign (free end: sigma = 0 for every
#    frequency component). Steps 3-4 are iterated; since the round trip takes at
#    least 2L/c0, the iteration is exact after ceil(T c0 / 2L) + 1 passes.
# 5. Gauge signals = superposition of all waves, each propagated FORWARD in its own
#    direction of travel from the point where it is defined (x_ref or the interface),
#    including the optional gauge-length average sin(kL/2)/(kL/2).
# 6. Interface ground truth from the same waves with the model-consistent operators:
#    v = -+(omega/k) eps per wave, N_E = A E eps, N_eff = A rho (omega/k)^2 eps.
#
# Direct impact (setup = "direct_impact")
# ---------------------------------------
# The input bar itself is the projectile: it moves at v0 (+x) with zero strain and hits the
# specimen at t_impact; there is no prescribed pulse. The bar state is the rigid-body motion v0
# plus strain waves a1, b1 (eps = a1 + b1, v = v0 + c1 (-a1 + b1) at the interface). For the
# specimen this is the same as a bar at rest with the extra incoming step wave
# s(t) = -v0/(2 c1) H(t - t_impact), since c1 (-(a1 + s) + (b1 - s)) = c1 (-a1 + b1) + v0. The
# specimen is therefore solved with a1 + s, and its outgoing wave is shifted back, b1 = N/(E1 A1) - a1.
# The gauges see only the waves a1, b1 (the rigid motion carries no strain); a1 is the far-end
# echo of b1. The bar-1 velocities (interface truth, velocity_at) include v0.

FORMAT_VERSION = "hopkinson-csv-1.0"


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


RENAMED_KEYS = {("bars", "length_incident"): "bars.length_input",
                ("bars", "length_transmitted"): "bars.length_output",
                ("bars", "material"): "bars.material_input",
                ("bars", "material_transmitted"): "bars.material_output",
                ("gauges", "incident"): "gauges.input", ("gauges", "transmitted"): "gauges.output"}


def check_renamed_keys(cfg: dict) -> None:
    """Refuse a case file that still uses the former key names, and name the new ones."""
    old = [f"{t}.{k} -> {new}" for (t, k), new in RENAMED_KEYS.items() if k in cfg.get(t, {})]
    if old:
        raise ValueError("the case file uses renamed keys: " + ", ".join(old))


class SHPBSimulation:
    """Holds the converged interface waves; evaluates the wave field anywhere on the bars."""

    def __init__(self, cfg: dict):
        check_renamed_keys(cfg)
        self.cfg = cfg
        bc = cfg["bars"]
        self.mat1 = BarMaterial(**bc["material_input"])
        self.mat2 = BarMaterial(**bc.get("material_output", bc["material_input"]))
        self.model1 = make_bar_model(cfg["bar_model"], self.mat1)
        self.model2 = make_bar_model(cfg["bar_model"], self.mat2)
        self.L1 = float(bc["length_input"])
        self.L2 = float(bc["length_output"])
        self.R = -1.0 if bc.get("far_end_reflections", True) else 0.0
        s = cfg["sampling"]
        self.fs = float(s["rate"])
        self.dt = 1.0 / self.fs
        # The simulation runs ``truth_margin`` beyond the stored record and is cropped in generate():
        # the velocity and A*sigma_eff operators are non-local in time (frequency domain), so on a
        # record that ends while waves are still present they ring in the last samples.
        self.n_out = int(round(s["duration"] * self.fs))
        self.n_margin = int(round(s.get("truth_margin", 0.0) * self.fs))
        self.n = self.n_out + self.n_margin
        self.t = np.arange(self.n) * self.dt
        pad = s.get("pad_factor", 4)
        self.sp = Spectral(self.n, self.dt, pad)
        self.sp.mask_distance = 2 * max(self.L1, self.L2)      # longest propagation path
        # same FFT length and excluded band, on the stored (cropped) grid: for re-propagating stored data
        self.sp_out = Spectral(self.n_out, self.dt, pad * self.n / self.n_out)
        self.sp_out.max_group_delay = self.sp.max_group_delay
        self.sp_out.mask_distance = self.sp.mask_distance
        self.setup = cfg.get("setup", "shpb")
        if self.setup == "direct_impact":
            imp = cfg["impact"]
            self.v0 = float(imp["velocity"])
            self.t_impact = float(imp.get("time", 20e-6))
            self.pulse = None
            self.d_ref = self.L1
        elif self.setup == "shtb":     # loading end solved in TensionSimulation; incident wave stored after solve()
            self.v0, self.t_impact = 0.0, None
            self.pulse = None
            self.d_ref = self.L1
        elif self.setup == "shpb":
            self.v0, self.t_impact = 0.0, None
            self.pulse = make_pulse(cfg["pulse"])
            self.d_ref = float(cfg["source"]["distance_from_interface"])
        else:
            raise ValueError(f"unknown setup {self.setup!r}")
        self.specimen = make_specimen(cfg["specimen"])
        n_sub = int(s.get("n_sub", 10))
        n_sub = max(n_sub, int(np.ceil(self.dt / self.specimen.max_dt()))) if np.isfinite(self.specimen.max_dt()) else n_sub
        self.n_sub = n_sub
        self.eps_ref = self.pulse(self.t) if self.pulse else np.zeros(self.n)

    def impact_step(self, t):
        """Equivalent incoming step wave s(t) of the moving input bar (zero for the SHPB setup)."""
        if not self.v0:
            return np.zeros_like(t)
        # mid value at the impact instant, so the trapezoidal rule of the specimen starts the load at t_impact
        h = np.where(np.isclose(t, self.t_impact, rtol=0.0, atol=1e-12), 0.5, (t > self.t_impact).astype(float))
        return -self.v0 / (2 * self.mat1.c0) * h

    # ------------------------------------------------------------------ solve
    def solve(self):
        sp, m1, m2 = self.sp, self.model1, self.model2
        cp = self.cfg.get("coupling", {})
        max_iter, tol = int(cp.get("max_iter", 80)), float(cp.get("tol", 1e-8))
        a1_inc = propagate(sp, m1, self.eps_ref, self.d_ref)
        b1 = np.zeros(self.n); b2 = np.zeros(self.n)
        hist = []
        t0 = time.time()
        for it in range(max_iter):
            b1_refl, b2_refl = b1, b2            # outgoing waves whose far-end echoes form a1, a2
            a1 = a1_inc + (self.R * propagate(sp, m1, b1_refl, 2 * self.L1) if self.R else 0.0)
            a2 = self.R * propagate(sp, m2, b2_refl, 2 * self.L2) if self.R else np.zeros(self.n)
            dtf = self.dt / self.n_sub
            # the step is added on the fine grid (band-limited interpolation of a step would ring)
            step = self.impact_step(np.arange(self.n * self.n_sub) * dtf)
            res = self.specimen.solve(sp.upsample(a1, self.n_sub) + step, sp.upsample(a2, self.n_sub),
                                      dtf, self.mat1, self.mat2)
            res = {k: v[:: self.n_sub] for k, v in res.items()}
            res["b1"] = res["b1"] + step[:: self.n_sub]     # b1 = N/(E1 A1) - a1
            scale = max(np.abs(res["b1"]).max(), np.abs(res["b2"]).max(), 1e-300)
            change = max(np.abs(res["b1"] - b1).max(), np.abs(res["b2"] - b2).max()) / scale
            b1, b2 = res["b1"], res["b2"]
            hist.append(float(change))
            if change < tol or not self.R:
                break
        converged = bool(hist[-1] < tol or not self.R)
        # Exact self-consistency: a1, a2 are the forward propagations of b1_refl, b2_refl, and the
        # gauge signals use the same b*_refl for the far-end echoes. b1_refl differs from the final
        # b1 by the last iteration change (stored in solve_info); this mismatch lives only in the
        # far-end reflection.
        self.b1_refl, self.b2_refl = b1_refl, b2_refl
        self.a1, self.a2, self.b1, self.b2 = a1, a2, b1, b2
        self.spec = res
        self.solve_info = {"iterations": len(hist), "converged": converged, "change_history": hist,
                           "final_relative_change": hist[-1], "n_sub": self.n_sub,
                           "nfft": sp.nfft, "wall_time_s": time.time() - t0}
        return self

    # ------------------------------------------------------------ wave field
    def strain_components(self, bar: int, d: float, gauge_length: float = 0.0):
        """(right-going, left-going) strain at distance d from the interface on bar 1 or 2."""
        sp = self.sp
        if bar == 1:
            if d > self.d_ref + 1e-12:
                raise ValueError("position beyond the source on the input bar")
            m, L = self.model1, self.L1
            right = propagate(sp, m, self.eps_ref, self.d_ref - d, gauge_length)
            if self.R:
                right = right + self.R * propagate(sp, m, self.b1_refl, 2 * L - d, gauge_length)
            left = propagate(sp, m, self.b1, d, gauge_length)
        else:
            m, L = self.model2, self.L2
            right = propagate(sp, m, self.b2, d, gauge_length)
            left = (self.R * propagate(sp, m, self.b2_refl, 2 * L - d, gauge_length)) if self.R else np.zeros(self.n)
        return right, left

    def strain_at(self, bar, d, gauge_length=0.0):
        r, l = self.strain_components(bar, d, gauge_length)
        return r + l

    def velocity_at(self, bar, d):
        r, l = self.strain_components(bar, d)
        m = self.model1 if bar == 1 else self.model2
        rigid = self.v0 if bar == 1 else 0.0
        return rigid + wave_velocity(self.sp, m, r, +1) + wave_velocity(self.sp, m, l, -1)

    # ---------------------------------------------------------- ground truth
    def interface_truth(self):
        out = {}
        for bar, m, (win, wout) in ((1, self.model1, (self.a1, self.b1)), (2, self.model2, (self.a2, self.b2))):
            eps = win + wout
            if bar == 1:   # a1 travels +x, b1 travels -x; v0: rigid motion of a moving input bar
                v = self.v0 + wave_velocity(self.sp, m, win, +1) + wave_velocity(self.sp, m, wout, -1)
            else:          # b2 travels +x, a2 travels -x
                v = wave_velocity(self.sp, m, wout, +1) + wave_velocity(self.sp, m, win, -1)
            out[f"bar{bar}"] = {
                "strain": eps, "velocity": v,
                "force_E": m.mat.E * m.mat.area * eps,
                "force_eff": force_eff(self.sp, m, eps),
                "wave_in": win, "wave_out": wout,
            }
        return out

    def gauge_records(self):
        g = self.cfg["gauges"]
        Lg = float(g["gauge_length"])
        chans = [(1, d) for d in g["input"]] + [(2, d) for d in g["output"]]
        names = [f"I{i+1}" for i in range(len(g["input"]))] + [f"O{i+1}" for i in range(len(g["output"]))]
        point = np.array([self.strain_at(b, d) for b, d in chans])
        avg = np.array([self.strain_at(b, d, Lg) for b, d in chans]) if g.get("averaging", True) else point.copy()
        return names, chans, point, avg

    def metadata(self):
        c1 = self.mat1.c0
        derived = {
            "bar1": self.model1.describe(), "bar2": self.model2.describe(),
            "specimen": self.specimen.describe(),
            "setup": self.setup,
            "pulse": self.pulse.describe() if self.pulse else None,
            "impact": {"velocity": self.v0, "time": self.t_impact} if self.setup == "direct_impact" else None,
            "area_ratio_bar_over_specimen": self.mat1.area / self.specimen.As,
            "round_trip_time_incident_bar": 2 * self.L1 / c1,
            "round_trip_time_transmitted_bar": 2 * self.L2 / self.mat2.c0,
            "transit_source_to_interface": self.d_ref / c1,
            "simulated_duration_incl_truth_margin": self.n * self.dt,
            "solve_info": self.solve_info,
        }
        return derived


def generate(cfg: dict) -> dict:
    """Run one case; returns (case dict for the write_* functions, solved simulation)."""
    if cfg.get("setup", "shpb") == "shtb":
        sim = TensionSimulation(cfg).solve()
    else:
        sim = SHPBSimulation(cfg).solve()
    n = sim.n_out

    def crop(x):
        if isinstance(x, dict):
            return {k: crop(v) for k, v in x.items()}
        return x[..., :n]

    names, chans, point, avg = sim.gauge_records()
    point, avg = crop(point), crop(avg)
    g = cfg["gauges"]
    case = {
        "format_version": FORMAT_VERSION,
        "kind": "synthetic",
        "name": cfg.get("name", "case"),
        "time": sim.t[:n],
        "sampling_rate": sim.fs,
        "channels": {
            "names": names,
            "bar": np.array([b for b, _ in chans]),
            "distance_from_interface": np.array([d for _, d in chans], dtype=float),
            "gauge_length": np.full(len(chans), float(g["gauge_length"])),
            "records": avg,               # gauge-averaged if [gauges] averaging = true
            "records_point": point,       # strain at the gauge centre
        },
        "source": {"eps_ref": crop(sim.eps_ref), "distance_from_interface": sim.d_ref},
        "interface": crop(sim.interface_truth()),
        "specimen": {k: crop(v) for k, v in sim.spec.items() if k not in ("b1", "b2")},
        "config": cfg,
        "derived": sim.metadata(),
    }
    return case, sim


def to_json(obj) -> str:
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))
    return json.dumps(obj, default=default, indent=1)




# ============================================================================================================
# SHTB (adapted from wavePINN src/wavepinn/tension.py; there the bars were joined by a threaded coupler,
# here the specimen is bonded between them)
# ============================================================================================================
# Layout along the global x axis (from the input towards the output bar):
#
#     flange | input bar (L1) | specimen (Ls) | output bar (L2) | free end
#       ^ striker (tube around the input bar, moving in -x at v_s0 < 0) hits the flange from the right
#
# Interface waves at the specimen faces, as in the pressure setups:
#     bar 1 (input):  a1 travels +x towards the specimen, b1 travels -x away from it;
#     bar 2 (output): b2 travels +x away from the specimen, a2 travels -x towards it.
# The specimen is one of the specimen models above with tension_cutoff = false: its faces are bonded to the
# bars and carry tension. It sees the bars through the same non-dispersive relations as in the pressure setups.
#
# Loading end (nonlinear, time domain)
# ------------------------------------
# The flange is a rigid mass m. At the loading end of bar 1 the outgoing wave f (travels +x) and
# the incoming wave g (travels -x) give N = E1 A1 (f + g), v = c1 (g - f). The striker moves at v_s0
# with zero strain; its waves at the contact face are p (incoming, travels -x) and q (outgoing,
# travels +x), on top of the rigid motion (same device as the direct impact). The striker face
# velocity is v_s0 + 2 c_s p + P/Z_s, with P >= 0 the compressive contact force. So
#
#     m dv/dt = 2 E1 A1 g - Z1 v - P,
#     P = Z_s (v - w) while in contact (P >= 0), w = v_s0 + 2 c_s p,   P = 0 while separated,
#
# integrated with the trapezoidal rule. Then f = g - v/c1 and q = -P/(E_s A_s) - p. The contact opens
# when P would become negative; the gap is tracked and the contact closes again when it reaches zero.
# The far end of the striker is free: p is the echo of q after 2 L_striker, with the sign flipped.
#
# Iteration
# ---------
# Each pass: g = H1(L1) b1 and p = -Hs(2 L_striker) q from the previous pass -> loading end -> f, q;
# a1 = H1(L1) f, a2 = R H2(2 L2) b2 (previous pass) -> specimen -> b1, b2. Each pass makes the solution
# exact for one more round trip; the passes stop when f, q, b1, b2 change by less than tol.

def solve_loading_end(g, p, dt, bar: BarMaterial, striker: BarMaterial, mass: float,
                      v_striker: float, t_impact: float):
    """Flange (rigid mass) between bar 1 and the striker; see the comment above.

    g, p: incoming waves at the bar end and at the striker face, on the time grid i*dt.
    Returns f, q, contact force P (compression +), flange velocity v, contact_open flag.
    """
    g = np.asarray(g, dtype=float); p = np.asarray(p, dtype=float)
    n = g.size
    EA1, Z1, c1 = bar.E * bar.area, bar.impedance, bar.c0
    Zs, cs = striker.impedance, striker.c0
    w = v_striker + 2.0 * cs * p                 # striker face velocity at zero contact force
    v = np.zeros(n); P = np.zeros(n); opened = np.zeros(n)
    h = 0.5 * dt
    gap, impacted = 0.0, False
    for i in range(1, n):
        v0 = v[i - 1]
        if not impacted and (i - 1) * dt >= t_impact - 1e-6 * dt:
            impacted = True                       # the striker reaches the flange at t_impact
        elif not impacted and i * dt > t_impact:
            # impact inside this step: free up to t_impact, then contact (keeps the records continuous
            # in t_impact, which the identification needs); g, w linear within the step
            th = (t_impact - (i - 1) * dt) / dt
            gm = g[i - 1] + th * (g[i] - g[i - 1]); wm = w[i - 1] + th * (w[i] - w[i - 1])
            hf, hc = th * h, (1.0 - th) * h
            vm = (mass * v0 + hf * (2 * EA1 * g[i - 1] - Z1 * v0 + 2 * EA1 * gm)) / (mass + hf * Z1)
            Fm = 2 * EA1 * gm - Z1 * vm - Zs * (vm - wm)
            v[i] = (mass * vm + hc * (Fm + 2 * EA1 * g[i] + Zs * w[i])) / (mass + hc * (Z1 + Zs))
            P[i] = Zs * (v[i] - w[i])
            impacted = True
            continue
        v1 = None
        if impacted and gap <= 0.0:
            # contact on the whole step (trapezoidal rule with the same active set at both ends)
            F0 = 2 * EA1 * g[i - 1] - Z1 * v0 - Zs * (v0 - w[i - 1])
            v1 = (mass * v0 + h * (F0 + 2 * EA1 * g[i] + Zs * w[i])) / (mass + h * (Z1 + Zs))
            P1 = Zs * (v1 - w[i])
            if P1 < 0.0:
                v1 = None                         # the striker separates during this step
            else:
                P[i], gap = P1, 0.0
        if v1 is None:
            F0 = 2 * EA1 * g[i - 1] - Z1 * v0
            v1 = (mass * v0 + h * (F0 + 2 * EA1 * g[i])) / (mass + h * Z1)
            if impacted:
                gap += h * ((w[i - 1] - v0) + (w[i] - v1))      # gap = x_striker_face - x_flange
                opened[i] = 1.0
        v[i] = v1
    f = g - v / c1
    q = -P / (striker.E * striker.area) - p
    return {"f": f, "q": q, "contact_force": P, "flange_velocity": v, "contact_open": opened}


class TensionSimulation(SHPBSimulation):
    """SHTB: striker-flange loading end, specimen bonded between the bars; see the comment above.

    The bars, the specimen, the sampling and the far end of the output bar are set up as in
    SHPBSimulation. The input bar has no free far end: its loading end is solved with the flange."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        if self.specimen.mat.tension_cutoff:
            raise ValueError("setup 'shtb' needs [specimen.material] tension_cutoff = false "
                             "(the specimen is bonded to the bars and carries tension)")
        st = cfg["striker"]
        self.mat_s = BarMaterial(**st["material"])
        self.model_s = make_bar_model(st.get("bar_model", cfg["bar_model"]), self.mat_s)
        self.L_striker = float(st["length"])
        self.v0_striker = float(st["velocity"])
        self.t_impact = float(st.get("impact_time", 20e-6))
        fl = cfg["flange"]
        self.flange_mass = float(fl.get("mass", fl["rho"] * np.pi * fl["diameter"] ** 2 / 4 * fl["length"]))

    def solve(self):
        sp, m1, m2, ms = self.sp, self.model1, self.model2, self.model_s
        cp = self.cfg.get("coupling", {})
        max_iter, tol = int(cp.get("max_iter", 80)), float(cp.get("tol", 1e-8))
        n, ns = self.n, self.n_sub
        dtf = self.dt / ns
        f = np.zeros(n); q = np.zeros(n); b1 = np.zeros(n); b2 = np.zeros(n)
        hist = []
        t0 = time.time()
        for it in range(max_iter):
            g = propagate(sp, m1, b1, self.L1)
            p = -propagate(sp, ms, q, 2 * self.L_striker)
            end = solve_loading_end(sp.upsample(g, ns), sp.upsample(p, ns), dtf, self.mat1,
                                    self.mat_s, self.flange_mass, self.v0_striker, self.t_impact)
            end = {k: v[::ns] for k, v in end.items()}
            b2_refl = b2                         # outgoing wave whose far-end echo forms a2
            a1 = propagate(sp, m1, end["f"], self.L1)
            a2 = self.R * propagate(sp, m2, b2_refl, 2 * self.L2) if self.R else np.zeros(n)
            res = self.specimen.solve(sp.upsample(a1, ns), sp.upsample(a2, ns), dtf, self.mat1, self.mat2)
            res = {k: v[::ns] for k, v in res.items()}
            new = (end["f"], end["q"], res["b1"], res["b2"])
            old = (f, q, b1, b2)
            scale = max(max(np.abs(x).max() for x in new), 1e-300)
            change = max(np.abs(x - y).max() for x, y in zip(new, old)) / scale
            f, q, b1, b2 = new
            hist.append(float(change))
            if change < tol:
                break
        # As in SHPBSimulation: g and b2_refl come from the previous pass; they differ from the
        # final b1, b2 by the last change (stored in solve_info).
        self.f, self.q = f, q
        self.eps_ref = f
        self.a1, self.a2, self.b1, self.b2 = a1, a2, b1, b2
        self.b1_refl, self.b2_refl = b1, b2_refl
        self.spec = {**res, "contact_force": end["contact_force"], "flange_velocity": end["flange_velocity"],
                     "striker_wave_out": q, "striker_wave_in": p, "contact_open": end["contact_open"]}
        self.solve_info = {"iterations": len(hist), "converged": bool(hist[-1] < tol),
                           "change_history": hist, "final_relative_change": hist[-1], "n_sub": ns,
                           "nfft": sp.nfft, "wall_time_s": time.time() - t0}
        return self

    def strain_components(self, bar: int, d: float, gauge_length: float = 0.0):
        """(right-going, left-going) strain at distance d from the specimen face on bar 1 or 2."""
        if bar == 2:
            return super().strain_components(bar, d, gauge_length)
        if d > self.L1 + 1e-12:
            raise ValueError("position beyond the loading end of the input bar")
        sp, m = self.sp, self.model1
        return propagate(sp, m, self.f, self.L1 - d, gauge_length), propagate(sp, m, self.b1, d, gauge_length)

    def metadata(self):
        c1 = self.mat1.c0
        Z1, Zs = self.mat1.impedance, self.mat_s.impedance
        return {
            "setup": self.setup,
            "bar1": self.model1.describe(), "bar2": self.model2.describe(),
            "specimen": self.specimen.describe(),
            "striker": {**self.model_s.describe(), "length": self.L_striker, "velocity": self.v0_striker,
                        "impact_time": self.t_impact, "round_trip_time": 2 * self.L_striker / self.mat_s.c0},
            "flange_mass": self.flange_mass,
            "area_ratio_bar_over_specimen": self.mat1.area / self.specimen.As,
            # first-contact estimates (rigid flange, no echoes): plateau strain and rise time
            "nominal_incident_strain": -self.v0_striker * Zs / (Z1 + Zs) / c1,
            "flange_time_constant": self.flange_mass / (Z1 + Zs),
            "round_trip_time_incident_bar": 2 * self.L1 / c1,
            "round_trip_time_transmitted_bar": 2 * self.L2 / self.mat2.c0,
            "simulated_duration_incl_truth_margin": self.n * self.dt,
            "solve_info": self.solve_info,
        }


# ============================================================================================================
# Output files (plain text). Contents: README.md, "Simulation output".
# ============================================================================================================
# simulation.toml  the case settings as read, and a [derived] table of quantities computed from them
# simulation.csv   time, gauge records and interface truth (the file the analysis reads)
# specimen.csv     specimen histories, the source wave and, for the SHTB, the loading end
# Both CSV files: '#' comment lines, then one header row, then one row per sample, comma-separated.

SIGN_CONVENTION = ("global x from the input bar towards the output bar; displacement and velocity positive "
                   "in +x; strain, stress and force positive in tension (SHTB contact force: compression +); "
                   "gauge positions are distances from the respective bar-specimen interface into the bar")
CSV_FMT = "%.12e"
INTERFACE_KEYS = (("wave_in", ""), ("wave_out", ""), ("strain", ""), ("velocity", "m/s"),
                  ("force_E", "N"), ("force_eff", "N"))
SPECIMEN_UNITS = {"stress": "Pa", "strain_rate": "1/s", "force_face1": "N", "force_face2": "N",
                  "velocity_face1": "m/s", "velocity_face2": "m/s", "contact_force": "N",
                  "flange_velocity": "m/s"}


def _column(name: str, unit: str) -> str:
    return f"{name}[{unit}]" if unit else name


def _write_csv(path, comments: list[str], names: list[str], cols: list) -> None:
    with open(path, "w") as f:
        for c in comments:
            f.write(f"# {c}\n")
        f.write(",".join(names) + "\n")
        np.savetxt(f, np.column_stack(cols), fmt=CSV_FMT, delimiter=",")


def _read_csv(path):
    """(comment lines, column names without units, data columns as a dict of arrays)."""
    lines = Path(path).read_text().splitlines()
    comments = [ln[1:].strip() for ln in lines if ln.startswith("#")]
    body = [ln for ln in lines if ln and not ln.startswith("#")]
    names = [n.split("[")[0] for n in body[0].split(",")]
    data = np.loadtxt(body[1:], delimiter=",", ndmin=2)
    if data.shape[1] != len(names):
        raise ValueError(f"{path}: {data.shape[1]} columns of data, {len(names)} names in the header")
    return comments, names, dict(zip(names, data.T))


def write_records_csv(case: dict, path, settings_name: str = "simulation.toml") -> None:
    """Gauge records (gauge-averaged, then at the gauge centre) and the interface truth of both bars."""
    ch, itf = case["channels"], case["interface"]
    names, cols = ["t[s]"], [case["time"]]
    for key, suffix in (("records", ""), ("records_point", "_point")):
        for i, n in enumerate(ch["names"]):
            names.append(n + suffix); cols.append(ch[key][i])
    for b in ("bar1", "bar2"):
        for key, unit in INTERFACE_KEYS:
            names.append(_column(f"{b}_{key}", unit)); cols.append(itf[b][key])
    comments = [
        FORMAT_VERSION,
        f"case {case['name']}; settings in {settings_name}; sampling rate {case['sampling_rate']:g} Hz",
        "units: SI (s, m, N, m/s); strain dimensionless",
        "sign convention: " + SIGN_CONVENTION,
        "I1, I2, ...: input-bar gauges, O1, O2, ...: output-bar gauges (order as in [gauges] of the settings); "
        "without suffix: averaged over the gauge length if [gauges] averaging = true; _point: at the gauge centre",
        "bar1_*, bar2_*: true quantities at the bar-specimen interface; wave_in travels towards the specimen, "
        "wave_out away from it; strain = wave_in + wave_out; velocity in the global frame; "
        "force_E = A E strain; force_eff = A rho (omega/k)^2 strain",
    ]
    _write_csv(path, comments, names, cols)


def read_records_csv(path) -> dict:
    """Read a records file; returns name, time, sampling_rate, channels (names, bar, records, records_point)
    and interface ({bar1, bar2}: the INTERFACE_KEYS)."""
    comments, names, data = _read_csv(path)
    if not comments or comments[0] != FORMAT_VERSION:
        raise ValueError(f"{path}: not a {FORMAT_VERSION} file (first line: {comments[:1]})")
    gauges = [n for n in names if n[:1] in "IO" and n[1:2].isdigit() and not n.endswith("_point")]
    missing = [n for n in ["t"] + [g + "_point" for g in gauges]
               + [f"{b}_{k}" for b in ("bar1", "bar2") for k, _ in INTERFACE_KEYS] if n not in data]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    t = data["t"]
    return {
        "format_version": FORMAT_VERSION,
        "name": comments[1].split(";")[0].removeprefix("case ").strip() if len(comments) > 1 else Path(path).stem,
        "time": t,
        "sampling_rate": 1.0 / (t[1] - t[0]),
        "channels": {
            "names": gauges,
            "bar": np.array([1 if g[0] == "I" else 2 for g in gauges]),
            "records": np.array([data[g] for g in gauges]),
            "records_point": np.array([data[g + "_point"] for g in gauges]),
        },
        "interface": {b: {k: data[f"{b}_{k}"] for k, _ in INTERFACE_KEYS} for b in ("bar1", "bar2")},
    }


def write_specimen_csv(case: dict, path, settings_name: str = "simulation.toml") -> None:
    """Specimen histories, the source wave and (SHTB) the loading end; not read by the analysis."""
    names, cols = ["t[s]"], [case["time"]]
    for k, v in case["specimen"].items():
        names.append(_column(k, SPECIMEN_UNITS.get(k, ""))); cols.append(v)
    names.append("source_eps_ref"); cols.append(case["source"]["eps_ref"])
    comments = [
        FORMAT_VERSION + " (specimen)",
        f"case {case['name']}; settings in {settings_name}",
        "units: SI (s, m, N, Pa, m/s); strain dimensionless",
        "sign convention: " + SIGN_CONVENTION,
        "specimen quantities as seen by the specimen model (engineering stress and strain); contact_open*: 1 "
        "where a face (or, SHTB, the striker) is separated",
        f"source_eps_ref: the right-going wave on the input bar at {case['source']['distance_from_interface']:g} m "
        "from the interface (SHPB: prescribed pulse; direct impact: zero; SHTB: wave leaving the loading end)",
    ]
    _write_csv(path, comments, names, cols)


def _plain(obj):
    """numpy -> Python types, tuples -> lists, None entries of tables dropped (TOML has no null)."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [_plain(v) for v in (obj.tolist() if isinstance(obj, np.ndarray) else obj)]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def _toml_key(k: str) -> str:
    return k if k and all(c.isalnum() or c in "_-" for c in k) and k.isascii() else json.dumps(k)


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if np.isnan(v):
            return "nan"
        if np.isinf(v):
            return "inf" if v > 0 else "-inf"
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"cannot write {type(v).__name__} to TOML")


def dump_toml(d: dict, _path: tuple = ()) -> str:
    """TOML text of a nested dict of scalars, strings, lists of scalars and tables (no arrays of tables)."""
    d = _plain(d)
    lines = [f"{_toml_key(k)} = {_toml_value(v)}" for k, v in d.items() if not isinstance(v, dict)]
    for k, v in d.items():
        if isinstance(v, dict):
            path = _path + (k,)
            lines += ["", "[" + ".".join(_toml_key(p) for p in path) + "]", dump_toml(v, path)]
    return "\n".join(ln for ln in lines if ln is not None).rstrip("\n") + ("\n" if not _path else "")


def write_settings_toml(cfg: dict, derived: dict, path) -> None:
    """The case settings and a [derived] table; read back and checked against the input."""
    doc = _plain({**cfg, "derived": derived})
    text = (f"# Settings of the simulation case {cfg.get('name', '')}, written by hopkinson_simulator.py.\n"
            "# All keys except [derived] are the case file as read; [derived] holds quantities computed from it.\n"
            "# SI units throughout.\n\n" + dump_toml(doc))
    if json.dumps(tomllib.loads(text), sort_keys=True) != json.dumps(doc, sort_keys=True):   # key order may differ
        raise RuntimeError("the TOML writer does not reproduce the settings")
    Path(path).write_text(text)


def load_config(path) -> dict:
    """Read one case file. ``name`` defaults to the file name without .toml."""
    path = Path(path)
    cfg = tomllib.loads(path.read_text())
    cfg.setdefault("name", path.stem)
    return cfg


# ============================================================================================================
# Figures (style from wavePINN src/wavepinn/plotstyle.py; plot_waves from src/wavepinn/analysis.py)
# ============================================================================================================
# Gnuplot 5 default line colours without the faint yellow: dark violet, green, sky blue, orange, blue, red.
COLORS = ["#9400D3", "#009E73", "#56B4E9", "#E69F00", "#0072B2", "#E51E10"]
INVALID = "#efefef"                  # shading of samples outside the valid window
US, PM = 1e6, 1e3                    # s -> µs, strain -> ‰


def apply_style() -> None:
    """Light grid, no top and right axis lines, the Gnuplot colours (call before the figures are made)."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.titlelocation": "left",
        "axes.edgecolor": "#888888", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "axes.prop_cycle": mpl.cycler(color=COLORS),
        "grid.color": "#e5e5e5", "grid.linewidth": 0.8,
        "xtick.color": "#888888", "ytick.color": "#888888",
        "xtick.labelcolor": "#444444", "ytick.labelcolor": "#444444",
        "xtick.labelsize": 9, "ytick.labelsize": 9, "lines.linewidth": 1.3,
        "legend.framealpha": 0.85, "legend.edgecolor": "white", "legend.fancybox": False, "legend.fontsize": 8.5,
        "savefig.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white",
    })


def _shade_open(ax, t, flag):
    """Shade the intervals where the contact on that face is open."""
    on = np.flatnonzero(np.diff(np.r_[0, (flag > 0).astype(int), 0]))
    for i0, i1 in zip(on[::2], on[1::2]):
        ax.axvspan(t[i0], t[min(i1, len(t) - 1)], color="#dddddd", lw=0, zorder=0)


def plot_waves(case: dict, path):
    """Simulation overview, saved to ``path``; returns the (open) figure. Left column input bar, right column
    output bar. Row 1: gauge records. Row 2: true interface waves; grey: specimen face separated from the bar."""
    import matplotlib.pyplot as plt
    t = case["time"] * US
    ch, itf, sp = case["channels"], case["interface"], case["specimen"]
    cfg = case["config"]
    setup = cfg.get("setup", "shpb")
    impact = setup == "direct_impact"
    fig, ax = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey="row" if not impact else False)
    for col, bar, title in ((0, 1, "input bar"), (1, 2, "output bar")):
        a = ax[0, col]
        for j, i in enumerate(np.flatnonzero(ch["bar"] == bar)):
            a.plot(t, ch["records"][i] * PM, lw=0.9, color=COLORS[j],
                   label=f"{ch['names'][i]}: {ch['distance_from_interface'][i] * 1e3:.0f} mm from the specimen")
        a.set(title=f"A) Gauge records, {title}", ylabel="strain [‰]")
        a = ax[1, col]
        w = itf[f"bar{bar}"]
        if bar == 1 and impact:
            a.plot(t, w["wave_out"] * PM, lw=1.0, color=COLORS[0], label="impact wave (← from specimen)")
            a.plot(t, w["wave_in"] * PM, lw=1.0, color=COLORS[1], label="far-end echo (→ toward specimen)")
        elif bar == 1:
            a.plot(t, w["wave_in"] * PM, lw=1.0, color=COLORS[0], label="incident (→ specimen)")
            a.plot(t, w["wave_out"] * PM, lw=1.0, color=COLORS[1], label="reflected (← from specimen)")
        else:
            a.plot(t, w["wave_out"] * PM, lw=1.0, color=COLORS[2], label="transmitted (→ from specimen)")
            a.plot(t, w["wave_in"] * PM, lw=1.0, color=COLORS[3], label="far-end echo (← toward specimen)")
        if f"contact_open_face{bar}" in sp and np.any(sp[f"contact_open_face{bar}"] > 0):
            _shade_open(a, t, sp[f"contact_open_face{bar}"])
        a.set(title=f"B) True waves at the interface, {title}", xlabel="t [µs]", ylabel="strain [‰]")
        for a in ax[:, col]:
            a.legend(loc="best")
    spc = f"specimen {cfg['specimen']['kind']} (Ls = {cfg['specimen']['length'] * 1e3:.0f} mm)"
    if setup == "shtb":
        loading = f"tension, striker at {cfg['striker']['velocity']:g} m/s"
    elif impact:
        loading = f"direct impact at {cfg['impact']['velocity']:g} m/s"
    else:
        loading = f"pulse {cfg['pulse']['preset']}"
    note = "; grey: specimen face separated from the bar" if any(
        np.any(sp.get(f"contact_open_face{b}", 0) > 0) for b in (1, 2)) else ""
    fig.suptitle(f"{cfg['name']}: model {cfg['bar_model']}, {loading}, {spc}; tension positive{note}")
    fig.tight_layout()
    fig.savefig(path)
    return fig


# ============================================================================================================
# Command line
# ============================================================================================================

def run(cfg_path: Path, out_root: Path, show: bool):
    """Simulate one case into <out_root>/<name>/; returns the open figure (or None)."""
    import matplotlib.pyplot as plt
    cfg = load_config(cfg_path)
    out = out_root / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    case, _ = generate(cfg)
    write_settings_toml(cfg, case["derived"], out / "simulation.toml")
    write_records_csv(case, out / "simulation.csv")
    write_specimen_csv(case, out / "specimen.csv")
    si = case["derived"]["solve_info"]
    print(f"{cfg['name']}: simulated in {time.time() - t0:.1f} s (coupling passes {si['iterations']}, "
          f"converged {si['converged']}) -> {out}/simulation.csv, simulation.toml, specimen.csv")
    fig = plot_waves(case, out / "waves.png")
    if show:
        fig.canvas.manager.set_window_title(f"{cfg['name']}: simulation")
    else:
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="+", type=Path, help="simulation case files (simulation_cases/*.toml)")
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
