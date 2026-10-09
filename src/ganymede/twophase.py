"""Stage 4: ice + water two-phase physics of the high-pressure (HP) ice layer.

Follows the reduced model of Kalousova et al. (2018, Icarus 299) -- zero compaction
length, so water percolates through the ice matrix driven only by the ice/water
density difference:

    water flux relative to ice   phi (v_w - v_i) = K(phi) (1 - phi) drho g / mu_w   (upward)
    permeability                 K(phi) = K0 (phi - phi_c)^n  for phi > phi_c, else 0
    porosity                     d phi/dt + div(phi v_w) = r / rho_w
    energy                       rho_i c_p dT/dt + ... + L r = k lap T,   T <= T_m(P)

Cold ice (T < T_m) holds no water beyond a fixed background porosity; temperate ice
(T = T_m) can hold water, and the energy equation then gives the melting rate r.
We solve the energy equation by operator splitting: a conduction step, then any
temperature above T_m is turned into melt (and water in ice below T_m freezes,
releasing latent heat). Total energy  E = sum (rho_i c_p T + rho_w L phi) dz  is
conserved exactly, up to the boundary fluxes and the latent heat of extracted water.

This module holds the 1D pieces (Stages 4a-4c); 2D coupling comes next.
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import brentq

YEAR = 3.15576e7  # s


@dataclass(frozen=True)
class HPIce:
    """Ice VI layer parameters, Kalousova et al. (2018) Tables 1-2, reference case H = 200 km."""
    H: float = 200e3          # layer thickness [m]
    g: float = 1.6            # gravity [m/s^2]
    rho_i: float = 1377.0     # mean ice VI density [kg/m^3]
    rho_w: float = 1280.0     # mean water density [kg/m^3]
    k: float = 1.58           # thermal conductivity [W/m/K]
    cp: float = 2850.0        # specific heat [J/kg/K]
    L: float = 360e3          # latent heat [J/kg]
    alpha: float = 1.46e-4    # thermal expansion [1/K]
    mu_w: float = 2e-3        # water viscosity [Pa s]
    K0: float = 1e-9          # reference permeability [m^2]
    n: float = 2.0            # permeability exponent
    phi_c: float = 0.01       # percolation threshold
    phi_bg: float = 0.005     # background porosity, always present, never extracted
    Tm_bot: float = 332.0     # melting T at the silicate interface [K]
    Tm_top: float = 309.0     # melting T at the ocean interface [K]
    q_s: float = 0.020        # heat flux from the silicates [W/m^2]

    @property
    def drho(self):
        return self.rho_i - self.rho_w

    @property
    def kappa(self):
        return self.k / (self.rho_i * self.cp)

    def Tm(self, z):
        """Melting temperature, linear between the two interfaces; z = height above the silicates."""
        return self.Tm_bot + (self.Tm_top - self.Tm_bot) * np.asarray(z) / self.H

    def with_(self, **kw):
        return replace(self, **kw)


# --- 4a: percolation ---------------------------------------------------------------

def permeability(phi, p):
    return p.K0 * np.clip(phi - p.phi_c, 0.0, None) ** p.n


def darcy_flux(phi, p):
    """Upward volume flux of water relative to the ice, phi (v_w - v_i) [m/s]."""
    return permeability(phi, p) * (1.0 - phi) * p.drho * p.g / p.mu_w


def darcy_speed(phi, p):
    """Characteristic speed dF/dphi of the porosity wave [m/s]."""
    x = np.clip(phi - p.phi_c, 0.0, None)
    c = p.K0 * p.drho * p.g / p.mu_w
    return c * (p.n * x ** (p.n - 1) * (1.0 - phi) - x ** p.n)


def percolate(phi, dz, dt, p, cfl=0.9):
    """Advance porosity by dt with the water flux alone (ice at rest), in 1D.

    Finite volumes, Godunov (= upwind, since dF/dphi >= 0 for small phi) with explicit
    sub-steps limited by the CFL condition. Bottom impermeable, free outflow at the top.
    Returns (phi, extracted water volume per unit area [m])."""
    phi = phi.copy()
    out = 0.0
    t = 0.0
    while t < dt:
        s = darcy_speed(phi, p).max()
        h = dt - t if s <= 0 else min(dt - t, cfl * dz / s)
        F = darcy_flux(phi, p)
        Ff = np.concatenate([[0.0], F])          # face fluxes: bottom = 0, then upwind from below
        phi -= h / dz * (Ff[1:] - Ff[:-1])
        out += h * F[-1]
        t += h
    return phi, out


def riemann_solution(z, t, phi_lo, phi_hi, z0, p):
    """Exact solution of the 1D percolation Riemann problem: porosity phi_lo below z0 and
    phi_hi above it at t = 0 (flux F convex on the states involved)."""
    xi = (np.asarray(z, float) - z0) / t
    F = lambda f: float(darcy_flux(np.array(f), p))
    if phi_lo > phi_hi:                          # wet below dry: an upward-moving shock
        s = (F(phi_lo) - F(phi_hi)) / (phi_lo - phi_hi)
        return np.where(xi < s, phi_lo, phi_hi)
    a_lo, a_hi = float(darcy_speed(np.array(phi_lo), p)), float(darcy_speed(np.array(phi_hi), p))
    out = np.where(xi <= a_lo, phi_lo, phi_hi).astype(float)
    fan = (xi > a_lo) & (xi < a_hi)              # rarefaction: dF/dphi = xi inside the fan
    lo = max(phi_lo, p.phi_c)
    out[fan] = [brentq(lambda f: float(darcy_speed(np.array(f), p)) - x, lo, phi_hi) for x in xi[fan]]
    return out


# --- 4b: temperate-ice energy -----------------------------------------------------

def conduction_step(T, dz, dt, p, q_bot, T_top):
    """Backward-Euler conduction on cell centres: heat flux q_bot into the bottom,
    T = T_top on the top face. Returns new T and the heat flux out of the top [W/m^2]."""
    N = len(T)
    a = p.k * dt / (p.rho_i * p.cp * dz**2)
    ab = np.zeros((3, N))
    ab[0, 1:] = -a                   # upper diagonal
    ab[1, :] = 1 + 2 * a
    ab[2, :-1] = -a                  # lower diagonal
    ab[1, 0] = 1 + a                 # flux bottom: no neighbour below
    ab[1, -1] = 1 + 3 * a            # Dirichlet top face at distance dz/2
    rhs = T + 0.0
    rhs[0] += q_bot * dt / (p.rho_i * p.cp * dz)
    rhs[-1] += 2 * a * T_top
    Tn = solve_banded((1, 1), ab, rhs)
    q_top = p.k * (Tn[-1] - T_top) / (dz / 2)
    return Tn, q_top


def phase_change(T, phi, Tm, p):
    """Turn temperature above T_m into melt; freeze water (above the background porosity)
    in ice below T_m, releasing latent heat. Conserves rho_i c_p T + rho_w L phi exactly."""
    T, phi = T.copy(), phi.copy()
    hot = T > Tm
    phi[hot] += p.rho_i * p.cp * (T[hot] - Tm[hot]) / (p.rho_w * p.L)
    T[hot] = Tm[hot]
    cold = (T < Tm) & (phi > p.phi_bg)
    m = np.minimum(p.rho_w * (phi[cold] - p.phi_bg),                 # water available [kg/m^3]
                   p.rho_i * p.cp * (Tm[cold] - T[cold]) / p.L)      # water the cold can freeze
    phi[cold] -= m / p.rho_w
    T[cold] += p.L * m / (p.rho_i * p.cp)
    return T, phi


def total_energy(T, phi, dz, p):
    """Energy per unit area [J/m^2] (sensible + latent, relative to 0 K and dry ice)."""
    return float(np.sum(p.rho_i * p.cp * T + p.rho_w * p.L * phi) * dz)


# --- 4c: a non-convecting HP ice column -------------------------------------------

def run_column(p, N=400, dt=1e3 * YEAR, t_end=30e6 * YEAR, T0=None, every=100):
    """Conductive HP-ice column heated from below, with melting, percolation and freezing.

    Starts at T0 (default: the ocean-interface melting point everywhere, as in Kalousova et
    al. 2018) and background porosity. Returns a dict of time series and snapshots.
    """
    dz = p.H / N
    z = (np.arange(N) + 0.5) * dz
    Tm = p.Tm(z)
    T = np.full(N, p.Tm_top if T0 is None else T0, dtype=float) * np.ones(N)
    phi = np.full(N, p.phi_bg)
    E0 = total_energy(T, phi, dz, p)
    heat_in = heat_out = latent_out = water_out = 0.0
    hist = {k: [] for k in ("t", "q_top", "water_flux", "temperate_top", "phi_max", "energy_error")}
    snaps = []
    nsteps = int(round(t_end / dt))
    for i in range(1, nsteps + 1):
        T, q_top = conduction_step(T, dz, dt, p, p.q_s, p.Tm_top)
        T, phi = phase_change(T, phi, Tm, p)
        phi, w = percolate(phi, dz, dt, p)
        T, phi = phase_change(T, phi, Tm, p)
        heat_in += p.q_s * dt
        heat_out += q_top * dt
        water_out += w
        latent_out += p.rho_w * p.L * w
        if i % every == 0 or i == nsteps:
            temperate = np.nonzero(T >= Tm - 1e-9)[0]
            # top of the temperate zone connected to the bottom
            gaps = np.nonzero(np.diff(temperate) > 1)[0]
            top = (temperate[gaps[0]] if len(gaps) else temperate[-1]) if len(temperate) and temperate[0] == 0 else -1
            E = total_energy(T, phi, dz, p)
            hist["t"].append(i * dt)
            hist["q_top"].append(q_top)
            hist["water_flux"].append(w / dt)
            hist["temperate_top"].append((top + 1) * dz if top >= 0 else 0.0)
            hist["phi_max"].append(phi.max())
            hist["energy_error"].append((E - E0 - heat_in + heat_out + latent_out) / (heat_in + 1e-30))
            snaps.append((i * dt, T.copy(), phi.copy()))
    return dict(z=z, Tm=Tm, hist={k: np.array(v) for k, v in hist.items()}, snaps=snaps,
                water_out=water_out, dz=dz)


def temperate_porosity(p, flux):
    """Porosity at which the Darcy flux carries a given upward water volume flux [m/s]."""
    return brentq(lambda f: float(darcy_flux(np.array(f), p)) - flux, p.phi_c, 0.5)


def front_height_energy_balance(t, p):
    """Height of the melt (temperate) front in the conductive column from energy balance,
    neglecting conduction ahead of the front: all basal heat goes into warming the ice from
    T_m(top) to T_m(z) and into filling the temperate zone with water at the porosity that
    carries the melt flux, q_s t = int_0^h [rho_i c_p (T_m(z) - T_m,top) + rho_w L (phi_t - phi_bg)] dz."""
    phi_t = temperate_porosity(p, p.q_s / (p.rho_w * p.L))
    dTm = p.Tm_bot - p.Tm_top
    a = p.rho_i * p.cp * dTm + p.rho_w * p.L * (phi_t - p.phi_bg)
    b = p.rho_i * p.cp * dTm / (2 * p.H)
    E = p.q_s * np.asarray(t, float)
    h = (a - np.sqrt(np.maximum(a * a - 4 * b * E, 0.0))) / (2 * b)    # a h - b h^2 = E
    t_break = (a * p.H - b * p.H**2) / p.q_s
    return np.where(np.asarray(t) < t_break, h, p.H), t_break
