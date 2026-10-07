"""Thermal structure of Ganymede's H2O layer, using SeaFreeze equations of state.

- Ice Ih shell: conductive, k = a/T, constant heat flux q (planar, no internal heating).
- Ocean and high-pressure ices: adiabatic (well mixed).
- Pressure from dP/dz = rho g, with g taken from a structure Profile.

Phase codes follow SeaFreeze: 0 liquid, 1 Ih, 2 II, 3 III, 5 V, 6 VI.
Units inside this module: P in MPa (SeaFreeze convention), T in K, depth z in m.
"""
from dataclasses import dataclass
import logging

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

from .constants import R_GANYMEDE, M_GANYMEDE, MOI_GANYMEDE
from .structure import (Profile, hydrostatic_profile, layered_density,
                        fit_three_layer, fit_with_h2o_profile)

PHASE_NAMES = {0: "liquid", 1: "ice Ih", 2: "ice II", 3: "ice III", 5: "ice V", 6: "ice VI"}
SF_NAMES = {0: "water1", 1: "Ih", 3: "III", 5: "V", 6: "VI"}
HP_ICES = (3, 5, 6)
K_ICE_IH = 651.0   # k = a / T for ice Ih [W m^-1] (Petrenko & Whitworth 1999)


def _fill_nan(a):
    """Replace NaN (outside a phase's fitted range) with the nearest valid value."""
    nan = np.isnan(a)
    if not nan.any():
        return a
    i = distance_transform_edt(nan, return_distances=False, return_indices=True)
    return a[tuple(i)]


class H2OTables:
    """SeaFreeze properties tabulated once on a (P, T) grid and interpolated afterwards.
    Calling SeaFreeze point by point is ~10x slower."""

    def __init__(self, P_max=2000.0, nP=801, T_min=100.0, T_max=350.0, nT=251):
        from seafreeze import seafreeze as sf
        logging.getLogger("lbftd").disabled = True
        self.P = np.linspace(0.1, P_max, nP)
        self.T = np.linspace(T_min, T_max, nT)
        PT = np.array([self.P, self.T], dtype=object)
        self.phase_grid = sf.whichphase(PT)
        self._phase = RegularGridInterpolator((self.P, self.T), self.phase_grid.astype(float),
                                              method="nearest")
        self._props = {}
        for code, name in SF_NAMES.items():
            out = sf.getProp(PT, name)
            self._props[code] = {
                k: RegularGridInterpolator((self.P, self.T), _fill_nan(np.asarray(getattr(out, k), float)))
                for k in ("rho", "alpha", "Cp")}

    def phase(self, P, T):
        return int(self._phase((P, T)))

    def props(self, code, P, T):
        """(rho, alpha, cp) for phase `code` at P [MPa], T [K]."""
        f = self._props[code]
        return float(f["rho"]((P, T))), float(f["alpha"]((P, T))), float(f["Cp"]((P, T)))


@dataclass
class Column:
    z: np.ndarray      # depth [m]
    P: np.ndarray      # pressure [MPa]
    T: np.ndarray      # temperature [K]
    phase: np.ndarray  # SeaFreeze phase code
    rho: np.ndarray    # density [kg m^-3]

    def thickness(self, codes):
        m = np.isin(self.phase, codes)
        return float(np.ptp(self.z[m])) if m.any() else 0.0

    @property
    def shell(self):
        return self.thickness([1])

    @property
    def ocean(self):
        return self.thickness([0])

    @property
    def hp_ice(self):
        return self.thickness(list(HP_ICES))

    def sequence(self):
        """Phases in order of appearance with depth."""
        _, first = np.unique(self.phase, return_index=True)
        return [int(p) for p in self.phase[np.sort(first)]]


def conductive_shell_thickness(q, T_surf, T_base, a=K_ICE_IH):
    """Closed-form thickness of a planar conductive shell with k = a/T:
    q = a ln(T_base/T_surf) / D."""
    return a * np.log(T_base / T_surf) / q


def h2o_column(q_surf, profile, h2o_thick, tables, T_surf=110.0, a=K_ICE_IH, dz=100.0):
    """March from the surface to the base of the H2O layer.

    Ice Ih: dT/dz = q T / a, integrated exactly over each step (T *= exp(q dz / a)).
    Ocean / HP ice: dT/dz = alpha g T / cp (explicit Euler; the adiabat is gentle).
    Below the ocean the column is kept solid: a 1D adiabat cannot represent basal melting.
    """
    z, P, T = 0.0, 0.1, T_surf
    rows, below_ocean = [], False
    while z <= h2o_thick:
        ph = tables.phase(P, T)
        if ph == 2:                      # cold-side ice II: treat as Ih here
            ph = 1
        if not below_ocean and ph in HP_ICES and rows and rows[-1][3] == 0:
            below_ocean = True
        if below_ocean and ph == 0:
            ph = rows[-1][3]
        rho, alpha, cp = tables.props(ph, P, T)
        g = np.interp(profile.r[-1] - z, profile.r, profile.g)
        rows.append((z, P, T, ph, rho))
        if ph == 1:
            T *= np.exp(q_surf * dz / a)
        else:
            T += alpha * g * T / cp * dz
        P += rho * g * dz / 1e6
        z += dz
    a_ = np.array(rows)
    return Column(z=a_[:, 0], P=a_[:, 1], T=a_[:, 2], phase=a_[:, 3].astype(int), rho=a_[:, 4])


@dataclass
class Model:
    q_surf: float
    r_core: float
    r_mantle: float
    profile: Profile
    column: Column
    iterations: int


def self_consistent_model(q_surf, tables, rho_core=5150.0, rho_mantle=3300.0, rho_h2o_init=1100.0,
                          R=R_GANYMEDE, M=M_GANYMEDE, moi=MOI_GANYMEDE,
                          tol=500.0, max_iter=20, nr=20001, verbose=False, **column_kw):
    """Iterate structure fit <-> SeaFreeze H2O column until the H2O thickness changes by < tol [m]."""
    rc, rm = fit_three_layer(R, M, moi, rho_core, rho_mantle, rho_h2o_init)
    r = np.linspace(0, R, nr)
    col = None
    for it in range(max_iter):
        rho = layered_density(r, [rc, rm, R], [rho_core, rho_mantle, rho_h2o_init])
        if col is not None:
            w = r >= rm
            rho[w] = np.interp(R - r[w], col.z, col.rho)
        prof = hydrostatic_profile(r, rho)
        col = h2o_column(q_surf, prof, R - rm, tables, **column_kw)
        rc_new, rm_new = fit_with_h2o_profile(R, M, moi, rho_core, rho_mantle, col.z, col.rho,
                                              guess=(rc / R, rm / R))
        change = abs(rm_new - rm)
        if verbose:
            print(f"iter {it}: H2O {(R - rm_new)/1e3:.1f} km, change {change/1e3:.2f} km")
        rc, rm = rc_new, rm_new
        if change < tol:
            break
    else:
        raise RuntimeError(f"self_consistent_model did not converge in {max_iter} iterations")
    rho = layered_density(r, [rc, rm, R], [rho_core, rho_mantle, rho_h2o_init])
    w = r >= rm
    rho[w] = np.interp(R - r[w], col.z, col.rho)
    prof = hydrostatic_profile(r, rho)
    col = h2o_column(q_surf, prof, R - rm, tables, **column_kw)
    return Model(q_surf, rc, rm, prof, col, it + 1)
