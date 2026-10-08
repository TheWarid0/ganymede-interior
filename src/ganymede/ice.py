"""Ice Ih rheology and convection with temperature-dependent viscosity and conductivity (Stage 2d).

Non-dimensional: T' = (T - T_s) / (T_b - T_s), lengths in shell thickness D, time in D^2 / kappa_b.
Viscosity and conductivity are normalised to 1 at the base (T' = 1); Ra uses the base viscosity.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from .stokes import Grid, stokes_matrix_fast, unpack
from .convection import T_ghost, vrms

R_GAS = 8.314
Q_DIFF_ICE = 59.4e3          # activation energy, volume diffusion creep [J/mol] (Goldsby & Kohlstedt 2001)
A_K_ICE = 651.0              # k = A_K_ICE / T  [W/m]


# --- material laws (dimensional) ------------------------------------------------

def eta_diffusion_creep(T, d):
    """Diffusion-creep viscosity of ice Ih [Pa s] for grain size d [m]
    (prefactor as used by Hammond et al. 2016)."""
    return 8.36e9 * (T / 250.0) * d**2 * np.exp(Q_DIFF_ICE / (R_GAS * T))


def onset_criterion(T_s, T_b, Q=Q_DIFF_ICE):
    """Stagnant-lid onset: Ra_cr = 20.9 theta^4 with theta = Q (T_b - T_s) / (R T_b^2) (Solomatov 1995)."""
    theta = Q * (T_b - T_s) / (R_GAS * T_b**2)
    return 20.9 * theta**4, theta


# --- normalised laws for the solver ---------------------------------------------

def arrhenius(T_s, T_b, Q=Q_DIFF_ICE, cap=1e8):
    """eta / eta_b = exp[(Q/R)(1/T - 1/T_b)], capped at `cap` (cold-lid viscosity above ~1e8 doesn't matter)."""
    dT = T_b - T_s
    return lambda Tp: np.minimum(np.exp(Q / R_GAS * (1 / (T_s + np.asarray(Tp) * dT) - 1 / T_b)), cap)


def conductivity_ice(T_s, T_b):
    """k / k_b for k = a / T."""
    dT = T_b - T_s
    return lambda Tp: T_b / (T_s + np.asarray(Tp, float) * dT)


def conductive_flux(T_s, T_b):
    """Steady conductive flux for k = a/T, in code units (k_b = 1, unit thickness and temperature drop)."""
    return (T_b / (T_b - T_s)) * np.log(T_b / T_s)


# --- discretisation ---------------------------------------------------------------

def energy_steady_matrix_k(g, vx, vz, k_c, k_bot, k_top, T_bot=1.0, T_top=0.0):
    """Steady div(v T) − div(k ∇T) = 0, k_c at cell centres, k_bot/k_top at the walls.
    Face conductivity = harmonic mean of neighbouring cells."""
    nx, nz, dx, dz = g.nx, g.nz, g.dx, g.dz
    idx = np.arange(nx * nz).reshape(nz, nx)
    R, C, V = [], [], []
    b = np.zeros(nx * nz)

    def put(r, c, v):
        r, c, v = np.broadcast_arrays(r, c, v)
        R.append(r.ravel()); C.append(c.ravel()); V.append(np.asarray(v, float).ravel())

    J, I = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
    faces = [(0, +1, lambda j, i: vx[j, i + 1], +1, dx), (0, -1, lambda j, i: vx[j, i], -1, dx),
             (+1, 0, lambda j, i: vz[j + 1, i], +1, dz), (-1, 0, lambda j, i: vz[j, i], -1, dz)]
    for dj, di, face_v, sgn, h in faces:
        inner = (J + dj >= 0) & (J + dj < nz) & (I + di >= 0) & (I + di < nx)
        j, i = J[inner], I[inner]
        r, u = idx[j, i], face_v(j, i)
        kf = 2 * k_c[j, i] * k_c[j + dj, i + di] / (k_c[j, i] + k_c[j + dj, i + di])
        put(r, r, sgn * u / (2 * h) + kf / h**2)
        put(r, idx[j + dj, i + di], sgn * u / (2 * h) - kf / h**2)
    for row, T_wall, k_wall in ((0, T_bot, k_bot), (nz - 1, T_top, k_top)):
        r = idx[row]
        kh = 2 * k_c[row] * k_wall / (k_c[row] + k_wall)
        put(r, r, 2 * kh / dz**2)
        b[r] += 2 * kh * T_wall / dz**2
    A = sp.csc_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(nx * nz, nx * nz))
    return A, b


def corner_T(T):
    """T at cell corners: mean of the four surrounding cells (ghost-padded at the walls)."""
    Tg = T_ghost(T)
    return 0.25 * (Tg[1:, 1:] + Tg[:-1, 1:] + Tg[1:, :-1] + Tg[:-1, :-1])


def top_flux(T, g, k_top):
    """Second-order surface heat flux: k_wall * dT/dz|wall, dT/dz = (8 T_wall − 9 T_N + T_{N−1}) / (3 dz), T_wall = 0."""
    return -k_top * ((-9 * T[-1] + T[-2]) / (3 * g.dz)).mean()


def _velocity(g, T, Ra_b, eta_fn):
    A = stokes_matrix_fast(g, eta_fn(T), eta_fn(corner_T(T)))
    Tg = T_ghost(T)
    bz = -Ra_b * 0.5 * (Tg[1:, 1:-1] + Tg[:-1, 1:-1]); bz[0] = bz[-1] = 0.0
    rhs = np.zeros(A.shape[0]); rhs[g.n_vx:g.n_vx + g.n_vz] = bz.ravel()
    vx, vz, _ = unpack(g, spsolve(A, rhs))
    return vx, vz


def _initial_T(g, aspect):
    Xc, Zc = np.meshgrid(g.xc, g.zc)
    return 1 - Zc + 0.1 * np.cos(np.pi * Xc / aspect) * np.sin(np.pi * Zc)


def steady_convection_general(nx, nz, Ra_b, eta_fn, k_fn=None, aspect=1.0, relax=0.3, tol=1e-6,
                              max_iter=800, T0=None):
    """Picard iteration to steady state with eta_fn(T) and k_fn(T), both normalised to 1 at the base.
    Returns g, T, vx, vz, iterations, k at cell centres."""
    g = Grid(nx, nz, L=aspect, H=1.0)
    T = _initial_T(g, aspect) if T0 is None else T0.copy()
    k_fn = k_fn or (lambda T: np.ones_like(np.asarray(T, float)))
    for it in range(1, max_iter + 1):
        vx, vz = _velocity(g, T, Ra_b, eta_fn)
        k_c = k_fn(T)
        Ae, be = energy_steady_matrix_k(g, vx, vz, k_c, float(k_fn(1.0)), float(k_fn(0.0)))
        T_new = spsolve(Ae, be).reshape(nz, nx)
        change = np.abs(T_new - T).max()
        T = relax * T_new + (1 - relax) * T
        if change < tol:
            return g, T, vx, vz, it, k_c
    raise RuntimeError(f"no steady state after {max_iter} iterations (last change {change:.1e})")


def convect_transient(nx, nz, Ra_b, eta_fn, k_fn, t_end, aspect=1.0, cfl=2.0, T0=None):
    """Time-dependent convection: Stokes rebuilt each step, backward-Euler energy step.
    Returns g, T, vx, vz, history [(t, surface flux, vrms)]."""
    g = Grid(nx, nz, L=aspect, H=1.0)
    T = _initial_T(g, aspect) if T0 is None else T0.copy()
    I = sp.identity(nx * nz, format="csc")
    k_top, k_bot = float(k_fn(0.0)), float(k_fn(1.0))
    t, hist = 0.0, []
    while t < t_end:
        vx, vz = _velocity(g, T, Ra_b, eta_fn)
        dt = cfl * min(g.dx, g.dz) / max(np.abs(vx).max(), np.abs(vz).max(), 1e-12)
        Ae, be = energy_steady_matrix_k(g, vx, vz, k_fn(T), k_bot, k_top)
        T = spsolve((I / dt + Ae).tocsc(), T.ravel() / dt + be).reshape(nz, nx)
        t += dt
        hist.append((t, top_flux(T, g, k_top), vrms(vx, vz)))
    return g, T, vx, vz, np.array(hist)
