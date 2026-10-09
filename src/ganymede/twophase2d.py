"""Stage 4d (prototype): two-phase convection in the HP ice layer, 2D.

Setup of Kalousova et al. (2018): box of height H and aspect 2, heat flux q_s into the
bottom, T = T_m(top) on the top, insulating sides; ice free-slip on top and sides,
no-slip on the bottom; water percolates vertically (zero compaction length, Eq. 1c-d),
leaves freely through the top, cannot cross the bottom or sides.

PROTOTYPE simplifications (to be removed in Stage 4d proper):
  * the ice matrix is treated as incompressible (div v_i = 0): the volume change on
    melting (Eq. 1a) and the bulk-viscosity term of Eq. (1b) are dropped. For porosities
    of ~1 % and a 7 % density difference this changes the matrix flow by < 0.1 %;
  * first-order upwind advection for T and porosity (numerically diffusive);
  * melt buoyancy -phi*drho*g and thermal buoyancy -rho_i*alpha*(T - T_m,top)*g,
    without the (1 - phi) factor.

Non-dimensionalisation: length H, time H^2/kappa, viscosity mu0,
theta = (T - Tm_top) / (Tm_bot - Tm_top), so the melting curve is theta_m = 1 - z.
"""
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, splu

from .convection import vrms
from .ice import corner_T
from .stokes import Grid, stokes_matrix_fast, unpack
from .twophase import YEAR, HPIce, darcy_flux, darcy_speed

R_GAS = 8.314


class Scales:
    def __init__(self, p, mu0=1e15, Q=30e3):
        self.p, self.mu0, self.Q = p, mu0, Q
        self.dT = p.Tm_bot - p.Tm_top
        self.t = p.H**2 / p.kappa                          # s
        self.v = p.kappa / p.H                              # m/s
        self.Ra = p.rho_i * p.alpha * p.g * self.dT * p.H**3 / (mu0 * p.kappa)
        self.Rphi = p.drho * p.g * p.H**3 / (mu0 * p.kappa)
        self.Qb = p.q_s * p.H / (p.k * self.dT)            # basal gradient, -dtheta/dz
        self.S = p.rho_i * p.cp * self.dT / (p.rho_w * p.L)  # porosity made per unit theta excess

    def T(self, theta):
        return self.p.Tm_top + theta * self.dT

    def viscosity(self, theta, z):
        """eta/mu0 = exp[Q/R (1/T - 1/T_m(z))] (Eq. 9), T <= T_m so eta >= mu0."""
        T = self.T(theta)
        Tm = self.T(1.0 - z)
        return np.minimum(np.exp(self.Q / R_GAS * (1.0 / T - 1.0 / Tm)), 1e6)


def stokes_matrix_noslip_bottom(g, eta_c, eta_n):
    """Free-slip box matrix with the bottom wall made no-slip: the shear stress on the
    bottom boundary becomes 2 eta vx / (dz/2 * 2) via a mirrored ghost (vx_ghost = -vx)."""
    A = stokes_matrix_fast(g, eta_c, eta_n)
    i = np.arange(1, g.nx)
    rows = g.ivx(0, i)
    D = sp.csc_matrix((-2 * eta_n[0, i] / g.dz**2, (rows, rows)), shape=A.shape)
    return (A + D).tocsc()


def _centre_to_vz(f):
    """Average a cell-centred field to interior horizontal faces (walls get 0)."""
    out = np.zeros((f.shape[0] + 1, f.shape[1]))
    out[1:-1] = 0.5 * (f[1:] + f[:-1])
    return out


def ice_velocity(g, theta, phi, sc, solver="direct"):
    """Ice velocity from buoyancy. solver: 'direct' (SciPy sparse LU) or 'stag' (PETSc DMStag,
    FGMRES + Schur split with geometric multigrid; ~3x faster at 128 x 256 and scales linearly)."""
    Zc = np.broadcast_to(g.zc[:, None], theta.shape)
    eta_c = sc.viscosity(theta, Zc)
    Zn = np.broadcast_to(g.zn[:, None], (g.nz + 1, g.nx + 1))
    eta_n = sc.viscosity(corner_T(theta), Zn)
    # density anomaly (rhs = drho_anomaly * g): warm and wet ice are lighter
    rho_vz = _centre_to_vz(-sc.Ra * theta - sc.Rphi * phi)
    if solver == "stag":
        from .stag_stokes import gather_fields, native_stokes_solve
        _, x, ix, _ = native_stokes_solve(g.nx, g.nz, eta_c, eta_n, rho_vz, L=g.L, H=g.H, rtol=1e-8,
                                          u_rtol=1e-6, noslip_bottom=True)
        vx, vz, _ = gather_fields(x, ix, g.nx, g.nz)
        return vx, vz, eta_c
    A = stokes_matrix_noslip_bottom(g, eta_c, eta_n)
    rhs = np.zeros(A.shape[0])
    rhs[g.n_vx:g.n_vx + g.n_vz] = rho_vz.ravel()
    vx, vz, _ = unpack(g, spsolve(A, rhs))
    return vx, vz, eta_c


def energy_matrix(g, vx, vz, dt, Qb):
    """Implicit step  (theta' - theta)/dt + div(v theta) - lap theta = 0, upwind advection.
    Bottom: -dtheta/dz = Qb; top: theta = 0; sides insulating. Returns A, b_extra."""
    nx, nz, dx, dz = g.nx, g.nz, g.dx, g.dz
    idx = np.arange(nx * nz).reshape(nz, nx)
    R, C, V = [], [], []

    def put(r, c, v):
        r, c, v = np.broadcast_arrays(r, c, v)
        R.append(r.ravel()); C.append(c.ravel()); V.append(np.asarray(v, float).ravel())

    J, I = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
    put(idx, idx, 1.0 / dt)
    faces = [(0, +1, lambda j, i: vx[j, i + 1], +1, dx), (0, -1, lambda j, i: vx[j, i], -1, dx),
             (+1, 0, lambda j, i: vz[j + 1, i], +1, dz), (-1, 0, lambda j, i: vz[j, i], -1, dz)]
    for dj, di, face_v, sgn, h in faces:
        inner = (J + dj >= 0) & (J + dj < nz) & (I + di >= 0) & (I + di < nx)
        j, i = J[inner], I[inner]
        r, nb = idx[j, i], idx[j + dj, i + di]
        un = sgn * face_v(j, i)                         # outward normal velocity
        put(r, r, np.maximum(un, 0) / h + 1 / h**2)
        put(r, nb, np.minimum(un, 0) / h - 1 / h**2)
    put(idx[-1], idx[-1], 2 / dz**2)                    # top wall theta = 0 (ghost)
    b = np.zeros(nx * nz)
    b[idx[0]] = Qb / dz                                 # bottom heat flux
    A = sp.csc_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(nx * nz,) * 2)
    return A, b


def phase_change(theta, phi, theta_m, sc):
    """theta above the melting curve becomes water; water above the background freezes in
    cold ice. Conserves theta + phi / S."""
    p = sc.p
    theta, phi = theta.copy(), phi.copy()
    hot = theta > theta_m
    phi[hot] += sc.S * (theta[hot] - theta_m[hot])
    theta[hot] = theta_m[hot]
    cold = (theta < theta_m) & (phi > p.phi_bg)
    m = np.minimum(phi[cold] - p.phi_bg, sc.S * (theta_m[cold] - theta[cold]))
    phi[cold] -= m
    theta[cold] += m / sc.S
    return theta, phi


def advect_porosity(g, phi, vx, vz, dt):
    """Explicit conservative upwind transport of phi by the (divergence-free) ice velocity."""
    pg = np.pad(phi, 1, mode="edge")
    up = lambda u, a, b: np.where(u > 0, a, b)
    Fx = vx * up(vx, pg[1:-1, :-1], pg[1:-1, 1:])      # faces (nz, nx+1); walls have vx = 0
    Fz = vz * up(vz, pg[:-1, 1:-1], pg[1:, 1:-1])      # faces (nz+1, nx); walls have vz = 0
    return phi - dt * ((Fx[:, 1:] - Fx[:, :-1]) / g.dx + (Fz[1:] - Fz[:-1]) / g.dz)


def percolate2d(phi, dz, dt, p, vscale, cfl=0.9):
    """Vertical Darcy percolation in every column (non-dimensional dz, dt; vscale = kappa/H).
    Returns phi and the extracted water volume per unit (non-dim) width, column by column."""
    phi = phi.copy()
    out = np.zeros(phi.shape[1])
    t = 0.0
    while t < dt:
        s = darcy_speed(phi, p).max() / vscale
        h = dt - t if s <= 0 else min(dt - t, cfl * dz / s)
        F = darcy_flux(phi, p) / vscale
        Ff = np.vstack([np.zeros((1, phi.shape[1])), F])
        phi -= h / dz * (Ff[1:] - Ff[:-1])
        out += h * F[-1]
        t += h
    return phi, out


def run(p=None, nx=200, nz=100, mu0=1e15, t_end_myr=20.0, cfl=0.5, snap_myr=(), log_every=20,
        theta0=None, phi0=None, verbose=True, wall_limit=None, solver="direct",
        t0_myr=0.0):
    """Time-dependent two-phase convection. Returns dict with grid, scales, history, snapshots."""
    p = p or HPIce()
    sc = Scales(p, mu0=mu0)
    g = Grid(nx, nz, L=2.0, H=1.0)
    Xc, Zc = np.meshgrid(g.xc, g.zc)
    theta_m = 1.0 - Zc
    theta = np.zeros((nz, nx)) if theta0 is None else theta0.copy()
    if theta0 is None:   # small kick to break symmetry
        theta += 0.01 * np.cos(np.pi * Xc / 2) * np.sin(np.pi * Zc)
    phi = np.full((nz, nx), p.phi_bg) if phi0 is None else phi0.copy()
    E = lambda th, ph: (th + ph / sc.S).sum() * g.dx * g.dz
    E0 = E(theta, phi)
    flux_in = flux_out = latent_out = 0.0
    t, step = t0_myr * 1e6 * YEAR / sc.t, 0
    t_end = t_end_myr * 1e6 * YEAR / sc.t
    snaps_left = sorted(snap_myr)
    hist, snaps = [], []
    w0 = time.time()
    while t < t_end - 1e-15:
        vx, vz, eta = ice_velocity(g, theta, phi, sc, solver=solver)
        vmax = max(np.abs(vx).max(), np.abs(vz).max(), 1e-12)
        dt = min(cfl * min(g.dx, g.dz) / vmax, t_end - t, 1e-4)
        A, b = energy_matrix(g, vx, vz, dt, sc.Qb)
        theta = spsolve(A, theta.ravel() / dt + b).reshape(nz, nx)
        q_top = 2 * theta[-1] / g.dz                     # -dtheta/dz at the top, per column
        theta, phi = phase_change(theta, phi, theta_m, sc)
        phi = advect_porosity(g, phi, vx, vz, dt)
        phi, w = percolate2d(phi, g.dz, dt, p, sc.v)
        theta, phi = phase_change(theta, phi, theta_m, sc)
        t += dt
        step += 1
        flux_in += sc.Qb * g.L * dt
        flux_out += q_top.sum() * g.dx * dt
        latent_out += w.sum() * g.dx / sc.S
        # dimensional diagnostics: heat flux [W/m^2] = nondim flux * k dT / H
        qd = p.k * sc.dT / p.H
        rec = dict(t_myr=t * sc.t / YEAR / 1e6, step=step,
                   q_cond=q_top.mean() * qd, q_water=w.sum() * g.dx / sc.S / g.L / dt * qd,
                   vrms_cm_yr=vrms(vx, vz) * sc.v * YEAR * 100,
                   phi_mean=phi.mean(), phi_max=phi.max(),
                   temperate=float(np.mean(theta >= theta_m - 1e-9)),
                   T_mean=sc.T(theta.mean()),
                   energy_error=(E(theta, phi) - E0 - flux_in + flux_out + latent_out) / max(flux_in, 1e-30))
        hist.append(rec)
        while snaps_left and rec["t_myr"] >= snaps_left[0] - 1e-9:
            snaps.append(dict(t_myr=rec["t_myr"], theta=theta.copy(), phi=phi.copy(), vx=vx.copy(),
                              vz=vz.copy(), dT=sc.T(theta) - sc.T(theta_m)))
            snaps_left.pop(0)
        if verbose and step % log_every == 0:
            print(f"step {step:5d} t={rec['t_myr']:7.3f} Myr  q_cond={rec['q_cond']*1e3:6.2f} "
                  f"q_water={rec['q_water']*1e3:6.2f} mW/m2  vrms={rec['vrms_cm_yr']:.3g} cm/yr  "
                  f"phi_mean={rec['phi_mean']*100:.3f}% max={rec['phi_max']*100:.2f}%  temperate={rec['temperate']:.3f}  "
                  f"T_mean={rec['T_mean']:.2f}  Eerr={rec['energy_error']:.1e}  wall={time.time()-w0:.0f}s", flush=True)
        if wall_limit and time.time() - w0 > wall_limit:
            break
    snaps.append(dict(t_myr=t * sc.t / YEAR / 1e6, theta=theta.copy(), phi=phi.copy(), vx=vx.copy(),
                      vz=vz.copy(), dT=sc.T(theta) - sc.T(theta_m)))
    return dict(g=g, sc=sc, hist=hist, snaps=snaps, theta=theta, phi=phi)
