"""Stage 5: two-phase (ice + water) convection in the ice Ih shell, connected to the ocean below.

Box of height D (the shell) and width aspect*D, z up from the ice/ocean interface.
  * top (surface):  T = T_s, free slip, impermeable
  * bottom (ocean): T = T_m(P_base) (SeaFreeze), free slip; meltwater drains freely into the ocean
  * sides:          insulating, free slip, impermeable
Ice Ih: Arrhenius diffusion creep normalised at the base (capped, as Stage 2d), k = 651/T.
Water is DENSER than ice Ih, so melt sinks: wet ice is negatively buoyant and water percolates
downward (zero compaction length Darcy law, as Stages 4a-d) and leaves into the ocean.
Temperate ice: T = T_m(z) from the SeaFreeze ice Ih melting curve; heat above it melts ice,
water in ice below it freezes (energy conserved exactly, as Stage 4b).
Optional tidal heating (Maxwell body, Tobie et al. 2003):  H = 2 eta edot0^2 / (1 + (omega eta / mu)^2).
Ocean connection (fixed domain): the heat conducted out of the ocean into the shell base, compared
with the heat the ocean delivers (q_ocean), gives the freezing (+) / melting (-) rate of the base
(Stefan condition), reported as shell growth in km/Myr.

Simplifications (as the Stage 4d prototype): incompressible ice matrix, first-order upwind
advection, constant heat capacity (its base value), no melt softening, fixed shell thickness.

Non-dimensional: length D, time D^2/kappa_b, viscosity eta_b (at the base), theta = (T - T_s)/(T_b - T_s).
"""
from dataclasses import dataclass, replace
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, bicgstab, spsolve

from .convection import vrms
from .ice import A_K_ICE, Q_DIFF_ICE, R_GAS, corner_T, eta_diffusion_creep
from .stokes import Grid, stokes_matrix_fast, unpack
from .twophase import YEAR


@dataclass(frozen=True)
class ShellParams:
    D: float = 100e3            # shell thickness [m]
    d: float = 1e-4             # grain size [m]
    T_s: float = 110.0          # surface temperature [K]
    g: float = 1.43             # gravity [m/s^2]
    rho_i: float = 930.0        # ice Ih [kg/m^3]
    rho_w: float = 1000.0       # water [kg/m^3]
    alpha: float = 1.6e-4       # thermal expansion [1/K]
    L: float = 334e3            # latent heat [J/kg]
    mu_w: float = 1.8e-3        # water viscosity [Pa s]
    K0: float = 1e-9            # permeability prefactor [m^2]
    n: float = 2.0              # permeability exponent
    phi_c: float = 0.01         # percolation threshold
    phi_bg: float = 0.0         # background porosity
    eta_cap: float = 1e8        # viscosity cap relative to the base
    edot0: float = 0.0          # tidal strain-rate amplitude [1/s] (0 = no tidal heating)
    omega: float = 1.016e-5     # Ganymede's orbital frequency [1/s]
    mu_e: float = 3.3e9         # ice shear modulus [Pa]
    q_ocean: float = 0.0        # heat flux delivered by the ocean to the shell base [W/m^2]

    def with_(self, **kw):
        return replace(self, **kw)


def P_melt_Ih(T):
    """Ice Ih melting pressure [Pa] at T [K], IAPWS R14-08 (Wagner et al. 2011); 251.165-273.16 K."""
    th = np.asarray(T, float) / 273.16
    return 611.657 * (1 + 1.195393937e6 * (1 - th**3) + 80818.3159 * (1 - th**25.75)
                      + 3338.26860 * (1 - th**103.750))


def T_melt_Ih(P):
    """Ice Ih melting temperature [K] at pressure P [Pa] (inverse of the IAPWS melting curve).
    Agrees with the SeaFreeze phase map (Stage 2d) to < 0.15 K."""
    from scipy.optimize import brentq
    if not 0 <= P <= P_melt_Ih(251.165):
        raise ValueError(f"P = {P/1e6:.1f} MPa is outside the ice Ih melting curve")
    return brentq(lambda T: P_melt_Ih(T) - max(P, 611.657), 251.165, 273.16, xtol=1e-6)


class ShellScales:
    def __init__(self, p, nz_table=60):
        self.p = p
        P_base = p.rho_i * p.g * p.D
        self.T_b = T_melt_Ih(P_base)
        self.dT = self.T_b - p.T_s
        self.k_b = A_K_ICE / self.T_b
        self.cp_b = 7.037 * self.T_b + 185.0
        self.kappa_b = self.k_b / (p.rho_i * self.cp_b)
        self.eta_b = eta_diffusion_creep(self.T_b, p.d)
        self.t = p.D**2 / self.kappa_b
        self.v = self.kappa_b / p.D
        self.Ra = p.rho_i * p.alpha * p.g * self.dT * p.D**3 / (self.kappa_b * self.eta_b)
        self.Rphi = (p.rho_w - p.rho_i) * p.g * p.D**3 / (self.kappa_b * self.eta_b)
        self.S = p.rho_i * self.cp_b * self.dT / (p.rho_w * p.L)
        self.qd = self.k_b * self.dT / p.D                 # flux scale [W/m^2]
        # melting curve T_m(z), z up from the base: tabulate once from SeaFreeze
        zt = np.linspace(0.0, 1.0, nz_table)
        self._zt = zt
        self._Tm = np.array([T_melt_Ih(p.rho_i * p.g * p.D * (1 - z)) for z in zt])

    def T(self, theta):
        return self.p.T_s + np.asarray(theta) * self.dT

    def theta_m(self, z):
        return (np.interp(z, self._zt, self._Tm) - self.p.T_s) / self.dT

    def viscosity(self, theta):
        T = self.T(np.clip(theta, 0.0, None))
        return np.minimum(np.exp(Q_DIFF_ICE / R_GAS * (1.0 / T - 1.0 / self.T_b)), self.p.eta_cap)

    def conductivity(self, theta):
        return self.T_b / self.T(np.clip(theta, 0.0, None))

    def tidal(self, eta_nd):
        """Non-dimensional tidal heating h = H D^2 / (k_b dT) for non-dimensional viscosity."""
        p = self.p
        if p.edot0 == 0.0:
            return np.zeros_like(eta_nd)
        eta = eta_nd * self.eta_b
        H = 2 * eta * p.edot0**2 / (1 + (p.omega * eta / p.mu_e) ** 2)
        return H * p.D**2 / (self.k_b * self.dT)

    def darcy_down(self, phi):
        """Downward water flux relative to the ice, non-dimensional (velocity scale kappa_b / D)."""
        p = self.p
        K = p.K0 * np.clip(phi - p.phi_c, 0.0, None) ** p.n
        return K * (1 - phi) * (p.rho_w - p.rho_i) * p.g / p.mu_w / self.v

    def darcy_speed(self, phi):
        p = self.p
        x = np.clip(phi - p.phi_c, 0.0, None)
        c = p.K0 * (p.rho_w - p.rho_i) * p.g / p.mu_w / self.v
        return c * (p.n * x ** (p.n - 1) * (1 - phi) - x**p.n)


def _centre_to_vz(f):
    out = np.zeros((f.shape[0] + 1, f.shape[1]))
    out[1:-1] = 0.5 * (f[1:] + f[:-1])
    return out


def velocity(g, theta, phi, sc, solver="stag", cache=None):
    """Free-slip box; buoyancy: warm ice rises, wet ice sinks (water denser than ice Ih).
    cache: dict kept between time steps; the PETSc solve starts from the previous solution."""
    eta_c = sc.viscosity(theta)
    eta_n = sc.viscosity(corner_T(theta))
    rho_vz = _centre_to_vz(-sc.Ra * theta + sc.Rphi * (phi - sc.p.phi_bg))
    if solver == "stag":
        from .stag_convection import allgather_fields
        from .stag_stokes import native_stokes_solve
        x0 = cache.get("x") if cache is not None else None
        _, x, ix, st = native_stokes_solve(g.nx, g.nz, eta_c, eta_n, rho_vz, L=g.L, H=g.H, rtol=1e-8,
                                           u_rtol=1e-6, x0=x0)
        if cache is not None:
            cache["x"] = x
            cache["iters"] = st["outer"]
        vx, vz, _ = allgather_fields(x, ix, g.nx, g.nz)      # every MPI rank gets the full field
        return vx, vz, eta_c
    A = stokes_matrix_fast(g, eta_c, eta_n)
    rhs = np.zeros(A.shape[0])
    rhs[g.n_vx:g.n_vx + g.n_vz] = rho_vz.ravel()
    vx, vz, _ = unpack(g, spsolve(A, rhs))
    return vx, vz, eta_c


def energy_matrix(g, vx, vz, dt, k_c, k_top, k_bot, h):
    """Implicit step (theta' - theta)/dt + div(v theta) - div(k grad theta) = h. Upwind advection,
    harmonic-mean face conductivity, theta = 0 on top, theta = 1 at the ocean, insulating sides."""
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
    for dj, di, face_v, sgn, hh in faces:
        inner = (J + dj >= 0) & (J + dj < nz) & (I + di >= 0) & (I + di < nx)
        j, i = J[inner], I[inner]
        r, nb = idx[j, i], idx[j + dj, i + di]
        un = sgn * face_v(j, i)
        kf = 2 * k_c[j, i] * k_c[j + dj, i + di] / (k_c[j, i] + k_c[j + dj, i + di])
        put(r, r, np.maximum(un, 0) / hh + kf / hh**2)
        put(r, nb, np.minimum(un, 0) / hh - kf / hh**2)
    b = h.ravel().copy()
    kt = 2 * k_c[-1] * k_top / (k_c[-1] + k_top)
    put(idx[-1], idx[-1], 2 * kt / dz**2)                          # theta = 0 at the surface
    kb = 2 * k_c[0] * k_bot / (k_c[0] + k_bot)
    put(idx[0], idx[0], 2 * kb / dz**2)                            # theta = 1 at the ocean
    b[idx[0]] += 2 * kb * 1.0 / dz**2
    A = sp.csc_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(nx * nz,) * 2)
    return A, b, kt, kb


def mpi_rank():
    """Rank in PETSc's world communicator (0 when running serially or without petsc4py).
    Under mpiexec every rank runs the whole time loop; only the Stokes solve is split between ranks."""
    try:
        from petsc4py import PETSc
        return PETSc.COMM_WORLD.getRank()
    except ImportError:
        return 0


def solve_energy(A, b, x0):
    """Implicit energy step: Jacobi-preconditioned BiCGSTAB from the previous temperature (the matrix
    is strongly diagonally dominant at CFL time steps; ~20x faster than a direct solve at 512 x 256).
    Falls back to a direct solve if it does not converge."""
    A = A.tocsr()
    d = A.diagonal()
    M = LinearOperator(A.shape, matvec=lambda v: v / d)
    x, info = bicgstab(A, b, x0=x0, rtol=1e-12, atol=0.0, M=M, maxiter=1000)
    if info != 0:
        x = spsolve(A.tocsc(), b)
    return x


def phase_change(theta, phi, theta_m, sc):
    """Heat above the melting curve melts ice; water (above background) in colder ice freezes.
    Conserves theta + phi / S exactly."""
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
    pg = np.pad(phi, 1, mode="edge")
    up = lambda u, a, b: np.where(u > 0, a, b)
    Fx = vx * up(vx, pg[1:-1, :-1], pg[1:-1, 1:])
    Fz = vz * up(vz, pg[:-1, 1:-1], pg[1:, 1:-1])
    return phi - dt * ((Fx[:, 1:] - Fx[:, :-1]) / g.dx + (Fz[1:] - Fz[:-1]) / g.dz)


def percolate_down(phi, dz, dt, sc, cfl=0.9):
    """Downward Darcy percolation in every column; the surface is impermeable, water leaving the
    bottom cell drains into the ocean. Returns phi and drained water volume per column (non-dim)."""
    phi = phi.copy()
    out = np.zeros(phi.shape[1])
    t = 0.0
    while t < dt:
        s = sc.darcy_speed(phi).max()
        h = dt - t if s <= 0 else min(dt - t, cfl * dz / s)
        F = sc.darcy_down(phi)                         # through the bottom face of each cell
        Fin = np.vstack([F[1:], np.zeros((1, phi.shape[1]))])   # from the cell above (none at the top)
        phi -= h / dz * (F - Fin)
        out += h * F[0]
        t += h
    return phi, out


def conductive_theta(z, sc):
    """Steady conduction with k = a/T between T_b (z = 0) and T_s (z = 1): T = T_b (T_s/T_b)^z."""
    return (sc.T_b * (sc.p.T_s / sc.T_b) ** z - sc.p.T_s) / sc.dT


def run(p=None, nx=128, nz=64, aspect=2.0, t_end_myr=50.0, cfl=0.5, snap_myr=(), log_every=100,
        theta0=None, phi0=None, t0_myr=0.0, solver="stag", verbose=True, wall_limit=None, dt_max=1e-3,
        seed=0):
    p = p or ShellParams()
    sc = ShellScales(p)
    g = Grid(nx, nz, L=aspect, H=1.0)
    Xc, Zc = np.meshgrid(g.xc, g.zc)
    theta_m = sc.theta_m(Zc)
    if theta0 is None:
        rng = np.random.default_rng(seed)
        theta = conductive_theta(Zc, sc) + 0.02 * np.sin(np.pi * Zc) * (np.cos(np.pi * Xc / aspect)
                                                                         + 0.3 * rng.standard_normal(Xc.shape))
        theta = np.clip(theta, 0.0, theta_m)
    else:
        theta = theta0.copy()
    phi = np.full((nz, nx), p.phi_bg) if phi0 is None else phi0.copy()
    E = lambda th, ph: (th + ph / sc.S).sum() * g.dx * g.dz
    E0 = E(theta, phi)
    acc = dict(base=0.0, top=0.0, tidal=0.0, drained=0.0)
    t, step = t0_myr * 1e6 * YEAR / sc.t, 0
    t_end = t_end_myr * 1e6 * YEAR / sc.t
    snaps_left = sorted(snap_myr)
    hist, snaps = [], []
    q_cond = A_K_ICE * np.log(sc.T_b / p.T_s) / p.D
    k_top, k_bot = float(sc.conductivity(0.0)), 1.0

    def snap(t_myr):
        return dict(t_myr=t_myr, theta=theta.copy(), phi=phi.copy(), vx=vx.copy(), vz=vz.copy(),
                    T=sc.T(theta), dT=sc.T(theta) - sc.T(theta_m), eta=sc.viscosity(theta) * sc.eta_b,
                    H_tidal=sc.tidal(sc.viscosity(theta)) * sc.k_b * sc.dT / p.D**2)

    w0 = time.time()
    cache = {}
    while t < t_end - 1e-15:
        vx, vz, eta = velocity(g, theta, phi, sc, solver=solver, cache=cache)
        vmax = max(np.abs(vx).max(), np.abs(vz).max(), 1e-12)
        dt = min(cfl * min(g.dx, g.dz) / vmax, t_end - t, dt_max)
        h = sc.tidal(eta)
        A, b, kt, kb = energy_matrix(g, vx, vz, dt, sc.conductivity(theta), k_top, k_bot, h)
        theta = solve_energy(A, theta.ravel() / dt + b, theta.ravel()).reshape(nz, nx)
        q_top = 2 * kt * theta[-1] / g.dz                    # out of the surface, per column
        q_base = 2 * kb * (1.0 - theta[0]) / g.dz             # out of the ocean into the shell
        theta, phi = phase_change(theta, phi, theta_m, sc)
        phi = advect_porosity(g, phi, vx, vz, dt)
        phi, w = percolate_down(phi, g.dz, dt, sc)
        theta, phi = phase_change(theta, phi, theta_m, sc)
        t += dt
        step += 1
        acc["base"] += q_base.sum() * g.dx * dt
        acc["top"] += q_top.sum() * g.dx * dt
        acc["tidal"] += h.sum() * g.dx * g.dz * dt
        acc["drained"] += w.sum() * g.dx / sc.S
        qb = q_base.mean() * sc.qd
        rec = dict(t_myr=t * sc.t / YEAR / 1e6, step=step,
                   q_surface=q_top.mean() * sc.qd, q_base=qb,
                   q_tidal=h.sum() * g.dx * g.dz / g.L * sc.qd,
                   q_drain=w.sum() * g.dx / sc.S / g.L / dt * sc.qd,
                   drain_mm_yr=w.sum() * g.dx / g.L / dt * sc.v * YEAR * 1e3,
                   Nu=q_top.mean() * sc.qd / q_cond,
                   growth_km_myr=(qb - p.q_ocean) / (p.rho_i * p.L) * YEAR * 1e6 / 1e3,
                   vrms_cm_yr=vrms(vx, vz) * sc.v * YEAR * 100,
                   phi_mean=phi.mean(), phi_max=phi.max(),
                   temperate=float(np.mean(theta >= theta_m - 1e-9)),
                   T_mean=float(sc.T(theta).mean()),
                   energy_error=(E(theta, phi) - E0 - acc["base"] + acc["top"] - acc["tidal"] + acc["drained"])
                   / max(acc["base"] + acc["tidal"], 1e-30))
        hist.append(rec)
        while snaps_left and rec["t_myr"] >= snaps_left[0] - 1e-9:
            snaps.append(snap(rec["t_myr"]))
            snaps_left.pop(0)
        if verbose and step % log_every == 0 and mpi_rank() == 0:
            print(f"step {step:6d} t={rec['t_myr']:8.3f} Myr  q_surf={rec['q_surface']*1e3:6.2f} "
                  f"q_base={rec['q_base']*1e3:6.2f} q_tidal={rec['q_tidal']*1e3:6.2f} q_drain={rec['q_drain']*1e3:6.2f} mW/m2  "
                  f"Nu={rec['Nu']:.3f}  vrms={rec['vrms_cm_yr']:.3g} cm/yr  phi_max={rec['phi_max']*100:.2f}%  "
                  f"temperate={rec['temperate']:.3f}  Eerr={rec['energy_error']:.1e}  stokes_it={cache.get('iters', '-')}  "
                  f"wall={time.time()-w0:.0f}s",
                  flush=True)
        if wall_limit and time.time() - w0 > wall_limit:
            break
    snaps.append(snap(t * sc.t / YEAR / 1e6))
    info = dict(T_b=sc.T_b, Ra=sc.Ra, eta_b=sc.eta_b, S=sc.S, t_scale_myr=sc.t / YEAR / 1e6,
                vscale_cm_yr=sc.v * YEAR * 100, D_km=p.D / 1e3, aspect=aspect, q_cond=q_cond)
    return dict(g=g, sc=sc, hist=hist, snaps=snaps, theta=theta, phi=phi, info=info)
