"""Thermal convection (Boussinesq, non-dimensional) on the staggered Stokes grid.

    -grad P + div(2 eta e) = -Ra T e_z,   div v = 0,   dT/dt + div(v T) = lap T
T at cell centres, T = 1 at the bottom, T = 0 at the top, insulating sides, free slip.
Two drivers: `convect` (explicit time stepping, constant viscosity, factorise once) and
`steady_convection` (Picard iteration to steady state, temperature-dependent viscosity).
Benchmarked against Blankenbach et al. (1989), cases 1a and 2a.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, spsolve

from .stokes import Grid, stokes_system, stokes_matrix_fast, unpack

def T_ghost(T, T_bot=1.0, T_top=0.0):
    """T at cell centres padded with ghost cells: fixed T at bottom and top, insulating sides."""
    Tg = np.pad(T, 1)
    Tg[1:-1, 0], Tg[1:-1, -1] = T[:, 0], T[:, -1]       # dT/dx = 0 on the side walls
    Tg[0, 1:-1] = 2 * T_bot - T[0]                        # wall value is the mean of ghost and first cell
    Tg[-1, 1:-1] = 2 * T_top - T[-1]
    return Tg

def dTdt(T, vx, vz, g):
    """-div(v T) + laplacian(T): conservative central differences, velocities on cell faces."""
    Tg = T_ghost(T)
    Tfx = 0.5 * (Tg[1:-1, 1:] + Tg[1:-1, :-1])          # T on vertical faces
    Tfz = 0.5 * (Tg[1:, 1:-1] + Tg[:-1, 1:-1])          # T on horizontal faces
    adv = (vx[:, 1:] * Tfx[:, 1:] - vx[:, :-1] * Tfx[:, :-1]) / g.dx \
        + (vz[1:] * Tfz[1:] - vz[:-1] * Tfz[:-1]) / g.dz
    lap = (Tg[1:-1, 2:] - 2 * T + Tg[1:-1, :-2]) / g.dx**2 + (Tg[2:, 1:-1] - 2 * T + Tg[:-2, 1:-1]) / g.dz**2
    return -adv + lap

def nusselt(T, g):
    """Mean heat flux through the top wall (T = 0), second-order one-sided gradient."""
    dTdz = (8 * 0.0 - 9 * T[-1] + T[-2]) / (3 * g.dz)
    return -dTdz.mean()

def vrms(vx, vz):
    vxc = 0.5 * (vx[:, 1:] + vx[:, :-1]); vzc = 0.5 * (vz[1:] + vz[:-1])
    return np.sqrt(np.mean(vxc**2 + vzc**2))

def convect(n, Ra, t_end=0.6, cfl=0.5):
    g = Grid(n, n)
    A, _ = stokes_system(g, np.ones((n, n)), np.ones((n + 1, n + 1)), np.zeros((n + 1, n)))
    lu = splu(A)                                          # constant viscosity: factorise ONCE
    Xc, Zc = np.meshgrid(g.xc, g.zc)
    T = 1 - Zc + 0.01 * np.cos(np.pi * Xc) * np.sin(np.pi * Zc)   # conductive profile + small kick
    b = np.zeros(A.shape[0])
    t, hist = 0.0, []
    while t < t_end:
        # Stokes: buoyancy rho' = -Ra T at interior vz points
        T_vz = 0.5 * (T_ghost(T)[1:, 1:-1] + T_ghost(T)[:-1, 1:-1])
        bz = -Ra * T_vz
        bz[0], bz[-1] = 0.0, 0.0                          # boundary rows (vz = 0)
        b[g.n_vx:g.n_vx + g.n_vz] = bz.ravel()
        vx, vz, _ = unpack(g, lu.solve(b))
        # energy: Heun (RK2) step with the velocity held fixed
        dt = min(cfl * g.dx / max(np.abs(vx).max(), np.abs(vz).max(), 1e-12), 0.2 * g.dx**2)
        k1 = dTdt(T, vx, vz, g)
        T = T + 0.5 * dt * (k1 + dTdt(T + dt * k1, vx, vz, g))
        t += dt
        hist.append((t, nusselt(T, g), vrms(vx, vz)))
    return g, T, vx, vz, np.array(hist)


def energy_steady_matrix(g, vx, vz, T_bot=1.0, T_top=0.0):
    """Steady advection–diffusion  div(v T) − ∇²T = 0  for T at cell centres.
    Central differences (fluxes on cell faces), fixed T at bottom/top via ghost cells, insulating sides."""
    nx, nz, dx, dz = g.nx, g.nz, g.dx, g.dz
    idx = np.arange(nx * nz).reshape(nz, nx)
    R, C, V = [], [], []
    b = np.zeros(nx * nz)
    def put(r, c, v):
        r, c, v = np.broadcast_arrays(r, c, v)
        R.append(r.ravel()); C.append(c.ravel()); V.append(np.asarray(v, float).ravel())

    J, I = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
    # Each interior face between cell P and neighbour N contributes, to row P:
    #   ± u (T_P + T_N) / (2 h)   (advective flux, + for the east/top face, − for west/bottom)
    #   − (T_N − T_P) / h²        (diffusive flux)
    faces = [  # (neighbour offset dj, di, face velocity, sign, grid spacing)
        (0, +1, lambda j, i: vx[j, i + 1], +1, dx),   # east
        (0, -1, lambda j, i: vx[j, i],     -1, dx),   # west
        (+1, 0, lambda j, i: vz[j + 1, i], +1, dz),   # top
        (-1, 0, lambda j, i: vz[j, i],     -1, dz),   # bottom
    ]
    for dj, di, face_v, sgn, h in faces:
        inner = (J + dj >= 0) & (J + dj < nz) & (I + di >= 0) & (I + di < nx)
        j, i = J[inner], I[inner]
        r, u = idx[j, i], face_v(j, i)
        put(r, r, sgn * u / (2 * h) + 1 / h**2)
        put(r, idx[j + dj, i + di], sgn * u / (2 * h) - 1 / h**2)
    # walls: velocity through every wall is 0; side walls are insulating (no flux);
    # top/bottom: ghost T = 2 T_wall − T_P gives a diffusive flux 2 (T_P − T_wall) / dz²
    for row, T_wall in ((0, T_bot), (nz - 1, T_top)):
        r = idx[row]
        put(r, r, 2 / dz**2)
        b[r] += 2 * T_wall / dz**2
    A = sp.csc_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(nx * nz, nx * nz))
    return A, b

def viscosity(T, b_visc):
    """eta = exp(−b T) at cell centres, and at corners from the mean T of the four surrounding cells."""
    Tg = T_ghost(T)
    T_corner = 0.25 * (Tg[1:, 1:] + Tg[:-1, 1:] + Tg[1:, :-1] + Tg[:-1, :-1])
    return np.exp(-b_visc * T), np.exp(-b_visc * T_corner)

def steady_convection(n, Ra, b_visc=0.0, relax=0.5, tol=1e-7, max_iter=300):
    """Picard iteration straight to steady state: Stokes with the current viscosity -> steady energy
    equation with that velocity -> relax T -> repeat until T stops changing."""
    g = Grid(n, n)
    Xc, Zc = np.meshgrid(g.xc, g.zc)
    T = 1 - Zc + 0.1 * np.cos(np.pi * Xc) * np.sin(np.pi * Zc)
    for it in range(1, max_iter + 1):
        eta_c, eta_n = viscosity(T, b_visc)
        A = stokes_matrix_fast(g, eta_c, eta_n)
        Tg = T_ghost(T)
        bz = -Ra * 0.5 * (Tg[1:, 1:-1] + Tg[:-1, 1:-1])
        bz[0] = bz[-1] = 0.0
        rhs = np.zeros(A.shape[0]); rhs[g.n_vx:g.n_vx + g.n_vz] = bz.ravel()
        vx, vz, _ = unpack(g, spsolve(A, rhs))
        Ae, be = energy_steady_matrix(g, vx, vz)
        T_new = spsolve(Ae, be).reshape(n, n)
        change = np.abs(T_new - T).max()
        T = relax * T_new + (1 - relax) * T
        if change < tol:
            return g, T, vx, vz, it
    raise RuntimeError(f"no steady state after {max_iter} iterations (last change {change:.1e})")
