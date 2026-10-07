"""2D Stokes flow on a staggered finite-difference grid (Harlow & Welch 1965; Gerya 2019).

Box [0, L] x [0, H], z positive up, gravity along -z, free-slip walls. Unknowns:
    vx at vertical cell faces, vz at horizontal faces, P at cell centres;
    shear viscosity at cell corners (eta_n), normal-stress viscosity at centres (eta_c).
`stokes_system` is the readable loop assembly; `stokes_matrix_fast` builds the identical
matrix with index arrays and is what the solvers use.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

class Grid:
    """Staggered grid on a box [0, L] x [0, H], z positive up.
    vx at vertical cell faces  (x = i dx,       z = (j+1/2) dz): shape (nz, nx+1)
    vz at horizontal faces     (x = (i+1/2) dx, z = j dz):       shape (nz+1, nx)
    P, eta_c at cell centres   (x = (i+1/2) dx, z = (j+1/2) dz): shape (nz, nx)
    eta_n at cell corners      (x = i dx,       z = j dz):       shape (nz+1, nx+1)"""
    def __init__(self, nx, nz, L=1.0, H=1.0):
        self.nx, self.nz, self.L, self.H = nx, nz, L, H
        self.dx, self.dz = L / nx, H / nz
        self.xc = (np.arange(nx) + 0.5) * self.dx
        self.zc = (np.arange(nz) + 0.5) * self.dz
        self.xn = np.arange(nx + 1) * self.dx
        self.zn = np.arange(nz + 1) * self.dz
        self.n_vx = nz * (nx + 1)
        self.n_vz = (nz + 1) * nx
        self.n_p = nz * nx

    def ivx(self, j, i): return j * (self.nx + 1) + i
    def ivz(self, j, i): return self.n_vx + j * self.nx + i
    def ip(self, j, i):  return self.n_vx + self.n_vz + j * self.nx + i

def stokes_system(g, eta_c, eta_n, rho_vz, grav=1.0):
    """Free-slip box. Solves  div(sigma) - grad P = rho g e_z,  div v = 0  (gravity points -z).
    rho_vz: density at vz points, shape (nz+1, nx). Returns vx, vz, P."""
    nx, nz, dx, dz = g.nx, g.nz, g.dx, g.dz
    rows, cols, vals = [], [], []
    b = np.zeros(g.n_vx + g.n_vz + g.n_p)
    def add(r, c, v):
        rows.append(r); cols.append(c); vals.append(v)

    # x-momentum at vx points
    for j in range(nz):
        for i in range(nx + 1):
            r = g.ivx(j, i)
            if i == 0 or i == nx:                     # no flow through side walls
                add(r, r, 1.0); continue
            # d(sigma_xx)/dx, sigma_xx = 2 eta_c dvx/dx at centres (j,i) and (j,i-1)
            eR, eL = eta_c[j, i], eta_c[j, i - 1]
            add(r, g.ivx(j, i + 1), 2 * eR / dx**2)
            add(r, g.ivx(j, i),    -2 * (eR + eL) / dx**2)
            add(r, g.ivx(j, i - 1), 2 * eL / dx**2)
            # d(sigma_xz)/dz, sigma_xz = eta_n (dvx/dz + dvz/dx) at nodes (j+1,i) top and (j,i) bottom
            if j + 1 < nz:                            # top node interior (free slip: sigma_xz = 0 on top wall)
                eT = eta_n[j + 1, i]
                add(r, g.ivx(j + 1, i), eT / dz**2)
                add(r, g.ivx(j, i),    -eT / dz**2)
                add(r, g.ivz(j + 1, i),     eT / (dx * dz))
                add(r, g.ivz(j + 1, i - 1), -eT / (dx * dz))
            if j > 0:                                 # bottom node interior
                eB = eta_n[j, i]
                add(r, g.ivx(j, i),    -eB / dz**2)
                add(r, g.ivx(j - 1, i), eB / dz**2)
                add(r, g.ivz(j, i),     -eB / (dx * dz))
                add(r, g.ivz(j, i - 1),  eB / (dx * dz))
            # -dP/dx
            add(r, g.ip(j, i),     -1.0 / dx)
            add(r, g.ip(j, i - 1),  1.0 / dx)

    # z-momentum at vz points
    for j in range(nz + 1):
        for i in range(nx):
            r = g.ivz(j, i)
            if j == 0 or j == nz:                     # no flow through top/bottom
                add(r, r, 1.0); continue
            eT, eB = eta_c[j, i], eta_c[j - 1, i]
            add(r, g.ivz(j + 1, i), 2 * eT / dz**2)
            add(r, g.ivz(j, i),    -2 * (eT + eB) / dz**2)
            add(r, g.ivz(j - 1, i), 2 * eB / dz**2)
            if i + 1 < nx:                            # right node interior
                eR = eta_n[j, i + 1]
                add(r, g.ivz(j, i + 1), eR / dx**2)
                add(r, g.ivz(j, i),    -eR / dx**2)
                add(r, g.ivx(j, i + 1),     eR / (dx * dz))
                add(r, g.ivx(j - 1, i + 1), -eR / (dx * dz))
            if i > 0:                                 # left node interior
                eL = eta_n[j, i]
                add(r, g.ivz(j, i),    -eL / dx**2)
                add(r, g.ivz(j, i - 1), eL / dx**2)
                add(r, g.ivx(j, i),     -eL / (dx * dz))
                add(r, g.ivx(j - 1, i),  eL / (dx * dz))
            add(r, g.ip(j, i),     -1.0 / dz)
            add(r, g.ip(j - 1, i),  1.0 / dz)
            b[r] = rho_vz[j, i] * grav

    # continuity at centres (one row replaced by P = 0 to fix the pressure constant)
    for j in range(nz):
        for i in range(nx):
            r = g.ip(j, i)
            if i == 0 and j == 0:
                add(r, r, 1.0); continue
            add(r, g.ivx(j, i + 1),  1.0 / dx)
            add(r, g.ivx(j, i),     -1.0 / dx)
            add(r, g.ivz(j + 1, i),  1.0 / dz)
            add(r, g.ivz(j, i),     -1.0 / dz)

    A = sp.csc_matrix((vals, (rows, cols)), shape=(b.size, b.size))
    return A, b

def unpack(g, s):
    vx = s[:g.n_vx].reshape(g.nz, g.nx + 1)
    vz = s[g.n_vx:g.n_vx + g.n_vz].reshape(g.nz + 1, g.nx)
    P = s[g.n_vx + g.n_vz:].reshape(g.nz, g.nx)
    return vx, vz, P

def solve_stokes(g, eta_c, eta_n, rho_vz, grav=1.0):
    A, b = stokes_system(g, eta_c, eta_n, rho_vz, grav)
    return unpack(g, spsolve(A, b))


def stokes_matrix_fast(g, eta_c, eta_n):
    """Vectorised assembly of the same matrix as stokes_system (no Python loops over cells)."""
    nx, nz, dx, dz = g.nx, g.nz, g.dx, g.dz
    R, C, V = [], [], []
    def put(r, c, v):
        r, c, v = np.broadcast_arrays(r, c, v)
        R.append(r.ravel()); C.append(c.ravel()); V.append(np.asarray(v, float).ravel())
    ivx = lambda j, i: j * (nx + 1) + i
    ivz = lambda j, i: g.n_vx + j * nx + i
    ip  = lambda j, i: g.n_vx + g.n_vz + j * nx + i

    # --- x-momentum ---
    J, I = np.meshgrid(np.arange(nz), np.arange(nx + 1), indexing="ij")
    wall = (I == 0) | (I == nx)
    put(ivx(J[wall], I[wall]), ivx(J[wall], I[wall]), 1.0)
    j, i = J[~wall], I[~wall]
    r = ivx(j, i)
    eR, eL = eta_c[j, i], eta_c[j, i - 1]
    put(r, ivx(j, i + 1), 2 * eR / dx**2); put(r, r, -2 * (eR + eL) / dx**2); put(r, ivx(j, i - 1), 2 * eL / dx**2)
    m = j + 1 < nz; jj, ii, rr = j[m], i[m], r[m]; eT = eta_n[jj + 1, ii]
    put(rr, ivx(jj + 1, ii), eT / dz**2); put(rr, rr, -eT / dz**2)
    put(rr, ivz(jj + 1, ii), eT / (dx * dz)); put(rr, ivz(jj + 1, ii - 1), -eT / (dx * dz))
    m = j > 0; jj, ii, rr = j[m], i[m], r[m]; eB = eta_n[jj, ii]
    put(rr, rr, -eB / dz**2); put(rr, ivx(jj - 1, ii), eB / dz**2)
    put(rr, ivz(jj, ii), -eB / (dx * dz)); put(rr, ivz(jj, ii - 1), eB / (dx * dz))
    put(r, ip(j, i), -1.0 / dx); put(r, ip(j, i - 1), 1.0 / dx)

    # --- z-momentum ---
    J, I = np.meshgrid(np.arange(nz + 1), np.arange(nx), indexing="ij")
    wall = (J == 0) | (J == nz)
    put(ivz(J[wall], I[wall]), ivz(J[wall], I[wall]), 1.0)
    j, i = J[~wall], I[~wall]
    r = ivz(j, i)
    eT, eB = eta_c[j, i], eta_c[j - 1, i]
    put(r, ivz(j + 1, i), 2 * eT / dz**2); put(r, r, -2 * (eT + eB) / dz**2); put(r, ivz(j - 1, i), 2 * eB / dz**2)
    m = i + 1 < nx; jj, ii, rr = j[m], i[m], r[m]; eR = eta_n[jj, ii + 1]
    put(rr, ivz(jj, ii + 1), eR / dx**2); put(rr, rr, -eR / dx**2)
    put(rr, ivx(jj, ii + 1), eR / (dx * dz)); put(rr, ivx(jj - 1, ii + 1), -eR / (dx * dz))
    m = i > 0; jj, ii, rr = j[m], i[m], r[m]; eL = eta_n[jj, ii]
    put(rr, rr, -eL / dx**2); put(rr, ivz(jj, ii - 1), eL / dx**2)
    put(rr, ivx(jj, ii), -eL / (dx * dz)); put(rr, ivx(jj - 1, ii), eL / (dx * dz))
    put(r, ip(j, i), -1.0 / dz); put(r, ip(j - 1, i), 1.0 / dz)

    # --- continuity ---
    J, I = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
    fix = (J == 0) & (I == 0)
    put(ip(0, 0), ip(0, 0), 1.0)
    j, i = J[~fix], I[~fix]; r = ip(j, i)
    put(r, ivx(j, i + 1), 1 / dx); put(r, ivx(j, i), -1 / dx); put(r, ivz(j + 1, i), 1 / dz); put(r, ivz(j, i), -1 / dz)

    n = g.n_vx + g.n_vz + g.n_p
    return sp.csc_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(n, n))
