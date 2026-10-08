"""Thermal convection on PETSc DMStag (Stage 3e, 2D): parallel Stokes + energy, steady Picard iteration.
Every rank keeps the full (small) T and velocity arrays (allgather) and assembles only its own rows.
Energy discretisation is identical to ganymede.ice.energy_steady_matrix_k, so results must match the SciPy code."""
import time
import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc

from ganymede.stokes import Grid
from ganymede.ice import energy_steady_matrix_k, corner_T, top_flux
from ganymede.convection import T_ghost, vrms
from ganymede.stag_stokes import native_stokes_solve, make_dmstag, StagIndex


def _comm_py(comm):
    try:
        return comm.tompi4py()
    except Exception:
        return None


def allgather_fields(x, ix, nx, nz):
    """vx, vz, P on every rank."""
    xa = x.getArray(); r0, _ = x.getOwnershipRange()
    pieces = [(k, *ix.owned(k), xa[ix(k, *ix.owned(k)) - r0]) for k in ("vx", "vz", "P")]
    mc = _comm_py(x.getComm())
    allp = mc.allgather(pieces) if mc is not None and mc.Get_size() > 1 else [pieces]
    out = {"vx": np.zeros((nz, nx + 1)), "vz": np.zeros((nz + 1, nx)), "P": np.zeros((nz, nx))}
    for pc in allp:
        for k, j, i, v in pc:
            out[k][j, i] = v
    return out["vx"], out["vz"], out["P"]


class CellLayout:
    """Cell-centred field (temperature) on a DMStag with one dof per element."""
    def __init__(self, nx, nz, L=1.0, H=1.0, comm=None):
        self.dm = make_dmstag(nx, nz, L, H, dofs=(0, 0, 1), comm=comm)
        self.ix = StagIndex(self.dm, nx, nz)
        self.nx, self.nz = nx, nz
        j, i = self.ix.owned("P")
        self.own_nat = j * nx + i                                   # natural index of owned cells
        self.own_glb = self.ix("P", j, i)                           # DMStag global index
        mc = _comm_py(self.dm.getComm())
        pairs = mc.allgather((self.own_nat, self.own_glb)) if mc is not None and mc.Get_size() > 1 else [(self.own_nat, self.own_glb)]
        self.nat2glb = np.empty(nx * nz, dtype=PETSc.IntType)
        for nat, glb in pairs:
            self.nat2glb[nat] = glb
        self.r0, self.r1 = self.dm.createGlobalVec().getOwnershipRange()

    def allgather(self, vec):
        a = vec.getArray(); full = np.zeros(self.nx * self.nz)
        mc = _comm_py(vec.getComm())
        loc = (self.own_nat, a[self.own_glb - self.r0])
        allp = mc.allgather(loc) if mc is not None and mc.Get_size() > 1 else [loc]
        for nat, v in allp:
            full[nat] = v
        return full.reshape(self.nz, self.nx)


def energy_solve(layout, vx, vz, k_c, k_bot, k_top, T_old=None, dt=None):
    """Steady (dt=None) or backward-Euler step of div(vT) - div(k grad T) = 0 on the DMStag cell layout.
    Rows: the owned rows of the SciPy matrix, columns renumbered to DMStag ordering."""
    g = Grid(layout.nx, layout.nz)
    A_nat, b_nat = energy_steady_matrix_k(g, vx, vz, k_c, k_bot, k_top)
    A_nat = A_nat.tocsr()
    if dt is not None:
        A_nat = A_nat + sp.identity(A_nat.shape[0], format="csr") / dt
        b_nat = b_nat + T_old.ravel() / dt
    order = np.argsort(layout.own_glb)                 # owned rows in increasing DMStag index
    rows_nat = layout.own_nat[order]
    A_loc = A_nat[rows_nat]
    A_loc = sp.csr_matrix((A_loc.data, layout.nat2glb[A_loc.indices], A_loc.indptr), shape=A_loc.shape)
    A_loc.sort_indices()
    n_loc, N = layout.r1 - layout.r0, layout.nx * layout.nz
    comm = layout.dm.getComm()
    A = PETSc.Mat().createAIJ(size=((n_loc, N), (n_loc, N)),
                              csr=(A_loc.indptr.astype(PETSc.IntType), A_loc.indices.astype(PETSc.IntType), A_loc.data), comm=comm)
    A.assemble()
    b = layout.dm.createGlobalVec(); b.setArray(b_nat[rows_nat])
    # Advection-dominated and non-symmetric: GMRES with additive Schwarz (overlapping subdomains, so hot plumes
    # crossing a process boundary are still seen by the preconditioner) and exact LU on each subdomain.
    # (Block Jacobi without overlap stalled: >2800 iterations at 128², no convergence on 4 processes.)
    ksp = PETSc.KSP().create(comm=comm); ksp.setOptionsPrefix("energy_"); ksp.setOperators(A)
    o = PETSc.Options()
    o["energy_ksp_type"] = "gmres"; o["energy_ksp_gmres_restart"] = 100
    o["energy_pc_type"] = "asm"; o["energy_pc_asm_overlap"] = 2; o["energy_sub_pc_type"] = "lu"
    ksp.setTolerances(rtol=1e-12, max_it=5000); ksp.setFromOptions()
    x = layout.dm.createGlobalVec(); ksp.solve(b, x)
    if ksp.getConvergedReason() <= 0:
        raise RuntimeError(f"energy solve did not converge: {ksp.getConvergedReason()}")
    return layout.allgather(x)


def steady_convection_stag(nx, nz, Ra_b, eta_fn, k_fn=None, relax=0.5, tol=1e-7, max_iter=800, comm=None):
    """Same Picard iteration as ganymede.ice.steady_convection_general, but Stokes and energy solved with PETSc
    on DMStag (MPI-parallel). eta_fn, k_fn: functions of T normalised to 1 at T = 1."""
    g = Grid(nx, nz)
    k_fn = k_fn or (lambda T: np.ones_like(np.asarray(T, float)))
    layout = CellLayout(nx, nz, comm=comm)
    Xc, Zc = np.meshgrid(g.xc, g.zc)
    T = 1 - Zc + 0.1 * np.cos(np.pi * Xc) * np.sin(np.pi * Zc)
    t0 = time.time()
    for it in range(1, max_iter + 1):
        Tg = T_ghost(T)
        rho = -Ra_b * 0.5 * (Tg[1:, 1:-1] + Tg[:-1, 1:-1])
        dm, x, ix, st = native_stokes_solve(nx, nz, eta_fn(T), eta_fn(corner_T(T)), rho, comm=comm)
        vx, vz, _ = allgather_fields(x, ix, nx, nz)
        k_c = k_fn(T)
        T_new = energy_solve(layout, vx, vz, k_c, float(k_fn(1.0)), float(k_fn(0.0)))
        change = np.abs(T_new - T).max()
        T = relax * T_new + (1 - relax) * T
        if change < tol:
            return g, T, vx, vz, dict(iterations=it, time=time.time() - t0, nproc=layout.dm.getComm().getSize(),
                                      Nu=top_flux(T, g, float(k_fn(0.0))), vrms=vrms(vx, vz))
    raise RuntimeError(f"no steady state after {max_iter} iterations (last change {change:.1e})")
