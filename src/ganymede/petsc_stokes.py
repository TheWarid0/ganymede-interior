"""Stokes solve with PETSc (Stage 3a): the staggered-grid system rearranged into symmetric saddle-point form
and solved with FGMRES + Schur-complement field split. Requires petsc4py (conda-forge: petsc petsc4py)."""
import time

import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc

from .stokes import stokes_matrix_fast, unpack


def stokes_symmetric(g, eta_c, eta_n, noslip_bottom=False):
    """The same staggered-grid discretisation, rearranged for iterative solvers:
      1. no pinned pressure: every cell keeps its continuity equation (the solver handles the constant via a null space)
      2. all rows multiplied by -1 -> velocity block symmetric positive definite, whole system [[K, G], [G^T, 0]]
      3. wall velocities (= 0) decoupled: their columns removed, their rows scaled like the interior
    With noslip_bottom, vx = 0 on the bottom wall (mirrored ghost: an extra -2 eta / dz^2 on the diagonal).
    Returns the matrix A, the row signs s (apply to the RHS too), and the wall-velocity rows."""
    A = stokes_matrix_fast(g, eta_c, eta_n).tocsr()
    if noslip_bottom:
        rows = g.ivx(0, np.arange(1, g.nx))
        A = A + sp.csr_matrix((-2 * eta_n[0, 1:g.nx] / g.dz**2, (rows, rows)), shape=A.shape)
    nv = g.n_vx + g.n_vz
    # 1. restore the continuity equation of cell (0, 0)
    r0 = nv
    A = A.tolil()
    A.rows[r0], A.data[r0] = [], []
    for col, val in ((g.ivx(0, 1), 1 / g.dx), (g.ivx(0, 0), -1 / g.dx), (g.ivz(1, 0), 1 / g.dz), (g.ivz(0, 0), -1 / g.dz)):
        A[r0, col] = val
    A = A.tocsr()
    # 2. flip signs
    s = -np.ones(A.shape[0])
    A = sp.diags(s) @ A
    # 3. wall velocities: rows that are pure identity rows in the velocity part
    d = A.diagonal()
    nnz_row = np.diff(A.indptr)
    walls = np.where((nnz_row[:nv] == 1) & np.isclose(d[:nv], -1.0))[0]
    scale = np.median(np.abs(np.delete(d[:nv], walls)))
    keep = np.ones(A.shape[0]); keep[walls] = 0.0
    A = (A @ sp.diags(keep)).tocsr()
    A = A + sp.csr_matrix((np.full(len(walls), scale), (walls, walls)), shape=A.shape)
    A = (A + sp.diags(np.zeros(A.shape[0]))).tocsr()     # explicit diagonal entries (PETSc wants them)
    A.eliminate_zeros(); A = (A + sp.diags(np.zeros(A.shape[0]))).tocsr()
    return A, s, walls


def petsc_stokes_solve(g, eta_c, eta_n, rho_vz, rtol=1e-10, u_pc="ilu", u_rtol=1e-8, max_it=300,
                       noslip_bottom=False, x0=None):
    """FGMRES on the full system, preconditioned by a Schur-complement split:
       velocity block K      -> inner CG + u_pc ('ilu' here; geometric multigrid later)
       Schur complement S    -> approximated by -diag(1/eta)  (the classic Stokes 'pressure mass matrix')
       constant pressure     -> declared as a null space.
    x0: optional initial guess (vx, vz, P flattened, e.g. the previous time step)."""
    A, s, walls = stokes_symmetric(g, eta_c, eta_n, noslip_bottom=noslip_bottom)
    b = np.zeros(A.shape[0])
    bz = rho_vz.copy(); bz[0] = bz[-1] = 0.0
    b[g.n_vx:g.n_vx + g.n_vz] = bz.ravel()
    b *= s; b[walls] = 0.0
    nv = g.n_vx + g.n_vz

    M = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
    null = M.createVecRight(); null.setArray(np.r_[np.zeros(nv), np.full(g.n_p, 1 / np.sqrt(g.n_p))])
    M.setNullSpace(PETSc.NullSpace().create(vectors=[null]))
    S = sp.diags(-1.0 / eta_c.ravel()).tocsr()
    S_pc = PETSc.Mat().createAIJ(size=S.shape, csr=(S.indptr, S.indices, S.data))

    ksp = PETSc.KSP().create()
    ksp.setOperators(M); ksp.setType("fgmres")
    pc = ksp.getPC(); pc.setType("fieldsplit")
    pc.setFieldSplitIS(("u", PETSc.IS().createStride(nv, 0, 1)), ("p", PETSc.IS().createStride(g.n_p, nv, 1)))
    pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
    pc.setFieldSplitSchurFactType(PETSc.PC.FieldSplitSchurFactType.UPPER)
    pc.setFieldSplitSchurPreType(PETSc.PC.FieldSplitSchurPreType.USER, S_pc)
    opts = PETSc.Options()
    opts["fieldsplit_u_ksp_type"] = "cg";      opts["fieldsplit_u_pc_type"] = u_pc
    opts["fieldsplit_u_ksp_rtol"] = u_rtol
    opts["fieldsplit_p_ksp_type"] = "preonly"; opts["fieldsplit_p_pc_type"] = "jacobi"
    pc.setFromOptions()
    ksp.setTolerances(rtol=rtol, max_it=max_it)

    x, rhs = M.createVecRight(), M.createVecLeft()
    rhs.setArray(b)
    if x0 is not None:
        x.setArray(x0)
        ksp.setInitialGuessNonzero(True)
    t0 = time.time(); ksp.solve(rhs, x); elapsed = time.time() - t0
    vx, vz, P = unpack(g, x.getArray().copy())
    return vx, vz, P, ksp.getIterationNumber(), ksp.getConvergedReason(), elapsed
