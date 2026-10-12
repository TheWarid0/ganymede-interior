"""Stage 7b: MPI-parallel Stokes solver for the annulus sector (PETSc DMStag, geometric multigrid).

The polar grid maps one-to-one onto a 2D DMStag: x <-> theta (nt cells), z <-> r (nr cells);
v_theta sits on LEFT faces, v_r on DOWN faces, P on elements. Curvature lives only in the matrix
coefficients, so the DMStag multigrid hierarchy (interpolation + Galerkin coarse operators) applies as
in Stage 3. The matrix is the Stage 7a operator weighted by control volumes, which makes it exactly
symmetric: [[K, G], [G^T, 0]] with K positive definite. Wall velocities (zero) are decoupled with a
scaled diagonal, the constant pressure is a null space.

Solver: FGMRES + Schur-complement field split; velocity block: CG + geometric multigrid
(Chebyshev/Jacobi smoothing); Schur complement ~ -diag(w_p / eta).

Current limitation: every rank assembles the global matrix with SciPy and keeps its own rows. Fine on a
laptop / a few nodes; for the full half annulus at 0.3 km resolution the assembly must become local.
"""
import time

import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc

from .annulus import stokes_matrix, volume_weights
from .stag_stokes import StagIndex, make_dmstag


def _comm_py(comm):
    try:
        return comm.tompi4py()
    except Exception:
        return None


class AnnulusLayout:
    """DMStag for an AnnulusGrid and the maps between natural (Stage 7a) and DMStag ordering."""

    def __init__(self, g, comm=None):
        self.g = g
        self.dm = make_dmstag(g.nt, g.nr, comm=comm)
        self.dm_v = make_dmstag(g.nt, g.nr, dofs=(0, 1, 0), comm=comm)
        self.comm = self.dm.getComm()
        ix = StagIndex(self.dm, g.nt, g.nr)
        self.ix = ix
        nr, nt = g.nr, g.nt
        n_vr, n_vt = g.n_vr, g.n_vt
        # owned unknowns: stag index, natural index (-1 = wall), kind code (0 v_r, 1 v_t, 2 P)
        st, na, kd = [], [], []
        j, i = ix.owned("vz")                                    # v_r: radial faces j = 0..nr
        st.append(ix("vz", j, i)); kd.append(np.zeros_like(j))
        na.append(np.where((j > 0) & (j < nr), (j - 1) * nt + i, -1))
        j, i = ix.owned("vx")                                    # v_theta: angular faces i = 0..nt
        st.append(ix("vx", j, i)); kd.append(np.ones_like(j))
        na.append(np.where((i > 0) & (i < nt), n_vr + j * (nt - 1) + (i - 1), -1))
        j, i = ix.owned("P")
        st.append(ix("P", j, i)); kd.append(np.full_like(j, 2))
        na.append(n_vr + n_vt + j * nt + i)
        self.own_stag = np.concatenate(st).astype(PETSc.IntType)
        self.own_nat = np.concatenate(na).astype(np.int64)
        self.own_kind = np.concatenate(kd)
        mc = _comm_py(self.comm)
        pairs = mc.allgather((self.own_nat, self.own_stag)) if mc is not None and mc.Get_size() > 1 \
            else [(self.own_nat, self.own_stag)]
        self.N_nat = n_vr + n_vt + g.n_p
        self.nat2stag = np.empty(self.N_nat, dtype=PETSc.IntType)
        for nat, stg in pairs:
            m = nat >= 0
            self.nat2stag[nat[m]] = stg[m]
        self.r0, self.r1 = self.dm.createGlobalVec().getOwnershipRange()

    def is_fields(self):
        u = np.sort(self.own_stag[self.own_kind < 2]).astype(PETSc.IntType)
        p = np.sort(self.own_stag[self.own_kind == 2]).astype(PETSc.IntType)
        return u, p

    def to_natural(self, x):
        """Solution vector (DMStag) -> natural-ordering array on every rank."""
        xa = x.getArray()
        vals = xa[self.own_stag - self.r0]
        m = self.own_nat >= 0
        mc = _comm_py(self.comm)
        parts = mc.allgather((self.own_nat[m], vals[m])) if mc is not None and mc.Get_size() > 1 \
            else [(self.own_nat[m], vals[m])]
        out = np.zeros(self.N_nat)
        for nat, v in parts:
            out[nat] = v
        return out


def solve_annulus_petsc(g, eta_c, eta_n, f_r, f_t, layout=None, rtol=1e-8, u_rtol=1e-6, max_it=300,
                        x0=None, verbose=False):
    """Parallel Stokes solve on the annulus sector. Returns (v_r, v_t, P) in natural layout on every rank,
    the PETSc solution vector (usable as x0 next time) and statistics."""
    L = layout or AnnulusLayout(g)
    nr, nt = g.nr, g.nt
    w = volume_weights(g)
    A = (sp.diags(w) @ stokes_matrix(g, eta_c, eta_n, fix_pressure=False)).tocsr()
    b_nat = w * np.concatenate([np.ravel(f_r), np.ravel(f_t), np.zeros(g.n_p)])
    nv = g.n_vr + g.n_vt
    wall_scale = float(np.median(np.abs(A.diagonal()[:nv])))
    # this rank's rows, in DMStag numbering
    N = L.dm.createGlobalVec().getSize()
    rows_local = L.own_stag - L.r0
    m = L.own_nat >= 0
    sub = A[L.own_nat[m]]                                   # this rank's equations (CSR rows)
    R = np.repeat(rows_local[m], np.diff(sub.indptr))
    C = L.nat2stag[sub.indices]
    V = sub.data
    walls = ~m                                              # wall velocities: decoupled, scaled identity
    pr = L.own_kind == 2                                    # explicit zero diagonal on pressure rows
    R = np.concatenate([R, rows_local[walls], rows_local[pr]])
    C = np.concatenate([C, L.own_stag[walls], L.own_stag[pr]])
    V = np.concatenate([V, np.full(walls.sum(), wall_scale), np.zeros(pr.sum())])
    loc = sp.coo_matrix((V, (R, C)), shape=(L.r1 - L.r0, N)).tocsr()
    loc.sort_indices()
    M = PETSc.Mat().createAIJ(size=((L.r1 - L.r0, N), (L.r1 - L.r0, N)),
                              csr=(loc.indptr.astype(PETSc.IntType), loc.indices.astype(PETSc.IntType), loc.data),
                              comm=L.comm)
    M.assemble()
    b = L.dm.createGlobalVec()
    bl = np.zeros(L.r1 - L.r0)
    bl[rows_local[m]] = b_nat[L.own_nat[m]]
    b.setArray(bl)
    is_u, is_p = L.is_fields()
    null = L.dm.createGlobalVec(); null.set(0.0)
    null.setValues(is_p, np.full(is_p.size, 1 / np.sqrt(g.n_p))); null.assemble()
    M.setNullSpace(PETSc.NullSpace().create(vectors=[null], comm=L.comm))
    # Schur preconditioner -diag(w_p / eta), in the pressure sub-block numbering
    mc = _comm_py(L.comm)
    off = mc.exscan(is_p.size) or 0 if mc is not None and mc.Get_size() > 1 else 0
    p_nat = L.own_nat[L.own_kind == 2]
    p_stag = L.own_stag[L.own_kind == 2]
    srt = np.argsort(p_stag)
    p_nat = p_nat[srt] - nv
    wp = w[nv:][p_nat] / np.ravel(eta_c)[p_nat]
    S = PETSc.Mat().createAIJ(size=((is_p.size, g.n_p), (is_p.size, g.n_p)), nnz=1, comm=L.comm)
    for r, v in zip(range(off, off + is_p.size), wp):
        S.setValue(r, r, -v)
    S.assemble()
    ksp = PETSc.KSP().create(comm=L.comm)
    ksp.setOperators(M)
    ksp.setType("fgmres")
    pc = ksp.getPC(); pc.setType("fieldsplit")
    pc.setFieldSplitIS(("u", PETSc.IS().createGeneral(is_u, comm=L.comm)),
                       ("p", PETSc.IS().createGeneral(is_p, comm=L.comm)))
    pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
    pc.setFieldSplitSchurFactType(PETSc.PC.FieldSplitSchurFactType.UPPER)
    pc.setFieldSplitSchurPreType(PETSc.PC.FieldSplitSchurPreType.USER, S)
    ksp.setTolerances(rtol=rtol, max_it=max_it)
    ksp.setUp()
    # DMStag can only coarsen if every rank's piece halves evenly, so check the local sizes
    lx, lz = (np.asarray(o) for o in L.dm.getOwnershipRanges()[:2])
    levels = 1
    while (np.all(lx % 2**levels == 0) and np.all(lz % 2**levels == 0)
           and min(nt, nr) // 2**levels >= 4 and min(lx.min(), lz.min()) // 2**levels >= 1):
        levels += 1
    ku, kp = pc.getFieldSplitSubKSP()
    kp.setType("preonly"); kp.getPC().setType("jacobi")
    ku.setType("cg"); ku.setTolerances(rtol=u_rtol)
    pmg = ku.getPC(); pmg.setOptionsPrefix("annulus_u_"); pmg.setType("mg"); pmg.setMGLevels(levels)
    hier = [L.dm_v]
    for _ in range(levels - 1):
        hier.append(hier[-1].coarsen())
    hier = hier[::-1]
    for l in range(1, levels):
        P_l, _ = hier[l - 1].createInterpolation(hier[l])
        pmg.setMGInterpolation(l, P_l)
    o = PETSc.Options()
    o["annulus_u_pc_mg_galerkin"] = "both"
    o["annulus_u_mg_levels_ksp_type"] = "chebyshev"; o["annulus_u_mg_levels_pc_type"] = "jacobi"
    o["annulus_u_mg_levels_ksp_max_it"] = 3
    o["annulus_u_mg_coarse_ksp_type"] = "preonly"; o["annulus_u_mg_coarse_pc_type"] = "redundant"
    o["annulus_u_mg_coarse_redundant_pc_type"] = "lu"
    pmg.setFromOptions()
    x = L.dm.createGlobalVec()
    if x0 is not None:
        x.setArray(x0.getArray()); ksp.setInitialGuessNonzero(True)
    t0 = time.time()
    ksp.solve(b, x)
    stats = dict(outer=ksp.getIterationNumber(), inner=ku.getIterationNumber(), reason=ksp.getConvergedReason(),
                 time=time.time() - t0, levels=levels, nproc=L.comm.getSize())
    xn = L.to_natural(x)
    vr = xn[:g.n_vr].reshape(nr - 1, nt)
    vt = xn[g.n_vr:nv].reshape(nr, nt - 1)
    P = xn[nv:].reshape(nr, nt)
    return (vr, vt, P), x, stats
