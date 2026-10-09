"""Native, MPI-parallel Stokes solver on PETSc DMStag (Stage 3b/3c).

Each MPI process assembles only the rows of the grid points it owns (plus reads one layer of ghost points).
Same equations as ganymede.petsc_stokes.stokes_symmetric: symmetric saddle point [[K, G], [G^T, 0]],
no pinned pressure (null space), wall velocities decoupled with a scaled diagonal.
Solver: FGMRES + Schur-complement field split; geometric multigrid (DMStag hierarchy, Galerkin coarse
operators, Chebyshev/Jacobi smoothing) on the velocity block; Schur complement ~ -diag(1/eta).

Run in parallel with e.g.  mpiexec -n 4 python scripts/parallel_stokes.py
"""
import time
import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc

SL = PETSc.DMStag.StencilLocation


def make_dmstag(nx, nz, L=1.0, H=1.0, dofs=(0, 1, 1), comm=None):
    dm = PETSc.DMStag().create(dim=2, dofs=dofs, sizes=(nx, nz),
                               boundary_types=(PETSc.DM.BoundaryType.NONE,) * 2,
                               stencil_type=PETSc.DMStag.StencilType.BOX, stencil_width=1,
                               comm=comm or PETSc.COMM_WORLD, setUp=True)
    dm.setUniformCoordinatesProduct(0.0, L, 0.0, H)
    return dm


class StagIndex:
    """Global indices of every unknown in this process's ghosted patch."""
    def __init__(self, dm, nx, nz):
        v = dm.createGlobalVec(); r0, r1 = v.getOwnershipRange()
        v.setArray(np.arange(r0, r1, dtype=float) + 1)
        loc = dm.createLocalVec(); dm.globalToLocal(v, loc)
        (self.gx0, self.gz0), (gnx, gnz) = dm.getGhostCorners()
        self.a = loc.getArray().reshape(gnz, gnx, -1).astype(PETSc.IntType) - 1
        (self.x0, self.z0), (mx, mz), (ex, ez) = dm.getCorners()
        self.x1, self.z1 = self.x0 + mx, self.z0 + mz          # owned elements [x0,x1) x [z0,z1)
        self.ex, self.ez = ex, ez                               # extra boundary points on the right/top
        self.nx, self.nz = nx, nz
        self.s = {"vx": dm.getLocationSlot(SL.LEFT, 0), "vz": dm.getLocationSlot(SL.DOWN, 0)}
        if dm.getDof()[-1] > 0:
            self.s["P"] = dm.getLocationSlot(SL.ELEMENT, 0)

    def __call__(self, kind, j, i):
        return self.a[np.asarray(j) - self.gz0, np.asarray(i) - self.gx0, self.s[kind]]

    def owned(self, kind):
        """(j, i) of the unknowns of `kind` owned by this process."""
        if kind == "vx":   # x = i in [0, nx], z-cells j in [0, nz)
            I = np.arange(self.x0, self.x1 + self.ex); J = np.arange(self.z0, self.z1)
        elif kind == "vz":
            I = np.arange(self.x0, self.x1); J = np.arange(self.z0, self.z1 + self.ez)
        else:
            I = np.arange(self.x0, self.x1); J = np.arange(self.z0, self.z1)
        J, I = np.meshgrid(J, I, indexing="ij")
        return J.ravel(), I.ravel()


def assemble_stokes(dm, nx, nz, dx, dz, eta_c, eta_n, rho_vz, wall_scale, noslip_bottom=False):
    """Build the symmetric Stokes matrix and RHS on DMStag. eta_c (nz,nx), eta_n (nz+1,nx+1), rho_vz (nz+1,nx)
    are full arrays (every rank reads only its ghosted patch)."""
    ix = StagIndex(dm, nx, nz)
    R, C, V = [], [], []

    def put(r, c, v, keep=None):
        r, c, v = np.broadcast_arrays(r, c, v)
        if keep is not None:
            keep = np.broadcast_to(keep, r.shape)
            r, c, v = r[keep], c[keep], v[keep]
        R.append(r.ravel()); C.append(c.ravel()); V.append(np.asarray(v, float).ravel())

    b_rows, b_vals = [], []

    # --- x-momentum (rows multiplied by -1) ---
    j, i = ix.owned("vx")
    wall = (i == 0) | (i == nx)
    r = ix("vx", j[wall], i[wall]); put(r, r, wall_scale)
    j, i = j[~wall], i[~wall]; r = ix("vx", j, i)
    eR, eL = eta_c[j, i], eta_c[j, i - 1]
    put(r, ix("vx", j, i + 1), -2 * eR / dx**2, keep=(i + 1 < nx))
    put(r, r, 2 * (eR + eL) / dx**2)
    put(r, ix("vx", j, i - 1), -2 * eL / dx**2, keep=(i - 1 > 0))
    m = j + 1 < nz; jj, ii, rr = j[m], i[m], r[m]; eT = eta_n[jj + 1, ii]
    put(rr, ix("vx", jj + 1, ii), -eT / dz**2); put(rr, rr, eT / dz**2)
    put(rr, ix("vz", jj + 1, ii), -eT / (dx * dz)); put(rr, ix("vz", jj + 1, ii - 1), eT / (dx * dz))
    m = j > 0; jj, ii, rr = j[m], i[m], r[m]; eB = eta_n[jj, ii]
    put(rr, rr, eB / dz**2); put(rr, ix("vx", jj - 1, ii), -eB / dz**2)
    put(rr, ix("vz", jj, ii), eB / (dx * dz)); put(rr, ix("vz", jj, ii - 1), -eB / (dx * dz))
    put(r, ix("P", j, i), 1.0 / dx); put(r, ix("P", j, i - 1), -1.0 / dx)
    if noslip_bottom:                                   # vx = 0 on the bottom wall via a mirrored ghost
        m = j == 0
        put(r[m], r[m], 2 * eta_n[0, i[m]] / dz**2)

    # --- z-momentum ---
    j, i = ix.owned("vz")
    wall = (j == 0) | (j == nz)
    r = ix("vz", j[wall], i[wall]); put(r, r, wall_scale)
    j, i = j[~wall], i[~wall]; r = ix("vz", j, i)
    eT, eB = eta_c[j, i], eta_c[j - 1, i]
    put(r, ix("vz", j + 1, i), -2 * eT / dz**2, keep=(j + 1 < nz))
    put(r, r, 2 * (eT + eB) / dz**2)
    put(r, ix("vz", j - 1, i), -2 * eB / dz**2, keep=(j - 1 > 0))
    m = i + 1 < nx; jj, ii, rr = j[m], i[m], r[m]; eR = eta_n[jj, ii + 1]
    put(rr, ix("vz", jj, ii + 1), -eR / dx**2); put(rr, rr, eR / dx**2)
    put(rr, ix("vx", jj, ii + 1), -eR / (dx * dz)); put(rr, ix("vx", jj - 1, ii + 1), eR / (dx * dz))
    m = i > 0; jj, ii, rr = j[m], i[m], r[m]; eL = eta_n[jj, ii]
    put(rr, rr, eL / dx**2); put(rr, ix("vz", jj, ii - 1), -eL / dx**2)
    put(rr, ix("vx", jj, ii), eL / (dx * dz)); put(rr, ix("vx", jj - 1, ii), -eL / (dx * dz))
    put(r, ix("P", j, i), 1.0 / dz); put(r, ix("P", j - 1, i), -1.0 / dz)
    b_rows.append(r); b_vals.append(-rho_vz[j, i])

    # --- continuity (all cells, rows multiplied by -1) ---
    j, i = ix.owned("P"); r = ix("P", j, i)
    put(r, ix("vx", j, i + 1), -1.0 / dx, keep=(i + 1 < nx))
    put(r, ix("vx", j, i), 1.0 / dx, keep=(i > 0))
    put(r, ix("vz", j + 1, i), -1.0 / dz, keep=(j + 1 < nz))
    put(r, ix("vz", j, i), 1.0 / dz, keep=(j > 0))
    put(r, r, 0.0)                                                   # explicit zero diagonal

    rows, cols, vals = np.concatenate(R), np.concatenate(C), np.concatenate(V)
    # Every rank assembled only its own rows, so build a local CSR block (global column indices) and hand it to
    # createAIJ. This works on all petsc4py versions (Mat.setPreallocationCOO only exists in recent ones).
    N = dm.createGlobalVec().getSize()
    r0, r1 = dm.createGlobalVec().getOwnershipRange()
    assert rows.size == 0 or (rows.min() >= r0 and rows.max() < r1), "assembled a row this rank does not own"
    local = sp.coo_matrix((vals, (rows - r0, cols)), shape=(r1 - r0, N)).tocsr()   # sums duplicates, keeps 0s
    local.sort_indices()
    A = PETSc.Mat().createAIJ(size=((r1 - r0, N), (r1 - r0, N)),
                              csr=(local.indptr.astype(PETSc.IntType), local.indices.astype(PETSc.IntType), local.data),
                              comm=dm.getComm())
    A.assemble()
    b = dm.createGlobalVec(); b.set(0.0)
    b.setValues(np.concatenate(b_rows).astype(PETSc.IntType), np.concatenate(b_vals))
    b.assemble()
    return A, b, ix


def _block_offset(n_local, comm):
    """Start index of this rank's chunk when per-rank blocks are concatenated."""
    try:
        mpicomm = comm.tompi4py()
        return mpicomm.exscan(n_local) or 0
    except Exception:
        return 0


def native_stokes_solve(nx, nz, eta_c, eta_n, rho_vz, L=1.0, H=1.0, rtol=1e-10, u_rtol=1e-8, max_it=300,
                        comm=None, noslip_bottom=False):
    """Assemble on DMStag (each rank its own patch) and solve with FGMRES + Schur fieldsplit,
    geometric multigrid on the velocity block. Returns (dm, x, index helper, stats)."""
    dx, dz = L / nx, H / nz
    dm = make_dmstag(nx, nz, L, H, comm=comm)
    dm_v = make_dmstag(nx, nz, L, H, dofs=(0, 1, 0), comm=comm)
    wall_scale = 2.0 * np.mean(eta_c) * (1 / dx**2 + 1 / dz**2) * 2.0
    A, b, ix = assemble_stokes(dm, nx, nz, dx, dz, eta_c, eta_n, rho_vz, wall_scale, noslip_bottom=noslip_bottom)
    pcomm = dm.getComm()

    jv, iv = ix.owned("vx"); jz, iz = ix.owned("vz"); jp, ip = ix.owned("P")
    is_u = np.sort(np.concatenate([ix("vx", jv, iv), ix("vz", jz, iz)])).astype(PETSc.IntType)
    p_glob = ix("P", jp, ip)
    order = np.argsort(p_glob)
    is_p = p_glob[order].astype(PETSc.IntType)

    null = dm.createGlobalVec(); null.set(0.0)
    n_p_total = nx * nz
    null.setValues(is_p, np.full(is_p.size, 1 / np.sqrt(n_p_total))); null.assemble()
    A.setNullSpace(PETSc.NullSpace().create(vectors=[null], comm=pcomm))

    # Schur preconditioner -diag(1/eta) in the pressure sub-block numbering
    off = _block_offset(is_p.size, pcomm)
    S = PETSc.Mat().createAIJ(size=((is_p.size, n_p_total), (is_p.size, n_p_total)), nnz=1, comm=pcomm)
    rows = np.arange(off, off + is_p.size, dtype=PETSc.IntType)
    for r, e in zip(rows, eta_c[jp[order], ip[order]]):
        S.setValue(r, r, -1.0 / e)
    S.assemble()

    ksp = PETSc.KSP().create(comm=pcomm); ksp.setOperators(A); ksp.setType("fgmres")
    pc = ksp.getPC(); pc.setType("fieldsplit")
    pc.setFieldSplitIS(("u", PETSc.IS().createGeneral(is_u, comm=pcomm)), ("p", PETSc.IS().createGeneral(is_p, comm=pcomm)))
    pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
    pc.setFieldSplitSchurFactType(PETSc.PC.FieldSplitSchurFactType.UPPER)
    pc.setFieldSplitSchurPreType(PETSc.PC.FieldSplitSchurPreType.USER, S)
    ksp.setTolerances(rtol=rtol, max_it=max_it)
    ksp.setUp()

    levels = 1
    while nx % 2**levels == 0 and nz % 2**levels == 0 and min(nx, nz) // 2**levels >= 4:
        levels += 1
    ku, kp = pc.getFieldSplitSubKSP()
    kp.setType("preonly"); kp.getPC().setType("jacobi")
    ku.setType("cg"); ku.setTolerances(rtol=u_rtol)
    pmg = ku.getPC(); pmg.setOptionsPrefix("u_"); pmg.setType("mg"); pmg.setMGLevels(levels)
    hier = [dm_v]
    for _ in range(levels - 1):
        hier.append(hier[-1].coarsen())
    hier = hier[::-1]
    for l in range(1, levels):
        P_l, _ = hier[l - 1].createInterpolation(hier[l])
        pmg.setMGInterpolation(l, P_l)
    o = PETSc.Options()
    o["u_pc_mg_galerkin"] = "both"
    o["u_mg_levels_ksp_type"] = "chebyshev"; o["u_mg_levels_pc_type"] = "jacobi"; o["u_mg_levels_ksp_max_it"] = 3
    o["u_mg_coarse_ksp_type"] = "preonly"; o["u_mg_coarse_pc_type"] = "redundant"
    o["u_mg_coarse_redundant_pc_type"] = "lu"
    pmg.setFromOptions()

    x = dm.createGlobalVec()
    t0 = time.time(); ksp.solve(b, x); elapsed = time.time() - t0
    stats = dict(outer=ksp.getIterationNumber(), inner=ku.getIterationNumber(),
                 reason=ksp.getConvergedReason(), time=elapsed, levels=levels, nproc=pcomm.getSize())
    return dm, x, ix, stats


def gather_fields(x, ix, nx, nz, comm=None):
    """Collect vx (nz,nx+1), vz (nz+1,nx), P (nz,nx) on rank 0 (None elsewhere)."""
    xa = x.getArray(); r0, _ = x.getOwnershipRange()
    pieces = []
    for kind in ("vx", "vz", "P"):
        j, i = ix.owned(kind)
        pieces.append((kind, j, i, xa[ix(kind, j, i) - r0]))
    try:
        mpicomm = x.getComm().tompi4py()
        allp = mpicomm.gather(pieces, root=0)
        rank = mpicomm.Get_rank()
    except Exception:
        allp, rank = [pieces], 0
    if rank != 0:
        return None
    out = {"vx": np.zeros((nz, nx + 1)), "vz": np.zeros((nz + 1, nx)), "P": np.zeros((nz, nx))}
    for pc in allp:
        for kind, j, i, v in pc:
            out[kind][j, i] = v
    return out["vx"], out["vz"], out["P"]


def ours_to_stag(g, dm):
    """Serial helper for tests: perm[k] = DMStag global index of unknown k in the SciPy ordering [vx..., vz..., P...]."""
    ix = StagIndex(dm, g.nx, g.nz)
    vx = ix("vx", *np.meshgrid(np.arange(g.nz), np.arange(g.nx + 1), indexing="ij"))
    vz = ix("vz", *np.meshgrid(np.arange(g.nz + 1), np.arange(g.nx), indexing="ij"))
    P = ix("P", *np.meshgrid(np.arange(g.nz), np.arange(g.nx), indexing="ij"))
    return np.concatenate([vx.ravel(), vz.ravel(), P.ravel()])
