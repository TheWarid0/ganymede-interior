"""3D Stokes on PETSc DMStag (Stage 3d): the 2D discretisation of stag_stokes.py, written once for all three
directions, MPI-parallel, geometric multigrid on the velocity block.

Axes: x (a=0), y (a=1), z (a=2, vertical, gravity along -z). Arrays are indexed [k, j, i] = [z, y, x].
Faces:  v_x at (k, j, i) with i in 0..nx (x = i, y = j+1/2, z = k+1/2), etc.
Edges:  shear stress sigma_ab lives on edges where the a- and b-indices are integers and the third is a cell index.
"""
import time
import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc

SL = PETSc.DMStag.StencilLocation
AX = ("x", "y", "z")
E = {0: np.array([0, 0, 1]), 1: np.array([0, 1, 0]), 2: np.array([1, 0, 0])}   # unit offsets in [k, j, i]
FACE_LOC = {0: SL.LEFT, 1: SL.DOWN, 2: SL.BACK}


def make_dmstag3d(n, L=(1.0, 1.0, 1.0), dofs=(0, 0, 1, 1), comm=None):
    nx, ny, nz = n
    dm = PETSc.DMStag().create(dim=3, dofs=dofs, sizes=(nx, ny, nz),
                               boundary_types=(PETSc.DM.BoundaryType.NONE,) * 3,
                               stencil_type=PETSc.DMStag.StencilType.BOX, stencil_width=1,
                               comm=comm or PETSc.COMM_WORLD, setUp=True)
    dm.setUniformCoordinatesProduct(0.0, L[0], 0.0, L[1], 0.0, L[2])
    return dm


class StagIndex3D:
    def __init__(self, dm, n):
        v = dm.createGlobalVec(); r0, r1 = v.getOwnershipRange()
        v.setArray(np.arange(r0, r1, dtype=float) + 1)
        loc = dm.createLocalVec(); dm.globalToLocal(v, loc)
        (gx0, gy0, gz0), (gnx, gny, gnz) = dm.getGhostCorners()
        self.g0 = np.array([gz0, gy0, gx0])
        self.a = loc.getArray().reshape(gnz, gny, gnx, -1).astype(PETSc.IntType) - 1
        (x0, y0, z0), (mx, my, mz), (ex, ey, ez) = dm.getCorners()
        self.lo = np.array([z0, y0, x0]); self.hi = self.lo + np.array([mz, my, mx]); self.extra = np.array([ez, ey, ex])
        self.n = np.array([n[2], n[1], n[0]])                      # [nz, ny, nx]
        self.slot = {a: dm.getLocationSlot(FACE_LOC[a], 0) for a in range(3)}
        if dm.getDof()[-1] > 0:
            self.slot["P"] = dm.getLocationSlot(SL.ELEMENT, 0)

    def __call__(self, kind, p):
        """p: (..., 3) integer array of [k, j, i]."""
        q = np.asarray(p) - self.g0
        return self.a[q[..., 0], q[..., 1], q[..., 2], self.slot[kind]]

    def owned(self, kind):
        """Owned unknowns of a face direction (0,1,2) or 'P', as an (N, 3) array of [k, j, i]."""
        hi = self.hi.copy()
        if kind != "P":
            ax = 2 - kind                        # face along x (kind 0) is the i axis = position 2 in [k, j, i]
            hi[ax] += self.extra[ax]
        K, J, I = np.meshgrid(*(np.arange(self.lo[d], hi[d]) for d in range(3)), indexing="ij")
        return np.stack([K.ravel(), J.ravel(), I.ravel()], axis=1)


def assemble_stokes3d(dm, n, h, eta_c, eta_edge, rho_vz, wall_scale):
    """Symmetric 3D Stokes system. n = (nx, ny, nz), h = (dx, dy, dz).
    eta_c: (nz, ny, nx).  eta_edge[(a, b)] with a < b: viscosity on the edges carrying sigma_ab:
      (0,1) xy: (nz, ny+1, nx+1)   (0,2) xz: (nz+1, ny, nx+1)   (1,2) yz: (nz+1, ny+1, nx)
    rho_vz: (nz+1, ny, nx)."""
    ix = StagIndex3D(dm, n)
    nk = ix.n                                    # [nz, ny, nx]
    pos = lambda a: 2 - a                        # axis a -> position in [k, j, i]
    R, C, V = [], [], []
    b_rows, b_vals = [], []

    def put(r, c, v, keep=None):
        r, c, v = np.broadcast_arrays(r, c, v)
        if keep is not None:
            keep = np.broadcast_to(keep, r.shape); r, c, v = r[keep], c[keep], v[keep]
        R.append(r.ravel()); C.append(c.ravel()); V.append(np.asarray(v, float).ravel())

    def interior_face(a, p):
        """True where the a-face p is not on a wall (a wall velocity is 0 and its column is dropped)."""
        return (p[:, pos(a)] > 0) & (p[:, pos(a)] < nk[pos(a)])

    eC = lambda p: eta_c[p[:, 0], p[:, 1], p[:, 2]]
    def eE(a, b, p):
        arr = eta_edge[(min(a, b), max(a, b))]
        return arr[p[:, 0], p[:, 1], p[:, 2]]

    for a in range(3):
        P_all = ix.owned(a)
        wall = ~interior_face(a, P_all)
        rw = ix(a, P_all[wall]); put(rw, rw, wall_scale)
        p = P_all[~wall]; r = ix(a, p); ea = E[a]; ha = h[a]
        # normal stress d(sigma_aa)/da (rows multiplied by -1)
        ep, em = eC(p), eC(p - ea)
        put(r, ix(a, p + ea), -2 * ep / ha**2, keep=interior_face(a, p + ea))
        put(r, r, 2 * (ep + em) / ha**2)
        put(r, ix(a, p - ea), -2 * em / ha**2, keep=interior_face(a, p - ea))
        # shear stresses d(sigma_ab)/db for the two other directions b
        for bdir in range(3):
            if bdir == a:
                continue
            eb, hb = E[bdir], h[bdir]
            for side, sgn in ((+1, -1.0), (0, +1.0)):        # upper edge (p + e_b) and lower edge (p)
                Eg = p + eb if side else p
                m = (Eg[:, pos(bdir)] > 0) & (Eg[:, pos(bdir)] < nk[pos(bdir)])     # edges on b-walls: stress 0
                Eg_m, r_m = Eg[m], r[m]
                eta = eE(a, bdir, Eg_m)
                # sigma_ab(E) = eta [ (v_a(E) - v_a(E - e_b)) / h_b + (v_b(E) - v_b(E - e_a)) / h_a ]
                # row contribution (after the -1 flip): sgn * sigma_ab(E) / h_b
                for col_p, coef, comp in ((Eg_m, 1 / hb, a), (Eg_m - eb, -1 / hb, a),
                                          (Eg_m, 1 / ha, bdir), (Eg_m - ea, -1 / ha, bdir)):
                    put(r_m, ix(comp, col_p), sgn * eta * coef / hb, keep=interior_face(comp, col_p))
        # pressure gradient
        put(r, ix("P", p), 1.0 / ha); put(r, ix("P", p - ea), -1.0 / ha)
        if a == 2:
            b_rows.append(r); b_vals.append(-rho_vz[p[:, 0], p[:, 1], p[:, 2]])

    # continuity (rows multiplied by -1)
    p = ix.owned("P"); r = ix("P", p)
    for a in range(3):
        ea, ha = E[a], h[a]
        put(r, ix(a, p + ea), -1.0 / ha, keep=interior_face(a, p + ea))
        put(r, ix(a, p), 1.0 / ha, keep=interior_face(a, p))
    put(r, r, 0.0)

    rows, cols, vals = np.concatenate(R), np.concatenate(C), np.concatenate(V)
    gv = dm.createGlobalVec(); N = gv.getSize(); r0, r1 = gv.getOwnershipRange()
    assert rows.size == 0 or (rows.min() >= r0 and rows.max() < r1)
    local = sp.coo_matrix((vals, (rows - r0, cols)), shape=(r1 - r0, N)).tocsr(); local.sort_indices()
    A = PETSc.Mat().createAIJ(size=((r1 - r0, N), (r1 - r0, N)),
                              csr=(local.indptr.astype(PETSc.IntType), local.indices.astype(PETSc.IntType), local.data),
                              comm=dm.getComm())
    A.assemble()
    b = dm.createGlobalVec(); b.set(0.0)
    if b_rows:
        b.setValues(np.concatenate(b_rows).astype(PETSc.IntType), np.concatenate(b_vals)); b.assemble()
    return A, b, ix


def _offset(n_local, comm):
    try:
        return comm.tompi4py().exscan(n_local) or 0
    except Exception:
        return 0


def stokes3d_solve(n, eta_c, eta_edge, rho_vz, L=(1.0, 1.0, 1.0), rtol=1e-10, u_rtol=1e-8, max_it=300, comm=None):
    nx, ny, nz = n
    h = (L[0] / nx, L[1] / ny, L[2] / nz)
    dm = make_dmstag3d(n, L, comm=comm)
    dm_v = make_dmstag3d(n, L, dofs=(0, 0, 1, 0), comm=comm)
    wall_scale = 2.0 * np.mean(eta_c) * sum(1 / hh**2 for hh in h) * 2.0
    A, b, ix = assemble_stokes3d(dm, n, h, eta_c, eta_edge, rho_vz, wall_scale)
    pcomm = dm.getComm()
    is_u = np.sort(np.concatenate([ix(a, ix.owned(a)) for a in range(3)])).astype(PETSc.IntType)
    pP = ix.owned("P"); pg = ix("P", pP); order = np.argsort(pg); is_p = pg[order].astype(PETSc.IntType)
    n_p = nx * ny * nz
    null = dm.createGlobalVec(); null.set(0.0); null.setValues(is_p, np.full(is_p.size, 1 / np.sqrt(n_p))); null.assemble()
    A.setNullSpace(PETSc.NullSpace().create(vectors=[null], comm=pcomm))
    off = _offset(is_p.size, pcomm)
    S = PETSc.Mat().createAIJ(size=((is_p.size, n_p), (is_p.size, n_p)), nnz=1, comm=pcomm)
    pe = pP[order]
    for rr, e in zip(range(off, off + is_p.size), eta_c[pe[:, 0], pe[:, 1], pe[:, 2]]):
        S.setValue(rr, rr, -1.0 / e)
    S.assemble()

    ksp = PETSc.KSP().create(comm=pcomm); ksp.setOperators(A); ksp.setType("fgmres")
    pc = ksp.getPC(); pc.setType("fieldsplit")
    pc.setFieldSplitIS(("u", PETSc.IS().createGeneral(is_u, comm=pcomm)), ("p", PETSc.IS().createGeneral(is_p, comm=pcomm)))
    pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
    pc.setFieldSplitSchurFactType(PETSc.PC.FieldSplitSchurFactType.UPPER)
    pc.setFieldSplitSchurPreType(PETSc.PC.FieldSplitSchurPreType.USER, S)
    ksp.setTolerances(rtol=rtol, max_it=max_it); ksp.setUp()
    levels = 1
    while all(m % 2**levels == 0 for m in n) and min(n) // 2**levels >= 4:
        levels += 1
    ku, kp = pc.getFieldSplitSubKSP()
    kp.setType("preonly"); kp.getPC().setType("jacobi")
    ku.setType("cg"); ku.setTolerances(rtol=u_rtol)
    pmg = ku.getPC(); pmg.setOptionsPrefix("u3_"); pmg.setType("mg"); pmg.setMGLevels(levels)
    hier = [dm_v]
    for _ in range(levels - 1):
        hier.append(hier[-1].coarsen())
    hier = hier[::-1]
    for l in range(1, levels):
        P_l, _ = hier[l - 1].createInterpolation(hier[l]); pmg.setMGInterpolation(l, P_l)
    o = PETSc.Options()
    o["u3_pc_mg_galerkin"] = "both"
    o["u3_mg_levels_ksp_type"] = "chebyshev"; o["u3_mg_levels_pc_type"] = "jacobi"; o["u3_mg_levels_ksp_max_it"] = 3
    o["u3_mg_coarse_ksp_type"] = "preonly"; o["u3_mg_coarse_pc_type"] = "redundant"; o["u3_mg_coarse_redundant_pc_type"] = "lu"
    pmg.setFromOptions()
    x = dm.createGlobalVec()
    t0 = time.time(); ksp.solve(b, x); dt = time.time() - t0
    stats = dict(outer=ksp.getIterationNumber(), inner=ku.getIterationNumber(), reason=ksp.getConvergedReason(),
                 time=dt, levels=levels, nproc=pcomm.getSize(), unknowns=x.getSize())
    return dm, x, ix, stats, A


def gather3d(x, ix, n):
    nx, ny, nz = n
    xa = x.getArray(); r0, _ = x.getOwnershipRange()
    pieces = [(kind, ix.owned(kind), xa[ix(kind, ix.owned(kind)) - r0]) for kind in (0, 1, 2, "P")]
    try:
        mc = x.getComm().tompi4py(); allp = mc.gather(pieces, root=0); rank = mc.Get_rank()
    except Exception:
        allp, rank = [pieces], 0
    if rank != 0:
        return None
    out = {0: np.zeros((nz, ny, nx + 1)), 1: np.zeros((nz, ny + 1, nx)), 2: np.zeros((nz + 1, ny, nx)), "P": np.zeros((nz, ny, nx))}
    for pc in allp:
        for kind, p, v in pc:
            out[kind][p[:, 0], p[:, 1], p[:, 2]] = v
    return out[0], out[1], out[2], out["P"]


def analytic3d(n, L=(1.0, 1.0, 1.0)):
    """rho = cos(kx x) cos(ky y) sin(kz z), eta = 1, free slip. Returns rho at vz points and exact vx, vy, vz."""
    nx, ny, nz = n
    kx, ky, kz = np.pi / L[0], np.pi / L[1], np.pi / L[2]
    K2 = kx**2 + ky**2 + kz**2; kh2 = kx**2 + ky**2
    D = kz / K2                                    # pressure amplitude (rho0 g = 1, eta = 1)
    A = -kh2 / K2**2                               # vz amplitude
    B, C = D * kx / K2, D * ky / K2                # vx, vy amplitudes
    xc = (np.arange(nx) + .5) * L[0] / nx; yc = (np.arange(ny) + .5) * L[1] / ny; zc = (np.arange(nz) + .5) * L[2] / nz
    xn = np.arange(nx + 1) * L[0] / nx; yn = np.arange(ny + 1) * L[1] / ny; zn = np.arange(nz + 1) * L[2] / nz
    Z, Y, X = np.meshgrid(zn, yc, xc, indexing="ij")
    rho = np.cos(kx * X) * np.cos(ky * Y) * np.sin(kz * Z)
    vz = A * np.cos(kx * X) * np.cos(ky * Y) * np.sin(kz * Z)
    Z, Y, X = np.meshgrid(zc, yc, xn, indexing="ij"); vx = B * np.sin(kx * X) * np.cos(ky * Y) * np.cos(kz * Z)
    Z, Y, X = np.meshgrid(zc, yn, xc, indexing="ij"); vy = C * np.cos(kx * X) * np.sin(ky * Y) * np.cos(kz * Z)
    return rho, vx, vy, vz


def edge_viscosity(fn, n, L=(1.0, 1.0, 1.0)):
    """Evaluate eta = fn(x, y, z) at cell centres and on the three edge families."""
    nx, ny, nz = n
    c = [(np.arange(m) + .5) * l / m for m, l in zip(n, L)]
    nd = [np.arange(m + 1) * l / m for m, l in zip(n, L)]
    ev = lambda zs, ys, xs: fn(*np.meshgrid(zs, ys, xs, indexing="ij")[::-1])
    eta_c = ev(c[2], c[1], c[0])
    eta_edge = {(0, 1): ev(c[2], nd[1], nd[0]), (0, 2): ev(nd[2], c[1], nd[0]), (1, 2): ev(nd[2], nd[1], c[0])}
    return eta_c, eta_edge
