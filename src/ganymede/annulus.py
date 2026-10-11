"""Stage 7a: Stokes flow in a 2D (cylindrical) annulus sector -- the bent ice shell.

Polar coordinates (r, theta); the sector spans theta in [theta0, theta0 + span] (a half annulus: span = pi)
and r in [r_in, r_out]. Staggered grid (the polar analogue of Stages 2-3):
    v_r      on radial faces    (r_f[i], theta_c[j])   -- interior faces only (v_r = 0 on r_in, r_out)
    v_theta  on angular faces   (r_c[i], theta_f[j])   -- interior faces only (v_theta = 0 on the side walls)
    P, eta_c at cell centres    (r_c[i], theta_c[j])
    tau_rtheta, eta_n at nodes  (r_f[i], theta_f[j])   -- zero on every wall (free slip)
Radial faces r_f may be spaced arbitrarily (refine towards the ocean); theta spacing is uniform.

The matrix is assembled from exact discrete operators (strain rates, stress divergence, gradient,
divergence) as sparse products, so compaction (a bulk-viscosity term) is a single extra term later.

Equations (no compaction yet):   -div(tau) + grad P = f,   div v = 0,   tau = 2 eta eps, with
    eps_rr = dv_r/dr,  eps_tt = (1/r) dv_t/dtheta + v_r/r,  2 eps_rt = r d(v_t/r)/dr + (1/r) dv_r/dtheta
    (div tau)_r = (1/r) d(r tau_rr)/dr + (1/r) d tau_rt/dtheta - tau_tt/r
    (div tau)_t = (1/r^2) d(r^2 tau_rt)/dr + (1/r) d tau_tt/dtheta
    div v       = (1/r) d(r v_r)/dr + (1/r) dv_t/dtheta
Unknowns are ordered [v_r, v_theta, P], each row-major with the radial index outermost.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


class AnnulusGrid:
    def __init__(self, r_faces, ntheta, span=np.pi, theta0=0.0):
        self.r_f = np.asarray(r_faces, float)
        self.nr = len(self.r_f) - 1
        self.nt = int(ntheta)
        self.r_c = 0.5 * (self.r_f[1:] + self.r_f[:-1])
        self.dr = np.diff(self.r_f)                       # cell widths (nr)
        self.drc = np.diff(self.r_c)                      # centre spacing at interior faces (nr-1)
        self.span, self.theta0 = span, theta0
        self.dt = span / self.nt
        self.t_f = theta0 + np.arange(self.nt + 1) * self.dt
        self.t_c = theta0 + (np.arange(self.nt) + 0.5) * self.dt
        self.n_vr = (self.nr - 1) * self.nt
        self.n_vt = self.nr * (self.nt - 1)
        self.n_p = self.nr * self.nt

    @classmethod
    def uniform(cls, r_in, r_out, nr, ntheta, **kw):
        return cls(np.linspace(r_in, r_out, nr + 1), ntheta, **kw)

    @classmethod
    def refined(cls, r_in, r_out, nr, ntheta, ratio=4.0, **kw):
        """Radial faces clustered towards both walls (tanh stretching); `ratio` ~ largest/smallest cell."""
        s = np.linspace(-1, 1, nr + 1)
        a = np.arccosh(ratio) if ratio > 1 else 1e-9
        x = np.tanh(a * s) / np.tanh(a)
        return cls(r_in + (r_out - r_in) * 0.5 * (x + 1), ntheta, **kw)

    # point coordinates of each unknown family
    def vr_points(self):
        return np.meshgrid(self.r_f[1:-1], self.t_c, indexing="ij")

    def vt_points(self):
        return np.meshgrid(self.r_c, self.t_f[1:-1], indexing="ij")

    def p_points(self):
        return np.meshgrid(self.r_c, self.t_c, indexing="ij")

    def node_points(self):
        return np.meshgrid(self.r_f, self.t_f, indexing="ij")

    def cell_area(self):
        """Area of each cell (for volume-weighted norms): 0.5 (r_out^2 - r_in^2) dtheta."""
        a = 0.5 * (self.r_f[1:] ** 2 - self.r_f[:-1] ** 2) * self.dt
        return np.repeat(a[:, None], self.nt, axis=1)


def _d(v):
    return sp.diags(np.asarray(v, float))


def operators(g):
    """Sparse 1D/2D building blocks (radial index outermost)."""
    nr, nt = g.nr, g.nt
    # radial: interior faces (nr-1) <-> centres (nr); boundary faces carry zeros
    Dr = sp.lil_matrix((nr, nr - 1))          # faces -> centres: (x[i+1] - x[i]) / dr[i]
    Ar = sp.lil_matrix((nr, nr - 1))          # faces -> centres: average
    for i in range(nr):
        if i < nr - 1:
            Dr[i, i] += 1 / g.dr[i]; Ar[i, i] += 0.5
        if i > 0:
            Dr[i, i - 1] -= 1 / g.dr[i]; Ar[i, i - 1] += 0.5
    Gr = sp.lil_matrix((nr - 1, nr))          # centres -> interior faces: (x[k+1] - x[k]) / drc[k]
    Mr = sp.lil_matrix((nr - 1, nr))          # centres -> interior faces: average
    for k in range(nr - 1):
        Gr[k, k + 1] = 1 / g.drc[k]; Gr[k, k] = -1 / g.drc[k]
        Mr[k, k + 1] = 0.5; Mr[k, k] = 0.5
    # angular: interior faces (nt-1) <-> centres (nt), uniform spacing
    Dt = sp.lil_matrix((nt, nt - 1))
    for j in range(nt):
        if j < nt - 1:
            Dt[j, j] += 1 / g.dt
        if j > 0:
            Dt[j, j - 1] -= 1 / g.dt
    Gt = sp.lil_matrix((nt - 1, nt))
    for k in range(nt - 1):
        Gt[k, k + 1] = 1 / g.dt; Gt[k, k] = -1 / g.dt
    Dr, Ar, Gr, Mr, Dt, Gt = (m.tocsr() for m in (Dr, Ar, Gr, Mr, Dt, Gt))
    Ir, It = sp.identity(nr), sp.identity(nt)
    Ir1, It1 = sp.identity(nr - 1), sp.identity(nt - 1)
    rc, rf = g.r_c, g.r_f[1:-1]
    K = sp.kron
    ops = dict(
        # strain rates at centres
        Err_vr=K(Dr, It),                                   # eps_rr from v_r
        Ett_vr=K(_d(1 / rc) @ Ar, It),                      # v_r / r
        Ett_vt=K(_d(1 / rc), Dt),                           # (1/r) dv_t/dtheta
        # engineering shear strain 2 eps_rt at interior nodes (nr-1) x (nt-1)
        Srt_vr=K(_d(1 / rf), Gt),                           # (1/r) dv_r/dtheta
        Srt_vt=K(_d(rf) @ Gr @ _d(1 / rc), It1),            # r d(v_t/r)/dr
        # stress divergence
        Fr_rr=K(_d(1 / rf) @ Gr @ _d(rc), It),              # (1/r) d(r tau_rr)/dr      centres -> v_r points
        Fr_rt=K(_d(1 / rf), Dt),                            # (1/r) d tau_rt/dtheta     nodes   -> v_r points
        Fr_tt=K(_d(1 / rf) @ Mr, It),                       # tau_tt / r                centres -> v_r points
        Ft_rt=K(_d(1 / rc**2) @ Dr @ _d(rf**2), It1),       # (1/r^2) d(r^2 tau_rt)/dr  nodes   -> v_t points
        Ft_tt=K(_d(1 / rc), Gt),                            # (1/r) d tau_tt/dtheta     centres -> v_t points
        # pressure gradient and divergence
        Gp_r=K(Gr, It),
        Gp_t=K(_d(1 / rc), Gt),
        Div_vr=K(_d(1 / rc) @ Dr @ _d(rf), It),
        Div_vt=K(_d(1 / rc), Dt),
    )
    return ops


def stokes_matrix(g, eta_c, eta_n, ops=None, fix_pressure=True):
    """Saddle-point matrix for  -div(2 eta eps) + grad P = f,  -div v = 0.
    eta_c (nr, nt) at centres; eta_n (nr+1, nt+1) at nodes (only interior nodes are used)."""
    o = ops or operators(g)
    Ec = _d(np.asarray(eta_c).ravel())
    En = _d(np.asarray(eta_n)[1:-1, 1:-1].ravel())
    # stresses as operators on (v_r, v_t)
    Trr_vr = 2 * Ec @ o["Err_vr"]
    Ttt_vr = 2 * Ec @ o["Ett_vr"]
    Ttt_vt = 2 * Ec @ o["Ett_vt"]
    Trt_vr = En @ o["Srt_vr"]
    Trt_vt = En @ o["Srt_vt"]
    # -(div tau)
    Krr = -(o["Fr_rr"] @ Trr_vr + o["Fr_rt"] @ Trt_vr - o["Fr_tt"] @ Ttt_vr)
    Krt = -(o["Fr_rt"] @ Trt_vt - o["Fr_tt"] @ Ttt_vt)
    Ktr = -(o["Ft_rt"] @ Trt_vr + o["Ft_tt"] @ Ttt_vr)
    Ktt = -(o["Ft_rt"] @ Trt_vt + o["Ft_tt"] @ Ttt_vt)
    Dvr, Dvt = -o["Div_vr"], -o["Div_vt"]
    A = sp.bmat([[Krr, Krt, o["Gp_r"]],
                 [Ktr, Ktt, o["Gp_t"]],
                 [Dvr, Dvt, None]], format="lil")
    if fix_pressure:                                       # pin one pressure (the constant is free)
        r0 = g.n_vr + g.n_vt
        A[r0, :] = 0
        A[r0, r0] = 1.0
    return A.tocsc()


def unpack(g, x):
    vr = x[:g.n_vr].reshape(g.nr - 1, g.nt)
    vt = x[g.n_vr:g.n_vr + g.n_vt].reshape(g.nr, g.nt - 1)
    P = x[g.n_vr + g.n_vt:].reshape(g.nr, g.nt)
    return vr, vt, P


def solve_stokes(g, eta_c, eta_n, f_r, f_t):
    """Direct solve (SciPy). f_r at v_r points (nr-1, nt), f_t at v_t points (nr, nt-1)."""
    A = stokes_matrix(g, eta_c, eta_n)
    b = np.concatenate([np.ravel(f_r), np.ravel(f_t), np.zeros(g.n_p)])
    return unpack(g, spsolve(A, b))
