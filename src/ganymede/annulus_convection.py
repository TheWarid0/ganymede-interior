"""Stage 7c: thermal convection in the annulus (cylindrical d = 1 or spherical annulus d = 2).

Nondimensional Boussinesq equations (length: shell thickness, time: thickness^2 / kappa,
temperature: Delta T), with constant gravity pointing inwards:
    -div(2 eta eps) + grad P = Ra T e_r,    div v = 0
    dT/dt + div(v T) = div(grad T) + H

Temperature lives at cell centres and is advanced with a finite-volume scheme on the same
control volumes as the pressure (measure r^d dr dtheta). The face velocities from the Stokes
solver are discretely divergence free on exactly these cells, so the advective fluxes conserve
heat and keep a uniform temperature uniform. Advection is central (second order) by default;
'upwind' is available for rough, high-Peclet transients. Time stepping is backward Euler
(unconditionally stable); steady states are polished with Picard iterations (dt -> infinity).

Boundaries: radial walls isothermal (T_in, T_out) or insulating (T_in=None); side walls of a sector
are insulating and free slip, i.e. mirror planes of a full-annulus solution.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu, spsolve

from .annulus import stokes_matrix, unpack


def _vol(g):
    """Control volume of each cell, r_c^d dr dtheta (the same weights as the continuity rows)."""
    return (g.r_c[:, None] ** g.d * g.dr[:, None] * g.dt) * np.ones((1, g.nt))


def _face_avg(g, Tc):
    """Cell-centre field -> interior radial faces (linear interpolation in r)."""
    w1 = g.dr[1:] / (g.dr[1:] + g.dr[:-1])          # weight of the cell below the face
    return w1[:, None] * Tc[:-1] + (1 - w1)[:, None] * Tc[1:]


def energy_operator(g, vr, vt, T_in=1.0, T_out=0.0, H=0.0, scheme="central"):
    """Return (A, b) with V dT/dt = A T + b  (A sparse, T raveled, radial index outermost).
    vr: (nr-1, nt) interior radial-face velocities; vt: (nr, nt-1) interior angular-face velocities."""
    nr, nt, d = g.nr, g.nt, g.d
    N = nr * nt
    idx = np.arange(N).reshape(nr, nt)
    rows, cols, vals = [], [], []
    b = np.zeros((nr, nt))

    def add(r, c, v):
        rows.append(np.ravel(r)); cols.append(np.ravel(c)); vals.append(np.ravel(v))

    # --- radial interior faces k (between cell k below and k+1 above) ---
    rf = g.r_f[1:-1]
    area = (rf ** d * g.dt)[:, None] * np.ones((1, nt))
    D = area / g.drc[:, None]                                # diffusive conductance
    lo, hi = idx[:-1], idx[1:]
    # flux F (outward, from lo to hi) = -D (T_hi - T_lo) + area * v * T_face
    if scheme == "central":
        w1 = (g.dr[1:] / (g.dr[1:] + g.dr[:-1]))[:, None] * np.ones((1, nt))
        a_lo, a_hi = area * vr * w1, area * vr * (1 - w1)
    else:
        a_lo, a_hi = area * np.maximum(vr, 0), area * np.minimum(vr, 0)
    c_lo = D + a_lo                                          # dF/dT_lo
    c_hi = -D + a_hi                                         # dF/dT_hi
    # cell lo loses F, cell hi gains F
    add(lo, lo, -c_lo); add(lo, hi, -c_hi)
    add(hi, lo, c_lo); add(hi, hi, c_hi)

    # --- angular interior faces j (between cell j left and j+1 right) ---
    area_t = (g.r_c ** (d - 1) * g.dr)[:, None] * np.ones((1, nt - 1))
    Dt = area_t / (g.r_c[:, None] * g.dt)
    le, ri = idx[:, :-1], idx[:, 1:]
    if scheme == "central":
        a_le, a_ri = 0.5 * area_t * vt, 0.5 * area_t * vt
    else:
        a_le, a_ri = area_t * np.maximum(vt, 0), area_t * np.minimum(vt, 0)
    c_le, c_ri = Dt + a_le, -Dt + a_ri
    add(le, le, -c_le); add(le, ri, -c_ri)
    add(ri, le, c_le); add(ri, ri, c_ri)

    # --- radial walls (v_r = 0 there: conduction only) ---
    for i, r_w, Tw, sgn in ((0, g.r_f[0], T_in, -1), (nr - 1, g.r_f[-1], T_out, +1)):
        if Tw is None:
            continue                                         # insulating
        Dw = r_w ** d * g.dt / (0.5 * g.dr[i])
        add(idx[i], idx[i], -Dw * np.ones(nt))
        b[i] += Dw * Tw
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
    b = b.ravel() + (H * _vol(g)).ravel()
    return A, b


def conductive_profile(g, T_in=1.0, T_out=0.0):
    r = g.r_c
    ri, ro = g.r_f[0], g.r_f[-1]
    if g.d == 1:
        T = T_out + (T_in - T_out) * np.log(r / ro) / np.log(ri / ro)
    else:
        T = T_out + (T_in - T_out) * (1 / r - 1 / ro) / (1 / ri - 1 / ro)
    return np.repeat(T[:, None], g.nt, axis=1)


def wall_flux(g, T, T_in=1.0, T_out=0.0):
    """Outward heat flux -dT/dr at the outer and inner walls (per unit area, each theta column), taken
    from the scheme's own conservative wall fluxes. In a steady state they equal the heat through every
    interior radius, so they inherit the second-order accuracy of the interior fluxes."""
    q_top = (T[-1] - T_out) / (0.5 * g.dr[-1])
    q_bot = np.zeros(g.nt) if T_in is None else (T_in - T[0]) / (0.5 * g.dr[0])
    return q_top, q_bot


def diagnostics(g, T, vr, vt, T_in=1.0, T_out=0.0):
    """Nusselt numbers (top and bottom, relative to the conductive state), v_rms, mean T."""
    ri, ro, d = g.r_f[0], g.r_f[-1], g.d
    q_top, q_bot = wall_flux(g, T, T_in, T_out)
    if g.d == 1:
        qc_top, qc_bot = 1 / (ro * np.log(ro / ri)), 1 / (ri * np.log(ro / ri))
    else:
        qc_top, qc_bot = ri / (ro * (ro - ri)), ro / (ri * (ro - ri))
    dT = (T_in if T_in is not None else 1.0) - T_out
    V = _vol(g)
    ur = 0.5 * (np.vstack([np.zeros((1, g.nt)), vr]) + np.vstack([vr, np.zeros((1, g.nt))]))
    ut = 0.5 * (np.hstack([np.zeros((g.nr, 1)), vt]) + np.hstack([vt, np.zeros((g.nr, 1))]))
    return dict(Nu_top=q_top.mean() / (qc_top * dT), Nu_bot=q_bot.mean() / (qc_bot * dT),
                q_top=q_top.mean(), q_bot=q_bot.mean(),
                vrms=np.sqrt(((ur**2 + ut**2) * V).sum() / V.sum()), Tmean=(T * V).sum() / V.sum())


class Convection:
    """Isoviscous or viscosity-field convection driver on an AnnulusGrid (SciPy direct solves)."""

    def __init__(self, g, Ra, eta_c=None, eta_n=None, T_in=1.0, T_out=0.0, H=0.0, scheme="central"):
        self.g, self.Ra, self.H, self.scheme = g, Ra, H, scheme
        self.T_in, self.T_out = T_in, T_out
        self.eta_c = np.ones((g.nr, g.nt)) if eta_c is None else eta_c
        self.eta_n = np.ones((g.nr + 1, g.nt + 1)) if eta_n is None else eta_n
        self.lu = splu(stokes_matrix(g, self.eta_c, self.eta_n).tocsc())   # constant viscosity: factor once
        self.V = _vol(g).ravel()

    def stokes(self, T):
        g = self.g
        f_r = self.Ra * _face_avg(g, T)
        b = np.concatenate([f_r.ravel(), np.zeros(g.n_vt), np.zeros(g.n_p)])
        vr, vt, P = unpack(g, self.lu.solve(b))
        return vr, vt, P

    def step(self, T, vr, vt, dt):
        A, b = energy_operator(self.g, vr, vt, self.T_in, self.T_out, self.H, self.scheme)
        V = self.V
        M = sp.diags(V / dt) - A
        return spsolve(M.tocsc(), V / dt * T.ravel() + b).reshape(T.shape)

    def steady_T(self, vr, vt):
        A, b = energy_operator(self.g, vr, vt, self.T_in, self.T_out, self.H, self.scheme)
        return spsolve(A.tocsc(), -b).reshape(self.g.nr, self.g.nt)

    def run(self, T0, t_end=1.0, cfl=4.0, dt_max=1e-2, tol=1e-6, picard=True, picard_tol=1e-9,
            relax=0.5, max_picard=400, log=None, every=50):
        """March to steady state (backward Euler, dt = cfl * min cell / v_max), then optionally polish
        with relaxed Picard iterations. Returns T, (vr, vt, P), history list of diagnostic dicts."""
        g = self.g
        T = T0.copy()
        t, n = 0.0, 0
        hist = []
        h = min(g.dr.min(), g.r_f[0] * g.dt)
        while t < t_end:
            vr, vt, P = self.stokes(T)
            vmax = max(np.abs(vr).max(), np.abs(vt).max(), 1e-12)
            dt = min(cfl * h / vmax, dt_max)
            Tn = self.step(T, vr, vt, dt)
            change = np.abs(Tn - T).max() / dt
            T = Tn; t += dt; n += 1
            if n % every == 0:
                dgn = diagnostics(g, T, vr, vt, self.T_in, self.T_out); dgn.update(t=t, dTdt=change)
                hist.append(dgn)
                if log:
                    log(f"t={t:.4f} Nu_top={dgn['Nu_top']:.4f} Nu_bot={dgn['Nu_bot']:.4f} "
                        f"vrms={dgn['vrms']:.3f} |dT/dt|={change:.2e}")
            if change < tol:
                break
        if picard:
            for k in range(max_picard):
                vr, vt, P = self.stokes(T)
                Tn = self.steady_T(vr, vt)
                diff = np.abs(Tn - T).max()
                T = relax * Tn + (1 - relax) * T
                if diff < picard_tol:
                    break
            if log:
                log(f"Picard: {k + 1} iterations, last change {diff:.1e}")
        vr, vt, P = self.stokes(T)
        dgn = diagnostics(g, T, vr, vt, self.T_in, self.T_out); dgn.update(t=t)
        hist.append(dgn)
        return T, (vr, vt, P), hist
