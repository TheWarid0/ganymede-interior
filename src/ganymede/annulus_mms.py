"""Manufactured solution for the annulus Stokes solver (needs sympy; used by the tests only).

Stream function psi = F(r) sin(k pi theta / span) on theta in [0, span]: v_r = r^-d dpsi/dtheta,
v_t = -r^(1-d) dpsi/dr is divergence free for both geometries (d = 1 cylindrical, d = 2 spherical annulus).
F is a polynomial chosen so that every wall is exactly free slip:
    v_r = 0 and r d(v_t/r)/dr = 0 at r_in and r_out  ->  F = 0 and F'' = d F'/r there;
    v_t = 0 and dv_r/dtheta = 0 at theta = 0, pi     ->  automatic for sin(k theta).
Viscosity varies by a factor exp(a) across the shell and laterally; pressure is a smooth field.
The body force follows from f = -div(2 eta eps) + grad P, evaluated symbolically.
"""
import numpy as np


def manufactured(r_in, r_out, k=2, a_visc=3.0, span=np.pi, d=1):
    import sympy as s
    r, t = s.symbols("r theta", positive=True)
    c0, c1 = s.symbols("c0 c1")
    F = (r - r_in) * (r - r_out) * (1 + c0 * r + c1 * r**2)
    Fp, Fpp = s.diff(F, r), s.diff(F, r, 2)
    sol = s.solve([(Fpp - d * Fp / r).subs(r, r_in), (Fpp - d * Fp / r).subs(r, r_out)], [c0, c1])
    F = F.subs(sol)
    kk = k * s.pi / s.nsimplify(span) if abs(span - np.pi) > 1e-12 else k
    psi = F * s.sin(kk * t)
    vr = s.diff(psi, t) / r**d
    vt = -s.diff(psi, r) * r**(1 - d)
    eta = s.exp(a_visc * (r - r_in) / (r_out - r_in)) * (1 + s.Rational(1, 2) * s.cos(t))
    P = r**2 * s.cos(t) + s.sin(2 * t) * r
    err = s.diff(vr, r)
    ett = s.diff(vt, t) / r + vr / r
    srt = r * s.diff(vt / r, r) + s.diff(vr, t) / r          # 2 eps_rt
    eoo = vr / r                                              # out-of-plane hoop strain (d = 2)
    trr, ttt, trt, too = 2 * eta * err, 2 * eta * ett, eta * srt, 2 * eta * eoo
    div_r = s.diff(r**d * trr, r) / r**d + s.diff(trt, t) / r - (ttt + (d - 1) * too) / r
    div_t = s.diff(r**(d + 1) * trt, r) / r**(d + 1) + s.diff(ttt, t) / r
    fr = -div_r + s.diff(P, r)
    ft = -div_t + s.diff(P, t) / r
    lam = lambda e: s.lambdify((r, t), e, "numpy")
    return dict(vr=lam(vr), vt=lam(vt), P=lam(P), eta=lam(eta), fr=lam(fr), ft=lam(ft),
                div=lam(s.simplify(s.diff(r**d * vr, r) / r**d + s.diff(vt, t) / r)))


def errors(g, sol, vr, vt, P):
    """Volume-weighted RMS errors (relative), pressure compared up to its mean."""
    R, T = g.vr_points(); e_vr = vr - sol["vr"](R, T)
    R2, T2 = g.vt_points(); e_vt = vt - sol["vt"](R2, T2)
    Rp, Tp = g.p_points(); A = g.cell_area()
    Pe = sol["P"](Rp, Tp)
    dp = (P - (P * A).sum() / A.sum()) - (Pe - (Pe * A).sum() / A.sum())
    vscale = max(np.abs(sol["vr"](R, T)).max(), np.abs(sol["vt"](R2, T2)).max())
    return (np.sqrt(np.mean(e_vr**2)) / vscale, np.sqrt(np.mean(e_vt**2)) / vscale,
            np.sqrt((dp**2 * A).sum() / A.sum()) / np.abs(Pe).max())
