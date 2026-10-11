"""Manufactured solution for the annulus Stokes solver (needs sympy; used by the tests only).

Stream function psi = F(r) sin(k theta) on theta in [0, pi]: v_r = (1/r) dpsi/dtheta, v_t = -dpsi/dr is
divergence free. F is a polynomial chosen so that every wall is exactly free slip:
    v_r = 0 and r d(v_t/r)/dr = 0 at r_in and r_out  ->  F = 0 and F'' = F'/r there;
    v_t = 0 and dv_r/dtheta = 0 at theta = 0, pi     ->  automatic for sin(k theta).
Viscosity varies by a factor exp(a) across the shell and laterally; pressure is a smooth field.
The body force follows from f = -div(2 eta eps) + grad P, evaluated symbolically.
"""
import numpy as np


def manufactured(r_in, r_out, k=2, a_visc=3.0):
    import sympy as s
    r, t = s.symbols("r theta", positive=True)
    c0, c1 = s.symbols("c0 c1")
    F = (r - r_in) * (r - r_out) * (1 + c0 * r + c1 * r**2)
    Fp, Fpp = s.diff(F, r), s.diff(F, r, 2)
    sol = s.solve([(Fpp - Fp / r).subs(r, r_in), (Fpp - Fp / r).subs(r, r_out)], [c0, c1])
    F = F.subs(sol)
    psi = F * s.sin(k * t)
    vr = s.diff(psi, t) / r
    vt = -s.diff(psi, r)
    eta = s.exp(a_visc * (r - r_in) / (r_out - r_in)) * (1 + s.Rational(1, 2) * s.cos(t))
    P = r**2 * s.cos(t) + s.sin(2 * t) * r
    err = s.diff(vr, r)
    ett = s.diff(vt, t) / r + vr / r
    srt = r * s.diff(vt / r, r) + s.diff(vr, t) / r          # 2 eps_rt
    trr, ttt, trt = 2 * eta * err, 2 * eta * ett, eta * srt
    div_r = s.diff(r * trr, r) / r + s.diff(trt, t) / r - ttt / r
    div_t = s.diff(r**2 * trt, r) / r**2 + s.diff(ttt, t) / r
    fr = -div_r + s.diff(P, r)
    ft = -div_t + s.diff(P, t) / r
    lam = lambda e: s.lambdify((r, t), e, "numpy")
    return dict(vr=lam(vr), vt=lam(vt), P=lam(P), eta=lam(eta), fr=lam(fr), ft=lam(ft),
                div=lam(s.simplify(s.diff(r * vr, r) / r + s.diff(vt, t) / r)))


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
