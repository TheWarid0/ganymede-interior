import numpy as np
import pytest

pytest.importorskip("sympy")

from ganymede.annulus import AnnulusGrid, solve_stokes
from ganymede.annulus_mms import errors, manufactured


def _solve(g, sol):
    Rc, Tc = g.p_points(); Rn, Tn = g.node_points()
    R, T = g.vr_points(); R2, T2 = g.vt_points()
    vr, vt, P = solve_stokes(g, sol["eta"](Rc, Tc), sol["eta"](Rn, Tn), sol["fr"](R, T), sol["ft"](R2, T2))
    return errors(g, sol, vr, vt, P), (vr, vt)


@pytest.mark.parametrize("r_in, aspect, kind", [(0.5, 2, "uniform"), (0.5, 2, "refined"),
                                                (1 - 150 / 2631.2, 8, "uniform"), (1 - 150 / 2631.2, 8, "refined")])
def test_second_order_convergence(r_in, aspect, kind):
    """Manufactured solution with exact free-slip walls and a viscosity varying by e^3 radially and x3
    laterally: velocity and pressure errors must fall 4x per grid doubling (thick and Ganymede-thin
    annuli, uniform and wall-refined radial grids)."""
    sol = manufactured(r_in, 1.0)
    errs = []
    for n in (16, 32):
        make = AnnulusGrid.uniform if kind == "uniform" else AnnulusGrid.refined
        g = make(r_in, 1.0, n, aspect * n) if kind == "uniform" else make(r_in, 1.0, n, aspect * n, ratio=4)
        errs.append(_solve(g, sol)[0])
    for e0, e1 in zip(*errs):
        assert e0 / e1 == pytest.approx(4.0, rel=0.15)
    assert max(errs[1]) < 4e-2


def test_discrete_divergence_free():
    sol = manufactured(0.5, 1.0)
    g = AnnulusGrid.uniform(0.5, 1.0, 16, 32)
    _, (vr, vt) = _solve(g, sol)
    vr_full = np.vstack([np.zeros((1, g.nt)), vr, np.zeros((1, g.nt))])
    vt_full = np.hstack([np.zeros((g.nr, 1)), vt, np.zeros((g.nr, 1))])
    rf = g.r_f[:, None]
    div = (np.diff(rf * vr_full, axis=0) / (g.r_c[:, None] * g.dr[:, None])
           + np.diff(vt_full, axis=1) / (g.r_c[:, None] * g.dt))
    div[0, 0] = 0.0                              # the pinned-pressure row
    assert np.abs(div).max() < 1e-9 * np.abs(vr).max()
