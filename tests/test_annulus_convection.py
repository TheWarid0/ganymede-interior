"""Stage 7c: energy equation and thermal convection in the annulus."""
import numpy as np
import pytest

from ganymede.annulus import AnnulusGrid
from ganymede.annulus_convection import Convection, conductive_profile, diagnostics, energy_operator


@pytest.mark.parametrize("d", [1, 2])
def test_conduction_second_order(d):
    """Pure conduction between isothermal walls: T and Nu - 1 converge at second order to the exact
    profile (ln r for the cylinder, 1/r for the spherical annulus)."""
    out = []
    for n in (16, 32):
        g = AnnulusGrid.uniform(1.22, 2.22, n, 2 * n, d=d)
        c = Convection(g, Ra=0.0)
        vr, vt = np.zeros((g.nr - 1, g.nt)), np.zeros((g.nr, g.nt - 1))
        T = c.steady_T(vr, vt)
        dg = diagnostics(g, T, vr, vt)
        out.append((np.abs(T - conductive_profile(g)).max(), abs(dg["Nu_top"] - 1), abs(dg["Nu_bot"] - 1)))
    for e0, e1 in zip(*out):
        assert e0 / e1 == pytest.approx(4.0, rel=0.1)


def test_spherical_conduction_matches_published_identity():
    """|Nu - 1| for 16 radial cells, r = 1.22..2.22 (same number MILEVA's docs list for this test)."""
    g = AnnulusGrid.uniform(1.22, 2.22, 16, 32, d=2)
    c = Convection(g, Ra=0.0)
    vr, vt = np.zeros((g.nr - 1, g.nt)), np.zeros((g.nr, g.nt - 1))
    assert abs(diagnostics(g, c.steady_T(vr, vt), vr, vt)["Nu_top"] - 1) == pytest.approx(8.1e-4, rel=0.02)


@pytest.mark.parametrize("scheme", ["central", "upwind"])
def test_advection_conserves_heat(scheme):
    """With insulating walls the operator conserves heat exactly (column sums 0) and keeps a uniform
    temperature uniform (row sums 0) for a divergence-free Stokes velocity."""
    g = AnnulusGrid.refined(1.22, 2.22, 12, 48, ratio=3, d=2)
    c = Convection(g, Ra=1e5)
    T0 = conductive_profile(g) + 0.1 * np.cos(3 * g.t_c)[None, :]
    vr, vt, _ = c.stokes(T0)
    A, _ = energy_operator(g, vr, vt, T_in=None, T_out=None, scheme=scheme)
    scale = abs(A).max()
    assert np.abs(np.asarray(A.sum(axis=0))).max() < 1e-12 * scale
    assert np.abs(A @ np.ones(g.n_p)).max() < 1e-9 * scale


@pytest.mark.slow
def test_hernlund_tackley_ra1e4():
    """Spherical annulus, basal heating, isoviscous, Ra = 1e4, five plumes, 32 x 256 cells on the half
    annulus: Nu = 4.18 and v_rms = 37.7 in Hernlund & Tackley (2008)."""
    g = AnnulusGrid.uniform(1.22, 2.22, 32, 256, d=2)
    c = Convection(g, Ra=1e4)
    T0 = conductive_profile(g) + 0.05 * np.cos(5 * g.t_c)[None, :] * np.sin(np.pi * (g.r_c - 1.22))[:, None]
    T, _, hist = c.run(T0, t_end=3.0, cfl=8.0, tol=1e-5)
    h = hist[-1]
    assert h["Nu_top"] == pytest.approx(h["Nu_bot"], rel=1e-6)
    assert h["Nu_top"] == pytest.approx(4.18, rel=0.02)
    assert h["vrms"] == pytest.approx(37.7, rel=0.03)
