import numpy as np
import pytest

pytest.importorskip("petsc4py")
from ganymede.stokes import Grid, solve_stokes
from ganymede.petsc_stokes import petsc_stokes_solve

K = np.pi


def _setup(n):
    g = Grid(n, n)
    Xz, Zz = np.meshgrid(g.xc, g.zn)
    Xx, Zx = np.meshgrid(g.xn, g.zc)
    return g, Xz, Zz, Xx, Zx


@pytest.mark.parametrize("n", [32, 64, 128])
def test_petsc_isoviscous_analytic_and_mesh_independent_iterations(n):
    g, Xz, Zz, Xx, Zx = _setup(n)
    vx, vz, P, its, reason, _ = petsc_stokes_solve(g, np.ones((n, n)), np.ones((n + 1, n + 1)), np.cos(K * Xz) * np.sin(K * Zz))
    A0 = K / (2 * K**2) ** 2
    err = max(np.abs(vx - A0 * K * np.sin(K * Xx) * np.cos(K * Zx)).max(),
              np.abs(vz + A0 * K * np.cos(K * Xz) * np.sin(K * Zz)).max()) / (A0 * K)
    assert reason > 0
    assert its <= 5
    assert err < 3.3e-3 * (16 / n) ** 2


@pytest.mark.parametrize("contrast", [1e3, 1e8])
def test_petsc_matches_direct_solver_with_viscosity_contrast(contrast):
    n = 64
    g, Xz, Zz, _, _ = _setup(n)
    eta_c = contrast ** np.meshgrid(g.xc, g.zc)[1]
    eta_n = contrast ** np.meshgrid(g.xn, g.zn)[1]
    rho = np.cos(K * Xz) * np.sin(K * Zz)
    vx0, vz0, _ = solve_stokes(g, eta_c, eta_n, rho)
    vx, vz, _, its, reason, _ = petsc_stokes_solve(g, eta_c, eta_n, rho)
    assert reason > 0 and its <= 15
    assert max(np.abs(vx - vx0).max(), np.abs(vz - vz0).max()) / np.abs(vz0).max() < 1e-6
