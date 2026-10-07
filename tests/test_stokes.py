import numpy as np
import pytest

from ganymede.stokes import Grid, solve_stokes, stokes_system, stokes_matrix_fast


def _sinusoidal_error(n):
    g = Grid(n, n)
    k = np.pi
    Xz, Zz = np.meshgrid(g.xc, g.zn)
    Xx, Zx = np.meshgrid(g.xn, g.zc)
    vx, vz, P = solve_stokes(g, np.ones((n, n)), np.ones((n + 1, n + 1)), np.cos(k * Xz) * np.sin(k * Zz))
    A = k / (2 * k**2) ** 2
    vx_ex = A * k * np.sin(k * Xx) * np.cos(k * Zx)
    vz_ex = -A * k * np.cos(k * Xz) * np.sin(k * Zz)
    err = max(np.abs(vx - vx_ex).max(), np.abs(vz - vz_ex).max()) / np.abs(vz_ex).max()
    div = np.abs((vx[:, 1:] - vx[:, :-1]) / g.dx + (vz[1:] - vz[:-1]) / g.dz).max()
    return err, div


def test_sinusoidal_density_second_order_and_divergence_free():
    e16, d16 = _sinusoidal_error(16)
    e32, d32 = _sinusoidal_error(32)
    assert e32 < 1e-3
    assert np.log2(e16 / e32) == pytest.approx(2.0, abs=0.05)
    assert max(d16, d32) < 1e-9          # round-off level (|v|/dx ~ 1)


@pytest.mark.parametrize("nx,nz", [(8, 11), (33, 36)])
def test_fast_assembly_identical_to_loop(nx, nz):
    rng = np.random.default_rng(0)
    g = Grid(nx, nz)
    eta_c = np.exp(3 * rng.normal(size=(nz, nx)))
    eta_n = np.exp(3 * rng.normal(size=(nz + 1, nx + 1)))
    A_loop, _ = stokes_system(g, eta_c, eta_n, np.zeros((nz + 1, nx)))
    assert abs(A_loop - stokes_matrix_fast(g, eta_c, eta_n)).max() == 0.0
