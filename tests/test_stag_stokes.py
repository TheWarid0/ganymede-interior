import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("petsc4py")
from ganymede.stokes import Grid, solve_stokes
from ganymede.petsc_stokes import stokes_symmetric
from ganymede.stag_stokes import make_dmstag, assemble_stokes, native_stokes_solve, gather_fields, ours_to_stag

K = np.pi
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "parallel_stokes.py"


def test_native_assembly_equals_scipy_matrix():
    rng = np.random.default_rng(1)
    nx, nz = 16, 12
    g = Grid(nx, nz); dm = make_dmstag(nx, nz)
    eta_c = np.exp(3 * rng.normal(size=(nz, nx))); eta_n = np.exp(3 * rng.normal(size=(nz + 1, nx + 1)))
    Xz, Zz = np.meshgrid(g.xc, g.zn); rho = np.cos(K * Xz) * np.sin(K * Zz)
    A0, s, walls = stokes_symmetric(g, eta_c, eta_n)
    perm = ours_to_stag(g, dm); n = A0.shape[0]
    Pm = sp.csr_matrix((np.ones(n), (perm, np.arange(n))), shape=(n, n))
    A_ref = (Pm @ A0 @ Pm.T).tocsr()
    w = perm[walls]
    A, b, _ = assemble_stokes(dm, nx, nz, g.dx, g.dz, eta_c, eta_n, rho, A_ref[w[0], w[0]])
    ai, aj, av = A.getValuesCSR()
    A_nat = sp.csr_matrix((av, aj, ai), shape=(n, n))
    assert abs(A_nat - A_ref).max() / abs(A_ref).max() < 1e-12


@pytest.mark.parametrize("n", [64, 128])
def test_native_solver_analytic_and_iterations(n):
    g = Grid(n, n)
    Xz, Zz = np.meshgrid(g.xc, g.zn); Xx, Zx = np.meshgrid(g.xn, g.zc)
    dm, x, ix, st = native_stokes_solve(n, n, np.ones((n, n)), np.ones((n + 1, n + 1)), np.cos(K * Xz) * np.sin(K * Zz))
    vx, vz, P = gather_fields(x, ix, n, n)
    A0 = K / (2 * K**2) ** 2
    err = max(np.abs(vx - A0 * K * np.sin(K * Xx) * np.cos(K * Zx)).max(),
              np.abs(vz + A0 * K * np.cos(K * Xz) * np.sin(K * Zz)).max()) / (A0 * K)
    assert st["reason"] > 0 and st["outer"] <= 5 and st["inner"] <= 8
    assert err < 3.3e-3 * (16 / n) ** 2


def test_native_solver_stiff_lid_matches_direct():
    n = 64; g = Grid(n, n); Xz, Zz = np.meshgrid(g.xc, g.zn)
    eta_c = 1e8 ** np.meshgrid(g.xc, g.zc)[1]; eta_n = 1e8 ** np.meshgrid(g.xn, g.zn)[1]
    rho = np.cos(K * Xz) * np.sin(K * Zz)
    vx0, vz0, _ = solve_stokes(g, eta_c, eta_n, rho)
    dm, x, ix, st = native_stokes_solve(n, n, eta_c, eta_n, rho)
    vx, vz, _ = gather_fields(x, ix, n, n)
    assert st["outer"] <= 12
    assert max(np.abs(vx - vx0).max(), np.abs(vz - vz0).max()) / np.abs(vz0).max() < 1e-5


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("mpiexec") is None, reason="needs mpiexec")
def test_parallel_runs_match_serial(tmp_path):
    out = {}
    for p in (1, 2, 4):
        f = tmp_path / f"np{p}.npy"
        subprocess.run(["mpiexec", "-n", str(p), sys.executable, str(SCRIPT), "--n", "128", "--save", str(f)],
                       check=True, capture_output=True, timeout=600)
        out[p] = np.load(f)
    for p in (2, 4):
        assert np.abs(out[p] - out[1]).max() / np.abs(out[1]).max() < 1e-10
