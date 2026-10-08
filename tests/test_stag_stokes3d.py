import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip("petsc4py")
from ganymede.stokes import Grid, solve_stokes
from ganymede.stag_stokes3d import stokes3d_solve, gather3d, analytic3d, edge_viscosity

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "parallel_stokes3d.py"
ONE = lambda x, y, z: np.ones_like(x)


def _error(m):
    n = (m, m, m)
    rho, vxe, vye, vze = analytic3d(n)
    ec, ee = edge_viscosity(ONE, n)
    dm, x, ix, st, A = stokes3d_solve(n, ec, ee, rho)
    vx, vy, vz, P = gather3d(x, ix, n)
    err = max(np.abs(vx - vxe).max(), np.abs(vy - vye).max(), np.abs(vz - vze).max()) / np.abs(vze).max()
    h = 1 / m
    div = np.abs((vx[:, :, 1:] - vx[:, :, :-1]) / h + (vy[:, 1:] - vy[:, :-1]) / h + (vz[1:] - vz[:-1]) / h).max()
    return err, div, st, A


def test_3d_matrix_symmetric():
    *_, A = _error(8)
    ai, aj, av = A.getValuesCSR()
    As = sp.csr_matrix((av, aj, ai))
    assert abs(As - As.T).max() == 0.0


def test_3d_analytic_second_order_divergence_free_mesh_independent():
    e8, d8, s8, _ = _error(8)
    e16, d16, s16, _ = _error(16)
    assert np.log2(e8 / e16) == pytest.approx(2.0, abs=0.05)
    assert max(d8, d16) < 1e-9
    assert s16["outer"] <= 5 and s16["inner"] <= 8


@pytest.mark.parametrize("contrast", [1e3, 1e8])
def test_3d_equals_2d_when_nothing_varies_in_y(contrast):
    m, ny = 16, 4
    n = (m, ny, m)
    ec, ee = edge_viscosity(lambda x, y, z: contrast**z, n)
    xc = (np.arange(m) + .5) / m; zn = np.arange(m + 1) / m; yc = (np.arange(ny) + .5) / ny
    Z, Y, X = np.meshgrid(zn, yc, xc, indexing="ij")
    dm, x, ix, st, _ = stokes3d_solve(n, ec, ee, np.cos(np.pi * X) * np.sin(np.pi * Z))
    vx, vy, vz, _ = gather3d(x, ix, n)
    g = Grid(m, m); Xz, Zz = np.meshgrid(g.xc, g.zn)
    vx2, vz2, _ = solve_stokes(g, contrast ** np.meshgrid(g.xc, g.zc)[1], contrast ** np.meshgrid(g.xn, g.zn)[1],
                               np.cos(np.pi * Xz) * np.sin(np.pi * Zz))
    sc = np.abs(vz2).max()
    assert max(np.abs(vx - vx2[:, None, :]).max(), np.abs(vz - vz2[:, None, :]).max()) / sc < 1e-5
    assert np.abs(vy).max() / sc < 1e-6


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("mpiexec") is None, reason="needs mpiexec")
def test_3d_parallel_matches_serial(tmp_path):
    out = {}
    for p in (1, 2):
        f = tmp_path / f"np{p}.npy"
        subprocess.run(["mpiexec", "-n", str(p), sys.executable, str(SCRIPT), "--n", "16", "--save", str(f)],
                       check=True, capture_output=True, timeout=600)
        out[p] = np.load(f)
    assert np.abs(out[2] - out[1]).max() / np.abs(out[1]).max() < 1e-10
