import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("petsc4py")
from ganymede.stag_convection import steady_convection_stag
from ganymede.ice import steady_convection_general, top_flux
from ganymede.convection import vrms

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "parallel_convection.py"


@pytest.mark.parametrize("b, nu", [(0.0, 4.9552), (np.log(1000), 11.6565)])
def test_dmstag_convection_identical_to_scipy(b, nu):
    eta = lambda T: np.exp(-b * np.asarray(T))
    g, T, vx, vz, st = steady_convection_stag(32, 32, 1e4, eta, tol=1e-7, relax=0.5)
    g2, T2, vx2, vz2, _, _ = steady_convection_general(32, 32, 1e4, eta, tol=1e-7, relax=0.5)
    assert np.abs(T - T2).max() < 1e-8
    assert st["Nu"] == pytest.approx(nu, abs=1e-4)
    assert st["vrms"] == pytest.approx(vrms(vx2, vz2), rel=1e-8)


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("mpiexec") is None, reason="needs mpiexec")
def test_dmstag_convection_parallel_matches_serial(tmp_path):
    out = {}
    for p in (1, 2):
        f = tmp_path / f"np{p}.npy"
        subprocess.run(["mpiexec", "-n", str(p), sys.executable, str(SCRIPT), "--case", "2a", "--n", "32", "--save", str(f)],
                       check=True, capture_output=True, timeout=600)
        out[p] = np.load(f)
    assert np.abs(out[2] - out[1]).max() < 1e-9
