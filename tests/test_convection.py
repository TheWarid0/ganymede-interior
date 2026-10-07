import numpy as np
import pytest

from ganymede.convection import convect, steady_convection, nusselt, vrms

NU_1A, VRMS_1A = 4.884409, 42.864947
NU_2A, VRMS_2A = 10.0660, 480.4334


def test_time_stepping_and_steady_solver_agree_case_1a():
    """Two independent algorithms must give the same steady state on the same grid."""
    g, T, vx, vz, _ = convect(32, 1e4)
    g2, T2, vx2, vz2, _ = steady_convection(32, 1e4)
    assert nusselt(T, g) == pytest.approx(nusselt(T2, g2), rel=1e-6)
    assert vrms(vx, vz) == pytest.approx(vrms(vx2, vz2), rel=1e-6)
    assert nusselt(T2, g2) == pytest.approx(4.9552, abs=1e-4)


def test_blankenbach_1a_richardson():
    res = {}
    for n in (32, 64):
        g, T, vx, vz, _ = steady_convection(n, 1e4)
        res[n] = nusselt(T, g), vrms(vx, vz)
    nu = (4 * res[64][0] - res[32][0]) / 3
    vr = (4 * res[64][1] - res[32][1]) / 3
    assert nu == pytest.approx(NU_1A, rel=1e-3)
    assert vr == pytest.approx(VRMS_1A, rel=1e-3)


def test_blankenbach_2a_regression_64():
    g, T, vx, vz, _ = steady_convection(64, 1e4, b_visc=np.log(1000))
    assert nusselt(T, g) == pytest.approx(10.3990, abs=1e-3)
    assert vrms(vx, vz) == pytest.approx(467.488, abs=1e-2)


@pytest.mark.slow
def test_blankenbach_2a_richardson_128():
    res = {}
    for n in (64, 128):
        g, T, vx, vz, _ = steady_convection(n, 1e4, b_visc=np.log(1000))
        res[n] = nusselt(T, g), vrms(vx, vz)
    nu = (4 * res[128][0] - res[64][0]) / 3
    vr = (4 * res[128][1] - res[64][1]) / 3
    assert nu == pytest.approx(NU_2A, rel=1e-3)
    assert vr == pytest.approx(VRMS_2A, rel=1e-3)
