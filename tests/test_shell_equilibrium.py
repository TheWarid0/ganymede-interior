import numpy as np
import pytest

from ganymede.shell_equilibrium import (A_K_ICE, equilibrium_thickness, max_thickness,
                                        shell_heat_flow)


def test_equilibrium_interpolation():
    D = np.array([10.0, 20.0, 40.0])
    q = np.array([40.0, 20.0, 10.0])
    assert equilibrium_thickness(D, q, 20.0) == pytest.approx(20.0)
    # log-linear between 20 and 40: q = 20 * 2^(-(D-20)/20) -> q = 10*sqrt(2) at D = 30
    assert equilibrium_thickness(D, q, 10 * np.sqrt(2)) == pytest.approx(30.0)
    assert equilibrium_thickness(D, q, 50.0) == 10.0
    assert equilibrium_thickness(D, q, 5.0) is None


def test_max_thickness_is_ih_iii_limit():
    assert max_thickness() / 1e3 == pytest.approx(156.1, abs=0.2)


def test_conductive_branch_matches_analytic():
    # thick grains, thin shell: far below onset, conductive flux 651 ln(Tb/Ts)/D
    r = shell_heat_flow(1e-3, 20e3, T_b=271.0)
    assert r["Nu"] == 1.0
    assert r["q_out"] == pytest.approx(A_K_ICE * np.log(271.0 / 110.0) / 20e3)


def test_melting_curve():
    seafreeze = pytest.importorskip("seafreeze")  # noqa: F841
    from ganymede.shell_equilibrium import T_melt_Ih
    assert T_melt_Ih(0.1e6) == pytest.approx(273.15, abs=0.05)
    assert T_melt_Ih(200e6) == pytest.approx(253.0, abs=1.0)
