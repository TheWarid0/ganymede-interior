import numpy as np
import pytest

from ganymede.constants import G, R_GANYMEDE, M_GANYMEDE, MOI_GANYMEDE
from ganymede.structure import hydrostatic_profile, layered_density, fit_three_layer


def test_uniform_sphere_matches_analytic():
    R, rho0 = 2.6e6, 2000.0
    r = np.linspace(0, R, 4001)
    prof = hydrostatic_profile(r, np.full_like(r, rho0))
    g_exact = 4 / 3 * np.pi * G * rho0 * r
    P_exact = 2 / 3 * np.pi * G * rho0**2 * (R**2 - r**2)
    assert np.allclose(prof.g, g_exact, rtol=1e-6, atol=1e-9)
    assert np.allclose(prof.P, P_exact, rtol=1e-5, atol=1.0)
    assert prof.moi == pytest.approx(0.4, rel=1e-5)


def test_three_layer_fit_reproduces_mass_and_moi():
    rc, rm = fit_three_layer(R_GANYMEDE, M_GANYMEDE, MOI_GANYMEDE,
                             rho_core=5150.0, rho_mantle=3300.0, rho_h2o=1100.0)
    r = np.linspace(0, R_GANYMEDE, 20001)
    rho = layered_density(r, [rc, rm, R_GANYMEDE], [5150.0, 3300.0, 1100.0])
    prof = hydrostatic_profile(r, rho)
    assert prof.mass == pytest.approx(M_GANYMEDE, rel=1e-3)
    assert prof.moi == pytest.approx(MOI_GANYMEDE, rel=1e-3)
