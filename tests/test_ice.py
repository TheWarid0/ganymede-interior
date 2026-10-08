import numpy as np
import pytest

from ganymede.ice import (arrhenius, conductivity_ice, conductive_flux, steady_convection_general,
                          convect_transient, top_flux, onset_criterion)
from ganymede.convection import steady_convection, nusselt

T_S, T_B = 110.0, 268.3


def test_conduction_with_k_over_T_matches_analytic():
    k = conductivity_ice(T_S, T_B)
    g, T, _, _, _, _ = steady_convection_general(8, 64, 0.0, lambda T: np.ones_like(T), k, tol=1e-12, relax=1.0)
    exact = T_B * (T_S / T_B) ** g.zc
    assert np.abs(T_S + T[:, 0] * (T_B - T_S) - exact).max() < 1e-3
    assert top_flux(T, g, float(k(0.0))) == pytest.approx(conductive_flux(T_S, T_B), rel=1e-4)


def test_general_solver_reduces_to_blankenbach_1a():
    one = lambda T: np.ones_like(np.asarray(T, float))
    g, T, _, _, _, _ = steady_convection_general(32, 32, 1e4, one, tol=1e-7, relax=0.5)
    assert top_flux(T, g, 1.0) == pytest.approx(4.9552, abs=1e-3)


def test_onset_criterion_values():
    Ra_cr, theta = onset_criterion(T_S, T_B)
    assert theta == pytest.approx(15.7, abs=0.1)
    assert Ra_cr == pytest.approx(1.27e6, rel=0.02)


@pytest.mark.slow
def test_transient_reaches_picard_steady_state():
    one = lambda T: np.ones_like(np.asarray(T, float))
    g, T, _, _, _ = convect_transient(32, 32, 1e4, one, one, t_end=0.6, cfl=0.5)
    g2, T2, _, _, _ = steady_convection(32, 1e4)
    assert np.abs(T - T2).max() < 1e-6


@pytest.mark.slow
def test_viscosity_cap_does_not_matter_above_1e8():
    k = conductivity_ice(T_S, T_B)
    out = []
    for cap in (1e8, 1e10):
        g, T, _, _, _, _ = steady_convection_general(64, 64, 4.6e7, arrhenius(T_S, T_B, cap=cap), k)
        out.append(top_flux(T, g, float(k(0.0))) / conductive_flux(T_S, T_B))
    assert out[0] == pytest.approx(1.950, abs=0.01)
    assert out[1] == pytest.approx(out[0], rel=2e-3)
