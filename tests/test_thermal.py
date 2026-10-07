import numpy as np
import pytest

from ganymede.constants import R_GANYMEDE, M_GANYMEDE, MOI_GANYMEDE
from ganymede.structure import (fit_three_layer, fit_with_h2o_profile,
                                hydrostatic_profile, layered_density)

seafreeze = pytest.importorskip("seafreeze")
from ganymede.thermal import (H2OTables, h2o_column, conductive_shell_thickness,
                              self_consistent_model)


@pytest.fixture(scope="module")
def tables():
    return H2OTables()


@pytest.fixture(scope="module")
def three_layer():
    rc, rm = fit_three_layer(R_GANYMEDE, M_GANYMEDE, MOI_GANYMEDE, 5150, 3300, 1100)
    r = np.linspace(0, R_GANYMEDE, 20001)
    prof = hydrostatic_profile(r, layered_density(r, [rc, rm, R_GANYMEDE], [5150, 3300, 1100]))
    return rc, rm, prof


def test_profile_fit_reduces_to_uniform_case():
    """With a uniform H2O density, the profile fit must equal the closed-form 3-layer fit."""
    rc, rm = fit_three_layer(R_GANYMEDE, M_GANYMEDE, MOI_GANYMEDE, 5150, 3300, 1100)
    z = np.linspace(0, 1e6, 101)
    rc2, rm2 = fit_with_h2o_profile(R_GANYMEDE, M_GANYMEDE, MOI_GANYMEDE, 5150, 3300,
                                    z, np.full_like(z, 1100.0), guess=(0.3, 0.7))
    assert rc2 == pytest.approx(rc, rel=1e-4)
    assert rm2 == pytest.approx(rm, rel=1e-5)


@pytest.mark.parametrize("q", [0.005, 0.015, 0.040])
def test_ice_shell_matches_closed_form(tables, three_layer, q):
    """Numerical Ih shell thickness = a ln(T_base/T_surf)/q, with T_base taken at the shell base."""
    rc, rm, prof = three_layer
    col = h2o_column(q, prof, R_GANYMEDE - rm, tables, dz=100.0)
    ih = col.phase == 1
    T_base = col.T[ih][-1]
    D_exact = conductive_shell_thickness(q, 110.0, T_base)
    assert col.shell == pytest.approx(D_exact, abs=150.0)     # within ~ one step


def test_shell_base_is_at_ih_melting(tables, three_layer):
    """The first ocean sample sits just past the Ih-liquid boundary (within a step's T jump)."""
    rc, rm, prof = three_layer
    col = h2o_column(0.015, prof, R_GANYMEDE - rm, tables)
    i = np.argmax(col.phase == 0)
    assert col.phase[i - 1] == 1
    assert tables.phase(col.P[i], col.T[i] - 1.0) == 1        # 1 K colder would be ice Ih


def test_self_consistent_model_matches_mass_and_moi(tables):
    model = self_consistent_model(0.015, tables)
    assert model.iterations <= 5
    assert model.profile.mass == pytest.approx(M_GANYMEDE, rel=1e-3)
    assert model.profile.moi == pytest.approx(MOI_GANYMEDE, rel=1e-3)
    assert model.column.sequence()[:2] == [1, 0]               # Ih shell over an ocean
    assert 780e3 < R_GANYMEDE - model.r_mantle < 830e3
