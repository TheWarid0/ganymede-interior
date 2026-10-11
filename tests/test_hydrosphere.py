import numpy as np
import pytest

pytest.importorskip("seafreeze")

from ganymede.constants import M_GANYMEDE
from ganymede.hydrosphere import (EOS, HydroParams, State, depression, enthalpy, evolve, march, solve_column,
                                  L_IH, S_EUT)


@pytest.fixture(scope="module")
def eos():
    return EOS()


def test_liquidus_matches_seafreeze_and_kalousova(eos):
    assert eos.liquidus(0.1)[0] == pytest.approx(273.15, abs=0.05)
    T, ph = eos.liquidus(1600.0)
    assert ph == 6 and T == pytest.approx(331.5, abs=0.6)        # ice VI at 1.6 GPa: Kalousova et al. 2018: 332 K
    assert eos.liquidus(150.0)[1] == 1 and eos.liquidus(400.0)[1] == 5


def test_freezing_point_depression_seawater():
    assert depression(0.035, 273.15, L_IH) == pytest.approx(1.9, abs=0.25)   # seawater: -1.9 C
    assert depression(0.0, 273.15, L_IH) == 0.0
    assert depression(1.0, 273.15, L_IH) == depression(S_EUT, 273.15, L_IH)


def test_column_holds_mass_and_cools_into_contraction(eos):
    """Pure water: a thicker shell (colder ocean) means more HP ice and a SMALLER Ganymede."""
    p = HydroParams(S0=0.0)
    march.M_h2o_guess = 0.0
    M, _ = march(eos, p, 2631.2e3, 38.7e3, 0, 0, 0, M_GANYMEDE)
    Mr = M_GANYMEDE - M
    cols = [solve_column(eos, p, State(D=D, s_top=0.0), M, Mr, 2631.2e3) for D in (20e3, 80e3)]
    for c in cols:
        m, _ = march(eos, p, c.R, c.D, 0, 0, 0, Mr)
        assert m == pytest.approx(M, rel=1e-6)
    assert cols[1].hp_thick > cols[0].hp_thick + 50e3
    assert cols[1].R < cols[0].R - 2e3


@pytest.mark.slow
def test_evolution_conserves_salt_and_energy(eos):
    rec, info = evolve(HydroParams(S0=0.01, f_sink=0.0), eos, t_end=300e6, dD=2000.0, verbose=False)
    assert max(abs(r["salt_error"]) for r in rec) < 1e-12
    assert max(abs(r["energy_error"]) for r in rec) < 1e-10
    assert rec[-1]["shell_km"] > rec[0]["shell_km"]
