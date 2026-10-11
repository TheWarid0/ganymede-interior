import numpy as np
import pytest

from ganymede.shell2p import (YEAR, P_melt_Ih, ShellParams, ShellScales, T_melt_Ih, conductive_theta,
                              percolate_down, phase_change, run)


def test_iapws_melting_curve():
    assert T_melt_Ih(0.1e6) == pytest.approx(273.153, abs=2e-3)
    assert P_melt_Ih(251.165) / 1e6 == pytest.approx(208.566, abs=0.01)    # Ih-III-liquid triple point
    sf = pytest.importorskip("seafreeze")  # noqa: F841
    from ganymede.shell_equilibrium import T_melt_Ih as T_sf
    for P in (50e6, 150e6):
        assert T_melt_Ih(P) == pytest.approx(T_sf(P), abs=0.15)


def test_conductive_shell():
    """Coarse grains (huge viscosity): no convection, the steady state is the k = a/T conductive
    profile and the surface flux is 651 ln(T_b/T_s)/D."""
    p = ShellParams(D=30e3, d=0.1)
    sc = ShellScales(p)
    r = run(p, nx=8, nz=40, aspect=0.2, t_end_myr=60, verbose=False, dt_max=2e-2, solver="direct")
    h = r["hist"][-1]
    assert h["q_surface"] == pytest.approx(651 * np.log(sc.T_b / p.T_s) / p.D, rel=2e-3)
    assert h["q_base"] == pytest.approx(h["q_surface"], rel=2e-3)
    z = (np.arange(40) + 0.5) / 40
    assert np.abs(r["theta"][:, 0] - conductive_theta(z, sc)).max() < 2e-3
    assert max(abs(x["energy_error"]) for x in r["hist"]) < 1e-9     # iterative energy solve, rtol 1e-12


def test_downward_percolation_conserves_water():
    sc = ShellScales(ShellParams(D=50e3))
    phi = np.zeros((20, 3))
    phi[12:15] = 0.03
    out_phi, out = percolate_down(phi, 1 / 20, 1e-4, sc)
    assert out_phi.sum() / 20 + out.sum() == pytest.approx(phi.sum() / 20, rel=1e-12)
    # water moved down: centre of mass lower
    z = np.arange(20)[:, None]
    assert (out_phi * z).sum() / out_phi.sum() < (phi * z).sum() / phi.sum()


def test_phase_change_conserves_energy():
    sc = ShellScales(ShellParams())
    rng = np.random.default_rng(0)
    tm = np.full(30, 1.02)
    th = tm + rng.normal(0, 0.02, 30)
    ph = rng.uniform(0, 0.01, 30)
    th2, ph2 = phase_change(th, ph, tm, sc)
    assert (th2 + ph2 / sc.S).sum() == pytest.approx((th + ph / sc.S).sum(), rel=1e-14)
    assert np.all(th2 <= tm + 1e-12)


def test_tidal_heating_peaks_at_maxwell_time():
    p = ShellParams(edot0=4e-11)
    sc = ShellScales(p)
    eta = np.logspace(12, 17, 501)
    h = sc.tidal(eta / sc.eta_b)
    assert eta[np.argmax(h)] == pytest.approx(p.mu_e / p.omega, rel=0.03)


@pytest.mark.slow
def test_shell_with_tidal_heating_conserves_energy():
    p = ShellParams(D=80e3, d=3e-4, edot0=6e-11)
    r = run(p, nx=32, nz=16, aspect=2.0, t_end_myr=5, verbose=False, solver="direct")
    assert max(abs(x["energy_error"]) for x in r["hist"]) < 1e-9
    assert r["hist"][-1]["q_tidal"] > 0
