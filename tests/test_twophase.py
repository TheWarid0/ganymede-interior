import numpy as np
import pytest
from scipy.special import erfc

from ganymede.twophase import (YEAR, HPIce, conduction_step, darcy_speed, front_height_energy_balance,
                               percolate, phase_change, riemann_solution, run_column, total_energy)

P = HPIce()


@pytest.mark.parametrize("lo, hi", [(0.03, 0.01), (0.03, 0.005), (0.012, 0.03)])
def test_percolation_riemann(lo, hi):
    """Shock (wet below dry) and rarefaction (dry below wet) against the exact solution;
    the error must fall with resolution and mass must be conserved."""
    errs = []
    for N in (200, 800):
        dz = 1000.0 / N
        z = (np.arange(N) + 0.5) * dz
        phi = np.where(z < 500, lo, hi)
        t = 0.3 * 1000.0 / float(darcy_speed(np.array(max(lo, hi)), P))
        out_phi, out = percolate(phi, dz, t, P)
        assert (out_phi.sum() + out / dz) == pytest.approx(phi.sum(), rel=1e-12)
        sel = z > 350        # away from the drainage fan growing from the impermeable bottom
        errs.append(np.abs(out_phi - riemann_solution(z, t, lo, hi, 500.0, P))[sel].mean())
    assert errs[1] < 0.6 * errs[0]
    assert errs[1] < 1e-4


def test_conduction_half_space():
    """Constant flux into a half-space: T = T0 + (2q/k) sqrt(kappa t) ierfc(z / 2 sqrt(kappa t))."""
    ierfc = lambda x: np.exp(-x**2) / np.sqrt(np.pi) - x * erfc(x)
    H, N, nt, T0 = 20e3, 400, 400, 300.0
    dz = H / N
    z = (np.arange(N) + 0.5) * dz
    T = np.full(N, T0)
    t_end = 0.1e6 * YEAR
    for _ in range(nt):
        T, _ = conduction_step(T, dz, t_end / nt, P, P.q_s, T0)
    s = np.sqrt(P.kappa * t_end)
    exact = T0 + 2 * P.q_s / P.k * s * ierfc(z / (2 * s))
    assert np.abs(T - exact).max() < 0.01      # of a ~16 K rise


def test_phase_change_conserves_energy():
    rng = np.random.default_rng(0)
    Tm = np.full(50, 320.0)
    T = Tm + rng.normal(0, 2, 50)
    phi = P.phi_bg + rng.uniform(0, 0.01, 50)
    T2, phi2 = phase_change(T, phi, Tm, P)
    assert total_energy(T2, phi2, 1.0, P) == pytest.approx(total_energy(T, phi, 1.0, P), rel=1e-14)
    assert np.all(T2 <= Tm + 1e-12)
    assert np.all(phi2 >= P.phi_bg - 1e-15)
    # cold cells keep only background water unless they ran out of cold to freeze it
    cold = T2 < Tm - 1e-9
    assert np.allclose(phi2[cold], P.phi_bg)


@pytest.mark.slow
def test_column_breakthrough_and_steady_state():
    """Conductive 200 km column: melt front reaches the ocean when the energy balance says,
    and at steady state the basal heat leaves as conduction + latent heat of extracted water."""
    r = run_column(P, N=200, dt=2e3 * YEAR, t_end=20e6 * YEAR, every=10)
    h = r["hist"]
    t_break = h["t"][np.argmax(h["temperate_top"] >= P.H - 1)]
    _, t_pred = front_height_energy_balance(0.0, P)
    assert t_break == pytest.approx(t_pred, rel=0.02)
    q_cond_melting_curve = P.k * (P.Tm_bot - P.Tm_top) / P.H
    assert h["water_flux"][-1] * P.rho_w * P.L + h["q_top"][-1] == pytest.approx(P.q_s, rel=1e-3)
    assert h["q_top"][-1] == pytest.approx(q_cond_melting_curve, rel=0.02)
    assert np.abs(h["energy_error"]).max() < 1e-9


def test_2d_prototype_conserves_energy_and_water():
    """A few hundred kyr of 2D two-phase convection on a coarse grid: energy (sensible + latent)
    balances the boundary fluxes and the extracted water, porosity never drops below background,
    and the ice never exceeds the melting point."""
    from ganymede.twophase2d import run
    r = run(nx=24, nz=12, t_end_myr=1.5, verbose=False)
    h = r["hist"]
    assert max(abs(x["energy_error"]) for x in h) < 1e-9
    assert r["phi"].min() >= P.phi_bg - 1e-12
    assert r["snaps"][-1]["dT"].max() <= 1e-9
    assert h[-1]["vrms_cm_yr"] > 1.0      # it convects


def test_2d_stag_solver_matches_direct():
    """No-slip-bottom Stokes on PETSc DMStag (geometric multigrid) = SciPy direct solve."""
    pytest.importorskip("petsc4py")
    from ganymede.stokes import Grid
    from ganymede.twophase2d import Scales, ice_velocity
    sc = Scales(P)
    g = Grid(32, 16, L=2.0)
    rng = np.random.default_rng(3)
    Zc = np.broadcast_to(g.zc[:, None], (16, 32))
    theta = np.clip(1 - Zc - 0.3 + 0.05 * rng.standard_normal((16, 32)), 0, None)
    phi = P.phi_bg + 0.01 * rng.random((16, 32))
    vx1, vz1, _ = ice_velocity(g, theta, phi, sc, solver="direct")
    vx2, vz2, _ = ice_velocity(g, theta, phi, sc, solver="stag")
    scale = max(np.abs(vx1).max(), np.abs(vz1).max())
    assert np.abs(vx2 - vx1).max() < 1e-5 * scale
    assert np.abs(vz2 - vz1).max() < 1e-5 * scale
    # no-slip bottom: the flow next to the bottom is much slower than with a free-slip bottom
    from scipy.sparse.linalg import spsolve
    from ganymede.ice import corner_T
    from ganymede.stokes import stokes_matrix_fast, unpack
    from ganymede.twophase2d import _centre_to_vz
    Zn = np.broadcast_to(g.zn[:, None], (17, 33))
    A = stokes_matrix_fast(g, sc.viscosity(theta, Zc), sc.viscosity(corner_T(theta), Zn))
    rhs = np.zeros(A.shape[0])
    rhs[g.n_vx:g.n_vx + g.n_vz] = _centre_to_vz(-sc.Ra * theta - sc.Rphi * phi).ravel()
    vx_free, _, _ = unpack(g, spsolve(A, rhs))
    assert np.abs(vx1[0]).mean() < 0.5 * np.abs(vx_free[0]).mean()
