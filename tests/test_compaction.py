import numpy as np
import pytest

from ganymede.compaction import (run, solitary_wave, solitary_wave_speed,
                                 physical_scales)


@pytest.mark.parametrize("A", [2.0, 5.0, 8.0])
def test_analytic_wave_satisfies_travelling_wave_ode(A):
    """-c(φ-1) + (φ³-1) + c φ³ φ'' = 0 with c = 2A + 1."""
    c = solitary_wave_speed(A)
    z = np.linspace(0, 200, 16001)
    dz = z[1] - z[0]
    phi = solitary_wave(A, z, 100.0)
    p = phi[1:-1]
    pzz = (phi[2:] - 2 * p + phi[:-2]) / dz**2
    res = -c * (p - 1) + (p**3 - 1) + c * p**3 * pzz
    assert phi.max() == pytest.approx(A)
    assert np.abs(res).max() < 1e-3


def _wave_error(dz, A=5.0, T=8.0, z0=50.0):
    c = solitary_wave_speed(A)
    z = np.arange(0, 200 + dz / 2, dz)
    phi = run(solitary_wave(A, z, z0), dz, T, dt=0.5 * dz / c)
    exact = solitary_wave(A, z, z0 + c * T)
    return np.sqrt(np.mean((phi - exact) ** 2)), phi, z


def test_solitary_wave_benchmark_second_order():
    e1, _, _ = _wave_error(0.2)
    e2, phi, z = _wave_error(0.1)
    assert e2 < 3e-3
    assert np.log2(e1 / e2) == pytest.approx(2.0, abs=0.15)
    assert z[np.argmax(phi)] == pytest.approx(50.0 + 11.0 * 8.0, abs=0.2)


def test_gaussian_conserves_mass_and_sheds_solitary_wave():
    dz = 0.1
    z = np.arange(0, 400 + dz / 2, dz)
    phi0 = 1 + 4 * np.exp(-((z - 40) / 6) ** 2)
    a = run(phi0, dz, 20.0, dt=0.5 * dz / 11)
    b = run(a, dz, 10.0, dt=0.5 * dz / 11)
    assert np.trapezoid(b - 1, z) == pytest.approx(np.trapezoid(phi0 - 1, z), rel=1e-8)
    A = b.max()
    speed = (z[np.argmax(b)] - z[np.argmax(a)]) / 10.0
    assert speed == pytest.approx(solitary_wave_speed(A), abs=0.05)


def test_crossing_time_independent_of_viscosity():
    """w0 depends on permeability only; δ grows as sqrt(η)."""
    s14 = physical_scales(1e-3, 1e14)
    s16 = physical_scales(1e-3, 1e16)
    assert s14.w0 == pytest.approx(s16.w0)
    assert s16.delta / s14.delta == pytest.approx(10.0, rel=1e-2)
