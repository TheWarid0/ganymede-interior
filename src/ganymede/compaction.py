"""1D two-phase compaction: water percolating through a viscously deforming ice matrix.

Dimensionless "magma equation" (McKenzie 1984; Barcilon & Richter 1986; Spiegelman 1993)
for constant matrix viscosity (m = 0) and permeability k ∝ φ^n:

    ∂φ/∂t + ∂(φ^n)/∂z − ∂/∂z( φ^n ∂/∂z ∂φ/∂t ) = 0

φ is porosity scaled by the background porosity φ0, z (positive up) by the compaction
length δ, and t by δ / w0, with w0 the background water velocity. Each time step solves
an elliptic equation for the compaction rate C = ∂φ/∂t, then advances φ.
"""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_banded

YEAR = 3.156e7  # s


# --- solver -------------------------------------------------------------------

def compaction_rate(phi, dz, n=3):
    """Solve  C − ∂z(K ∂z C) = −∂z K,  K = φ^n,  for C = ∂φ/∂t.

    Second-order finite differences with K on cell faces (arithmetic mean);
    C = 0 at both ends (far-field background porosity).
    """
    K = phi**n
    Kf = 0.5 * (K[1:] + K[:-1])
    N = phi.size
    ab = np.zeros((3, N))
    rhs = np.zeros(N)
    ab[1, :] = 1.0
    ab[1, 1:-1] += (Kf[1:] + Kf[:-1]) / dz**2
    ab[0, 2:] = -Kf[1:] / dz**2
    ab[2, :-2] = -Kf[:-1] / dz**2
    rhs[1:-1] = -(Kf[1:] - Kf[:-1]) / dz
    return solve_banded((1, 1), ab, rhs)


def run(phi0, dz, t_end, dt, n=3):
    """Advance φ from t = 0 to t_end with SSP-RK3 (third order in time)."""
    phi = np.array(phi0, dtype=float)
    t = 0.0
    while t < t_end - 1e-12:
        h = min(dt, t_end - t)
        k1 = phi + h * compaction_rate(phi, dz, n)
        k2 = 0.75 * phi + 0.25 * (k1 + h * compaction_rate(k1, dz, n))
        phi = phi / 3 + 2 / 3 * (k2 + h * compaction_rate(k2, dz, n))
        t += h
    return phi


# --- analytic solitary wave (n = 3, m = 0) ------------------------------------

def solitary_wave_speed(A):
    """Speed of a solitary wave of amplitude A (n = 3, m = 0)."""
    return 2.0 * A + 1.0


def solitary_wave_profile(A):
    """Closed-form solitary wave (Barcilon & Richter 1986), parametrised by s = sqrt(A − φ)
    so the crest and the tails are both resolved. Returns (distance from crest, φ)."""
    a = np.sqrt(A - 1.0)
    s = np.concatenate([np.linspace(0, 0.9 * a, 2000, endpoint=False),
                        a * (1 - np.geomspace(0.1, 1e-10, 2000))])
    dist = np.sqrt(A + 0.5) * (2 * s - np.log((a - s) / (a + s)) / a)
    keep = np.concatenate([[True], np.diff(dist) > 0])   # round-off guard in the far tail
    return dist[keep], A - s[keep] ** 2


def solitary_wave(A, z, z0):
    """Solitary wave of amplitude A centred at z0, sampled on z.
    Interpolates s = sqrt(A − φ) with a cubic spline: s is smooth in distance, whereas
    linear interpolation of φ puts a kink at the crest."""
    dist, phi = solitary_wave_profile(A)
    spline = CubicSpline(dist, np.sqrt(A - phi))
    d = np.abs(np.asarray(z, dtype=float) - z0)
    s = np.where(d < dist[-1], spline(np.minimum(d, dist[-1])), np.sqrt(A - 1.0))
    return A - s**2


# --- physical scales ----------------------------------------------------------

@dataclass
class CompactionScales:
    k0: float      # background permeability [m^2]
    delta: float   # compaction length [m]
    w0: float      # background water velocity [m/s]
    t0: float      # time scale delta / w0 [s]


def physical_scales(d, eta, phi0=0.01, C=1000.0, n=3, mu=1.8e-3, drho=120.0, g=1.4):
    """Dimensional scales for grain size d [m] and matrix shear viscosity eta [Pa s].
    Permeability k0 = d² φ0^n / C; bulk viscosity ζ = η / φ0.
    Defaults are rough values for ice VI with liquid water."""
    k0 = d**2 * phi0**n / C
    zeta = eta / phi0
    delta = np.sqrt(k0 * (zeta + 4.0 / 3.0 * eta) / mu)
    w0 = k0 * drho * g / (mu * phi0)
    return CompactionScales(k0=k0, delta=delta, w0=w0, t0=delta / w0)
