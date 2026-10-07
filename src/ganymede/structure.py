"""Radial structure of a spherically symmetric, self-gravitating body.

Given a density profile rho(r), compute the enclosed mass m(r), gravity g(r),
hydrostatic pressure P(r) and the normalised moment of inertia.
"""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import fsolve

from .constants import G


@dataclass
class Profile:
    r: np.ndarray    # radius [m], increasing from 0 to R
    rho: np.ndarray  # density [kg m^-3]
    m: np.ndarray    # enclosed mass [kg]
    g: np.ndarray    # gravitational acceleration [m s^-2]
    P: np.ndarray    # pressure [Pa]

    @property
    def mass(self):
        return self.m[-1]

    @property
    def moi(self):
        """Normalised moment of inertia C / (M R^2)."""
        R = self.r[-1]
        I = 8 * np.pi / 3 * np.trapezoid(self.rho * self.r**4, self.r)
        return I / (self.mass * R**2)


def hydrostatic_profile(r, rho):
    """Integrate mass, gravity and pressure for a given density profile.

    m(r) = int_0^r 4 pi r'^2 rho dr'
    g(r) = G m(r) / r^2
    P(r) = int_r^R rho g dr'        (P = 0 at the surface)
    """
    r = np.asarray(r, dtype=float)
    rho = np.asarray(rho, dtype=float)
    # Mass per shell from the exact shell volume 4/3 pi (r_out^3 - r_in^3), with
    # the cell-mean density. (The trapezoid rule on 4 pi r^2 rho is biased near
    # r = 0, where r^2 is strongly curved -- the uniform-sphere test catches it.)
    rho_mid = 0.5 * (rho[1:] + rho[:-1])
    dm = 4 * np.pi / 3 * rho_mid * (r[1:] ** 3 - r[:-1] ** 3)
    m = np.concatenate([[0.0], np.cumsum(dm)])
    g = np.zeros_like(r)
    g[1:] = G * m[1:] / r[1:] ** 2
    # integrate from the surface inwards
    integrand = rho * g
    P_from_centre = cumulative_trapezoid(integrand, r, initial=0.0)
    P = P_from_centre[-1] - P_from_centre
    return Profile(r=r, rho=rho, m=m, g=g, P=P)


def layered_density(r, outer_radii, densities):
    """Piecewise-constant density. Layer i spans (outer_radii[i-1], outer_radii[i]]."""
    rho = np.empty_like(r, dtype=float)
    inner = 0.0
    for r_out, d in zip(outer_radii, densities):
        rho[(r >= inner) & (r <= r_out)] = d
        inner = r_out
    return rho


def fit_three_layer(R, M, moi, rho_core, rho_mantle, rho_h2o, guess=(0.25, 0.7)):
    """Find core and mantle outer radii so a 3-layer body matches M and C/MR^2.

    Densities are fixed inputs; the two radii are the unknowns. Uses the
    closed-form mass and moment of inertia of uniform spherical shells.
    Returns (r_core, r_mantle) in metres.
    """
    def residuals(x):
        rc, rm = x[0] * R, x[1] * R
        mass = 4 * np.pi / 3 * (rho_core * rc**3
                                + rho_mantle * (rm**3 - rc**3)
                                + rho_h2o * (R**3 - rm**3))
        inertia = 8 * np.pi / 15 * (rho_core * rc**5
                                    + rho_mantle * (rm**5 - rc**5)
                                    + rho_h2o * (R**5 - rm**5))
        return [mass / M - 1.0, inertia / (M * R**2) / moi - 1.0]

    sol, info, ier, msg = fsolve(residuals, guess, full_output=True)
    if ier != 1 or not (0 < sol[0] < sol[1] < 1):
        raise ValueError(f"No physical 3-layer solution for these densities: {msg} {sol}")
    return sol[0] * R, sol[1] * R


def fit_with_h2o_profile(R, M, moi, rho_core, rho_mantle, z_w, rho_w, guess):
    """Like fit_three_layer, but the H2O layer has a depth-dependent density rho_w(z),
    given at depths z_w below the surface. Unknowns: core radius and mantle top radius.
    `guess` is (r_core/R, r_mantle/R)."""
    def residuals(x):
        rc, rm = x[0] * R, x[1] * R
        r_w = np.linspace(rm, R, 2000)
        rho_r = np.interp(R - r_w, z_w, rho_w)
        m_w = np.trapezoid(4 * np.pi * r_w**2 * rho_r, r_w)
        I_w = np.trapezoid(8 * np.pi / 3 * r_w**4 * rho_r, r_w)
        m = 4 * np.pi / 3 * (rho_core * rc**3 + rho_mantle * (rm**3 - rc**3)) + m_w
        I = 8 * np.pi / 15 * (rho_core * rc**5 + rho_mantle * (rm**5 - rc**5)) + I_w
        return [m / M - 1.0, I / (M * R**2) / moi - 1.0]

    sol, info, ier, msg = fsolve(residuals, guess, full_output=True)
    if ier != 1 or not (0 < sol[0] < sol[1] < 1):
        raise ValueError(f"No physical solution with this H2O profile: {msg} {sol}")
    return sol[0] * R, sol[1] * R
