"""Equilibrium thickness of a floating ice Ih shell (Stage 2d).

For a shell of thickness D the base sits at the ice Ih melting point at the
base pressure P_b = rho g D (SeaFreeze), the surface at T_s. We compute the heat
leaving the top, q_out(D), from steady 2D convection (or conduction below the
onset of convection). A shell is in thermal equilibrium where q_out(D) equals
the heat supplied from below, q_in; it is stable where q_out decreases with D.

The ice Ih--liquid boundary ends at the Ih--III--liquid triple point
(~207.6 MPa, ~251 K), so ice Ih cannot float deeper than D_max = P_triple/(rho g).
"""
import time

import numpy as np
from scipy.optimize import brentq

from .ice import (A_K_ICE, arrhenius, conductive_flux, conductivity_ice, eta_diffusion_creep,
                  onset_criterion, steady_convection_general, top_flux)

T_SURFACE = 110.0     # K
RHO_ICE = 930.0       # kg m^-3, held fixed through the shell
G_SHELL = 1.43        # m s^-2, surface gravity, held fixed
ALPHA_ICE = 1.6e-4    # 1/K
P_TRIPLE_IH_III = 207.6e6  # Pa, Ih--III--liquid triple point


def heat_capacity_ice(T):
    """Ice Ih specific heat [J/kg/K], linear fit cp = 7.037 T + 185."""
    return 7.037 * T + 185.0


def T_melt_Ih(P):
    """Melting temperature of ice Ih [K] at pressure P [Pa], from the SeaFreeze phase map."""
    import logging

    from seafreeze import seafreeze as sf
    logging.getLogger("lbftd").disabled = True
    if not 0 <= P <= P_TRIPLE_IH_III:
        raise ValueError(f"P = {P/1e6:.1f} MPa is outside the ice Ih melting curve")

    def is_Ih(T):
        pt = np.empty(1, dtype=object)
        pt[0] = (P / 1e6, T)
        return int(sf.whichphase(pt)[0]) == 1

    return brentq(lambda T: 0.5 - is_Ih(T), 240.0, 274.0, xtol=1e-4)


def max_thickness(rho=RHO_ICE, g=G_SHELL):
    """Deepest possible ice Ih base: the Ih--III--liquid triple point."""
    return P_TRIPLE_IH_III / (rho * g)


def rayleigh_basal(D, T_b, d, T_s=T_SURFACE, rho=RHO_ICE, g=G_SHELL, alpha=ALPHA_ICE):
    """Rayleigh number with the viscosity and diffusivity of the warm base."""
    kappa = (A_K_ICE / T_b) / (rho * heat_capacity_ice(T_b))
    return rho * alpha * g * (T_b - T_s) * D**3 / (kappa * eta_diffusion_creep(T_b, d))


def shell_heat_flow(d, D, n=64, aspect=1.0, T_s=T_SURFACE, T_b=None, T0=None,
                    relax=0.1, tol=1e-6, max_iter=4000):
    """Heat flux q_out [W/m^2] leaving the top of an ice Ih shell of thickness D [m],
    grain size d [m]. Returns a dict (D, d, T_b, Ra, Ra_cr, Nu, q_out, status, ...).

    Shells far below the onset of convection (Ra < Ra_cr / 2) are taken as conductive
    without running the solver. Pass `T0` to start from a given temperature field
    (e.g. a convective state, to follow the convective branch where both are stable).
    """
    if T_b is None:
        T_b = T_melt_Ih(RHO_ICE * G_SHELL * D)
    Ra = rayleigh_basal(D, T_b, d, T_s=T_s)
    Ra_cr, _ = onset_criterion(T_s, T_b)
    q_cond = A_K_ICE * np.log(T_b / T_s) / D    # conductive flux for k = 651/T
    out = dict(d=d, D=D, T_b=T_b, Ra=Ra, Ra_cr=Ra_cr, n=n, aspect=aspect)
    t0 = time.time()
    if Ra < 0.5 * Ra_cr and T0 is None:
        out.update(Nu=1.0, q_out=q_cond, status="conductive (Ra < Ra_cr/2)", T=None)
    else:
        k = conductivity_ice(T_s, T_b)
        g, T, vx, vz, it, _ = steady_convection_general(
            int(round(n * aspect)), n, Ra, arrhenius(T_s, T_b), k, aspect=aspect,
            relax=relax, tol=tol, max_iter=max_iter, T0=T0)
        Nu = top_flux(T, g, float(k(0.0))) / conductive_flux(T_s, T_b)
        out.update(Nu=Nu, q_out=Nu * q_cond, status=f"steady ({it} it)", T=T)
    out["time"] = time.time() - t0
    return out


def equilibrium_thickness(D, q_out, q_in):
    """Thinnest D at which q_out(D) falls to q_in (a stable equilibrium), interpolating
    linearly in log q between samples. Returns None if q_out never reaches q_in in range
    (the shell would thicken beyond the sampled D), and D[0] if q_out(D[0]) <= q_in."""
    D = np.asarray(D, float)
    lq = np.log(np.asarray(q_out, float))
    l0 = np.log(q_in)
    if lq[0] <= l0:
        return float(D[0])
    for i in range(len(D) - 1):
        if lq[i] > l0 >= lq[i + 1]:
            return float(D[i] + (l0 - lq[i]) / (lq[i + 1] - lq[i]) * (D[i + 1] - D[i]))
    return None
