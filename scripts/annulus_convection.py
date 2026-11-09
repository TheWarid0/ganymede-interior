"""Stage 7c: thermal convection in a half annulus (mirror symmetric, so it represents a full annulus).

Hernlund & Tackley (2008) spherical-annulus benchmark, basal heating, isoviscous, Ra = 1e4:
    python scripts/annulus_convection.py --nr 64 --plumes 5
    python scripts/annulus_convection.py --nr 64 --plumes 5 --geometry cylindrical
    python scripts/annulus_convection.py --nr 64 --plumes 5 --vtk paraview/ht_ra1e4.vts

Geometry: inner radius 1.22, outer 2.22 (shell thickness 1, Earth-like radius ratio 0.55).
--plumes m seeds a perturbation cos(m theta): the steady state then has m plumes in the full annulus
(the half annulus with free-slip, insulating side walls holds the mirror-symmetric half of it).
Prints Nu (top and bottom), v_rms and mean temperature.
"""
import argparse
import json
import os
import time

import numpy as np

from ganymede.annulus import AnnulusGrid
from ganymede.annulus_convection import Convection, conductive_profile


def main(a):
    d = 2 if a.geometry == "spherical" else 1
    nt = a.ntheta or 8 * a.nr
    g = AnnulusGrid.uniform(a.r_in, a.r_in + 1.0, a.nr, nt, span=np.pi, d=d)
    c = Convection(g, Ra=a.Ra, scheme=a.scheme)
    T0 = conductive_profile(g) + 0.05 * np.cos(a.plumes * g.t_c)[None, :] * np.sin(np.pi * (g.r_c - a.r_in))[:, None]
    t0 = time.time()
    T, (vr, vt, P), hist = c.run(T0, t_end=a.t_end, cfl=a.cfl, tol=a.tol, log=print if a.verbose else None)
    h = hist[-1]
    print(f"{a.geometry} annulus, Ra = {a.Ra:g}, grid {a.nr} x {nt} (half annulus), seeded with {a.plumes} plumes")
    print(f"Nu_top = {h['Nu_top']:.4f}, Nu_bottom = {h['Nu_bot']:.4f}, v_rms = {h['vrms']:.3f}, "
          f"mean T = {h['Tmean']:.4f}  ({time.time() - t0:.0f} s)")
    if a.Ra == 1e4:
        ref = "spherical annulus: Nu = 4.18, v_rms = 37.7" if a.geometry == "spherical" else \
              "scaled cylinder (use --r_in 0.4327): Nu = 3.99, v_rms = 35.6"
        print(f"Hernlund & Tackley (2008), {ref}")
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w") as f:
            json.dump(dict(vars(a), ntheta=nt, **{k: float(v) for k, v in h.items()}), f, indent=1)
    if a.vtk:
        from ganymede.annulus_vtk import stokes_fields, write_vts
        os.makedirs(os.path.dirname(a.vtk) or ".", exist_ok=True)
        fields = stokes_fields(g, vr, vt, P, c.eta_c)
        fields["temperature"] = T
        write_vts(a.vtk, g, fields)
        print(f"wrote {a.vtk}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", choices=["spherical", "cylindrical"], default="spherical")
    ap.add_argument("--Ra", type=float, default=1e4)
    ap.add_argument("--nr", type=int, default=32)
    ap.add_argument("--ntheta", type=int, default=None, help="cells over the half annulus (default 8 nr)")
    ap.add_argument("--r_in", type=float, default=1.22)
    ap.add_argument("--plumes", type=int, default=5)
    ap.add_argument("--scheme", choices=["central", "upwind"], default="central")
    ap.add_argument("--t_end", type=float, default=3.0)
    ap.add_argument("--cfl", type=float, default=8.0)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--vtk", default=None)
    main(ap.parse_args())
