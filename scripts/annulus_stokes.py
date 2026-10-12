"""Stage 7b: solve the annulus-sector Stokes benchmark (exact manufactured solution) at any resolution.

    python scripts/annulus_stokes.py --nr 32
    mpiexec -n 4 python scripts/annulus_stokes.py --nr 64 --vtk paraview/annulus_test.vts

Geometry: Ganymede's ice shell, --shell_km thick (default 150) on a 2631.2 km radius, over --span_deg of
arc (default 180: a half annulus). By default the angular resolution gives square cells
(n_theta ~ n_r * arc length / thickness), rounded so the multigrid can coarsen. Prints the error against
the exact solution, solver iterations and time; --vtk writes the curved shell for ParaView.
"""
import argparse
import os

import numpy as np
from petsc4py import PETSc

from ganymede.annulus import AnnulusGrid
from ganymede.annulus_mms import errors, manufactured
from ganymede.annulus_petsc import AnnulusLayout, solve_annulus_petsc

R_G = 2631.2


def main(a):
    rank = PETSc.COMM_WORLD.getRank()
    span = np.radians(a.span_deg)
    r_in = 1 - a.shell_km / R_G
    nt = a.ntheta
    if nt is None:                                   # square cells, divisible by 2^(levels-1)
        q = 2 ** max(int(np.log2(a.nr)) - 2, 0)
        nt = max(q, int(round(a.nr * span * (1 - 0.5 * a.shell_km / R_G) / (a.shell_km / R_G) / q)) * q)
    make = AnnulusGrid.uniform if a.refine <= 1 else (lambda *x, **k: AnnulusGrid.refined(*x, ratio=a.refine, **k))
    g = make(r_in, 1.0, a.nr, nt, span=span)
    sol = manufactured(r_in, 1.0, k=a.k, span=span)
    Rc, Tc = g.p_points(); Rn, Tn = g.node_points(); R, T = g.vr_points(); R2, T2 = g.vt_points()
    eta_c = sol["eta"](Rc, Tc)
    layout = AnnulusLayout(g)
    (vr, vt, P), x, st = solve_annulus_petsc(g, eta_c, sol["eta"](Rn, Tn), sol["fr"](R, T), sol["ft"](R2, T2),
                                             layout=layout)
    e = errors(g, sol, vr, vt, P)
    if rank == 0:
        cell_r = a.shell_km / a.nr
        cell_t = R_G * span / nt
        print(f"grid {a.nr} x {nt} cells ({g.nr * g.nt / 1e6:.2f} M), cells {cell_r:.2f} km (radial) x "
              f"{cell_t:.2f} km (angular), {st['nproc']} process(es)")
        print(f"solver: {st['outer']} outer / {st['inner']} inner iterations, {st['time']:.2f} s, "
              f"{st['levels']} multigrid levels, converged reason {st['reason']}")
        print(f"error vs exact solution: v_r {e[0]:.2e}, v_theta {e[1]:.2e}, P {e[2]:.2e}")
        if a.vtk:
            from ganymede.annulus_vtk import stokes_fields, write_vts
            os.makedirs(os.path.dirname(a.vtk) or ".", exist_ok=True)
            write_vts(a.vtk, g, stokes_fields(g, vr, vt, P, eta_c), scale=R_G)
            print(f"wrote {a.vtk} (open in ParaView; coordinates in km)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nr", type=int, default=32, help="radial cells")
    ap.add_argument("--ntheta", type=int, default=None, help="angular cells (default: square cells)")
    ap.add_argument("--shell_km", type=float, default=150.0)
    ap.add_argument("--span_deg", type=float, default=180.0)
    ap.add_argument("--refine", type=float, default=1.0, help="radial refinement ratio towards the walls (1 = uniform)")
    ap.add_argument("--k", type=int, default=4, help="angular wavenumber of the test flow")
    ap.add_argument("--vtk", default=None)
    main(ap.parse_args())
