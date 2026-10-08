"""3D Stokes benchmark on DMStag, for running under MPI:

    mpiexec -n 1 python scripts/parallel_stokes3d.py --n 32
    mpiexec -n 8 python scripts/parallel_stokes3d.py --n 64

Isoviscous case: compared with the analytic solution for rho = cos(pi x) cos(pi y) sin(pi z) in a free-slip unit cube.
--contrast C: viscosity eta = C^z (stiff top, like a stagnant lid).
"""
import argparse
import numpy as np
from petsc4py import PETSc
from ganymede.stag_stokes3d import stokes3d_solve, gather3d, analytic3d, edge_viscosity

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=32)
ap.add_argument("--contrast", type=float, default=1.0)
ap.add_argument("--save", default=None)
args = ap.parse_args()

n = (args.n,) * 3
rho, vxe, vye, vze = analytic3d(n)
eta_c, eta_edge = edge_viscosity(lambda x, y, z: args.contrast ** z, n)
dm, x, ix, st, _ = stokes3d_solve(n, eta_c, eta_edge, rho)
fields = gather3d(x, ix, n)
if PETSc.COMM_WORLD.getRank() == 0:
    vx, vy, vz, P = fields
    msg = (f"{args.n}³ ({st['unknowns']:,} unknowns) on {st['nproc']} process(es): outer {st['outer']}, inner {st['inner']}, "
           f"converged {st['reason'] > 0}, solve {st['time']:.2f} s")
    if args.contrast == 1.0:
        err = max(np.abs(vx - vxe).max(), np.abs(vy - vye).max(), np.abs(vz - vze).max()) / np.abs(vze).max()
        msg += f", error vs analytic {err:.2e}"
    print(msg)
    if args.save:
        np.save(args.save, np.concatenate([a.ravel() for a in fields]))
