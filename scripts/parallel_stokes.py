"""Stokes benchmark on DMStag, for running under MPI:

    mpiexec -n 1 python scripts/parallel_stokes.py
    mpiexec -n 4 python scripts/parallel_stokes.py

Solves the sinusoidal-density problem (analytic solution) and a stiff-lid case, prints iteration counts and
timings, and saves the gathered solution so runs on different process counts can be compared.
"""
import argparse
import numpy as np
from petsc4py import PETSc
from ganymede.stag_stokes import native_stokes_solve, gather_fields

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--contrast", type=float, default=1.0, help="viscosity contrast top/bottom (eta = contrast^z)")
ap.add_argument("--save", default=None, help="save gathered solution to this .npy file")
args = ap.parse_args()

n, k = args.n, np.pi
xc = (np.arange(n) + 0.5) / n; xn = np.arange(n + 1) / n
Xz, Zz = np.meshgrid(xc, xn)
eta_c = args.contrast ** np.meshgrid(xc, xc)[1]
eta_n = args.contrast ** np.meshgrid(xn, xn)[1]
dm, x, ix, st = native_stokes_solve(n, n, eta_c, eta_n, np.cos(k * Xz) * np.sin(k * Zz))
fields = gather_fields(x, ix, n, n)

if PETSc.COMM_WORLD.getRank() == 0:
    vx, vz, P = fields
    msg = f"{n}² on {st['nproc']} process(es): outer {st['outer']}, inner {st['inner']}, converged {st['reason'] > 0}, solve {st['time']:.2f} s"
    if args.contrast == 1.0:
        A0 = k / (2 * k**2) ** 2
        Xx, Zx = np.meshgrid(xn, xc)
        err = max(np.abs(vx - A0 * k * np.sin(k * Xx) * np.cos(k * Zx)).max(),
                  np.abs(vz + A0 * k * np.cos(k * Xz) * np.sin(k * Zz)).max()) / (A0 * k)
        msg += f", error vs analytic {err:.2e}"
    print(msg)
    if args.save:
        np.save(args.save, np.concatenate([vx.ravel(), vz.ravel(), P.ravel()]))
