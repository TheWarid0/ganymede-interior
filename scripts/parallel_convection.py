"""Blankenbach convection benchmark on DMStag (Stokes + energy with PETSc), for running under MPI:

    mpiexec -n 1 python scripts/parallel_convection.py --case 2a --n 128
    mpiexec -n 4 python scripts/parallel_convection.py --case 2a --n 128

Case 1a: Ra = 1e4, constant viscosity (Nu = 4.884409, vrms = 42.864947).
Case 2a: Ra = 1e4, eta = exp(-ln(1000) T) (Nu = 10.0660, vrms = 480.4334).
"""
import argparse
import numpy as np
from petsc4py import PETSc
from ganymede.stag_convection import steady_convection_stag

REF = {"1a": (0.0, 4.884409, 42.864947), "2a": (np.log(1000), 10.0660, 480.4334)}
ap = argparse.ArgumentParser()
ap.add_argument("--case", default="1a", choices=REF)
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--save", default=None)
args = ap.parse_args()
b, nu_ref, vr_ref = REF[args.case]
g, T, vx, vz, st = steady_convection_stag(args.n, args.n, 1e4, lambda T: np.exp(-b * np.asarray(T)), tol=1e-7, relax=0.5)
if PETSc.COMM_WORLD.getRank() == 0:
    print(f"case {args.case}, {args.n}² on {st['nproc']} process(es): Nu = {st['Nu']:.4f} (ref {nu_ref}), "
          f"vrms = {st['vrms']:.3f} (ref {vr_ref}), {st['iterations']} Picard iterations, {st['time']:.1f} s")
    if args.save:
        np.save(args.save, T)
