"""Figure for Stage 7c: steady Ra = 1e4 convection in the spherical annulus (Hernlund & Tackley 2008 case),
half annulus computed, mirrored to the full annulus for display.

    python scripts/plot_annulus_convection.py --nr 48 --plumes 5 --out figures/annulus_ht_ra1e4.png
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from ganymede.annulus import AnnulusGrid
from ganymede.annulus_convection import Convection, conductive_profile


def main(a):
    g = AnnulusGrid.uniform(1.22, 2.22, a.nr, 8 * a.nr, span=np.pi, d=2)
    c = Convection(g, Ra=a.Ra)
    T0 = conductive_profile(g) + 0.05 * np.cos(a.plumes * g.t_c)[None, :] * np.sin(np.pi * (g.r_c - 1.22))[:, None]
    T, _, hist = c.run(T0, t_end=3.0, cfl=8.0, tol=1e-5)
    h = hist[-1]
    # mirror theta -> -theta to show the full annulus
    th = np.concatenate([-g.t_f[::-1], g.t_f[1:]])
    Tf = np.hstack([T[:, ::-1], T])
    R, TH = np.meshgrid(g.r_f, th, indexing="ij")
    X, Y = R * np.sin(TH), R * np.cos(TH)
    fig, ax = plt.subplots(figsize=(6, 6))
    pc = ax.pcolormesh(X, Y, Tf, cmap="RdBu_r", vmin=0, vmax=1, shading="flat", rasterized=True)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"Spherical annulus, Ra = {a.Ra:g}, {a.plumes} plumes\n"
                 f"Nu = {h['Nu_top']:.3f}, v_rms = {h['vrms']:.2f}  (H&T 2008: 4.18, 37.7)", fontsize=10)
    fig.colorbar(pc, ax=ax, shrink=0.6, label="temperature")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=150, bbox_inches="tight")
    print(f"wrote {a.out}: Nu {h['Nu_top']:.4f} vrms {h['vrms']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nr", type=int, default=48)
    ap.add_argument("--Ra", type=float, default=1e4)
    ap.add_argument("--plumes", type=int, default=5)
    ap.add_argument("--out", default="figures/annulus_ht_ra1e4.png")
    main(ap.parse_args())
