"""Stage 4d prototype: two-phase convection in the HP ice layer (Kalousova et al. 2018 setup).

    python scripts/hp_convection.py run  --nz 64 --t_end 25 --out results/hp_convection_ref.pkl
    python scripts/hp_convection.py plot results/hp_convection_ref.pkl --prefix figures/hp_convection

`run` integrates in time (64 x 128 cells: ~0.4 s per step, ~1.3 kyr per step) and pickles the
history and snapshots; `plot` makes the snapshot panels (T - T_m and porosity) and the
heat-budget time series.
"""
import argparse
import os
import pickle

import numpy as np

from ganymede.twophase import HPIce

INK, MUTED, GRID, BG = "#1f1f1e", "#6b6a63", "#e4e3dc", "#fcfcfb"
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORANGE = ["#fcfcfb", "#fde6da", "#f8c2a6", "#f29a72", "#eb6834", "#b8461b", "#6e2408"]


def cmd_run(a):
    from ganymede.twophase2d import run
    p = HPIce(H=a.H * 1e3, q_s=a.qs * 1e-3, phi_c=a.phi_c / 100)
    snaps = tuple(float(s) for s in a.snaps)
    prev, kw = None, {}
    if a.restart:                                     # continue a previous run from its final state
        prev = pickle.load(open(a.restart, "rb"))
        last = prev["snaps"][-1]
        kw = dict(theta0=last["theta"], phi0=last["phi"], t0_myr=last["t_myr"])
        a.nz = last["theta"].shape[0]
    r = run(p, nx=2 * a.nz, nz=a.nz, mu0=a.mu0, t_end_myr=a.t_end, snap_myr=snaps, log_every=100,
            wall_limit=a.wall, solver=a.solver, **kw)
    r.pop("g")
    r.pop("sc")
    if prev is not None:
        r["hist"] = prev["hist"] + r["hist"]
        r["snaps"] = prev["snaps"][:-1] + r["snaps"]   # drop the duplicated restart state
        r["params"] = prev.get("params", {})
    r.setdefault("params", {})
    r["params"].update(H=a.H, qs=a.qs, mu0=a.mu0, phi_c=a.phi_c, nz=a.nz)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    pickle.dump(r, open(a.out, "wb"))


def _style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def cmd_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    r = pickle.load(open(a.results, "rb"))
    par = r.get("params", dict(H=200, qs=20, mu0=1e15, phi_c=1))
    H = par["H"]
    cm_T = LinearSegmentedColormap.from_list("cold", BLUE[::-1][:-1] + ["#f0efec"])    # 0 K (melting) = pale
    cm_phi = LinearSegmentedColormap.from_list("wet", ORANGE)
    snaps = r["snaps"]
    if a.times:
        snaps = [min(snaps, key=lambda s: abs(s["t_myr"] - t)) for t in a.times]
    else:
        idx = np.unique(np.linspace(0, len(snaps) - 1, min(4, len(snaps))).round().astype(int))
        snaps = [snaps[i] for i in idx]
    uniq = []
    for s in snaps:                                   # the final state can repeat the last snapshot
        if not uniq or abs(s["t_myr"] - uniq[-1]["t_myr"]) > 1e-6:
            uniq.append(s)
    snaps = uniq

    # --- snapshots -------------------------------------------------------------------
    n = len(snaps)
    fig, axes = plt.subplots(n, 2, figsize=(10, 1.95 * n + 0.9), dpi=150, squeeze=False)
    fig.patch.set_facecolor(BG)
    extent = (0, 2 * H, 0, H)
    dT_min = -max(5.0, np.ceil(max(-s["dT"].min() for s in snaps)))
    for row, s in enumerate(snaps):
        a1, a2 = axes[row]
        im1 = a1.imshow(s["dT"], origin="lower", extent=extent, cmap=cm_T, vmin=dT_min, vmax=0, aspect="equal")
        im2 = a2.imshow(s["phi"] * 100, origin="lower", extent=extent, cmap=cm_phi, vmin=0.5, vmax=a.phimax,
                        aspect="equal")
        for ax in (a1, a2):
            _style(ax)
            ax.set_yticks([0, H / 2, H])
            if row < n - 1:
                ax.set_xticklabels([])
        a1.set_ylabel(f"{s['t_myr']:.1f} Myr\nheight (km)", color=INK, fontsize=8.5)
        a2.set_yticklabels([])
    axes[-1, 0].set_xlabel("distance (km)", color=INK, fontsize=8.5)
    axes[-1, 1].set_xlabel("distance (km)", color=INK, fontsize=8.5)
    axes[0, 0].set_title("T − T_m (K): how far below melting", color=INK, fontsize=9.5, loc="left")
    axes[0, 1].set_title("porosity (% water)", color=INK, fontsize=9.5, loc="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    for col, im in ((0, im1), (1, im2)):
        box = axes[-1, col].get_position()
        cax = fig.add_axes([box.x0 + 0.1 * box.width, 0.035, 0.8 * box.width, 0.012])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=MUTED, labelsize=8)
    fig.suptitle(f"Two-phase convection in HP ice (prototype): H = {H:g} km, q_s = {par['qs']:g} mW m⁻², "
                 f"μ₀ = {par['mu0']:.0e} Pa s, φ_c = {par['phi_c']:g} %", color=INK, fontsize=10, x=0.01, ha="left")
    fig.savefig(a.prefix + "_snapshots.png", facecolor=BG)
    plt.close(fig)

    # --- heat budget -------------------------------------------------------------------
    h = r["hist"]
    t = np.array([x["t_myr"] for x in h])
    qc = np.array([x["q_cond"] for x in h]) * 1e3
    qw = np.array([x["q_water"] for x in h]) * 1e3
    win = max(1, int(np.searchsorted(t, a.smooth_kyr / 1e3)))   # running mean: water leaves in bursts
    smooth = lambda y: np.convolve(y, np.ones(win), mode="same") / np.convolve(np.ones_like(y), np.ones(win), mode="same")
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    fig.patch.set_facecolor(BG)
    _style(ax)
    ax.grid(True, color=GRID, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.axhline(par["qs"], color=MUTED, lw=0.8, ls=(0, (4, 3)), label="heat in from the silicates")
    ax.plot(t, smooth(qw), color="#eb6834", lw=2, label="carried out by meltwater")
    ax.plot(t, qc, color="#2a78d6", lw=2, label="conducted into the ocean")
    ax.plot(t, smooth(qw) + qc, color=INK, lw=1, label="total out")
    ax.plot(t, par["qs"] - smooth(qw) - qc, color="#1baf7a", lw=1.5, label="stored: warming the layer")
    ax.set_xlim(0, t[-1])
    stored = par["qs"] - smooth(qw) - qc
    ax.set_ylim(min(0, stored.min() * 1.1), max(par["qs"], (smooth(qw) + qc).max()) * 1.08)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("Time (Myr)", color=INK)
    ax.set_ylabel("Heat flux (mW m⁻²)", color=INK)
    ax.set_title("Where the basal heat goes", color=INK, fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    ax.set_title(f"Where the basal heat goes ({a.smooth_kyr:g} kyr running mean)", color=INK, fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(a.prefix + "_budget.png", facecolor=BG)

    late = t > t[-1] / 2
    print(f"t_end = {t[-1]:.2f} Myr; second half: conduction {qc[late].mean():.2f}, water {qw[late].mean():.2f}, "
          f"total {qc[late].mean() + qw[late].mean():.2f} mW/m2 (in: {par['qs']}); "
          f"mean porosity {np.mean([x['phi_mean'] for x, l in zip(h, late) if l])*100:.3f} %, "
          f"temperate fraction {np.mean([x['temperate'] for x, l in zip(h, late) if l]):.3f}, "
          f"vrms {np.mean([x['vrms_cm_yr'] for x, l in zip(h, late) if l]):.1f} cm/yr, "
          f"max |energy error| {max(abs(x['energy_error']) for x in h):.1e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--H", type=float, default=200.0, help="layer thickness [km]")
    r.add_argument("--qs", type=float, default=20.0, help="basal heat flux [mW/m2]")
    r.add_argument("--mu0", type=float, default=1e15, help="viscosity at the melting point [Pa s]")
    r.add_argument("--phi_c", type=float, default=1.0, help="percolation threshold [%%]")
    r.add_argument("--nz", type=int, default=64)
    r.add_argument("--t_end", type=float, default=25.0, help="[Myr]")
    r.add_argument("--snaps", nargs="*", default=[1, 3, 6, 10, 15, 20, 25])
    r.add_argument("--solver", choices=["direct", "stag"], default="stag",
                   help="Stokes solver: SciPy direct or PETSc DMStag multigrid (needs petsc4py)")
    r.add_argument("--wall", type=float, default=None, help="stop after this many seconds")
    r.add_argument("--restart", default=None, help="results file to continue from")
    r.add_argument("--out", default="results/hp_convection_ref.pkl")
    r.set_defaults(f=cmd_run)
    pl = sub.add_parser("plot")
    pl.add_argument("results")
    pl.add_argument("--prefix", default="figures/hp_convection")
    pl.add_argument("--times", type=float, nargs="*", default=None)
    pl.add_argument("--smooth_kyr", type=float, default=100.0, help="running-mean window for the budget [kyr]")
    pl.add_argument("--phimax", type=float, default=2.0, help="porosity colour-scale max [%%]")
    pl.set_defaults(f=cmd_plot)
    args = ap.parse_args()
    args.f(args)
