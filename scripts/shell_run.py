"""Stage 5: two-phase ice Ih shell over the ocean.

    python scripts/shell_run.py run --D 80 --grain 0.3 --edot 4e-11 --nz 64 --t_end 30 --out results/shell.pkl
    python scripts/shell_run.py run --restart results/shell.pkl --refine 2 --t_end 40 --snap_every 0.25 --out results/shell_hr.pkl
    python scripts/shell_run.py plot results/shell_hr.pkl --prefix figures/shell
    python scripts/to_paraview.py results/shell_hr.pkl
    mpiexec -n 4 python scripts/shell_run.py run --restart results/shell_restart.pkl --t_end 26 --out results/shell_long.pkl

Under mpiexec every rank runs the time loop and the Stokes solve (the expensive part) is split between
ranks; results agree with a serial run to the solver tolerance (~1e-7). `slim` shrinks a results file to
what a restart needs.

--edot is the tidal strain-rate amplitude [1/s]: ~2e-12 for present-day Ganymede (negligible
heating), ~4e-11 for a past high-eccentricity period (e ~ 0.02). --refine 2 restarts on a grid
twice as fine (fields interpolated), so a run can be spun up cheaply and finished at high resolution.
"""
import argparse
import os
import pickle

import numpy as np

from ganymede.shell2p import ShellParams

INK, MUTED, GRID, BG = "#1f1f1e", "#6b6a63", "#e4e3dc", "#fcfcfb"
ORANGE = ["#fcfcfb", "#fde6da", "#f8c2a6", "#f29a72", "#eb6834", "#b8461b", "#6e2408"]


def _refine(f, k):
    """Bilinear-ish refinement of a cell-centred field by an integer factor (conservative mean kept)."""
    from scipy.ndimage import zoom
    return zoom(f, k, order=1, mode="nearest")


def cmd_run(a):
    from ganymede.shell2p import run
    kw, prev = {}, None
    if a.restart:
        prev = pickle.load(open(a.restart, "rb"))
        par = prev["params"]
        for key in ("D", "grain", "edot", "phi_c", "K0", "q_ocean", "aspect"):
            if getattr(a, key) is None:
                setattr(a, key, par[key])
        last = prev["snaps"][-1]
        th, ph = last["theta"], last["phi"]
        if a.refine > 1:
            th, ph = _refine(th, a.refine), np.clip(_refine(ph, a.refine), 0, None)
        kw = dict(theta0=th, phi0=ph, t0_myr=last["t_myr"])
        a.nz = th.shape[0]
    defaults = dict(D=80.0, grain=0.3, edot=0.0, phi_c=1.0, K0=1e-9, q_ocean=0.0, aspect=2.0)
    for key, v in defaults.items():
        if getattr(a, key) is None:
            setattr(a, key, v)
    p = ShellParams(D=a.D * 1e3, d=a.grain * 1e-3, edot0=a.edot, phi_c=a.phi_c / 100, K0=a.K0,
                    q_ocean=a.q_ocean * 1e-3)
    snaps = set(float(s) for s in a.snaps)
    t_start = kw.get("t0_myr", 0.0)
    if a.snap_every:
        snaps |= set(np.round(np.arange(t_start + a.snap_every, a.t_end + 1e-9, a.snap_every), 6))
    r = run(p, nx=int(round(a.aspect * a.nz)), nz=a.nz, aspect=a.aspect, t_end_myr=a.t_end,
            snap_myr=tuple(sorted(snaps)), solver=a.solver, wall_limit=a.wall, **kw)
    r.pop("g")
    r.pop("sc")
    r["params"] = dict(D=a.D, grain=a.grain, edot=a.edot, phi_c=a.phi_c, K0=a.K0, q_ocean=a.q_ocean,
                       aspect=a.aspect, nz=a.nz)
    r["events"] = list(prev.get("events", [])) if prev is not None else []
    if prev is not None and a.refine > 1:
        r["events"].append((kw["t0_myr"], f"grid refined to {a.nz} rows"))
    if prev is not None:
        r["hist"] = prev["hist"] + r["hist"]
        r["snaps"] = [s for s in prev["snaps"][:-1] if a.refine == 1] + r["snaps"]
    from ganymede.shell2p import mpi_rank
    if mpi_rank() == 0:                                # under mpiexec, only rank 0 writes
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        pickle.dump(r, open(a.out, "wb"))


def cmd_slim(a):
    """Keep only what a restart needs (parameters, history, final state): small enough to copy around."""
    r = pickle.load(open(a.results, "rb"))
    last = r["snaps"][-1]
    r["snaps"] = [dict(t_myr=last["t_myr"], theta=last["theta"], phi=last["phi"], vx=last["vx"], vz=last["vz"],
                       T=last["T"], dT=last["dT"], eta=last["eta"], H_tidal=last["H_tidal"])]
    r.pop("theta", None)
    r.pop("phi", None)
    pickle.dump(r, open(a.out, "wb"))
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB), restart time {last['t_myr']:.3f} Myr")


def _style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def cmd_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm

    r = pickle.load(open(a.results, "rb"))
    par, info = r["params"], r["info"]
    D = par["D"]
    snaps = []
    for s in r["snaps"]:
        if not snaps or abs(s["t_myr"] - snaps[-1]["t_myr"]) > 1e-6:
            snaps.append(s)
    if a.times:
        snaps = [min(snaps, key=lambda s: abs(s["t_myr"] - t)) for t in a.times]
    else:
        idx = np.unique(np.linspace(0, len(snaps) - 1, min(3, len(snaps))).round().astype(int))
        snaps = [snaps[i] for i in idx]
    cm_T = LinearSegmentedColormap.from_list("T", ["#0d366b", "#256abf", "#6da7ec", "#cde2fb", "#fde6da",
                                                   "#f29a72", "#b8461b"])
    cm_phi = LinearSegmentedColormap.from_list("wet", ORANGE)
    n = len(snaps)
    fig, axes = plt.subplots(n, 3, figsize=(13, 2.0 * n + 1.0), dpi=150, squeeze=False)
    fig.patch.set_facecolor(BG)
    nz, nx = snaps[0]["theta"].shape
    extent = (0, par["aspect"] * D, 0, D)
    phimax = max(a.phimax, 0.1)
    for row, s in enumerate(snaps):
        ims = []
        ims.append(axes[row, 0].imshow(s["T"], origin="lower", extent=extent, cmap=cm_T,
                                       vmin=110, vmax=info["T_b"], aspect="equal"))
        ims.append(axes[row, 1].imshow(s["phi"] * 100, origin="lower", extent=extent, cmap=cm_phi,
                                       vmin=0, vmax=phimax, aspect="equal"))
        cm_eta = LinearSegmentedColormap.from_list("eta", ["#f0efec", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"])
        ims.append(axes[row, 2].imshow(s["eta"], origin="lower", extent=extent, cmap=cm_eta, aspect="equal",
                                       norm=LogNorm(vmin=s["eta"].min(), vmax=s["eta"].min() * 1e4)))
        for ax in axes[row]:
            _style(ax)
            if row < n - 1:
                ax.set_xticklabels([])
        axes[row, 0].set_ylabel(f"{s['t_myr']:.2f} Myr\nheight above ocean (km)", color=INK, fontsize=8.5)
        for ax in axes[row, 1:]:
            ax.set_yticklabels([])
    for ax in axes[-1]:
        ax.set_xlabel("distance (km)", color=INK, fontsize=8.5)
    titles = ["temperature (K)", "water content (%)", "ice viscosity (Pa s)"]
    for ax, ttl in zip(axes[0], titles):
        ax.set_title(ttl, color=INK, fontsize=9.5, loc="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    for col, im in enumerate(ims):
        box = axes[-1, col].get_position()
        cax = fig.add_axes([box.x0 + 0.1 * box.width, 0.035, 0.8 * box.width, 0.012])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=MUTED, labelsize=7.5)
    fig.suptitle(f"Ice Ih shell over the ocean: D = {D:g} km, grain {par['grain']:g} mm, "
                 f"tidal strain rate {par['edot']:.1e} s⁻¹, {nx} × {nz} cells ({D/nz:.2g} km)",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.savefig(a.prefix + "_snapshots.png", facecolor=BG)
    plt.close(fig)

    h = r["hist"]
    t = np.array([x["t_myr"] for x in h])
    get = lambda k: np.array([x[k] for x in h]) * 1e3
    win = max(1, int(np.searchsorted(t, t[0] + a.smooth_kyr / 1e3)))
    smooth = lambda y: np.convolve(y, np.ones(win), mode="same") / np.convolve(np.ones_like(y), np.ones(win), mode="same")
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    fig.patch.set_facecolor(BG)
    _style(ax)
    ax.grid(True, color=GRID, lw=0.6)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.plot(t, smooth(get("q_surface")), color="#2a78d6", lw=2, label="out of the surface")
    ax.plot(t, smooth(get("q_base")), color="#1baf7a", lw=2, label="from the ocean into the shell")
    ax.plot(t, smooth(get("q_tidal")), color=INK, lw=1.2, label="tidal heating in the shell")
    ax.plot(t, smooth(get("q_drain")), color="#eb6834", lw=2, label="meltwater drained into the ocean")
    ax.axhline(info["q_cond"] * 1e3, color=MUTED, lw=0.8, ls=(0, (4, 3)), label="conductive shell")
    for t_ev, label in r.get("events", []):
        ax.axvline(t_ev, color=MUTED, lw=0.8, ls=(0, (1, 2)))
        ax.text(t_ev, 0.5, " " + label, color=MUTED, fontsize=7, rotation=90, va="bottom", ha="right")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(bottom=min(0, smooth(get("q_base")).min() * 1.1))
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_xlabel("Time (Myr)", color=INK)
    ax.set_ylabel("Heat flux (mW m⁻²)", color=INK)
    ax.set_title(f"Shell heat budget ({a.smooth_kyr:g} kyr running mean)", color=INK, fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(a.prefix + "_budget.png", facecolor=BG)

    late = t > t[0] + (t[-1] - t[0]) / 2
    dt = np.gradient(t)
    avg = lambda k: float(np.sum(np.array([x[k] for x in h])[late] * dt[late]) / dt[late].sum())
    print(f"t = {t[0]:.1f}-{t[-1]:.1f} Myr; second half: surface {avg('q_surface')*1e3:.2f}, from ocean "
          f"{avg('q_base')*1e3:.2f}, tidal {avg('q_tidal')*1e3:.2f}, drained {avg('q_drain')*1e3:.3f} mW/m2 "
          f"({avg('drain_mm_yr'):.3g} mm/yr of water), Nu {avg('Nu'):.3f}, growth {avg('growth_km_myr'):.3f} km/Myr, "
          f"temperate fraction {avg('temperate'):.4f}, max porosity {max(x['phi_max'] for x in h)*100:.2f} %, "
          f"max |energy error| {max(abs(x['energy_error']) for x in h):.1e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--D", type=float, default=None, help="shell thickness [km] (default 80)")
    r.add_argument("--grain", type=float, default=None, help="grain size [mm] (default 0.3)")
    r.add_argument("--edot", type=float, default=None, help="tidal strain-rate amplitude [1/s] (default 0)")
    r.add_argument("--phi_c", type=float, default=None, help="percolation threshold [%%] (default 1)")
    r.add_argument("--K0", type=float, default=None, help="permeability prefactor [m^2] (default 1e-9)")
    r.add_argument("--q_ocean", type=float, default=None, help="ocean heat flux into the base [mW/m2]")
    r.add_argument("--aspect", type=float, default=None, help="box width / depth (default 2)")
    r.add_argument("--nz", type=int, default=64)
    r.add_argument("--t_end", type=float, default=30.0, help="[Myr]")
    r.add_argument("--snaps", nargs="*", default=[])
    r.add_argument("--snap_every", type=float, default=1.0, help="snapshot interval [Myr]")
    r.add_argument("--restart", default=None)
    r.add_argument("--refine", type=int, default=1, help="restart on a grid this many times finer")
    r.add_argument("--solver", choices=["direct", "stag"], default="stag")
    r.add_argument("--wall", type=float, default=None, help="stop after this many seconds")
    r.add_argument("--out", default="results/shell.pkl")
    r.set_defaults(f=cmd_run)
    pl = sub.add_parser("plot")
    pl.add_argument("results")
    pl.add_argument("--prefix", default="figures/shell")
    pl.add_argument("--times", type=float, nargs="*", default=None)
    pl.add_argument("--phimax", type=float, default=2.0)
    pl.add_argument("--smooth_kyr", type=float, default=200.0)
    pl.set_defaults(f=cmd_plot)
    sl = sub.add_parser("slim", help="shrink a results file to a restart file")
    sl.add_argument("results")
    sl.add_argument("--out", required=True)
    sl.set_defaults(f=cmd_slim)
    args = ap.parse_args()
    args.f(args)
