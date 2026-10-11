"""Stage 6a: does a freezing salty ocean make Ganymede grow?

    python scripts/hydro_evolution.py run mixed  --f_sink 0 --t_end 4.5 --out results/hydro_mixed.json
    python scripts/hydro_evolution.py run brine  --f_sink 1 --t_end 1.5 --out results/hydro_brine.json
    python scripts/hydro_evolution.py plot results/hydro_mixed.json results/hydro_brine.json --fig figures/hydro_evolution.png

`run` integrates the column model (ganymede.hydrosphere) through a cooling history; `plot` compares runs:
radius change, ice Ih shell, HP ice and ocean salinity against time, and radius change against shell
thickness (the clean test: at the same shell thickness, which run is bigger?).
"""
import argparse
import json
import os

import numpy as np

INK, MUTED, GRID, BG = "#1f1f1e", "#6b6a63", "#e4e3dc", "#fcfcfb"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def cmd_run(a):
    from ganymede.hydrosphere import EOS, HydroParams, evolve
    p = HydroParams(S0=a.S0 / 1e3, f_sink=a.f_sink, k_eff=a.k_eff, c_brine=a.c_brine, Nu_brine=a.Nu_brine,
                    grain=a.grain, q_rock0=a.q_rock0 * 1e-3, tau_rock=a.tau_rock * 1e9)
    rec, info = evolve(p, EOS(), t_end=a.t_end * 1e9, dD=2000.0, dhp=3000.0, dt_max=100e6, log_every=5)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(label=a.label, rec=rec, info=info, params=vars(a) | dict(f=None)), open(a.out, "w"), default=str)


def cmd_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = []
    for path in a.results:
        d = json.load(open(path))
        rec = []
        for r in d["rec"]:                                        # stop before the eutectic / energy drift
            if r["S_top"] >= 0.2 or abs(r["energy_error"]) > 1e-2:
                break
            rec.append(r)
        cyc = [r for r in rec if r["mode"] != "brine"]            # states right after each overturn
        label = d.get("label") or os.path.splitext(os.path.basename(path))[0]
        runs.append((label, cyc if len(cyc) > 10 else rec, rec))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=150)
    fig.patch.set_facecolor(BG)
    panels = [("time (Gyr)", "radius change (km)"), ("time (Gyr)", "ice Ih shell (km)"),
              ("time (Gyr)", "high-pressure ice (km)"), ("ice Ih shell (km)", "radius change (km)")]
    for ax, (xl, yl) in zip(axes.flat, panels):
        ax.set_facecolor(BG)
        ax.grid(True, color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel(xl, color=INK, fontsize=9)
        ax.set_ylabel(yl, color=INK, fontsize=9)
        ax.axhline(0, color=MUTED, lw=0.6) if "radius" in yl else None
    for (label, rec, full), c in zip(runs, COLORS):
        t = np.array([r["t_myr"] for r in rec]) / 1e3
        R = np.array([r["R_km"] for r in rec])
        dR = R - R[0]
        shell = np.array([r["shell_km"] for r in rec])
        hp = np.array([r["hp_km"] for r in rec])
        axes[0, 0].plot(t, dR, color=c, lw=2, label=label)
        axes[0, 1].plot(t, shell, color=c, lw=2)
        axes[1, 0].plot(t, hp, color=c, lw=2)
        axes[1, 1].plot(shell, dR, color=c, lw=2)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    axes[0, 0].set_title("Ganymede's radius", color=INK, fontsize=9.5, loc="left")
    axes[0, 1].set_title("ice Ih shell (freezing at the top of the ocean: expands)", color=INK, fontsize=9.5, loc="left")
    axes[1, 0].set_title("HP ice (freezing at the ocean floor: contracts)", color=INK, fontsize=9.5, loc="left")
    axes[1, 1].set_title("radius change at the same shell thickness", color=INK, fontsize=9.5, loc="left")
    fig.suptitle("Freezing Ganymede's ocean: well-mixed vs sinking brine (column model, Stage 6a; brine runs shown "
                 "after each overturn)",
                 color=INK, fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.fig) or ".", exist_ok=True)
    fig.savefig(a.fig, facecolor=BG)
    for label, _, rec in runs:
        r0, r1 = rec[0], rec[-1]
        print(f"{label:28s} t {r1['t_myr']/1e3:5.2f} Gyr  shell {r0['shell_km']:5.1f}->{r1['shell_km']:5.1f} km  "
              f"HP {r0['hp_km']:5.1f}->{r1['hp_km']:5.1f} km  dR {r1['R_km']-r0['R_km']:+6.2f} km  "
              f"areal strain {2*(r1['R_km']-r0['R_km'])/r0['R_km']*100:+.2f} %  S_top {r1['S_top']*1e3:.0f} g/kg  "
              f"max |E err| {max(abs(r['energy_error']) for r in rec):.0e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("label")
    r.add_argument("--S0", type=float, default=10.0, help="initial ocean salinity [g/kg]")
    r.add_argument("--f_sink", type=float, default=1.0, help="fraction of rejected brine that sinks (0 = well mixed)")
    r.add_argument("--k_eff", type=float, default=0.1, help="fraction of salt kept in new shell ice (mushy base)")
    r.add_argument("--c_brine", type=float, default=3.0, help="sinking brine salinity / ocean salinity")
    r.add_argument("--Nu_brine", type=float, default=1.0, help="heat transport through the brine layer / conduction")
    r.add_argument("--grain", type=float, default=0.3, help="ice grain size for the shell heat flow [mm]")
    r.add_argument("--q_rock0", type=float, default=30.0, help="initial rock heat flux [mW/m2]")
    r.add_argument("--tau_rock", type=float, default=2.0, help="decay time of the rock heat flux [Gyr]")
    r.add_argument("--t_end", type=float, default=4.5, help="[Gyr]")
    r.add_argument("--out", required=True)
    r.set_defaults(f=cmd_run)
    pl = sub.add_parser("plot")
    pl.add_argument("results", nargs="+")
    pl.add_argument("--fig", default="figures/hydro_evolution.png")
    pl.set_defaults(f=cmd_plot)
    args = ap.parse_args()
    args.f(args)
