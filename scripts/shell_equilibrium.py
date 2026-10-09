"""Stage 2d: equilibrium thickness of the ice Ih shell vs grain size and basal heat flux.

    python scripts/shell_equilibrium.py run  --grain 0.1 --out results/shell_equilibrium.json
    python scripts/shell_equilibrium.py plot results/shell_equilibrium.json --fig figures/shell_equilibrium.png

`run` sweeps shell thickness D for one grain size (mm) and appends to the JSON file
(each point is a steady 2D convection solve at 64^2, ~1-3 min). `plot` draws
q_out(D) and prints the equilibrium-thickness table as Markdown.
"""
import argparse
import json
import os

import numpy as np

from ganymede.shell_equilibrium import (A_K_ICE, T_SURFACE, T_melt_Ih, RHO_ICE, G_SHELL,
                                        equilibrium_thickness, max_thickness, shell_heat_flow)

D_KM = [20, 30, 45, 60, 80, 100, 120, 140, 155]
Q_IN = [5, 10, 15, 20, 30, 40]          # mW m^-2
COLORS = {0.1: "#2a78d6", 0.3: "#eb6834", 1.0: "#1baf7a"}   # validated categorical slots 1-3
MARKERS = {0.1: "o", 0.3: "s", 1.0: "^"}


def conductive_equilibrium(q_in):
    """Thickness of a purely conductive shell carrying q_in [W/m^2]: D = 651 ln(T_b(D)/T_s) / q_in."""
    D = 20e3
    for _ in range(20):
        D = A_K_ICE * np.log(T_melt_Ih(RHO_ICE * G_SHELL * D) / T_SURFACE) / q_in
    return D


def cmd_run(a):
    rows = json.load(open(a.out)) if os.path.exists(a.out) else []
    for D in a.D:
        r = shell_heat_flow(a.grain * 1e-3, D * 1e3, n=a.n, aspect=a.aspect)
        r.pop("T")
        print(f"d={a.grain} mm D={D:5.0f} km T_b={r['T_b']:.1f} K Ra/Ra_cr={r['Ra']/r['Ra_cr']:7.2f} "
              f"Nu={r['Nu']:.3f} q_out={r['q_out']*1e3:.2f} mW/m2  {r['status']}", flush=True)
        rows.append(r)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(rows, open(a.out, "w"), indent=1)


def curves(rows):
    out = {}
    for r in rows:
        if r.get("aspect", 1.0) != 1.0 or r.get("branch"):
            continue
        out.setdefault(round(r["d"] * 1e3, 3), []).append(r)
    return {d: sorted(v, key=lambda r: r["D"]) for d, v in sorted(out.items())}


def equilibrium_table(rows):
    lines = ["| q_in (mW m⁻²) | " + " | ".join(f"{d:g} mm" for d in curves(rows)) + " |",
             "|---|" + "---|" * len(curves(rows))]
    for q in Q_IN:
        cells = []
        for d, c in curves(rows).items():
            D = np.array([r["D"] for r in c]) / 1e3
            qo = np.array([r["q_out"] for r in c]) * 1e3
            De = equilibrium_thickness(D, qo, q)
            if De is None:
                cells.append(f"≥ {max_thickness()/1e3:.0f} (Ih–III limit)")
            elif De == D[0] and c[0]["Nu"] < 1.01:
                cells.append(f"{conductive_equilibrium(q*1e-3)/1e3:.0f} (cond.)")
            else:
                i = np.searchsorted(D, De)
                conv = max(c[max(i - 1, 0)]["Nu"], c[min(i, len(c) - 1)]["Nu"]) > 1.01
                cells.append(f"{De:.0f}" + ("" if conv else " (cond.)"))
        lines.append(f"| {q} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cmd_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = json.load(open(a.results))
    ink, muted, grid = "#1f1f1e", "#6b6a63", "#e4e3dc"
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    Dmax = max_thickness() / 1e3
    ax.axvspan(Dmax, 175, color="#ecebe5", lw=0)
    ax.text(Dmax + 1.5, 9, "ice III\nat base", color=muted, fontsize=8, va="top")
    for q in (5, 15, 40):
        ax.axhline(q, color=muted, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.text(171, q * 1.04, f"q_in = {q}", color=muted, fontsize=8, ha="right", va="bottom")
    for d, c in curves(rows).items():
        D = np.array([r["D"] for r in c]) / 1e3
        qo = np.array([r["q_out"] for r in c]) * 1e3
        conv = np.array([r["Nu"] > 1.01 for r in c])
        col = COLORS.get(d, ink)
        ax.plot(D, qo, color=col, lw=2, zorder=3, label=f"{d:g} mm")
        ax.scatter(D[conv], qo[conv], s=60, marker=MARKERS.get(d, "o"), color=col,
                   edgecolor="#fcfcfb", linewidth=1.5, zorder=4)
        ax.scatter(D[~conv], qo[~conv], s=60, marker=MARKERS.get(d, "o"), facecolor="#fcfcfb",
                   edgecolor=col, linewidth=1.5, zorder=4)
        ax.text(D[-1] - 2, qo[-1] * 1.08, f"{d:g} mm", color=ink, fontsize=9, ha="right", va="bottom")
    for r in rows:   # second (convective) steady state where the shell is bistable
        if r.get("branch") == "convective" and r.get("aspect", 1.0) == 1.0:
            d = round(r["d"] * 1e3, 3)
            ax.scatter(r["D"] / 1e3, r["q_out"] * 1e3, s=60, marker=MARKERS.get(d, "o"),
                       color=COLORS.get(d, ink), edgecolor="#fcfcfb", linewidth=1.5, zorder=5)
            ax.annotate("convective branch\n(bistable)", (r["D"] / 1e3, r["q_out"] * 1e3),
                        xytext=(-58, 16), textcoords="offset points", fontsize=8, color=muted,
                        arrowprops=dict(arrowstyle="-", color=muted, lw=0.6))
    ax.set_yscale("log")
    ax.set_xlim(15, 175)
    ax.set_ylim(3, 50)
    ax.set_yticks([3, 5, 10, 15, 20, 30, 40])
    ax.set_yticklabels(["3", "5", "10", "15", "20", "30", "40"])
    ax.set_xlabel("Ice Ih shell thickness D (km)", color=ink)
    ax.set_ylabel("Heat flux out of the top, q_out (mW m⁻²)", color=ink)
    ax.set_title("Ice Ih shell: equilibrium where q_out = q_in (filled = convecting, open = conductive)",
                 fontsize=9.5, color=ink, loc="left")
    ax.grid(True, which="major", color=grid, lw=0.6)
    ax.minorticks_off()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(muted)
    ax.tick_params(colors=muted)
    leg = ax.legend(title="grain size", loc="lower left", frameon=False, fontsize=8, title_fontsize=8)
    for h in leg.legend_handles:
        h.set_linewidth(2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.fig) or ".", exist_ok=True)
    fig.savefig(a.fig, facecolor=fig.get_facecolor())
    print(equilibrium_table(rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--grain", type=float, required=True, help="grain size [mm]")
    r.add_argument("--D", type=float, nargs="+", default=D_KM, help="thicknesses [km]")
    r.add_argument("--n", type=int, default=64)
    r.add_argument("--aspect", type=float, default=1.0)
    r.add_argument("--out", default="results/shell_equilibrium.json")
    r.set_defaults(f=cmd_run)
    pl = sub.add_parser("plot")
    pl.add_argument("results")
    pl.add_argument("--fig", default="figures/shell_equilibrium.png")
    pl.set_defaults(f=cmd_plot)
    a = p.parse_args()
    a.f(a)
