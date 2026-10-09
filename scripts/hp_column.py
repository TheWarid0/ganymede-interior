"""Stage 4c: melt generation and extraction in a non-convecting HP ice column (Kalousova et al. 2018 parameters).

    python scripts/hp_column.py --fig figures/hp_column.png

A 200 km ice VI layer starts at the ocean-interface melting point and is heated from below at
q_s. With no convection, the basal heat melts ice, the water percolates up and refreezes in the
cold ice above, warming it, until a temperate (partially molten) channel reaches the ocean.
"""
import argparse
import os

import numpy as np

from ganymede.twophase import YEAR, HPIce, front_height_energy_balance, run_column


def main(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = HPIce(H=a.H * 1e3, q_s=a.qs * 1e-3)
    r = run_column(p, N=a.N, dt=1e3 * YEAR, t_end=a.t_end * 1e6 * YEAR, every=50)
    h = r["hist"]
    t_myr = h["t"] / YEAR / 1e6
    t_break = t_myr[np.argmax(h["temperate_top"] >= p.H - 1)]
    _, t_pred = front_height_energy_balance(0.0, p)
    tt = np.linspace(0, a.t_end, 400)
    h_pred, _ = front_height_energy_balance(tt * 1e6 * YEAR, p)
    print(f"melt reaches the ocean after {t_break:.2f} Myr (energy balance: {t_pred/YEAR/1e6:.2f} Myr)")
    print(f"steady state: water carries {h['water_flux'][-1]*p.rho_w*p.L*1e3:.2f} mW/m2, "
          f"conduction {h['q_top'][-1]*1e3:.3f} mW/m2, water flux {h['water_flux'][-1]*YEAR*1e3:.2f} mm/yr, "
          f"max porosity {h['phi_max'][-1]*100:.2f} %, energy error {np.abs(h['energy_error']).max():.1e}")

    ink, muted, grid, bg = "#1f1f1e", "#6b6a63", "#e4e3dc", "#fcfcfb"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.4), dpi=150, gridspec_kw=dict(width_ratios=[1.25, 1]))
    fig.patch.set_facecolor(bg)
    for ax in (ax1, ax2):
        ax.set_facecolor(bg)
        ax.grid(True, color=grid, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(muted)
        ax.tick_params(colors=muted)

    ax1.plot(tt, h_pred / 1e3, color=muted, lw=1.2, ls=(0, (4, 3)), label="energy balance")
    ax1.plot(t_myr, h["temperate_top"] / 1e3, color="#2a78d6", lw=2, label="model")
    ax1.axvline(t_break, color=muted, lw=0.8)
    ax1.text(t_break + 0.4, 20, f"melt reaches\nthe ocean\n{t_break:.1f} Myr", color=ink, fontsize=8)
    ax1.set_xlim(0, a.t_end)
    ax1.set_ylim(0, p.H / 1e3 * 1.04)
    ax1.set_xlabel("Time (Myr)", color=ink)
    ax1.set_ylabel("Top of the temperate (wet) zone, km above silicates", color=ink)
    ax1.set_title("(a) Melt front rising through the HP ice", fontsize=9.5, color=ink, loc="left")
    ax1.legend(frameon=False, fontsize=8, loc="upper left")

    shades = ["#9ec5f4", "#5598e7", "#256abf", "#0d366b"]
    times = [3, 7, 11, a.t_end]
    for c, tm in zip(shades, times):
        j = np.argmin([abs(s[0] / YEAR / 1e6 - tm) for s in r["snaps"]])
        t_s, T, phi = r["snaps"][j]
        ax2.plot(T - r["Tm"], r["z"] / 1e3, color=c, lw=2, label=f"{t_s/YEAR/1e6:.0f} Myr")
    phi_t = r["snaps"][-1][2][r["z"] < p.H / 2].mean()
    ax2.text(-1.2, p.H / 1e3 * 0.39, f"temperate ice:\nT = T_m,\n{phi_t*100:.2f} % water", color=ink,
             fontsize=8, ha="right", va="center")
    ax2.set_xlim(-25, 1)
    ax2.set_ylim(0, p.H / 1e3)
    ax2.set_xlabel("T − T_m (K)", color=ink)
    ax2.set_ylabel("Height above silicates (km)", color=ink)
    ax2.set_title("(b) Temperature below the melting point", fontsize=9.5, color=ink, loc="left")
    ax2.legend(frameon=False, fontsize=8, loc="lower left", title="time", title_fontsize=8)
    fig.suptitle(f"Non-convecting HP ice layer, H = {a.H:g} km, q_s = {a.qs:g} mW m⁻² (Kalousová et al. 2018 parameters)",
                 fontsize=10, color=ink, x=0.01, ha="left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.fig) or ".", exist_ok=True)
    fig.savefig(a.fig, facecolor=bg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=float, default=200.0, help="layer thickness [km]")
    ap.add_argument("--qs", type=float, default=20.0, help="basal heat flux [mW/m2]")
    ap.add_argument("--N", type=int, default=400)
    ap.add_argument("--t_end", type=float, default=25.0, help="[Myr]")
    ap.add_argument("--fig", default="figures/hp_column.png")
    main(ap.parse_args())
