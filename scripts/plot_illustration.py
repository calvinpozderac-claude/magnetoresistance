#!/usr/bin/env python3
"""Illustration: the pyramid landscape with a drift-kick path, and MSD curves."""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mrdiff import SquarePyramid, simulate, trajectory  # noqa: E402


def main(out="figures/pyramid_illustration.png"):
    pot = SquarePyramid()
    tau, B = 1.0, 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), constrained_layout=True)

    # -- (a) landscape + trajectory ---------------------------------------
    ax = axes[0]
    lim = 4.0
    g = np.linspace(-0.001, lim, 801)
    X, Y = np.meshgrid(g, g)
    V = pot.value(X, Y)
    ax.imshow(V, origin="lower", extent=[0, lim, 0, lim], cmap="RdBu_r",
              vmin=-1, vmax=1, alpha=0.8)
    ax.contour(X, Y, V, levels=np.linspace(-0.9, 0.9, 13), colors="k",
               linewidths=0.4, alpha=0.35)

    r_c = 0.35
    arcs = trajectory(pot, r_c=r_c, tau=tau, n_steps=90, B=B, seed=3,
                      start=(1.2, 1.35))
    for k, arc in enumerate(arcs):
        ax.plot(arc[0], arc[1], color="k", lw=1.5, solid_capstyle="round",
                zorder=3, label="contour drift ($\\tau$)" if k == 0 else None)
        if k + 1 < len(arcs):
            ax.plot([arc[0, -1], arcs[k + 1][0, 0]],
                    [arc[1, -1], arcs[k + 1][1, 0]], color="0.15", lw=0.9,
                    ls=(0, (2, 1.6)), zorder=2,
                    label="impurity hop ($r_c$)" if k == 0 else None)
    ax.plot(*arcs[0][:, 0], "o", ms=6, mfc="w", mec="k", zorder=4)
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="$x/a$", ylabel="$y/a$",
           title=f"Square-pyramid $V(x,y)$ and a drift--kick path ($r_c={r_c}$)")
    ax.set_aspect("equal")
    ax.legend(frameon=True, framealpha=0.9, fontsize=9, loc="upper right")

    # -- (b) MSD curves ----------------------------------------------------
    ax = axes[1]
    for r_c, n_steps, nw, col in [(0.05, 60000, 1024, "#3b6ea5"),
                                  (0.2, 12000, 2048, "#4a9b5c"),
                                  (1.0, 4000, 4096, "#d08b28"),
                                  (3.0, 3000, 4096, "#b33a3a")]:
        res = simulate(pot, r_c=r_c, tau=tau, n_steps=n_steps, n_walkers=nw,
                       B=B, seed=99, n_snapshots=140)
        ax.plot(res.lags, res.tamsd, lw=1.7, color=col,
                label=rf"$r_c={r_c}$,  $D={res.D:.3g}$")
        ax.plot(res.lags, 4 * res.D * res.lags, lw=0.9, ls="--", color="0.4",
                zorder=1)
    ax.set(xscale="log", yscale="log", xlabel=r"lag time  $\Delta t/\tau$",
           ylabel=r"$\langle |{\bf r}(t+\Delta t)-{\bf r}(t)|^2 \rangle / a^2$",
           title="Time-averaged MSD (dashed: $4D\\Delta t$ fits)")
    ax.grid(True, which="both", alpha=0.25, lw=0.6)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=190)
    print("wrote", out)


if __name__ == "__main__":
    main()
