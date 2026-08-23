#!/usr/bin/env python3
"""The Gaussian random landscape: contours, the percolating level, and a path."""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mrdiff import GaussianRandomField, trajectory  # noqa: E402


def field_panel(ax, pot, lim, n=600):
    g = np.linspace(-lim, lim, n)
    X, Y = np.meshgrid(g, g)
    V = pot.value(X.ravel(), Y.ravel()).reshape(X.shape)
    s = np.abs(V).max()
    ax.imshow(V, origin="lower", extent=[-lim, lim, -lim, lim], cmap="RdBu_r",
              vmin=-s, vmax=s, alpha=0.85)
    ax.contour(X, Y, V, levels=np.linspace(-s, s, 17), colors="k",
               linewidths=0.35, alpha=0.3)
    ax.contour(X, Y, V, levels=[0.0], colors="k", linewidths=1.4, alpha=0.9)
    ax.set_aspect("equal")
    return V


def main(out="figures/random_illustration.png"):
    xi0, tau, B = 1.0, 1.0, 1.0
    pot = GaussianRandomField(xi0=xi0, Gamma=1 / np.sqrt(2), n_modes=64, seed=5)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6), constrained_layout=True)

    # (a) close-up: contours, the V = 0 percolating level, and one path
    ax = axes[0]
    field_panel(ax, pot, lim=6.0)
    arcs = trajectory(pot, r_c=0.3, tau=tau, n_steps=110, B=B, seed=2,
                      start=(0.4, 0.6), n_sub=8, collisions="poisson")
    for k, arc in enumerate(arcs):
        ax.plot(arc[0], arc[1], color="k", lw=1.4, zorder=3,
                label="contour drift" if k == 0 else None)
        if k + 1 < len(arcs):
            ax.plot([arc[0, -1], arcs[k + 1][0, 0]],
                    [arc[1, -1], arcs[k + 1][1, 0]], color="0.15", lw=0.8,
                    ls=(0, (2, 1.6)), zorder=2,
                    label="impurity hop ($r_c$)" if k == 0 else None)
    ax.plot(*arcs[0][:, 0], "o", ms=6, mfc="w", mec="k", zorder=4)
    ax.set(xlim=(-6, 6), ylim=(-6, 6), xlabel=r"$x/\xi_0$", ylabel=r"$y/\xi_0$",
           title="Gaussian random $V$: contours, the $V=0$ network, and a path")
    ax.legend(frameon=True, framealpha=0.9, fontsize=9, loc="upper right")

    # (b) the same landscape at larger scale, with the percolating level alone
    ax = axes[1]
    lim = 20.0
    g = np.linspace(-lim, lim, 900)
    X, Y = np.meshgrid(g, g)
    V = pot.value(X.ravel(), Y.ravel()).reshape(X.shape)
    ax.contourf(X, Y, V, levels=[-1e9, 0.0, 1e9], colors=["#dce6f2", "#f7dfd6"])
    ax.contour(X, Y, V, levels=[0.0], colors="k", linewidths=0.8)
    arcs = trajectory(pot, r_c=0.3, tau=8.0, n_steps=90, B=B, seed=4,
                      start=(0.0, 0.0), n_samples_per_arc=140, n_sub=64,
                      collisions="poisson")
    for k, arc in enumerate(arcs):
        ax.plot(arc[0], arc[1], color="#8c1d1d", lw=1.2, zorder=3)
        if k + 1 < len(arcs):
            ax.plot([arc[0, -1], arcs[k + 1][0, 0]],
                    [arc[1, -1], arcs[k + 1][1, 0]], color="#8c1d1d", lw=0.7,
                    ls=(0, (2, 1.6)), alpha=0.7, zorder=2)
    ax.plot(0, 0, "o", ms=6, mfc="w", mec="k", zorder=4)
    ax.set_aspect("equal")
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel=r"$x/\xi_0$",
           ylabel=r"$y/\xi_0$",
           title=r"Hills (red) and valleys (blue); a $\tau=8T_0$ walk"
                 "\n"
                 r"rides the long contours near the percolating level")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=185)
    print("wrote", out)


if __name__ == "__main__":
    main()
