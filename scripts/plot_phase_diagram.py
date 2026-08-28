#!/usr/bin/env python3
"""Phase diagram of the (r_c, tau) plane from a run_grid.py sweep.

Panel (a): local exponent  d ln D / d ln tau   (at fixed r_c)
Panel (b): local exponent  d ln D / d ln r_c   (at fixed tau)
Panel (c): D rescaled by the Case-2 prediction, to show where it plateaus

With xi0 = v0 = 1 the predicted boundaries are

    Case 1 <-> 2 :  r_c = sqrt(v0 xi0 tau)          = tau^(1/2)
    Case 2 <-> 3 :  r_c = xi0 (xi0 / v0 tau)^(3/7)  = tau^(-3/7)

which cross at the triple point (tau, r_c) = (1, 1).

    Case 1   r_c > sqrt(tau)                    D = r_c^2 / (2 tau)   slope_tau = -1
    Case 2   r_c < sqrt(tau), r_c < tau^(-3/7)  D ~ (r_c^2/tau)^(3/13) slope_tau = -3/13
    Case 3   tau^(-3/7) < r_c < sqrt(tau)       D ~ tau^(-3/7)         slope_tau = -3/7
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

SLOPES = {1: -1.0, 2: -3.0 / 13.0, 3: -3.0 / 7.0}


def local_slope(logD, logx, axis):
    """Central-difference d logD / d logx along `axis`, NaN-tolerant."""
    out = np.full(logD.shape, np.nan)
    n = logD.shape[axis]
    take = lambda k: np.take(logD, k, axis=axis)
    for k in range(n):
        lo, hi = max(k - 1, 0), min(k + 1, n - 1)
        if lo == hi:
            continue
        num = take(hi) - take(lo)
        den = logx[hi] - logx[lo]
        idx = [slice(None)] * logD.ndim
        idx[axis] = k
        out[tuple(idx)] = num / den
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/grid_D.npz")
    p.add_argument("--out", default="figures/phase_diagram.png")
    args = p.parse_args()

    z = np.load(args.data)
    rc, tau, D, done = z["r_c"], z["tau"], z["D"], z["done"]
    D = np.where(done, D, np.nan)
    print(f"{int(done.sum())}/{done.size} cells present")

    logD = np.log(D)
    s_tau = local_slope(logD, np.log(tau), axis=1)
    s_rc = local_slope(logD, np.log(rc), axis=0)

    # predicted boundaries (xi0 = v0 = 1)
    tt = np.logspace(np.log10(tau[0]) - 0.3, np.log10(tau[-1]) + 0.3, 200)
    b12 = np.sqrt(tt)                 # Case 1 <-> 2
    b23 = tt ** (-3.0 / 7.0)          # Case 2 <-> 3

    def frame(ax, title):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.plot(tt, b12, "w-", lw=2.4)
        ax.plot(tt, b12, "k--", lw=1.3, label=r"$r_c=\sqrt{v_0\xi_0\tau}$")
        ax.plot(tt, b23, "w-", lw=2.4)
        ax.plot(tt, b23, "k:", lw=1.6, label=r"$r_c=\xi_0(\xi_0/v_0\tau)^{3/7}$")
        ax.plot([1.0], [1.0], "ko", ms=5, mfc="w", mew=1.4)
        ax.set_xlim(tau[0] / 1.8, tau[-1] * 1.8)
        ax.set_ylim(rc[0] / 1.8, rc[-1] * 1.8)
        ax.set_xlabel(r"$\tau$  [$\xi_0/v_0$]")
        ax.set_title(title, fontsize=10)

    # cell edges for pcolormesh
    def edges(a):
        la = np.log10(a)
        e = np.empty(la.size + 1)
        e[1:-1] = 0.5 * (la[1:] + la[:-1])
        e[0] = la[0] - 0.5 * (la[1] - la[0])
        e[-1] = la[-1] + 0.5 * (la[-1] - la[-2])
        return 10.0 ** e

    te, re = edges(tau), edges(rc)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

    # ---- (a) tau exponent -------------------------------------------------
    ax = axes[0]
    m = ax.pcolormesh(te, re, s_tau, cmap="viridis", vmin=-1.05, vmax=0.0,
                      shading="flat")
    cb = fig.colorbar(m, ax=ax)
    cb.set_label(r"$d\ln D/d\ln\tau$")
    for v, lab in ((-1.0, "-1"), (-3 / 7, "-3/7"), (-3 / 13, "-3/13")):
        cb.ax.axhline(v, color="r", lw=1.0)
        cb.ax.text(1.6, v, lab, color="r", va="center", fontsize=8)
    frame(ax, r"(a) local $\tau$ exponent"
              "\n" r"Case 1 $=-1$,  Case 2 $=-3/13$,  Case 3 $=-3/7$")
    ax.set_ylabel(r"$r_c$  [$\xi_0$]")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.85)

    # ---- (b) r_c exponent -------------------------------------------------
    ax = axes[1]
    m = ax.pcolormesh(te, re, s_rc, cmap="magma", vmin=-0.1, vmax=2.1,
                      shading="flat")
    cb = fig.colorbar(m, ax=ax)
    cb.set_label(r"$d\ln D/d\ln r_c$")
    for v, lab in ((2.0, "2"), (6 / 13, "6/13"), (0.0, "0")):
        cb.ax.axhline(v, color="c", lw=1.0)
        cb.ax.text(1.6, v, lab, color="c", va="center", fontsize=8)
    frame(ax, r"(b) local $r_c$ exponent"
              "\n" r"Case 1 $=2$,  Case 2 $=6/13$,  Case 3 $=0$")

    # ---- (c) regime classification ---------------------------------------
    ax = axes[2]
    ok = np.isfinite(s_tau)
    cls = np.full(s_tau.shape, np.nan)
    keys = np.array([SLOPES[1], SLOPES[2], SLOPES[3]])
    d = np.abs(s_tau[..., None] - keys[None, None, :])
    cls[ok] = (np.argmin(d, axis=-1) + 1.0)[ok]
    cmap = ListedColormap(["#3b6fb6", "#e0a03c", "#b1453f"])
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)
    m = ax.pcolormesh(te, re, cls, cmap=cmap, norm=norm, shading="flat")
    cb = fig.colorbar(m, ax=ax, ticks=[1, 2, 3])
    cb.ax.set_yticklabels(["Case 1", "Case 2", "Case 3"])
    frame(ax, "(c) nearest regime by measured slope"
              "\n(colour = data, lines = prediction)")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=170)
    print("wrote", args.out)

    # ---- text table -------------------------------------------------------
    print("\nD(r_c, tau)")
    print("r_c \\ tau " + "".join(f"{t:>10.3g}" for t in tau))
    for i, r in enumerate(rc):
        print(f"{r:9.3g} " + "".join(
            ("       ---" if not done[i, j] else f"{D[i, j]:10.4g}")
            for j in range(tau.size)))
    print("\nd lnD / d ln tau")
    print("r_c \\ tau " + "".join(f"{t:>10.3g}" for t in tau))
    for i, r in enumerate(rc):
        print(f"{r:9.3g} " + "".join(
            ("       ---" if not np.isfinite(s_tau[i, j]) else f"{s_tau[i,j]:10.3f}")
            for j in range(tau.size)))
    print("\nd lnD / d ln r_c")
    print("r_c \\ tau " + "".join(f"{t:>10.3g}" for t in tau))
    for i, r in enumerate(rc):
        print(f"{r:9.3g} " + "".join(
            ("       ---" if not np.isfinite(s_rc[i, j]) else f"{s_rc[i,j]:10.3f}")
            for j in range(tau.size)))


if __name__ == "__main__":
    main()
