#!/usr/bin/env python3
"""Measure nu and d_h for the level sets of the simulated landscape.

Tracing contours one at a time censors exactly the large ones (they are the
slowest to close), which is fatal for an exponent that describes how contours
diverge.  Working on the grid instead, every cluster at a given level is found
at once, with no size cut:

  nu   from the percolation correlation length of the finite clusters,
       xi(V) = sqrt( sum_s 2 R_s^2 s^2 / sum_s s^2 )  ~  |V|^(-nu)
  d_h  from the boundary length of each cluster against its radius of gyration
       at the critical level, P ~ R^(d_h)

Both are the ingredients of D ~ tau^-(1 - (2 - 1/nu)/d_h) = tau^(-2/7).
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mrdiff import PeriodicGaussianField  # noqa: E402


def cluster_stats(mask, dx):
    """Label the mask and return per-cluster (size, R_gyration, perimeter)."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0), lab, n
    idx = lab.ravel()
    size = np.bincount(idx, minlength=n + 1)[1:].astype(float)
    ny, nx = mask.shape
    yy, xx = np.divmod(np.arange(idx.size), nx)
    sx = np.bincount(idx, weights=xx, minlength=n + 1)[1:]
    sy = np.bincount(idx, weights=yy, minlength=n + 1)[1:]
    sxx = np.bincount(idx, weights=xx * xx, minlength=n + 1)[1:]
    syy = np.bincount(idx, weights=yy * yy, minlength=n + 1)[1:]
    cx, cy = sx / size, sy / size
    rg2 = (sxx / size - cx ** 2) + (syy / size - cy ** 2)
    # boundary length: count cluster/non-cluster edges in both directions
    b = np.zeros(n + 1)
    for sh, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        nb = np.roll(mask, sh, axis=ax)
        edge = mask & ~nb
        b += np.bincount(lab[edge], minlength=n + 1)
    return size * dx ** 2, np.sqrt(np.maximum(rg2, 0)) * dx, b[1:] * dx, lab, n


def spanning(lab, n):
    """Labels present on both opposite edges (a cluster that crosses the box)."""
    out = np.zeros(n + 1, bool)
    for a, b in ((lab[0], lab[-1]), (lab[:, 0], lab[:, -1])):
        out[np.intersect1d(np.unique(a), np.unique(b))] = True
    out[0] = False
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=float, default=400.0)
    p.add_argument("--dx", type=float, default=0.2)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--out", default="figures/percolation_exponents.png")
    args = p.parse_args()

    levels = np.geomspace(0.02, 1.2, 22)
    xi_all, P_all, R_all = [], [], []
    for sd in range(args.seeds):
        f = PeriodicGaussianField(xi0=1.0, Gamma=1.0, L=args.L, dx=args.dx,
                                  seed=700 + sd, precompute=False)
        Vg = f.V
        xis = []
        for lev in levels:
            # both tails; the field is symmetric so average the two
            acc = []
            for m in (Vg > lev, Vg < -lev):
                size, rg, per, lab, n = cluster_stats(m, f.dx)
                if n == 0:
                    continue
                keep = ~spanning(lab, n)[1:]
                s, r = size[keep], rg[keep]
                if s.sum() <= 0:
                    continue
                acc.append(np.sqrt(np.sum(2 * r ** 2 * s ** 2) / np.sum(s ** 2)))
            xis.append(np.mean(acc) if acc else np.nan)
        xi_all.append(xis)
        # hull dimension at the critical level
        size, rg, per, lab, n = cluster_stats(Vg > 0.0, f.dx)
        keep = (~spanning(lab, n)[1:]) & (rg > 0.6)
        P_all.append(per[keep]); R_all.append(rg[keep])
        print(f"  seed {sd}: {n} clusters at V=0, "
              f"{keep.sum()} finite ones used", flush=True)

    xi = np.nanmean(np.array(xi_all), axis=0)
    P = np.concatenate(P_all); R = np.concatenate(R_all)

    def fit(xv, yv):
        c, cov = np.polyfit(np.log(xv), np.log(yv), 1, cov=True)
        return c[0], np.sqrt(cov[0, 0]), c[1]

    # nu: use levels where xi is well inside the box (no finite-size cut)
    m = np.isfinite(xi) & (xi < args.L / 8) & (levels < 0.7)
    nu, nu_e, nu_c = fit(levels[m], xi[m])
    # d_h: the fractal range, clusters well above the correlation length of V
    md = (R > 2.0)
    dh, dh_e, dh_c = fit(R[md], P[md])
    alpha = 1 - (2 + 1 / nu) / dh          # nu is fitted as a negative slope
    print(f"\n  nu   = {-nu:.3f} +- {nu_e:.3f}    [2-D percolation 4/3 = 1.333]")
    print(f"  d_h  = {dh:.3f} +- {dh_e:.3f}    [2-D percolation 7/4 = 1.750]")
    print(f"  => D ~ tau^-{alpha:.3f}          [2/7 = 0.286]")

    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.9), constrained_layout=True)
    a = ax[0]
    a.plot(levels[m], xi[m], "o", ms=6, mfc="white", mew=1.5, color="#1f5fa9",
           label="finite-cluster correlation length")
    a.plot(levels[~m & np.isfinite(xi)], xi[~m & np.isfinite(xi)], "s", ms=5,
           mfc="#dfe7f0", mew=1.0, color="#1f5fa9",
           label=r"excluded ($\xi > L/8$ or $|V| > 0.7\Gamma$)")
    xs = np.geomspace(levels[m].min(), levels[m].max(), 30)
    a.plot(xs, np.exp(nu_c) * xs ** nu, lw=2, color="#c2582a",
           label=rf"fit: $\xi \propto |V|^{{{nu:.2f}}}$")
    a.plot(xs, xs ** (-4 / 3) * np.exp(nu_c) * xs[0] ** (nu + 4 / 3) / 1.0,
           lw=1.4, ls="--", color="0.35", label=r"$|V|^{-4/3}$")
    a.set(xscale="log", yscale="log", xlabel=r"$|V|/\Gamma$",
          ylabel=r"$\xi/\xi_0$", title=r"$\nu$: contours diverge at $V\to0$")
    a.grid(True, which="both", alpha=0.2); a.legend(frameon=False, fontsize=8.5)

    a = ax[1]
    a.plot(R, P, ".", ms=1.5, alpha=0.2, color="#3f8f52")
    xs = np.geomspace(2.0, R.max(), 30)
    a.plot(xs, np.exp(dh_c) * xs ** dh, lw=2, color="#c2582a",
           label=rf"fit: $P \propto R^{{{dh:.2f}}}$")
    a.plot(xs, xs ** 1.75 * np.exp(dh_c) * xs[0] ** (dh - 1.75) / 1.0, lw=1.4,
           ls="--", color="0.35", label=r"$R^{7/4}$")
    a.set(xscale="log", yscale="log",
          xlabel=r"cluster radius of gyration $R/\xi_0$",
          ylabel=r"boundary length $P/\xi_0$",
          title=r"$d_h$: hull dimension at $V=0$")
    a.grid(True, which="both", alpha=0.2); a.legend(frameon=False, fontsize=9)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=190)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
