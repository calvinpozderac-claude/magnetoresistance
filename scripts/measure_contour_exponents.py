#!/usr/bin/env python3
"""Measure the two percolation exponents directly on the simulated landscape.

    nu   : how the contour size diverges as the level approaches percolation,
           diameter ~ |V|^(-nu)
    d_h  : how convoluted the contour is, perimeter ~ diameter^(d_h)

Both are quoted as exact 2-D percolation values (nu = 4/3, d_h = 7/4) in the
D ~ tau^(-2/7) argument; this checks them on the field the walkers actually see.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mrdiff import PeriodicGaussianField  # noqa: E402


def trace_contours(pot, x, y, h=0.0625, s_max=4000.0, B=1.0,
                   r_in=1.5, r_out=6.0):
    """Follow each contour once round, returning (perimeter, diameter, closed).

    Perimeter is the arclength actually travelled; diameter is the largest
    extent of the bounding box swept out.  Contours still open after ``s_max``
    of arclength are reported as unclosed and excluded from the fits.
    """
    x = np.array(x, float, copy=True)
    y = np.array(y, float, copy=True)
    rx, ry = x.copy(), y.copy()
    gx, gy = pot.grad(x, y)
    v0 = np.hypot(gx, gy) / B
    rin = np.maximum(r_in * v0 * h, 1e-14)
    rout = r_out * v0 * h

    s = np.zeros(x.shape)
    xmin, xmax = x.copy(), x.copy()
    ymin, ymax = y.copy(), y.copy()
    left = np.zeros(x.shape, bool)
    done = np.zeros(x.shape, bool)

    while True:
        ia = np.flatnonzero(~done & (s < s_max))
        if ia.size == 0:
            break
        xa, ya = x[ia], y[ia]
        gxa, gya = pot.grad(xa, ya)
        va = np.hypot(gxa, gya) / B
        xn, yn = pot._rk4_project(xa, ya, h, B)
        s[ia] += np.hypot(xn - xa, yn - ya)
        x[ia], y[ia] = xn, yn
        xmin[ia] = np.minimum(xmin[ia], xn); xmax[ia] = np.maximum(xmax[ia], xn)
        ymin[ia] = np.minimum(ymin[ia], yn); ymax[ia] = np.maximum(ymax[ia], yn)
        d = np.hypot(xn - rx[ia], yn - ry[ia])
        left[ia] |= d > rout[ia]
        close = left[ia] & (d < rin[ia])
        if close.any():
            done[ia[close]] = True
    diam = np.maximum(xmax - xmin, ymax - ymin)
    return s, diam, done


def fit(xv, yv):
    c, cov = np.polyfit(np.log(xv), np.log(yv), 1, cov=True)
    return c[0], np.sqrt(cov[0, 0]), c[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--L", type=float, default=400.0)
    p.add_argument("--h", type=float, default=0.0625)
    p.add_argument("--s-max", type=float, default=4000.0)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--out", default="figures/contour_exponents.png")
    args = p.parse_args()

    V, per, dia = [], [], []
    for sd in range(args.seeds):
        pot = PeriodicGaussianField(xi0=1.0, Gamma=1 / np.sqrt(2), L=args.L,
                                    dx=0.2, seed=300 + sd)
        rng = np.random.default_rng(sd)
        x, y = rng.uniform(0, args.L, (2, args.n))
        lev = pot.value(x, y)
        s, d, closed = trace_contours(pot, x, y, h=args.h, s_max=args.s_max)
        print(f"  seed {sd}: {100*closed.mean():.0f}% of contours closed within "
              f"arclength {args.s_max:g}")
        V.append(np.abs(lev[closed]) / (1 / np.sqrt(2)))   # |V| / Gamma
        per.append(s[closed])
        dia.append(d[closed])
    V = np.concatenate(V); per = np.concatenate(per); dia = np.concatenate(dia)

    # d_h from perimeter vs diameter, over the fractal range (diam > a few xi0)
    m = dia > 3.0
    dh, dh_e, _ = fit(dia[m], per[m])
    # nu from diameter vs level, binned in |V| so every decade counts equally
    bins = np.geomspace(max(V.min(), 1e-3), 1.0, 16)
    idx = np.digitize(V, bins) - 1
    bx, by = [], []
    for b in range(len(bins) - 1):
        sel = idx == b
        if sel.sum() > 30:
            bx.append(np.sqrt(bins[b] * bins[b + 1]))
            by.append(np.median(dia[sel]))
    bx, by = np.array(bx), np.array(by)
    mm = (bx < 0.35) & (by > 2.0)
    nu, nu_e, nu_c = fit(bx[mm], by[mm])

    print(f"\n  d_h  (perimeter ~ diameter^d_h) = {dh:.3f} +- {dh_e:.3f}"
          f"   [2-D percolation hull: 7/4 = 1.750]")
    print(f"  nu   (diameter ~ |V|^-nu)       = {-nu:.3f} +- {nu_e:.3f}"
          f"   [2-D percolation:      4/3 = 1.333]")
    print(f"  => D ~ tau^-(1 - (2 - 1/nu)/d_h) = tau^"
          f"-{1 - (2 + 1/nu)/dh:.3f}   [2/7 = 0.286]")

    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.9), constrained_layout=True)
    a = ax[0]
    a.plot(dia, per, ".", ms=1.6, alpha=0.25, color="#1f5fa9")
    xs = np.geomspace(3, dia.max(), 30)
    a.plot(xs, np.exp(np.polyval([dh, _fitc(dia[m], per[m], dh)], np.log(xs))),
           lw=2, color="#c2582a", label=rf"fit: $\propto \Lambda^{{{dh:.2f}}}$")
    a.plot(xs, xs ** 1.75 * (per[m].mean() / dia[m].mean() ** 1.75), lw=1.4,
           ls="--", color="0.35", label=r"$\Lambda^{7/4}$")
    a.set(xscale="log", yscale="log", xlabel=r"contour diameter $\Lambda/\xi_0$",
          ylabel=r"contour perimeter $L/\xi_0$",
          title=r"$d_h$: how convoluted a contour is")
    a.grid(True, which="both", alpha=0.2); a.legend(frameon=False, fontsize=9)

    a = ax[1]
    a.plot(V, dia, ".", ms=1.6, alpha=0.2, color="#3f8f52")
    a.plot(bx, by, "o", ms=6, mfc="white", mew=1.5, color="#1f5fa9",
           label="median per level bin")
    xs = np.geomspace(bx.min(), 0.6, 30)
    a.plot(xs, np.exp(nu_c) * xs ** nu, lw=2, color="#c2582a",
           label=rf"fit: $\propto |V|^{{{nu:.2f}}}$")
    a.plot(xs, xs ** (-4 / 3) * by[mm][-1] / bx[mm][-1] ** (-4 / 3), lw=1.4,
           ls="--", color="0.35", label=r"$|V|^{-4/3}$")
    a.set(xscale="log", yscale="log", xlabel=r"$|V|/\Gamma$",
          ylabel=r"contour diameter $\Lambda/\xi_0$",
          title=r"$\nu$: how contours grow towards percolation")
    a.grid(True, which="both", alpha=0.2); a.legend(frameon=False, fontsize=9)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=190)
    print("wrote", args.out)


def _fitc(xv, yv, slope):
    return np.mean(np.log(yv) - slope * np.log(xv))


if __name__ == "__main__":
    main()
