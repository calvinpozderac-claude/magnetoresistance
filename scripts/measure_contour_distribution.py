#!/usr/bin/env python3
"""The contour a walker actually sits on, sampled uniformly in space.

The heuristics differ in one factor: the probability that a walker is on a
contour of size xi(eps).  Taking Prob = L w / xi^2 (the area of one contour's
tube inside a cell of size xi) gives P(Lambda_contour >= Lambda) ~ Lambda^-1;
taking it as the fraction of area with |V| < eps gives Lambda^(-1/nu) =
Lambda^-3/4.  Both are testable: label the level sets on the grid, look up the
cluster each random point belongs to, and histogram its size.

The same data give a parameter-free prediction for D(tau): a walker on a
contour of diameter Lambda and arclength L displaces Lambda if v tau >= L, and
Lambda (v tau / L)^(1/d_h) otherwise, so

    D(tau) = < min(Lambda, Lambda (v tau/L)^(1/d_h))^2 > / 4 tau .
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mrdiff import PeriodicGaussianField  # noqa: E402


def sample_contours(pot, n_pts, n_lev, rng):
    """For uniformly random points, the diameter and perimeter of the level-set
    component through each."""
    V = pot.V
    x, y = rng.uniform(0, pot.L, (2, n_pts))
    lev = pot.value(x, y)
    ix = (np.floor(x / pot.dx).astype(int)) % pot.n
    iy = (np.floor(y / pot.dx).astype(int)) % pot.n

    edges = np.quantile(np.abs(lev), np.linspace(0, 1, n_lev + 1))
    edges[0] = 0.0
    diam = np.full(n_pts, np.nan)
    per = np.full(n_pts, np.nan)
    for j in range(n_lev):
        lo, hi = edges[j], edges[j + 1]
        thr = 0.5 * (lo + hi)
        for sign in (+1, -1):
            sel = np.flatnonzero((np.abs(lev) >= lo) & (np.abs(lev) < hi)
                                 & (np.sign(lev) == sign))
            if sel.size == 0:
                continue
            mask = (V >= thr) if sign > 0 else (V <= -thr)
            lab, nlab = ndimage.label(mask)
            if nlab == 0:
                continue
            idx = lab.ravel()
            size = np.bincount(idx, minlength=nlab + 1).astype(float)
            ny, nx = mask.shape
            yy, xx = np.divmod(np.arange(idx.size), nx)
            sx = np.bincount(idx, weights=xx, minlength=nlab + 1)
            sy = np.bincount(idx, weights=yy, minlength=nlab + 1)
            sxx = np.bincount(idx, weights=xx * xx, minlength=nlab + 1)
            syy = np.bincount(idx, weights=yy * yy, minlength=nlab + 1)
            with np.errstate(invalid="ignore", divide="ignore"):
                cx, cy = sx / size, sy / size
                rg2 = (sxx / size - cx ** 2) + (syy / size - cy ** 2)
            b = np.zeros(nlab + 1)
            for sh, ax in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
                edge = mask & ~np.roll(mask, sh, axis=ax)
                b += np.bincount(lab[edge], minlength=nlab + 1)
            lab_pt = lab[ix[sel], iy[sel]]
            ok = lab_pt > 0
            s = sel[ok]
            diam[s] = 2.0 * np.sqrt(np.maximum(rg2[lab_pt[ok]], 0)) * pot.dx
            per[s] = b[lab_pt[ok]] * pot.dx
    good = np.isfinite(diam) & (diam > 0)
    return diam[good], per[good]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=float, default=400.0)
    p.add_argument("--n-pts", type=int, default=60000)
    p.add_argument("--n-lev", type=int, default=24)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--dh", type=float, default=1.75)
    p.add_argument("--out", default="data/contour_distribution.npz")
    args = p.parse_args()

    D, P = [], []
    for sd in range(args.seeds):
        pot = PeriodicGaussianField(xi0=1.0, Gamma=1 / np.sqrt(2), L=args.L,
                                    dx=0.2, seed=1200 + sd, precompute=False)
        d, q = sample_contours(pot, args.n_pts, args.n_lev,
                               np.random.default_rng(sd))
        D.append(d); P.append(q)
        print(f"  seed {sd}: {d.size} contours, median diameter {np.median(d):.2f}",
              flush=True)
    D = np.concatenate(D); P = np.concatenate(P)

    print("\nTail of the contour-size distribution through a random point:")
    print("  P(Lambda_c >= Lambda)   ~ Lambda^(-a)")
    xs = np.geomspace(2.0, np.quantile(D, 0.999), 14)
    cdf = np.array([(D >= t).mean() for t in xs])
    m = cdf > 3e-4
    a = np.polyfit(np.log(xs[m]), np.log(cdf[m]), 1)[0]
    for t, c in zip(xs[m], cdf[m]):
        print(f"    Lambda >= {t:7.2f}   P = {c:.5f}")
    print(f"  fitted a = {-a:.3f}    [tube counting L w/xi^2 -> 1.00;"
          f"  level-layer -> 1/nu = 0.75]")

    np.savez(args.out, diam=D, per=P)
    print("\nParameter-free D(tau) from these contours (v0 = 1, xi0 = 1):")
    print(f"{'tau':>8} {'D_pred':>10} {'frac completing':>16}")
    for tau in (10.0, 30.0, 100.0, 300.0, 600.0):
        s = tau  # v0 = 1
        disp = np.where(P <= s, D, D * (s / P) ** (1 / args.dh))
        print(f"{tau:8.4g} {np.mean(disp ** 2) / (4 * tau):10.5f} "
              f"{(P <= s).mean():16.3f}")
    t = np.array([30., 100., 300., 600.])
    dp = np.array([np.mean(np.where(P <= s, D, D * (s / P) ** (1 / args.dh)) ** 2)
                   / (4 * s) for s in t])
    print(f"  fitted exponent over tau = 30-600: {np.polyfit(np.log(t), np.log(dp), 1)[0]:+.3f}"
          f"   [theirs -3/7 = -0.429, mine -2/7 = -0.286]")


if __name__ == "__main__":
    main()
