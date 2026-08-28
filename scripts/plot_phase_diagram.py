#!/usr/bin/env python3
r"""Map the (r_c, tau) plane and test the three-regime picture.

With xi0 = v0 = 1 the notes predict

    Case 1   r_c > sqrt(tau)                     D = c r_c^2 / tau
    Case 2   r_c < sqrt(tau), r_c < tau^(-3/7)   D ~ (r_c^2/tau)^(3/13)
    Case 3   tau^(-3/7) < r_c < sqrt(tau)        D ~ tau^(-3/7)

so the boundaries are r_c = sqrt(v0 xi0 tau) and r_c = xi0 (xi0/v0 tau)^(3/7),
crossing at the triple point (tau, r_c) = (1, 1).

Cases 1 and 2 both make D a function of the single variable u = r_c^2/(v0 xi0 tau);
Cases 2 and 3 both make D tau^(3/7) a function of the single variable
X = r_c (v0 tau/xi0)^(3/7).  Those two collapses use every cell at once and are
far less noise-sensitive than differentiating cell by cell.

    python scripts/plot_phase_diagram.py --data data/grid_D.npz data/grid_hi_*.npz
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
from matplotlib.colors import LogNorm, Normalize  # noqa: E402

SLOPE = {1: -1.0, 2: -3.0 / 13.0, 3: -3.0 / 7.0}


def merge(paths):
    """Union several run_grid.py outputs; later files win on shared cells."""
    grids = [np.load(q) for q in paths]
    rc = np.unique(np.concatenate([g["r_c"] for g in grids]))
    tau = np.unique(np.concatenate([g["tau"] for g in grids]))
    D = np.full((rc.size, tau.size), np.nan)
    E = np.full((rc.size, tau.size), np.nan)
    done = np.zeros(D.shape, dtype=bool)
    for g in grids:
        ii = np.searchsorted(rc, g["r_c"])
        jj = np.searchsorted(tau, g["tau"])
        for a, i in enumerate(ii):
            for b, j in enumerate(jj):
                if g["done"][a, b]:
                    D[i, j], E[i, j], done[i, j] = g["D"][a, b], g["D_err"][a, b], True
    return rc, tau, D, E, done


def window_slope(logD, logx, axis, half=1):
    """d logD / d logx by least squares over a +-half window along `axis`."""
    out = np.full(logD.shape, np.nan)
    n = logD.shape[axis]
    for k in range(n):
        lo, hi = max(k - half, 0), min(k + half, n - 1)
        if hi - lo < 1:
            continue
        sl = [slice(None)] * logD.ndim
        sl[axis] = slice(lo, hi + 1)
        y = np.moveaxis(logD[tuple(sl)], axis, -1)
        x = logx[lo:hi + 1]
        good = np.isfinite(y)
        m = good.sum(-1)
        xm = np.where(m > 1, (np.where(good, x, 0.0)).sum(-1) / np.maximum(m, 1), np.nan)
        ym = np.where(m > 1, (np.where(good, y, 0.0)).sum(-1) / np.maximum(m, 1), np.nan)
        dx = np.where(good, x - xm[..., None], 0.0)
        dy = np.where(good, y - ym[..., None], 0.0)
        s = np.where(m > 1, (dx * dy).sum(-1) / np.where((dx * dx).sum(-1) > 0,
                                                         (dx * dx).sum(-1), np.nan), np.nan)
        idx = [slice(None)] * logD.ndim
        idx[axis] = k
        out[tuple(idx)] = s
    return out


def edges(a):
    la = np.log10(a)
    e = np.empty(la.size + 1)
    e[1:-1] = 0.5 * (la[1:] + la[:-1])
    e[0] = la[0] - 0.5 * (la[1] - la[0])
    e[-1] = la[-1] + 0.5 * (la[-1] - la[-2])
    return 10.0 ** e


def powerfit(x, y, w=None):
    """Least-squares slope of log y vs log x."""
    lx, ly = np.log(x), np.log(y)
    if w is None:
        w = np.ones_like(lx)
    W = w.sum()
    mx, my = (w * lx).sum() / W, (w * ly).sum() / W
    sxx = (w * (lx - mx) ** 2).sum()
    slope = (w * (lx - mx) * (ly - my)).sum() / sxx
    resid = ly - my - slope * (lx - mx)
    dof = max(lx.size - 2, 1)
    err = np.sqrt((w * resid ** 2).sum() / W / sxx * lx.size / dof)
    return slope, err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", default=["data/grid_D.npz"])
    p.add_argument("--out", default="figures/phase_diagram.png")
    args = p.parse_args()

    rc, tau, D, E, done = merge(args.data)
    D = np.where(done, D, np.nan)
    E = np.where(done, E, np.nan)
    print(f"{int(done.sum())}/{done.size} cells present "
          f"({rc.size} r_c x {tau.size} tau)")

    s_tau = window_slope(np.log(D), np.log(tau), axis=1)
    s_rc = window_slope(np.log(D), np.log(rc), axis=0)

    R, T = np.meshgrid(rc, tau, indexing="ij")
    b12 = R > np.sqrt(T)                      # Case 1
    b23 = (~b12) & (R < T ** (-3.0 / 7.0))    # Case 2
    pred = np.where(b12, 1, np.where(b23, 2, 3))

    tt = np.logspace(np.log10(tau[0]) - 0.4, np.log10(tau[-1]) + 0.4, 200)
    te, re = edges(tau), edges(rc)

    def frame(ax, title, legend=False):
        ax.set_xscale("log"); ax.set_yscale("log")
        for style, lab in ((("k--", 1.4), r"$r_c=\sqrt{v_0\xi_0\tau}$"),):
            ax.plot(tt, np.sqrt(tt), "w-", lw=2.6)
            ax.plot(tt, np.sqrt(tt), style[0], lw=style[1], label=lab)
        ax.plot(tt, tt ** (-3.0 / 7.0), "w-", lw=2.6)
        ax.plot(tt, tt ** (-3.0 / 7.0), "k:", lw=1.8,
                label=r"$r_c=\xi_0(\xi_0/v_0\tau)^{3/7}$")
        ax.plot([1.0], [1.0], "o", ms=6, mfc="w", mec="k", mew=1.5)
        ax.set_xlim(tau[0] / 1.6, tau[-1] * 1.6)
        ax.set_ylim(rc[0] / 1.6, rc[-1] * 1.6)
        ax.set_xlabel(r"$\tau\ \ [\xi_0/v_0]$")
        ax.set_title(title, fontsize=10)
        for i in range(rc.size):
            for j in range(tau.size):
                if done[i, j]:
                    ax.text(tau[j], rc[i], "123"[pred[i, j] - 1], ha="center",
                            va="center", fontsize=7, color="0.25", alpha=0.9)
        if legend:
            ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 9.4))

    # ---- (a) D itself ------------------------------------------------------
    ax = axes[0, 0]
    m = ax.pcolormesh(te, re, D, cmap="cividis", norm=LogNorm(), shading="flat")
    fig.colorbar(m, ax=ax).set_label(r"$D\ \ [v_0\xi_0]$")
    frame(ax, r"(a) measured $D$  (digits = predicted regime)", legend=True)
    ax.set_ylabel(r"$r_c\ \ [\xi_0]$")

    # ---- (b) local tau exponent -------------------------------------------
    ax = axes[0, 1]
    m = ax.pcolormesh(te, re, s_tau, cmap="viridis", vmin=-1.05, vmax=-0.1,
                      shading="flat")
    cb = fig.colorbar(m, ax=ax)
    cb.set_label(r"$d\ln D/d\ln\tau$   (3-point fit)")
    for v, lab in ((-1.0, "-1"), (-3 / 7, "-3/7"), (-3 / 13, "-3/13")):
        cb.ax.axhline(v, color="r", lw=1.1)
        cb.ax.text(1.7, v, lab, color="r", va="center", fontsize=8)
    frame(ax, "(b) local $\\tau$ exponent\n"
              r"Case 1 $=-1$,  Case 2 $=-3/13$,  Case 3 $=-3/7$")

    # ---- (c) collapse of Cases 1 and 2:  D = D(r_c^2/tau) -----------------
    ax = axes[1, 0]
    u = R ** 2 / T
    keep = np.isfinite(D) & (pred != 3)
    sc = ax.scatter(u[keep], D[keep], c=T[keep], norm=LogNorm(), cmap="plasma",
                    s=34, edgecolor="k", linewidth=0.3, zorder=3)
    fig.colorbar(sc, ax=ax).set_label(r"$\tau$")
    ax.set_xscale("log"); ax.set_yscale("log")
    k1 = keep & (pred == 1)
    k2 = keep & (pred == 2)
    if k1.sum() > 2:
        a1, e1 = powerfit(u[k1], D[k1])
        c1 = np.exp(np.mean(np.log(D[k1]) - a1 * np.log(u[k1])))
        xx = np.logspace(np.log10(u[k1].min()), np.log10(u[k1].max()), 20)
        ax.plot(xx, c1 * xx ** a1, "k-", lw=1.4,
                label=f"Case 1 fit  {a1:.3f}$\\pm${e1:.3f}  (=1)")
    if k2.sum() > 2:
        a2, e2 = powerfit(u[k2], D[k2])
        c2 = np.exp(np.mean(np.log(D[k2]) - a2 * np.log(u[k2])))
        xx = np.logspace(np.log10(u[k2].min()), np.log10(u[k2].max()), 20)
        ax.plot(xx, c2 * xx ** a2, "k--", lw=1.4,
                label=f"Case 2 fit  {a2:.3f}$\\pm${e2:.3f}  (=3/13={3/13:.3f})")
    ax.axvline(1.0, color="0.5", lw=1.0, ls=":")
    ax.set_xlabel(r"$u=r_c^2/(v_0\xi_0\tau)$")
    ax.set_ylabel(r"$D\ \ [v_0\xi_0]$")
    ax.set_title("(c) Cases 1+2 collapse: $D$ depends on $r_c^2/\\tau$ alone",
                 fontsize=10)
    ax.legend(fontsize=8, loc="upper left")

    # ---- (d) collapse of Cases 2 and 3 ------------------------------------
    ax = axes[1, 1]
    X = R * T ** (3.0 / 7.0)
    Y = D * T ** (3.0 / 7.0)
    keep = np.isfinite(D) & (pred != 1)
    sc = ax.scatter(X[keep], Y[keep], c=T[keep], norm=LogNorm(), cmap="plasma",
                    s=34, edgecolor="k", linewidth=0.3, zorder=3)
    fig.colorbar(sc, ax=ax).set_label(r"$\tau$")
    ax.set_xscale("log"); ax.set_yscale("log")
    k2 = keep & (pred == 2)
    k3 = keep & (pred == 3)
    if k2.sum() > 2:
        a2, e2 = powerfit(X[k2], Y[k2])
        c2 = np.exp(np.mean(np.log(Y[k2]) - a2 * np.log(X[k2])))
        xx = np.logspace(np.log10(X[k2].min()), np.log10(X[k2].max()), 20)
        ax.plot(xx, c2 * xx ** a2, "k--", lw=1.4,
                label=f"Case 2 fit  {a2:.3f}$\\pm${e2:.3f}  (=6/13={6/13:.3f})")
    if k3.sum() > 2:
        a3, e3 = powerfit(X[k3], Y[k3])
        c3 = np.exp(np.mean(np.log(Y[k3]) - a3 * np.log(X[k3])))
        xx = np.logspace(np.log10(X[k3].min()), np.log10(X[k3].max()), 20)
        ax.plot(xx, c3 * xx ** a3, "k-", lw=1.4,
                label=f"Case 3 fit  {a3:.3f}$\\pm${e3:.3f}  (=0)")
    ax.axvline(1.0, color="0.5", lw=1.0, ls=":")
    ax.set_xlabel(r"$X=r_c\,(v_0\tau/\xi_0)^{3/7}/\xi_0$")
    ax.set_ylabel(r"$D\,(v_0\tau/\xi_0)^{3/7}/(v_0\xi_0)$")
    ax.set_title("(d) Cases 2+3 collapse: $D\\tau^{3/7}$ depends on "
                 "$r_c\\tau^{3/7}$ alone", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=165)
    print("wrote", args.out)

    # ---------------- text tables ------------------------------------------
    def table(name, M, fmt="{:10.4g}"):
        print(f"\n{name}")
        print("r_c \\ tau " + "".join(f"{t:>10.3g}" for t in tau))
        for i, r in enumerate(rc):
            print(f"{r:9.3g} " + "".join(
                "       ---" if not np.isfinite(M[i, j]) else fmt.format(M[i, j])
                for j in range(tau.size)))

    table("D(r_c, tau)", D)
    table("d lnD / d ln tau", s_tau, "{:10.3f}")
    table("d lnD / d ln r_c", s_rc, "{:10.3f}")

    print("\nmean exponents inside each predicted regime "
          "(cells strictly inside, i.e. not on a boundary):")
    for c in (1, 2, 3):
        sel = np.isfinite(s_tau) & (pred == c)
        # keep only cells whose whole 3-point tau window stays in the regime
        inner = np.zeros_like(sel)
        for i in range(rc.size):
            for j in range(tau.size):
                lo, hi = max(j - 1, 0), min(j + 1, tau.size - 1)
                inner[i, j] = sel[i, j] and (pred[i, lo:hi + 1] == c).all()
        if inner.sum():
            v = s_tau[inner]
            print(f"  Case {c}: <d lnD/d ln tau> = {v.mean():+.3f} "
                  f"+- {v.std(ddof=1)/np.sqrt(v.size):.3f}  "
                  f"(n={v.size}, predicted {SLOPE[c]:+.3f})")
        selr = np.isfinite(s_rc) & (pred == c)
        innerr = np.zeros_like(selr)
        for i in range(rc.size):
            for j in range(tau.size):
                lo, hi = max(i - 1, 0), min(i + 1, rc.size - 1)
                innerr[i, j] = selr[i, j] and (pred[lo:hi + 1, j] == c).all()
        if innerr.sum():
            v = s_rc[innerr]
            pr = {1: 2.0, 2: 6 / 13, 3: 0.0}[c]
            print(f"           <d lnD/d ln r_c> = {v.mean():+.3f} "
                  f"+- {v.std(ddof=1)/np.sqrt(v.size):.3f}  "
                  f"(n={v.size}, predicted {pr:+.3f})")


if __name__ == "__main__":
    main()
