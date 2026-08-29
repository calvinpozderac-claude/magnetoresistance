#!/usr/bin/env python3
r"""D as a function of r_c inside regime 3.

Regime 3 is the window   xi_0 (xi_0/v_0 tau)^(3/7)  <  r_c  <  sqrt(v_0 xi_0 tau)
in which the notes predict D ~ v_0 xi_0 (xi_0/v_0 tau)^(3/7), i.e. D independent
of r_c.  With xi_0 = v_0 = 1 the window is tau^(-3/7) < r_c < sqrt(tau); it opens
at the triple point tau = 1 and widens as tau^(13/14).

Panel (a) is the raw D(r_c) at fixed tau, with the Case-3 window of each curve
marked; panel (b) divides out the predicted tau^(-3/7) so every Case-3 point
should fall on one horizontal line.

    python scripts/plot_D_vs_rc_case3.py --data data/grid*.npz
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def merge(paths):
    """Union several run_grid.py outputs; later files win on shared cells."""
    grids = [np.load(q) for q in paths]
    rc = np.unique(np.concatenate([g["r_c"] for g in grids]))
    tau = np.unique(np.concatenate([g["tau"] for g in grids]))
    shape = (rc.size, tau.size)
    D, E, LL = (np.full(shape, np.nan) for _ in range(3))
    done = np.zeros(shape, dtype=bool)
    for g in grids:
        ii = np.searchsorted(rc, g["r_c"])
        jj = np.searchsorted(tau, g["tau"])
        for a, i in enumerate(ii):
            for b, j in enumerate(jj):
                if g["done"][a, b]:
                    D[i, j], E[i, j] = g["D"][a, b], g["D_err"][a, b]
                    LL[i, j], done[i, j] = g["loglog"][a, b], True
    return rc, tau, D, E, LL, done


def wpowerfit(x, y, sy):
    """Weighted least-squares slope of log y vs log x, with its error."""
    lx, ly, w = np.log(x), np.log(y), (y / np.maximum(sy, 1e-12)) ** 2
    W = w.sum()
    mx, my = (w * lx).sum() / W, (w * ly).sum() / W
    sxx = (w * (lx - mx) ** 2).sum()
    b = (w * (lx - mx) * (ly - my)).sum() / sxx
    # scale the formal error by the scatter, so a bad fit reports honestly
    res = ly - my - b * (lx - mx)
    chi2 = (w * res ** 2).sum() / max(lx.size - 2, 1)
    return b, np.sqrt(max(chi2, 1.0) / sxx), np.exp(my - b * mx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", default=sorted(glob.glob("data/grid*.npz")))
    p.add_argument("--out", default="figures/D_vs_rc_case3.png")
    p.add_argument("--ll-tol", type=float, default=0.15)
    p.add_argument("--split", type=float, default=1.0,
                   help="r_c at which to break the regime-3 fit (in xi_0)")
    p.add_argument("--margin", type=float, default=0.0,
                   help="ln-distance a point must keep from both window edges")
    args = p.parse_args()

    rc, tau, D, E, LL, done = merge(args.data)
    done &= np.abs(LL - 1.0) <= args.ll_tol
    D, E = np.where(done, D, np.nan), np.where(done, E, np.nan)

    R, T = np.meshgrid(rc, tau, indexing="ij")
    lo, hi = T ** (-3.0 / 7.0), np.sqrt(T)          # window edges
    inside = done & (np.log(R / lo) > args.margin) & (np.log(hi / R) > args.margin)

    taus = [t for j, t in enumerate(tau) if inside[:, j].sum() >= 3]
    cmap = plt.get_cmap("viridis")
    cols = {t: cmap(k / max(len(taus) - 1, 1)) for k, t in enumerate(taus)}

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12.4, 5.3))

    # ---------------- (a) raw D(r_c) --------------------------------------
    for t in taus:
        j = int(np.searchsorted(tau, t))
        m = done[:, j]
        c = cols[t]
        ax.plot(rc[m], D[m, j], "-", color=c, lw=1.0, alpha=0.45, zorder=2)
        out = m & ~inside[:, j]
        ax.plot(rc[out], D[out, j], "o", color=c, ms=5, mfc="none", mew=1.1,
                zorder=3)
        ins = inside[:, j]
        ax.errorbar(rc[ins], D[ins, j], yerr=E[ins, j], fmt="o", color=c, ms=6,
                    capsize=2, lw=1.0, zorder=4,
                    label=fr"$\tau={t:g}$")
        # mark this curve's Case-3 window on the r_c axis
        ax.plot([lo[0, j], hi[0, j]], [D[ins, j].min() * 0.55] * 2, "-",
                color=c, lw=3.0, alpha=0.65, solid_capstyle="butt", zorder=1)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$r_c\ \ [\xi_0]$")
    ax.set_ylabel(r"$D\ \ [v_0\xi_0]$")
    ax.set_title("(a) $D(r_c)$ at fixed $\\tau$.  Filled = inside regime 3,\n"
                 "open = outside; bars under each curve span its regime-3 window",
                 fontsize=10)
    # guide slopes anchored to the largest-tau curve
    j = int(np.searchsorted(tau, taus[-1]))
    ins = inside[:, j]
    x0, y0 = rc[ins][0], D[ins, j][0]
    xg = np.array([x0, x0 * 6])
    ax.plot(xg, y0 * 1.7 * (xg / x0) ** 0.0, "k-", lw=1.2)
    ax.text(xg[1] * 1.1, y0 * 1.7, r"$r_c^{0}$ (Case 3)", fontsize=9, va="center")
    ax.plot(xg, y0 * 0.42 * (xg / x0) ** (6 / 13), "k--", lw=1.2)
    ax.text(xg[1] * 1.1, y0 * 0.42 * 6 ** (6 / 13), r"$r_c^{6/13}$ (Case 2)",
            fontsize=9, va="center")
    ax.legend(fontsize=8, loc="upper left", ncol=2)

    # ---------------- (b) scaled: D tau^(3/7) ------------------------------
    allx, ally, alle = [], [], []
    for t in taus:
        j = int(np.searchsorted(tau, t))
        ins = inside[:, j]
        y = D[ins, j] * t ** (3.0 / 7.0)
        e = E[ins, j] * t ** (3.0 / 7.0)
        bx.errorbar(rc[ins], y, yerr=e, fmt="o-", color=cols[t], ms=6, lw=1.0,
                    capsize=2, label=fr"$\tau={t:g}$")
        allx.append(rc[ins]); ally.append(y); alle.append(e)
    allx, ally, alle = (np.concatenate(v) for v in (allx, ally, alle))
    # the data break at r_c ~ xi_0, which is neither predicted boundary, so fit
    # the two sides separately as well as the whole regime
    lowm, higm = allx <= args.split, allx >= 2 * args.split
    for sel, style, tag in ((lowm, "k-", r"r_c\lesssim\xi_0"),
                            (higm, "k-.", r"r_c\gtrsim\xi_0")):
        if sel.sum() < 3:
            continue
        b, eb, amp = wpowerfit(allx[sel], ally[sel], alle[sel])
        xx = np.logspace(np.log10(allx[sel].min()), np.log10(allx[sel].max()), 20)
        bx.plot(xx, amp * xx ** b, style, lw=1.7, zorder=6,
                label=f"${tag}$:  $r_c^{{{b:+.3f}\\pm{eb:.3f}}}$")
    bx.axvline(args.split, color="0.35", lw=1.2, ls=":")
    bx.text(args.split * 1.07, 0.04, r"$r_c=\xi_0$", fontsize=9, color="0.3",
            va="bottom", ha="left", transform=bx.get_xaxis_transform())
    b, eb, amp = wpowerfit(allx, ally, alle)
    bx.axhline(np.exp(np.mean(np.log(ally[lowm]))), color="0.45", ls="--", lw=1.2,
               label=r"constant (Case 3 prediction, $r_c^0$)")
    bx.set_xscale("log"); bx.set_yscale("log")
    bx.set_xlabel(r"$r_c\ \ [\xi_0]$")
    bx.set_ylabel(r"$D\,(v_0\tau/\xi_0)^{3/7}/(v_0\xi_0)$")
    bx.set_title(r"(b) regime-3 points only, scaled by the predicted $\tau^{-3/7}$"
                 "\n" r"Case 3 says this is a single $r_c$-independent constant",
                 fontsize=10)
    bx.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=170)
    print("wrote", args.out)

    # ---------------- numbers ---------------------------------------------
    print(f"\nregime-3 window  tau^(-3/7) < r_c < sqrt(tau), margin={args.margin:g}")
    print(f"{'tau':>8} {'window':>18} {'n':>3}  d lnD/d ln r_c")
    for t in taus:
        j = int(np.searchsorted(tau, t))
        ins = inside[:, j]
        s = ""
        if ins.sum() >= 3:
            bb, ee, _ = wpowerfit(rc[ins], D[ins, j], E[ins, j])
            s = f"{bb:+.3f} +- {ee:.3f}"
        print(f"{t:8g} {lo[0,j]:8.3f}..{hi[0,j]:<8.2f} {int(ins.sum()):3d}  {s}")
    print(f"\nall regime-3 points together ({allx.size}): "
          f"d lnD/d ln r_c = {b:+.3f} +- {eb:.3f}   (predicted 0)")
    print(f"split at r_c = {args.split:g} xi_0:")
    for lab, sel in ((f"r_c <= {args.split:g}", lowm),
                     (f"r_c >= {2*args.split:g}", higm)):
        if sel.sum() < 3:
            continue
        bb, ee, _ = wpowerfit(allx[sel], ally[sel], alle[sel])
        print(f"   {lab:>12}:  n={int(sel.sum()):3d}   "
              f"d lnD/d ln r_c = {bb:+.3f} +- {ee:.3f}   "
              f"<D tau^3/7> = {np.exp(np.mean(np.log(ally[sel]))):.3f}")
    print("sub-xi_0 branch, tau by tau (this is the one that must flatten):")
    for t in taus:
        j = int(np.searchsorted(tau, t))
        sel = inside[:, j] & (rc <= args.split)
        if sel.sum() < 3:
            continue
        bb, ee, _ = wpowerfit(rc[sel], D[sel, j], E[sel, j])
        print(f"   tau={t:6g}:  n={int(sel.sum()):2d}   {bb:+.3f} +- {ee:.3f}")
    for cut in (0.0, 0.7, 1.2, 1.8):
        sel = done & (np.log(R / lo) > cut) & (np.log(hi / R) > cut)
        if sel.sum() < 4:
            continue
        bb, ee, _ = wpowerfit(R[sel], D[sel] * T[sel] ** (3 / 7),
                              E[sel] * T[sel] ** (3 / 7))
        print(f"   margin > {cut:.1f}:  n={int(sel.sum()):3d}   "
              f"d lnD/d ln r_c = {bb:+.3f} +- {ee:.3f}")


if __name__ == "__main__":
    main()
