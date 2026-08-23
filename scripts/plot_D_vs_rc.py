#!/usr/bin/env python3
"""Log-log plot of D(r_c) from a sweep produced by scripts/run_pyramid.py."""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def power_law_exponent(x, y, lo, hi):
    """Local log-log slope fitted over x in [lo, hi]."""
    m = (x >= lo) & (x <= hi) & (y > 0)
    if m.sum() < 2:
        return np.nan
    return np.polyfit(np.log(x[m]), np.log(y[m]), 1)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/D_vs_rc_pyramid.npz")
    p.add_argument("--out", default="figures/D_vs_rc_pyramid.png")
    args = p.parse_args()

    d = np.load(args.data)
    rc, D, D_err = d["r_c"], d["D"], d["D_err"]
    tau, B, a = float(d["tau"]), float(d["B"]), float(d["a"])

    n_small = max(3, int((rc < 0.2 * a).sum()))
    lo_exp = power_law_exponent(rc, D, rc.min(), rc[n_small - 1])
    hi_exp = power_law_exponent(rc, D, 2.0 * a, rc.max())
    # prefactor of the small-r_c law D = C a^2 r_c / tau
    C = np.median(D[:n_small] * tau / (a ** 2 * rc[:n_small]))

    fig, ax = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)

    # asymptotes
    x_lo = np.geomspace(rc.min() * 0.7, 1.5 * a, 50)
    ax.plot(x_lo, C * a ** 2 * x_lo / tau, ls="--", lw=1.3, color="0.45",
            zorder=1, label=rf"$D = {C:.2f}\,a^2 r_c/\tau$   (slope 1)")
    x_hi = np.geomspace(0.6 * a, rc.max() * 1.4, 50)
    ax.plot(x_hi, x_hi ** 2 / (4 * tau), ls=":", lw=1.6, color="0.25", zorder=1,
            label=r"$D = r_c^2/4\tau$   (free walk, slope 2)")
    ax.axvline(a, color="0.8", lw=1.0, zorder=0)
    ax.text(a * 1.04, D.min() * 1.15, "$r_c = a$", color="0.5", fontsize=9)

    ax.errorbar(rc, D, yerr=D_err, fmt="o", ms=5.5, lw=1.2, capsize=2.5,
                color="#1f5fa9", mfc="white", mew=1.4, zorder=3,
                label="simulation")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"cyclotron radius  $r_c$   (units of pyramid size $a$)")
    ax.set_ylabel(r"diffusion coefficient  $D$   (units of $a^2/\tau$)")
    ax.set_title("Guiding-centre diffusion on a square-pyramid potential\n"
                 rf"$B={B:g}$, $\tau={tau:g}$, $V_0={float(d['V0']):g}$, "
                 rf"$a={a:g}$", fontsize=11)
    ax.grid(True, which="both", alpha=0.25, lw=0.6)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")

    txt = (rf"fitted slopes:  $r_c \ll a$: {lo_exp:.2f}" "\n"
           rf"$\qquad\qquad\quad\ \ r_c \gg a$: {hi_exp:.2f}")
    ax.text(0.98, 0.05, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.5, color="0.25")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(f"small-r_c exponent {lo_exp:.3f} (prefactor C={C:.3f}), "
          f"large-r_c exponent {hi_exp:.3f}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
