#!/usr/bin/env python3
"""D(tau) on the Gaussian random potential, on its own axes.

    python scripts/plot_random_D_vs_tau.py --data data/D_vs_tau_random_full.npz

Draws the measured D against the collision-limited asymptote r_c^2/2 tau and
fits a power law in each regime window.
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

from mrdiff import theory  # noqa: E402


def fit_window(tau, D, lo, hi):
    m = (tau >= lo) & (tau <= hi)
    if m.sum() < 2:
        return None
    c = np.polyfit(np.log(tau[m]), np.log(D[m]), 1)
    r = np.log(D[m]) - np.polyval(c, np.log(tau[m]))
    dof = max(m.sum() - 2, 1)
    se = np.sqrt(np.sum(r ** 2) / dof
                 / np.sum((np.log(tau[m]) - np.log(tau[m]).mean()) ** 2))
    return c, se, m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", default="figures/D_vs_tau_random_only.png")
    p.add_argument("--windows", nargs="*", type=float,
                   default=[1e-4, 3e-3, 0.3, 30.0, 100.0, 1e9],
                   help="pairs of (lo, hi) tau bounds, one pair per regime fit")
    p.add_argument("--period", type=float, default=None,
                   help="median orbital period, drawn as a vertical marker")
    args = p.parse_args()

    tau = np.concatenate([np.load(f)["tau"] for f in args.data])
    D = np.concatenate([np.load(f)["D"] for f in args.data])
    err = np.concatenate([np.load(f)["D_err"] for f in args.data])
    ll = np.concatenate([np.load(f)["loglog_slope"] for f in args.data])
    d0 = np.load(args.data[0])
    r_c, xi, T0 = float(d0["r_c"]), float(d0["xi"]), float(d0["T0"])
    o = np.argsort(tau)
    tau, D, err, ll = tau[o], D[o], err[o], ll[o]
    Dn, errn = theory.to_note(D), theory.to_note(err)

    fig, ax = plt.subplots(figsize=(8.0, 6.0), constrained_layout=True)

    # collision-limited asymptote: the walk is just r_c hops, the drift is idle.
    # Drawn only where it is within reach of the data, so the plot is not
    # stretched over the decades where it has long since been left behind.
    x = np.geomspace(tau.min() * 0.6,
                     min(tau.max() * 1.6, 30.0 * r_c ** 2 / Dn.min()), 60)
    ax.plot(x, theory.D_collision(x, r_c), ls="--", lw=1.4, color="0.35",
            zorder=1, label=r"$D = r_c^2/2\tau$   (free walk, slope $-1$)")

    cols = ["#c2582a", "#3f8f52", "#7b53a6"]
    pairs = list(zip(args.windows[::2], args.windows[1::2]))
    for k, (lo, hi) in enumerate(pairs):
        f = fit_window(tau, Dn, lo, hi)
        if f is None:
            continue
        c, se, m = f
        xx = np.geomspace(tau[m].min(), tau[m].max(), 30)
        ax.plot(xx, np.exp(np.polyval(c, np.log(xx))), lw=2.4,
                color=cols[k % len(cols)], alpha=0.5, zorder=2,
                label=rf"$\tau/T_0\in[{lo:g},\,{min(hi, tau.max()):g}]$: "
                      rf"$D\propto\tau^{{{c[0]:.2f}\pm{se:.2f}}}$")

    if args.period:
        ax.axvline(args.period, color="0.7", lw=1.0, zorder=0)
        ax.annotate(r"median orbit period", xy=(args.period * 1.1, 0.04),
                    xycoords=("data", "axes fraction"), fontsize=8.5,
                    color="0.45", rotation=90, va="bottom")

    good = np.abs(ll - 1.0) <= 0.1
    ax.errorbar(tau[good] / T0, Dn[good], yerr=errn[good], fmt="o", ms=6, lw=1.2,
                capsize=2.5, color="#1f5fa9", mfc="white", mew=1.5, zorder=4,
                label="simulation")
    if (~good).any():
        ax.errorbar(tau[~good] / T0, Dn[~good], yerr=errn[~good], fmt="s", ms=5.5,
                    lw=1.2, capsize=2.5, color="#1f5fa9", mfc="#c9d8ea", mew=1.2,
                    zorder=4,
                    label=r"$|d\ln\,\mathrm{MSD}/d\ln t - 1| > 0.1$")

    ax.set(xscale="log", yscale="log", ylim=(0.4 * Dn.min(), 3.0 * Dn.max()),
           xlabel=r"$\tau/T_0$",
           ylabel=r"$D$   (notes' convention, $\langle\Delta r^2\rangle = 2Dt$)")
    ax.set_title("Gaussian random potential: diffusion coefficient vs. mean "
                 "free time\n"
                 rf"$\xi_0=1$, $r_c/\xi_0={r_c:g}$, $T_0=\xi_0/v_d=1$, "
                 r"Poisson collisions", fontsize=11)
    ax.grid(True, which="both", alpha=0.22, lw=0.6)
    ax.legend(frameon=False, fontsize=9.5, loc="lower left")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=195)
    print("wrote", args.out)
    for lo, hi in pairs:
        f = fit_window(tau, Dn, lo, hi)
        if f:
            print(f"  tau in [{lo:g}, {hi:g}]: exponent {f[0][0]:+.3f} +- {f[1]:.3f}")


if __name__ == "__main__":
    main()
