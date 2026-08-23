#!/usr/bin/env python3
"""D(tau) against the analytic regimes of the project notes.

    python scripts/plot_D_vs_tau.py --data data/D_vs_tau_pyramid_rc0.1_poisson.npz ...

Everything is drawn in the notes' convention <|dr|^2> = 2 D t (= twice the
standard 2-D coefficient), so the notes' formulas can be read straight off.
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

COLORS = ["#1f5fa9", "#c2582a", "#3f8f52", "#7b53a6"]


def load(path):
    d = np.load(path, allow_pickle=False)
    return dict(tau=d["tau"], D=d["D"], D_err=d["D_err"], r_c=float(d["r_c"]),
                xi=float(d["xi"]), T0=float(d["T0"]), ll=d["loglog_slope"],
                potential=str(d["potential"]), collisions=str(d["collisions"]),
                path=path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--out", default="figures/D_vs_tau.png")
    p.add_argument("--title", default=None)
    p.add_argument("--theory", default="notes",
                   choices=["notes", "none"],
                   help="overlay the notes' regimes (uses each run's xi, T0)")
    p.add_argument("--ratio", action="store_true",
                   help="add a lower panel of D_sim / D_notes")
    p.add_argument("--ratio-pair", action="store_true",
                   help="lower panel shows series 1 / series 2 instead")
    p.add_argument("--fit-power", action="store_true",
                   help="fit and draw D ~ tau^-alpha for non-pyramid series")
    args = p.parse_args()

    runs = [load(f) for f in args.data]

    if args.ratio:
        fig, (ax, axr) = plt.subplots(
            2, 1, figsize=(7.6, 7.6), sharex=True,
            gridspec_kw=dict(height_ratios=[2.6, 1.0], hspace=0.06),
            constrained_layout=True)
    else:
        fig, ax = plt.subplots(figsize=(7.6, 5.8), constrained_layout=True)
        axr = None

    for i, r in enumerate(runs):
        col = COLORS[i % len(COLORS)]
        u = r["tau"] / r["T0"]
        Dn = theory.to_note(r["D"])
        err = theory.to_note(r["D_err"])
        suffix = "" if r["collisions"] == "poisson" else r", fixed $\tau$"
        lbl = "{}, $r_c/\\xi={:g}${}".format(r["potential"], r["r_c"], suffix)
        ax.errorbar(u, Dn, yerr=err, fmt="o", ms=4.5, lw=1.1, capsize=2,
                    color=col, mfc="white", mew=1.3, zorder=3, label=lbl)

        if args.fit_power and r["potential"] != "pyramid":
            a, b = np.polyfit(np.log(u), np.log(Dn), 1)
            uu = np.geomspace(u.min(), u.max(), 50)
            ax.plot(uu, np.exp(b) * uu ** a, lw=1.3, ls="-", color=col,
                    alpha=0.55, zorder=2,
                    label=rf"fit: $D \propto \tau^{{{a:.2f}}}$")

        if args.theory == "notes" and r["potential"] == "pyramid":
            uu = np.geomspace(u.min(), u.max(), 400)
            tt = uu * r["T0"]
            xi, rc, t0 = r["xi"], r["r_c"], r["T0"]
            b1, b2 = theory.regime_boundaries(rc, xi, t0)
            ax.plot(uu, theory.D_piecewise(tt, rc, xi, t0), lw=1.3, ls="--",
                    color=col, alpha=0.75, zorder=2,
                    label="notes, 3-regime table" if i == 0 else None)
            ax.plot(uu, theory.D_eq7(tt, rc, xi, t0), lw=1.0, ls=":",
                    color=col, alpha=0.75, zorder=2,
                    label="notes, eq. (7)" if i == 0 else None)
            if i == 0:
                for b, txt in ((b1 / t0, r"$\tau/T_0=\pi r_c^2/16\xi^2$"),
                               (b2 / t0, r"$\tau/T_0=4/\pi$")):
                    if u.min() < b < u.max():
                        ax.axvline(b, color="0.75", lw=0.9, zorder=0)
                        ax.text(b * 1.08, 0.98, txt, transform=ax.get_xaxis_transform(),
                                fontsize=8, color="0.45", rotation=90, va="top")

        if axr is not None and not args.ratio_pair:
            pred = theory.D_piecewise(r["tau"], r["r_c"], r["xi"], r["T0"])
            axr.errorbar(u, Dn / pred, yerr=err / pred, fmt="o", ms=4, lw=1.0,
                         capsize=2, color=col, mfc="white", mew=1.2)

    if axr is not None and args.ratio_pair and len(runs) >= 2:
        a, b = runs[0], runs[1]
        ua, ub = a["tau"] / a["T0"], b["tau"] / b["T0"]
        # only compare where the two sweeps actually overlap in tau
        m = (ua >= ub.min() * 0.999) & (ua <= ub.max() * 1.001)
        Db = np.exp(np.interp(np.log(ua[m]), np.log(ub), np.log(b["D"])))
        axr.plot(ua[m], a["D"][m] / Db, "o-", ms=4.5, lw=1.2, color=COLORS[0],
                 mfc="white", mew=1.3)
        axr.set_yscale("log")

    ax.set(xscale="log", yscale="log",
           ylabel=r"$D$   (notes' convention, $\langle \Delta r^2\rangle = 2Dt$)")
    ax.grid(True, which="both", alpha=0.22, lw=0.6)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title(args.title or "Diffusion coefficient vs. mean free time",
                 fontsize=11.5)
    if axr is not None:
        axr.axhline(1.0, color="0.4", lw=1.0, ls="--")
        if args.ratio_pair:
            axr.set(xscale="log", xlabel=r"$\tau/T_0$",
                    ylabel="random / pyramid")
        else:
            axr.set(xscale="log", ylim=(0.5, 1.6),
                    xlabel=r"$\tau/T_0$", ylabel="sim / notes")
        axr.grid(True, which="both", alpha=0.22, lw=0.6)
    else:
        ax.set_xlabel(r"$\tau/T_0$")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=195)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
