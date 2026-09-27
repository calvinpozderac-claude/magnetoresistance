#!/usr/bin/env python3
r"""Phase diagrams in lab units for the raw and the orbit-averaged landscape.

Every boundary is the same condition in the landscape's own units,

    1 <-> 2   r_c = sqrt(v_eff xi_eff tau)        i.e. tau = r_c^2/(v_eff xi_eff)
    2 <-> 3   r_c = xi_eff (xi_eff/v_eff tau)^(3/7)  i.e. tau = (xi_eff/v_eff)(xi_eff/r_c)^(7/3)
    3a <-> 3b r_c = xi_eff

but for the averaged landscape xi_eff and v_eff are themselves functions of r_c
(measured in data/avg_landscape.json), so the lines bend: v_eff ~ r_c^(-1/2)
makes the drift-dominated wedge close up at large r_c.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from analyse_regimes import load, regime, RAW_V, RAW_XI  # noqa: E402
import glob  # noqa: E402

rows = json.load(open("data/avg_landscape.json"))
_r = np.array([r["r_c"] for r in rows]); _o = np.argsort(_r)
_r = _r[_o]
_xi = np.array([r["xi_pred"] for r in rows])[_o]
_v = np.array([r["v_pred"] for r in rows])[_o]


def eff(rc, averaged):
    if not averaged:
        return np.full_like(np.atleast_1d(rc), RAW_XI, dtype=float), \
               np.full_like(np.atleast_1d(rc), RAW_V, dtype=float)
    lr = np.log(np.maximum(np.atleast_1d(rc), 1e-6))
    return (np.exp(np.interp(lr, np.log(np.maximum(_r, 1e-6)), np.log(_xi))),
            np.exp(np.interp(lr, np.log(np.maximum(_r, 1e-6)), np.log(_v))))


COL = {"1": "#3b6fb6", "2": "#e0a03c", "3a": "#5aa469", "3b": "#b1453f"}
NAME = {"1": "1  free walk  $D=r_c^2/4\\tau$",
        "2": "2  $D\\sim(r_c^2/\\tau)^{3/13}$",
        "3a": "3a  $D\\sim\\tau^{-3/7}$, $r_c$-independent",
        "3b": "3b  $D\\sim r_c^{1/2}\\tau^{-3/7}$"}


def main():
    raw = load(sorted(glob.glob("data/grid*.npz")))
    avg = load(sorted(glob.glob("data/avg_[A-Z]*.npz")))
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), sharey=True)
    for ax, (title, d, averaged) in zip(
            axes, [("raw potential", raw, False),
                   ("orbit-averaged potential", avg, True)]):
        rr = np.logspace(-1.3, 1.4, 400)
        xi, v = eff(rr, averaged)
        t12 = rr ** 2 / (v * xi)                       # 1 <-> 2
        t23 = (xi / v) * (xi / rr) ** (7.0 / 3.0)      # 2 <-> 3
        ax.plot(t12, rr, "k--", lw=1.6, label=r"$r_c=\sqrt{v_{\rm eff}\xi_{\rm eff}\tau}$")
        ax.plot(t23, rr, "k:", lw=2.0,
                label=r"$r_c=\xi_{\rm eff}(\xi_{\rm eff}/v_{\rm eff}\tau)^{3/7}$")
        cross = rr[np.argmin(np.abs(rr - xi))]
        ax.axhline(cross, color="k", ls="-.", lw=1.4,
                   label=r"$r_c=\xi_{\rm eff}$  (3a$\,|\,$3b)")
        lab = regime(d, 0.0)
        for g in ("1", "2", "3a", "3b"):
            m = lab == g
            if m.any():
                ax.plot(d["tau"][m], d["r_c"][m], "o", color=COL[g], ms=7,
                        mec="k", mew=0.4, ls="none", label=NAME[g], zorder=4)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(2e-2, 6e3); ax.set_ylim(0.04, 40)
        ax.set_xlabel(r"$\tau\ \ [\xi_0/v_0]$")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.15, which="both")
    axes[0].set_ylabel(r"$r_c\ \ [\xi_0]$")
    axes[1].legend(fontsize=8, loc="lower right", framealpha=0.92)
    fig.suptitle("Regimes in lab units.  Points are measured cells, coloured by "
                 "the regime they fall in;\nlines are the boundaries computed "
                 r"from each landscape's own $\xi_{\rm eff}(r_c)$, $v_{\rm eff}(r_c)$",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/phase_both.png", dpi=165)
    print("wrote figures/phase_both.png")


if __name__ == "__main__":
    main()
