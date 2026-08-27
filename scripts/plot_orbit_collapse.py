#!/usr/bin/env python3
"""Bare and orbit-averaged landscapes on one curve, once rescaled.

Plots y = D_contour/(v' xi') against u = tau/T0', with v' and xi' measured from
whichever potential the walker is actually following.  If orbit averaging only
renormalises the landscape, the two sets fall on the same curve.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rows = []
for f in ("data/orbit_average.json", "data/orbit_average_tau30.json",
          "data/orbit_average_tau100.json"):
    if os.path.exists(f):
        rows += json.load(open(f))
for r in rows:
    r["vx"] = r["v_eff"] * r["xi_eff"]
    r["u"] = r["tau"] / (r["xi_eff"] / r["v_eff"])
    r["y"] = (r["D"] - r["free"]) / r["vx"]

fig, ax = plt.subplots(figsize=(7.6, 5.6), constrained_layout=True)
b = [r for r in rows if r["mode"] == "bare" and r["r_c"] == 2.0]
a = [r for r in rows if r["mode"] == "averaged" and r["r_c"] == 2.0]
ub = np.array(sorted(r["u"] for r in b))
yb = np.array([r["y"] for r in sorted(b, key=lambda r: r["u"])])
sl = np.polyfit(np.log(ub), np.log(yb), 1)
xs = np.geomspace(1.3, 130, 40)
ax.plot(xs, np.exp(np.polyval(sl, np.log(xs))), lw=1.6, color="0.45", ls="--",
        zorder=1, label=rf"bare fit: $\propto (\tau/T_0')^{{{sl[0]:.2f}}}$")

other = [r for r in rows if r["r_c"] != 2.0]
ax.plot([r["u"] for r in other if r["mode"] == "bare"],
        [r["y"] for r in other if r["mode"] == "bare"], "o", ms=5,
        color="#9bb4d0", mfc="#9bb4d0", zorder=2,
        label=r"bare, other $r_c$ (0.1 - 4)")
ax.plot([r["u"] for r in other if r["mode"] == "averaged"],
        [r["y"] for r in other if r["mode"] == "averaged"], "s", ms=5,
        color="#d8b48c", mfc="#d8b48c", zorder=2,
        label=r"averaged, other $r_c$")
ax.plot([r["u"] for r in b], [r["y"] for r in b], "o", ms=9, color="#1f5fa9",
        mfc="white", mew=2, zorder=4, label=r"bare, $r_c=2\xi_0$")
ax.plot([r["u"] for r in a], [r["y"] for r in a], "s", ms=9, color="#c2582a",
        mfc="white", mew=2, zorder=4, label=r"orbit-averaged, $r_c=2\xi_0$")

am = max(a, key=lambda r: r["u"])
yi = float(np.exp(np.interp(np.log(am["u"]), np.log(ub), np.log(yb))))
ax.annotate(f"matched $\\tau/T_0'$: {am['y']/yi:.3f}",
            xy=(am["u"], am["y"]), xytext=(am["u"] * 1.25, am["y"] * 1.7),
            fontsize=9.5, color="0.25",
            arrowprops=dict(arrowstyle="->", color="0.5", lw=1))

ax.set(xscale="log", yscale="log", xlabel=r"$\tau/T_0'$",
       ylabel=r"$D_{\rm contour}\,/\,(v'\xi')$")
ax.set_title("Orbit averaging renormalises the landscape; it does not\n"
             "change the law: both fall on one curve in the field's own units",
             fontsize=11)
ax.grid(True, which="both", alpha=0.22, lw=0.6)
ax.legend(frameon=False, fontsize=9, loc="lower left")
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/orbit_average_collapse.png", dpi=195)
print("wrote figures/orbit_average_collapse.png   (matched-u ratio "
      f"{am['y']/yi:.3f}, bare slope {sl[0]:+.3f})")
