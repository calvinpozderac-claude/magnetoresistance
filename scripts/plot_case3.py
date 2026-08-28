#!/usr/bin/env python3
"""Does Case 3 appear at large r_c?  Its two signatures, tested separately."""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)

# ---- (a) the tau exponent -------------------------------------------------
a = ax[0]
d = np.load("data/D_vs_tau_case3_rc2.npz")
t2, D2, e2 = d["tau"], d["D"], d["D_err"]
rows = []
for f in ("data/orbit_average.json", "data/orbit_average_tau30.json",
          "data/orbit_average_tau100.json"):
    rows += json.load(open(f))
b = sorted([r for r in rows if r["mode"] == "bare" and r["r_c"] == 2.0],
           key=lambda r: r["tau"])
t1 = np.array([r["tau"] for r in b]); D1 = np.array([r["D"] for r in b])

a.errorbar(t1, D1, fmt="s", ms=7, color="#3f8f52", mfc="white", mew=1.8,
           label=r"$L=400$ box,  $\tau=10-100$")
a.errorbar(t2, D2, yerr=e2, fmt="o", ms=8, color="#1f5fa9", mfc="white", mew=2,
           capsize=3, label=r"$L=800$ box,  $\tau=100-1000$")
xs = np.geomspace(60, 1600, 20)
a.plot(xs, D2[0] * (xs / t2[0]) ** (-3 / 7), lw=2, color="#c2582a",
       label=r"Case 3:  $\tau^{-3/7}$")
xs2 = np.geomspace(8, 200, 20)
a.plot(xs2, D1[0] * (xs2 / t1[0]) ** (-3 / 13), lw=1.6, ls=":", color="0.4",
       label=r"Case 2:  $\tau^{-3/13}$")
a.set(xscale="log", yscale="log", xlabel=r"$\tau/T_0$", ylabel=r"$D$",
      title=r"$r_c=2\xi_0$ (no averaging): $\tau$ exponent")
a.grid(True, which="both", alpha=0.22)
a.legend(frameon=False, fontsize=9, loc="lower left")
a.text(0.97, 0.95, "fitted  $-0.427\\pm0.021$\n(Case 3: $-0.4286$)",
       transform=a.transAxes, ha="right", va="top", fontsize=10, color="0.2",
       bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.8"))

# ---- (b) r_c independence --------------------------------------------------
a = ax[1]
res = {0.5: (0.11455, 0.00402), 1.0: (0.13978, 0.00865), 2.0: (0.17061, 0.00292)}
rc = np.array(list(res)); Dt = np.array([v[0] for v in res.values()])
er = np.array([v[1] for v in res.values()])
a.errorbar(rc, Dt, yerr=er, fmt="o", ms=9, color="#1f5fa9", mfc="white", mew=2,
           capsize=3, label=r"simulation, $\tau=100\,T_0$")
xs = np.geomspace(0.4, 2.6, 20)
a.plot(xs, np.full_like(xs, Dt[1]), lw=2, color="#c2582a",
       label=r"Case 3:  $D$ independent of $r_c$")
a.plot(xs, Dt[1] * (xs / rc[1]) ** (6 / 13), lw=1.6, ls=":", color="0.4",
       label=r"Case 2:  $r_c^{6/13}$")
a.plot(xs, Dt[1] * (xs / rc[1]) ** 0.287, lw=1.4, ls="--", color="#3f8f52",
       label=r"fit:  $r_c^{+0.29}$")
a.set(xscale="log", yscale="log", xlabel=r"$r_c/\xi_0$", ylabel=r"$D$",
      ylim=(0.07, 0.30),
      title=r"$\tau=100\,T_0$: is $D$ independent of $r_c$?")
a.grid(True, which="both", alpha=0.22)
a.legend(frameon=False, fontsize=9, loc="upper left")

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/case3.png", dpi=195)
print("wrote figures/case3.png")
