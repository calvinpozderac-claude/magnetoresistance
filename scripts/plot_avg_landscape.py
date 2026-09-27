#!/usr/bin/env python3
"""How orbit averaging renormalises the landscape."""
from __future__ import annotations
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

rows = json.load(open("data/avg_landscape.json"))
rc = np.array([r["r_c"] for r in rows])
G = np.array([r["Gamma_eff"] for r in rows])
v = np.array([r["v_eff"] for r in rows])
xi = np.array([r["xi_eff"] for r in rows])
Gp = np.array([r["Gamma_pred"] for r in rows])
vp = np.array([r["v_pred"] for r in rows])
xip = np.array([r["xi_pred"] for r in rows])
m = rc > 0

fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.5))
a = ax[0]
a.plot(rc[m], G[m] / G[0], "o", color="#20558a", ms=6, label=r"$\Gamma_{\rm eff}/\Gamma$")
a.plot(rc[m], Gp[m] / Gp[0], "-", color="#20558a", lw=1.2, alpha=0.6)
a.plot(rc[m], v[m] / v[0], "s", color="#c2492f", ms=6, label=r"$v_{\rm eff}/v_0$")
a.plot(rc[m], vp[m] / vp[0], "-", color="#c2492f", lw=1.2, alpha=0.6)
x = np.logspace(0.3, 1.6, 20)
a.plot(x, (2 * np.pi) ** -0.25 * x ** -0.5, "k--", lw=1.2)
a.text(x[-1], (2 * np.pi) ** -0.25 * x[-1] ** -0.5 * 1.25,
       r"$(2\pi)^{-1/4}(r_c/\xi_0)^{-1/2}$", fontsize=9, ha="right")
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel(r"$r_c\ \ [\xi_0]$"); a.set_ylabel("ratio to the bare field")
a.set_title("(a) averaging weakens the landscape\n"
            "points = measured field, lines = exact Bessel integrals", fontsize=10)
a.legend(fontsize=9)

b = ax[1]
b.plot(rc[m], xi[m], "o", color="#2e7d5b", ms=6, label=r"measured $\xi_{\rm eff}$")
b.plot(rc[m], xip[m], "-", color="#2e7d5b", lw=1.2, alpha=0.6, label="exact")
b.axhline(np.sqrt(2), color="k", ls="--", lw=1.2)
b.text(rc[m][1], np.sqrt(2) * 1.02, r"$\sqrt{2}\,\xi_0$", fontsize=10)
b.axhline(1.0, color="0.6", ls=":", lw=1.2)
b.text(rc[m][1], 1.015, r"$\xi_0$", fontsize=10, color="0.4")
b.set_xscale("log"); b.set_ylim(0.9, 1.75)
b.set_xlabel(r"$r_c\ \ [\xi_0]$")
b.set_ylabel(r"$\xi_{\rm eff}=\sqrt{2}\,\Gamma_{\rm eff}/v_{\rm eff}\ \ [\xi_0]$")
b.set_title("(b) but does not coarsen it:\n"
            r"$\xi_{\rm eff}$ saturates at $\sqrt{2}\,\xi_0$", fontsize=10)
b.legend(fontsize=9, loc="lower right")

fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/avg_landscape.png", dpi=170)
print("wrote figures/avg_landscape.png")
