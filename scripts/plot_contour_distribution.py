#!/usr/bin/env python3
"""Why the measured D(tau) exponent sits between -3/13 and -3/7.

The whole large-tau argument rests on Prob(eps), which fixes the tail exponent a
of the contour-size distribution seen by a uniformly placed walker,
P(Lambda_c >= Lambda) ~ Lambda^-a.  Tube counting (L w / xi^2) gives a = 1;
weighting by the area fraction with |V| < eps gives a = 1/nu = 0.75.  The
displacement then follows as <dr^2> ~ Lambda_tau^(2-a), so

    D ~ tau^((2-a)/d_h - 1),   Lambda_tau = (v tau)^(1/d_h)

and the exponent is whatever a is AT THE SCALE Lambda_tau -- which is the
catch: a only reaches 1 once Lambda_tau is a few tens of xi0.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("data/contour_distribution.npz")
D = d["diam"]
dh = 1.75

fig, ax = plt.subplots(1, 2, figsize=(11.8, 4.9), constrained_layout=True)

a0 = ax[0]
xs = np.geomspace(1.5, 300, 40)
cdf = np.array([(D >= t).mean() for t in xs])
m = cdf > 3e-4
a0.plot(xs[m], cdf[m], "o", ms=4.5, color="#1f5fa9", mfc="white", mew=1.3,
        label=r"measured $P(\Lambda_c \geq \Lambda)$")
for a, c, lbl in ((1.0, "#c2582a", r"$a=1$  (tube counting, $Lw/\xi^2$)"),
                  (0.75, "#3f8f52", r"$a=1/\nu=0.75$  (level-layer)")):
    ref = cdf[m][6] * (xs[m] / xs[m][6]) ** (-a)
    a0.plot(xs[m], ref, lw=1.6, ls="--", color=c, label=lbl)
a0.set(xscale="log", yscale="log", xlabel=r"$\Lambda/\xi_0$",
       ylabel=r"$P(\Lambda_c \geq \Lambda)$",
       title="Contour size seen by a uniformly placed walker")
a0.grid(True, which="both", alpha=0.2)
a0.legend(frameon=False, fontsize=9)

a1 = ax[1]
e = np.geomspace(2, 200, 30)
c2 = np.array([(D >= t).mean() for t in e])
loc = -np.gradient(np.log(c2), np.log(e))
tau = e ** dh
a1.plot(tau, (2 - loc) / dh - 1, "-", lw=2, color="#1f5fa9",
        label=r"implied $d\ln D/d\ln\tau$ from measured $a(\Lambda_\tau)$")
a1.axhline(-3 / 7, ls="--", lw=1.5, color="#c2582a",
           label=r"$-3/7$  (asymptotic, $a=1$)")
a1.axhline(-3 / 13, ls=":", lw=1.5, color="0.4", label=r"$-3/13$  (Case 2)")
meas = [(10, -0.24), (30, -0.28), (100, -0.30), (300, -0.32)]
a1.plot([m[0] for m in meas], [m[1] for m in meas], "s", ms=7, color="#3f8f52",
        mfc="white", mew=1.6, label="measured from the dynamics")
a1.set(xscale="log", xlabel=r"$\tau/T_0$", ylabel=r"$d\ln D/d\ln \tau$",
       ylim=(-0.75, 0.0),
       title=r"the effective exponent tracks $a$ at the scale $\Lambda_\tau$")
a1.grid(True, which="both", alpha=0.2)
a1.legend(frameon=False, fontsize=8.5, loc="lower left")

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/contour_distribution.png", dpi=190)
print("wrote figures/contour_distribution.png")
