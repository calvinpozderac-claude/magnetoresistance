#!/usr/bin/env python3
r"""What orbit averaging does to the landscape.

V'(r) = (1/2pi) int dtheta V(r + r_c u(theta)) multiplies every Fourier mode of
V by J_0(|k| r_c), so the averaged field is still Gaussian but with spectrum
S(k) J_0(k r_c)^2.  For the Gaussian correlator <V(0)V(r)> = Gamma^2
exp(-r^2/2 xi0^2), i.e. S(k) ~ exp(-k^2 xi0^2 / 2),

    Gamma_eff^2 / Gamma^2 = exp(-x) I_0(x),          x = r_c^2 / xi0^2
    v_eff^2   / v_0^2     = <k^2>_{S J_0^2} / <k^2>_S

and it is natural to define the averaged field's own correlation length from
its gradient, xi_eff = sqrt(2) Gamma_eff / v_eff, which reduces to xi0 when
r_c -> 0.  Large r_c: <J_0^2> -> 1/(pi k r_c), giving

    Gamma_eff/Gamma -> (2 pi)^(-1/4) (xi0/r_c)^(1/2)
    v_eff/v_0       -> (1/sqrt 2) Gamma_eff/Gamma
    xi_eff          -> sqrt(2) xi0                    (a constant!)

so averaging mostly *weakens* the landscape rather than coarsening it.
This script checks all three against the actual periodic field.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.special import i0e, j0

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mrdiff import PeriodicGaussianField  # noqa: E402


def quad_ratio(rc, xi0, power):
    """<k^power>_{S J0^2} / <k^power>_S  for S = exp(-k^2 xi0^2 / 2)."""
    k = np.linspace(0.0, 60.0 / xi0, 400001)
    S = np.exp(-0.5 * (k * xi0) ** 2) * k ** (1 + power)
    return np.trapezoid(S * j0(k * rc) ** 2, k) / np.trapezoid(S, k)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rc", type=float, nargs="+",
                   default=[0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0,
                            4.0, 5.7, 8.0, 11.3, 16.0, 22.6, 32.0])
    p.add_argument("--L", type=float, default=200.0)
    p.add_argument("--dx", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=5150)
    p.add_argument("--out", default="data/avg_landscape.json")
    args = p.parse_args()

    xi0, Gam = 1.0, 1 / np.sqrt(2)
    v0 = np.sqrt(2) * Gam / xi0            # rms |grad V| of the bare field
    rows = []
    for rc in args.rc:
        pot = PeriodicGaussianField(xi0=xi0, Gamma=Gam, L=args.L, dx=args.dx,
                                    seed=args.seed, ring_average=rc,
                                    precompute=False)
        G = float(pot.V.std())
        v = float(np.sqrt((pot.Vx ** 2 + pot.Vy ** 2).mean()))
        xi = np.sqrt(2.0) * G / v
        x = (rc / xi0) ** 2
        Gp = Gam * np.sqrt(i0e(x))                       # exp(-x) I0(x), stable
        vp = v0 * np.sqrt(quad_ratio(rc, xi0, 2))
        rows.append(dict(r_c=rc, Gamma_eff=G, v_eff=v, xi_eff=xi,
                         Gamma_pred=float(Gp), v_pred=float(vp),
                         xi_pred=float(np.sqrt(2) * Gp / vp)))
        print(f"r_c={rc:6.3g}  Gamma_eff={G:8.5f} (pred {Gp:8.5f})  "
              f"v_eff={v:8.5f} (pred {vp:8.5f})  "
              f"xi_eff={xi:7.4f} (pred {np.sqrt(2)*Gp/vp:7.4f})", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
