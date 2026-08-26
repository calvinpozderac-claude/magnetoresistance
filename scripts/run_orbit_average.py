#!/usr/bin/env python3
"""Does orbit averaging change the dynamics at large r_c?

For each r_c the same walk is run twice: once on the bare potential (contours of
V, the model used so far) and once on the orbit-averaged potential (contours of
V' = ring average of V over the cyclotron orbit, which is what a guiding centre
actually follows).  Printed alongside is the free-walk value r_c^2/4tau and the
prediction from the renormalised landscape,

    D  ~  0.26 v' xi' (tau/T0')^(-2/7)  +  r_c^2/4tau

with v' = rms|grad V'|/B, xi' = sqrt(2) rms(V')/rms|grad V'|, T0' = xi'/v'.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mrdiff import PeriodicGaussianField, simulate  # noqa: E402


def landscape_params(pot, rng, n=40000):
    x, y = rng.uniform(0, pot.L, (2, n))
    v, gx, gy = pot.value_grad(x, y)
    G = float(np.std(v))
    g = float(np.sqrt(np.mean(gx ** 2 + gy ** 2)))
    xi = float(np.sqrt(2.0) * G / g)
    return G, g, xi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tau", type=float, default=10.0)
    p.add_argument("--rc", type=float, nargs="+",
                   default=[0.1, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--L", type=float, default=200.0)
    p.add_argument("--dx", type=float, default=0.2)
    p.add_argument("--n-steps", type=int, default=800)
    p.add_argument("--n-walkers", type=int, default=384)
    p.add_argument("--n-real", type=int, default=2)
    p.add_argument("--h", type=float, default=0.125)
    p.add_argument("--out", default="data/orbit_average.json")
    args = p.parse_args()

    rows = []
    print(f"tau = {args.tau}   (bare xi0 = Gamma = B = 1)")
    print(f"{'r_c':>6} {'mode':>9} {'v_eff':>7} {'xi_eff':>7} {'D':>10} {'err':>9} "
          f"{'slope':>6} {'free walk':>10} {'predicted':>10} {'s':>6}")
    for rc in args.rc:
        for mode in ("bare", "averaged"):
            Ds, t0 = [], time.time()
            for r in range(args.n_real):
                pot = PeriodicGaussianField(
                    xi0=1.0, Gamma=1 / np.sqrt(2), L=args.L, dx=args.dx,
                    seed=9100 + r, ring_average=(rc if mode == "averaged" else 0.0))
                if r == 0:
                    G, g, xi = landscape_params(pot, np.random.default_rng(3))
                res = simulate(pot, r_c=rc, tau=args.tau, n_steps=args.n_steps,
                               n_walkers=args.n_walkers, box=args.L, h=args.h,
                               loop_detect=True, collisions="poisson",
                               seed=41 + r)
                Ds.append(res.D)
                if r == 0:
                    ll = res.fit_slope_loglog
            D = float(np.mean(Ds))
            err = float(np.std(Ds, ddof=1) / np.sqrt(args.n_real)) if len(Ds) > 1 else 0.0
            free = rc ** 2 / (4 * args.tau)
            T0p = xi / g
            pred = 0.26 * g * xi * (args.tau / T0p) ** (-2 / 7) + free
            print(f"{rc:6.3g} {mode:>9} {g:7.3f} {xi:7.3f} {D:10.5f} {err:9.5f} "
                  f"{ll:6.2f} {free:10.5f} {pred:10.5f} {time.time()-t0:6.0f}",
                  flush=True)
            rows.append(dict(r_c=rc, mode=mode, v_eff=g, xi_eff=xi, D=D, err=err,
                             loglog=float(ll), free=free, pred=pred, tau=args.tau))
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            json.dump(rows, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
