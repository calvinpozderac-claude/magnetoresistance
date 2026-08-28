#!/usr/bin/env python3
"""Map the (r_c, tau) plane: D on a grid, to locate the three regimes.

Writes after every cell and resumes from what is already there, so an
interrupted run loses at most one cell.  Re-invoking with the same --out
continues where it stopped.

    python scripts/run_grid.py --out data/grid_D.npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mrdiff import PeriodicGaussianField, simulate  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rc", type=float, nargs="+",
                   default=[0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0])
    p.add_argument("--tau", type=float, nargs="+",
                   default=[0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0])
    p.add_argument("--L", type=float, default=400.0)
    p.add_argument("--dx", type=float, default=0.2)
    p.add_argument("--h", type=float, default=0.125)
    p.add_argument("--n-walkers", type=int, default=192)
    p.add_argument("--t-target", type=float, default=2000.0)
    p.add_argument("--min-steps", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--seed", type=int, default=5150)
    p.add_argument("--out", default="data/grid_D.npz")
    args = p.parse_args()

    rc = np.array(args.rc, dtype=float)
    tau = np.array(args.tau, dtype=float)
    shape = (rc.size, tau.size)
    D = np.full(shape, np.nan)
    E = np.full(shape, np.nan)
    LL = np.full(shape, np.nan)
    NS = np.zeros(shape, dtype=int)
    done = np.zeros(shape, dtype=bool)

    if os.path.exists(args.out):          # resume
        z = np.load(args.out)
        if z["r_c"].shape == rc.shape and z["tau"].shape == tau.shape:
            D, E, LL, done = z["D"], z["D_err"], z["loglog"], z["done"]
            NS = z["n_steps"]
            print(f"resuming: {done.sum()}/{done.size} cells already done",
                  flush=True)

    # one landscape, reused for every cell, so cells differ only in (r_c, tau)
    pot = PeriodicGaussianField(xi0=1.0, Gamma=1 / np.sqrt(2), L=args.L,
                                dx=args.dx, seed=args.seed)
    print(f"grid {rc.size} r_c x {tau.size} tau on L={args.L:g} box", flush=True)

    def save():
        np.savez(args.out, r_c=rc, tau=tau, D=D, D_err=E, loglog=LL,
                 done=done, n_steps=NS, L=args.L, h=args.h,
                 n_walkers=args.n_walkers)

    # cheapest cells first, so an interrupted run still covers the plane
    order = sorted(np.ndindex(shape), key=lambda ij: tau[ij[1]])
    for i, j in order:
        if done[i, j]:
            continue
        t = tau[j]
        n_steps = int(np.clip(args.t_target / t, args.min_steps, args.max_steps))
        h = min(args.h, t / 4.0)
        t0 = time.time()
        res = simulate(pot, r_c=rc[i], tau=t, n_steps=n_steps,
                       n_walkers=args.n_walkers, box=args.L, h=h,
                       loop_detect=True, collisions="poisson",
                       seed=args.seed + 17 * i + j)
        D[i, j], E[i, j], LL[i, j], NS[i, j] = (res.D, res.D_err,
                                                res.fit_slope_loglog, n_steps)
        done[i, j] = True
        save()
        print(f"  r_c={rc[i]:6.3g} tau={t:8.3g}  n_steps={n_steps:6d}  "
              f"D={res.D:10.5g} +- {res.D_err:8.3g}  ll={res.fit_slope_loglog:5.2f}"
              f"  {time.time()-t0:6.0f}s   [{done.sum()}/{done.size}]", flush=True)
    print("grid complete ->", args.out)


if __name__ == "__main__":
    main()
