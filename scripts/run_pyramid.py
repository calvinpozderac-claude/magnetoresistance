#!/usr/bin/env python3
"""Diffusion coefficient vs cyclotron radius on the square-pyramid landscape.

    python scripts/run_pyramid.py [--tau 1.0] [--n-rc 26] ...

Sweeps r_c at fixed B, measures D from the slope of the mean squared
displacement, and writes both the raw sweep (data/*.npz) and a log-log plot
(figures/*.png).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mrdiff import SquarePyramid, simulate  # noqa: E402


def sizing(r_c, a=1.0, steps_coeff=200.0, max_steps=200_000, min_steps=3000,
           work_budget=4.0e7, min_walkers=512, max_walkers=8192):
    """How long a run does this r_c need?

    Small kicks randomise the walker's *contour level* only diffusively: it
    takes ~ (a / r_c)^2 kicks to wander across the full range of V and reach the
    percolating V = 0 network, so the sub-diffusive transient grows as r_c^-2
    and the run length must grow with it.  Walker count is then set by a fixed
    total-work budget.
    """
    n_steps = int(np.clip(steps_coeff * (a / r_c) ** 2, min_steps, max_steps))
    n_walkers = int(np.clip(work_budget / n_steps, min_walkers, max_walkers))
    return n_steps, n_walkers


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tau", type=float, default=1.0, help="mean free time")
    p.add_argument("--B", type=float, default=1.0, help="magnetic field")
    p.add_argument("--V0", type=float, default=1.0, help="pyramid apex height")
    p.add_argument("--a", type=float, default=1.0, help="pyramid (cell) size")
    p.add_argument("--rc-min", type=float, default=0.03)
    p.add_argument("--rc-max", type=float, default=5.0)
    p.add_argument("--n-rc", type=int, default=26)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--out", default="data/D_vs_rc_pyramid.npz")
    args = p.parse_args()

    pot = SquarePyramid(V0=args.V0, a=args.a)
    rcs = np.geomspace(args.rc_min, args.rc_max, args.n_rc)

    D = np.empty_like(rcs)
    D_err = np.empty_like(rcs)
    ll = np.empty_like(rcs)
    n_steps_arr = np.empty(rcs.size, dtype=int)
    n_walk_arr = np.empty(rcs.size, dtype=int)

    print(f"square pyramid: V0={args.V0} a={args.a} | B={args.B} tau={args.tau} "
          f"| drift speed = {pot.drift_speed_factor / args.B:g}")
    print(f"{'r_c':>8} {'n_steps':>8} {'walkers':>8} {'D':>12} {'err':>9} "
          f"{'dlnMSD/dlnt':>12} {'s':>6}")
    t_all = time.time()
    for k, r_c in enumerate(rcs):
        n_steps, n_walkers = sizing(r_c, a=args.a)
        t0 = time.time()
        res = simulate(pot, r_c=r_c, tau=args.tau, n_steps=n_steps,
                       n_walkers=n_walkers, B=args.B, seed=args.seed + k,
                       n_snapshots=140)
        D[k], D_err[k], ll[k] = res.D, res.D_err, res.fit_slope_loglog
        n_steps_arr[k], n_walk_arr[k] = n_steps, n_walkers
        flag = "" if abs(ll[k] - 1.0) < 0.12 else "  <- not yet diffusive?"
        print(f"{r_c:8.4f} {n_steps:8d} {n_walkers:8d} {D[k]:12.6g} "
              f"{D_err[k]:9.3g} {ll[k]:12.3f} {time.time() - t0:6.1f}{flag}")
    print(f"total {time.time() - t_all:.1f} s")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, r_c=rcs, D=D, D_err=D_err, loglog_slope=ll,
             n_steps=n_steps_arr, n_walkers=n_walk_arr, tau=args.tau, B=args.B,
             V0=args.V0, a=args.a)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
