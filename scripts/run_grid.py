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
    p.add_argument("--orbit-average", action="store_true",
                   help="follow contours of the orbit-averaged potential, "
                        "V'(r) = <V(r + r_c u)>_theta, rebuilding the landscape "
                        "for every r_c (its modes are scaled by J_0(|k| r_c))")
    p.add_argument("--fixed-step", action="store_true",
                   help="keep the integration step h fixed instead of scaling "
                        "it as h/v_eff to hold the spatial step constant")
    args = p.parse_args()

    rc = np.array(args.rc, dtype=float)
    tau = np.array(args.tau, dtype=float)
    shape = (rc.size, tau.size)
    D = np.full(shape, np.nan)
    E = np.full(shape, np.nan)
    LL = np.full(shape, np.nan)
    NS = np.zeros(shape, dtype=int)
    done = np.zeros(shape, dtype=bool)
    # landscape statistics, per r_c row (they only vary when averaging is on)
    GA = np.full(rc.size, np.nan)
    VE = np.full(rc.size, np.nan)
    XE = np.full(rc.size, np.nan)

    if os.path.exists(args.out):          # resume
        z = np.load(args.out)
        if z["r_c"].shape == rc.shape and z["tau"].shape == tau.shape:
            D, E, LL, done = z["D"], z["D_err"], z["loglog"], z["done"]
            NS = z["n_steps"]
            if "v_eff" in z:
                GA, VE, XE = z["Gamma_eff"], z["v_eff"], z["xi_eff"]
            print(f"resuming: {done.sum()}/{done.size} cells already done",
                  flush=True)

    def build(i):
        """The landscape row i walks on, and its rms gradient."""
        pot = PeriodicGaussianField(
            xi0=1.0, Gamma=1 / np.sqrt(2), L=args.L, dx=args.dx,
            seed=args.seed, ring_average=rc[i] if args.orbit_average else 0.0)
        GA[i] = pot.V.std()
        VE[i] = np.sqrt((pot.Vx ** 2 + pot.Vy ** 2).mean())
        XE[i] = np.sqrt(2.0) * GA[i] / VE[i]
        return pot

    print(f"grid {rc.size} r_c x {tau.size} tau on L={args.L:g} box"
          f"{'  [orbit-averaged]' if args.orbit_average else ''}", flush=True)

    def save():
        np.savez(args.out, r_c=rc, tau=tau, D=D, D_err=E, loglog=LL,
                 done=done, n_steps=NS, L=args.L, h=args.h,
                 n_walkers=args.n_walkers, Gamma_eff=GA, v_eff=VE, xi_eff=XE,
                 orbit_average=args.orbit_average)

    if args.orbit_average:
        # the landscape depends on r_c, so walk the grid r_c-major and rebuild
        # once per row; within a row do the cheap (small tau) cells first
        order = sorted(np.ndindex(shape), key=lambda ij: (ij[0], tau[ij[1]]))
    else:
        # one landscape for every cell: cheapest cells first, so an interrupted
        # run still covers the plane
        order = sorted(np.ndindex(shape), key=lambda ij: tau[ij[1]])

    pot, row = None, None
    for i, j in order:
        if done[i, j]:
            continue
        # without averaging the landscape does not depend on r_c, so build once
        if pot is None or (args.orbit_average and i != row):
            pot, row = build(i), i
            if not args.orbit_average:
                GA[:], VE[:], XE[:] = GA[i], VE[i], XE[i]
            print(f"  landscape r_c={rc[i]:g}: Gamma_eff={GA[i]:.4f} "
                  f"v_eff={VE[i]:.4f} xi_eff={XE[i]:.4f}", flush=True)
        t = tau[j]
        n_steps = int(np.clip(args.t_target / t, args.min_steps, args.max_steps))
        # hold the *spatial* substep at args.h * v_0, not the time step: the
        # averaged landscape drifts slower, so a fixed h would oversample it
        hstep = args.h if args.fixed_step else args.h / max(VE[i], 1e-6)
        h = min(hstep, t / 4.0)
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
