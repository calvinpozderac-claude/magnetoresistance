#!/usr/bin/env python3
"""Diffusion coefficient vs the mean free time tau, at fixed cyclotron radius.

This is the sweep the project notes plot: D(tau) at r_c << xi, in the notes'
units xi = Gamma = B = 1, so that the drift speed is 1 and T0 = xi/v_d = 1.

    python scripts/run_tau_sweep.py --potential pyramid --rc 0.1
    python scripts/run_tau_sweep.py --potential random  --rc 0.1
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mrdiff import (GaussianRandomField, PeriodicGaussianField,  # noqa: E402
                    SquarePyramid, simulate, theory)


def make_potential(name, xi, Gamma, seed, n_modes, L=600.0, dx=0.2):
    """Both landscapes are set up with the same |grad V| = Gamma/xi and the same
    characteristic length xi, so that T0 = xi/v_d = 1 in the units used here."""
    if name == "pyramid":
        # cell size a = 2 xi, apex height Gamma  =>  |grad V| = 2*Gamma/a = Gamma/xi
        return SquarePyramid(V0=Gamma, a=2.0 * xi)
    if name == "random":
        # rms |grad V| = sqrt(2) Gamma / xi0; scale Gamma so it matches Gamma/xi
        return GaussianRandomField(xi0=xi, Gamma=Gamma / np.sqrt(2.0),
                                   n_modes=n_modes, seed=seed)
    if name == "periodic":
        # same statistics, but on an explicit periodic box of side L with every
        # grid mode inside the Gaussian envelope present
        return PeriodicGaussianField(xi0=xi, Gamma=Gamma / np.sqrt(2.0),
                                     L=L, dx=dx, seed=seed)
    raise ValueError(name)


def D_predict(tau, r_c, xi, t0, potential, D_scale=1.0):
    """Rough D used only to choose run lengths (standard convention).

    On the pyramid the notes' own table is the natural estimate.  On the random
    field contours near the percolating level are unbounded, D falls far more
    slowly than 1/tau, and the pyramid formula badly underestimates it at large
    tau -- an empirical power law calibrated on a few pilot runs is used instead.
    """
    if potential in ("random", "periodic"):
        return np.maximum(r_c ** 2 / (4.0 * tau), D_scale * 0.22 * tau ** -0.22)
    return D_scale * theory.to_standard(theory.D_piecewise(tau, r_c, xi, t0))


def sizing(tau, r_c, xi, t0, target_cells=100.0, min_steps=2000,
           max_steps=400_000, work_budget=6.0e7, min_walkers=256,
           max_walkers=8192, max_time=None, D_scale=1.0, potential="pyramid"):
    """Run long enough to diffuse across many cells, cheap enough to finish.

    The run length is set from the *predicted* D so that the total simulated
    time is ``target_cells`` times what it takes to spread over one cell
    (2 xi)^2; the walker count then fills a fixed work budget.
    """
    D_pred = D_predict(tau, r_c, xi, t0, potential, D_scale)
    t_total = target_cells * (2.0 * xi) ** 2 / (4.0 * D_pred)
    if max_time is not None:
        t_total = min(t_total, max_time)
    n_steps = int(np.clip(t_total / tau, min_steps, max_steps))
    n_walkers = int(np.clip(work_budget / n_steps, min_walkers, max_walkers))
    return n_steps, n_walkers


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--potential", choices=["pyramid", "random", "periodic"],
                   default="pyramid")
    p.add_argument("--box-L", type=float, default=600.0,
                   help="periodic landscape: side of the box, in units of xi0")
    p.add_argument("--box-dx", type=float, default=0.2,
                   help="periodic landscape: grid spacing")
    p.add_argument("--rc", type=float, default=0.1, help="cyclotron radius / xi")
    p.add_argument("--xi", type=float, default=1.0)
    p.add_argument("--Gamma", type=float, default=1.0)
    p.add_argument("--B", type=float, default=1.0)
    p.add_argument("--tau-min", type=float, default=1e-4)
    p.add_argument("--tau-max", type=float, default=1e2)
    p.add_argument("--n-tau", type=int, default=22)
    p.add_argument("--n-modes", type=int, default=64, help="random field only")
    p.add_argument("--n-real", type=int, default=1,
                   help="independent disorder realisations (random field only)")
    p.add_argument("--target-cells", type=float, default=100.0,
                   help="run until the walkers have spread over this many "
                        "(2 xi)^2 cells")
    p.add_argument("--D-scale", type=float, default=1.0,
                   help="multiplier on the predicted D used only for sizing")
    p.add_argument("--max-steps", type=int, default=400_000)
    p.add_argument("--work-budget", type=float, default=6.0e7)
    p.add_argument("--max-substeps", type=int, default=400,
                   help="cap on RK4 substeps per drift (random field only)")
    p.add_argument("--loop-detect", action="store_true",
                   help="time each closed orbit and take the drift modulo its "
                        "period, so that tau >> orbital period stays cheap")
    p.add_argument("--substep-time", type=float, default=0.125,
                   help="RK4 time step.  On the random field the potential "
                        "leaks across contours once the step exceeds ~0.25 "
                        "xi0/v_rms; 0.125 was checked against 0.0625.")
    p.add_argument("--fixed-time", type=float, default=None,
                   help="run every tau for this same total simulated time, so "
                        "that D is measured over one common lag window and the "
                        "shape of D(tau) carries no scale-dependent bias")
    p.add_argument("--min-steps", type=int, default=2000)
    p.add_argument("--min-walkers", type=int, default=256)
    p.add_argument("--max-walkers", type=int, default=8192)
    p.add_argument("--collisions", choices=["poisson", "fixed"], default="poisson",
                   help="exponential drift times of mean tau (the notes' "
                        "'collide on average after tau'), or exactly tau")
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    t0 = theory.T0(args.xi, args.Gamma, args.B)
    taus = np.geomspace(args.tau_min, args.tau_max, args.n_tau)
    out = args.out or (f"data/D_vs_tau_{args.potential}_rc{args.rc:g}"
                       f"_{args.collisions}.npz")

    D = np.zeros(taus.size)
    D_err = np.zeros(taus.size)
    ll = np.zeros(taus.size)
    n_steps_a = np.zeros(taus.size, dtype=int)
    n_walk_a = np.zeros(taus.size, dtype=int)

    def save(path, upto):
        """Write the sweep after every point, so a run that is interrupted
        still leaves usable data behind."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        sl = slice(0, upto)
        np.savez(path, tau=taus[sl], D=D[sl], D_err=D_err[sl],
                 loglog_slope=ll[sl], r_c=args.rc, xi=args.xi, Gamma=args.Gamma,
                 B=args.B, T0=t0, potential=args.potential,
                 collisions=args.collisions, n_steps=n_steps_a[sl],
                 n_walkers=n_walk_a[sl], n_modes=args.n_modes,
                 n_real=args.n_real, substep_time=args.substep_time,
                 box_L=args.box_L,
                 fixed_time=(args.fixed_time if args.fixed_time else 0.0))

    print(f"{args.potential}: xi={args.xi} Gamma={args.Gamma} B={args.B} "
          f"| T0={t0:g} | r_c={args.rc} | collisions={args.collisions} "
          f"| realisations={args.n_real}")
    print(f"{'tau':>10} {'tau/T0':>9} {'n_steps':>8} {'walkers':>7} {'n/h':>5} "
          f"{'D(std)':>11} {'err':>9} {'slope':>6} {'s':>6}")
    t_all = time.time()
    for k, tau in enumerate(taus):
        if args.fixed_time is not None:
            n_steps = int(np.clip(args.fixed_time / tau, args.min_steps,
                                  args.max_steps))
            n_walkers = args.max_walkers
        else:
            n_steps, n_walkers = sizing(tau, args.rc, args.xi, t0,
                                        target_cells=args.target_cells,
                                        max_steps=args.max_steps,
                                        work_budget=args.work_budget,
                                        min_steps=args.min_steps,
                                        min_walkers=args.min_walkers,
                                        max_walkers=args.max_walkers,
                                        D_scale=getattr(args, "D_scale"),
                                        potential=args.potential)
        n_walkers = max(n_walkers // args.n_real, 64) if args.n_real > 1 else n_walkers
        # A fixed RK4 step, small enough that the projection holds the walker
        # on its contour, but never fewer than 4 steps across a short drift.
        h = min(args.substep_time, tau / 4.0)
        n_sub = int(np.ceil(tau / h))
        tk = time.time()
        Ds = []
        for r in range(args.n_real):
            pot = make_potential(args.potential, args.xi, args.Gamma,
                                 args.seed + 1000 * r, args.n_modes,
                                 L=args.box_L, dx=args.box_dx)
            box = (4.0 * args.xi if args.potential == "pyramid"
                   else args.box_L if args.potential == "periodic"
                   else 12.0 * args.xi)
            res = simulate(pot, r_c=args.rc, tau=tau, n_steps=n_steps,
                           n_walkers=n_walkers, B=args.B, seed=args.seed + k + 7 * r,
                           box=box, h=h, loop_detect=args.loop_detect,
                           collisions=args.collisions)
            Ds.append(res.D)
            if r == 0:
                D_err[k], ll[k] = res.D_err, res.fit_slope_loglog
        D[k] = float(np.mean(Ds))
        if args.n_real > 1:
            D_err[k] = float(np.std(Ds, ddof=1) / np.sqrt(args.n_real))
        n_steps_a[k], n_walk_a[k] = n_steps, n_walkers
        print(f"{tau:10.4g} {tau / t0:9.4g} {n_steps:8d} {n_walkers:7d} {n_sub:5.4g} "
              f"{D[k]:11.5g} {D_err[k]:9.3g} {ll[k]:6.3f} {time.time() - tk:6.1f}")
    print(f"total {time.time() - t_all:.1f} s")
    print("wrote", out)


def _unused(out, taus, D, D_err, ll, args, t0, n_steps_a, n_walk_a):
    np.savez(out, tau=taus, D=D, D_err=D_err, loglog_slope=ll, r_c=args.rc,
             xi=args.xi, Gamma=args.Gamma, B=args.B, T0=t0,
             potential=args.potential, collisions=args.collisions,
             n_steps=n_steps_a, n_walkers=n_walk_a,
             n_modes=args.n_modes, n_real=args.n_real,
             fixed_time=(args.fixed_time if args.fixed_time else 0.0))
    print("wrote", out)


if __name__ == "__main__":
    main()
