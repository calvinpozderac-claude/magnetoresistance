"""Drift--kick random walk and the diffusion coefficient it produces.

Model
-----
Between impurity collisions the guiding centre of a cyclotron orbit follows an
equipotential contour of V (see :mod:`mrdiff.potentials`).  A collision randomises
the momentum, which re-centres the cyclotron orbit somewhere on the circle of
radius r_c about the current position; the guiding centre therefore hops by a
vector of length r_c in a uniformly random direction and then follows whatever
contour it lands on.  One elementary step is

    1. drift along the contour through r for a time tau  (mean free time),
    2. hop:  r -> r + r_c (cos theta, sin theta),  theta ~ U[0, 2 pi).

Repeating this and measuring the slope of <|r(t) - r(0)|^2> gives

    D = lim_{t -> inf} <|r(t) - r(0)|^2> / (4 t)        (2-D).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class WalkResult:
    r_c: float
    tau: float
    msd_times: np.ndarray      # log-spaced times, measured from the start
    msd: np.ndarray            # <|r(t) - r(0)|^2>, single time origin
    lags: np.ndarray           # lag times for the time-averaged MSD
    tamsd: np.ndarray          # <|r(t+dt) - r(t)|^2>, averaged over origins
    D: float                   # diffusion coefficient from the linear fit
    D_err: float               # standard error over independent walker batches
    fit_slope_loglog: float    # d log(TAMSD) / d log(dt) over the fit window
    fit_window: tuple          # (first, last) lag time included in the fit
    n_walkers: int
    n_steps: int
    positions: np.ndarray | None = field(default=None, repr=False)


def simulate(potential, r_c, tau, n_steps, n_walkers=1024, B=1.0, seed=0,
             n_snapshots=128, fit_lags=(0.05, 0.5), n_batches=8, box=None,
             n_msd_points=140, n_sub=None, collisions="fixed"):
    """Run the drift--kick walk and extract D.

    Parameters
    ----------
    potential : Potential
        Landscape whose contours the guiding centres follow.
    r_c : float
        Cyclotron radius, i.e. the length of the hop at each impurity collision.
    tau : float
        Time spent drifting on a contour between collisions.
    n_steps : int
        Number of drift+kick steps per walker.
    n_walkers : int
        Independent walkers (the ensemble average is over these).
    n_snapshots : int
        Positions are stored at this many equally spaced times; the diffusion
        coefficient comes from the *time-averaged* MSD built from them, i.e.
        <|r(t+dt) - r(t)|^2> averaged over every available origin t as well as
        over walkers.  Using all origins rather than only t = 0 cuts the
        variance of D substantially at small r_c, where the run has to be long
        anyway.
    fit_lags : (float, float)
        Fractions of the total run duration bracketing the lags used for the
        linear fit.  The lower end discards the sub-diffusive transient (which
        lasts ~ (a/r_c)^2 steps), the upper end keeps enough distinct origins
        for the average to be meaningful.
    n_msd_points : int
        The single-origin MSD is additionally recorded on a log-spaced time
        grid spanning the whole run, which resolves the sub-diffusive transient
        that the (evenly spaced) time-averaged MSD starts after.
    n_batches : int
        Walkers are split into this many batches; the scatter of the per-batch
        diffusion coefficients gives the error bar.
    collisions : {"fixed", "poisson"}
        ``"fixed"`` drifts for exactly ``tau`` between kicks.  ``"poisson"``
        draws each drift time from an exponential distribution of mean ``tau``,
        i.e. a Poisson collision process at rate 1/tau -- the notes' "collide on
        average after tau".  The two differ once tau approaches the orbital
        period: with a fixed time the walker performs a rigid rotation of its
        closed contour every step, which is not ergodic on the contour.  Times
        are still counted as ``n_steps * tau``, which is exact in the diffusive
        regime because the MSD is linear in t and <sum of drift times> = n tau.
    n_sub : int or None
        RK4 substeps per drift, for potentials that integrate their contours
        numerically.  Ignored by potentials with an analytic propagator.
    box : float or None
        Walkers are seeded uniformly in ``[0, box)^2``.  Defaults to one lattice
        period of the potential (2a), which is the stationary distribution: the
        drift is incompressible and the hop is isotropic, so a spatially uniform
        density is invariant under the dynamics.
    """
    rng = np.random.default_rng(seed)
    if box is None:
        box = 2.0 * getattr(potential, "a", 1.0)

    n_snapshots = int(min(n_snapshots, n_steps))
    every = max(1, n_steps // n_snapshots)
    n_snapshots = n_steps // every

    pos = rng.random((2, n_walkers)) * box
    x, y = pos[0].copy(), pos[1].copy()
    x0, y0 = x.copy(), y.copy()
    snaps = np.empty((n_snapshots, 2, n_walkers))

    msd_steps = np.unique(np.geomspace(1, n_steps, n_msd_points).astype(int))
    msd = np.empty(msd_steps.size)
    next_msd = 0

    if collisions not in ("fixed", "poisson"):
        raise ValueError(collisions)

    for step in range(1, n_steps + 1):
        dt = rng.exponential(tau, n_walkers) if collisions == "poisson" else tau
        x, y = potential.propagate(x, y, dt, B=B, n_sub=n_sub)
        theta = rng.random(n_walkers) * (2.0 * np.pi)
        x = x + r_c * np.cos(theta)
        y = y + r_c * np.sin(theta)
        if step % every == 0 and step // every <= n_snapshots:
            k = step // every - 1
            snaps[k, 0] = x
            snaps[k, 1] = y
        if next_msd < msd_steps.size and step == msd_steps[next_msd]:
            msd[next_msd] = ((x - x0) ** 2 + (y - y0) ** 2).mean()
            next_msd += 1

    msd_times = msd_steps * tau

    # time-averaged MSD: for each lag, average over every available origin
    lags = np.arange(1, n_snapshots) * every * tau
    tamsd_w = np.empty((lags.size, n_walkers))
    for d in range(1, n_snapshots):
        diff = snaps[d:] - snaps[:-d]
        tamsd_w[d - 1] = (diff ** 2).sum(axis=1).mean(axis=0)
    tamsd = tamsd_w.mean(axis=1)

    D, slope_ll, window = _fit_D(lags, tamsd, fit_lags)
    batch = np.arange(n_walkers) % n_batches
    D_b = np.array([_fit_D(lags, tamsd_w[:, batch == b].mean(axis=1),
                           fit_lags)[0] for b in range(n_batches)])
    D_err = D_b.std(ddof=1) / np.sqrt(n_batches)

    return WalkResult(r_c=r_c, tau=tau, msd_times=msd_times, msd=msd, lags=lags,
                      tamsd=tamsd, D=D, D_err=D_err, fit_slope_loglog=slope_ll,
                      fit_window=window, n_walkers=n_walkers, n_steps=n_steps,
                      positions=np.stack([x, y]))


def _fit_D(lags, msd, fit_lags):
    """Least-squares slope of MSD vs lag over the requested lag window.

    Also returns d log(MSD) / d log(dt) across the same window -- a check that
    the fit really sits in the diffusive regime and not in the sub-diffusive
    transient.
    """
    lo, hi = fit_lags[0] * lags[-1], fit_lags[1] * lags[-1]
    sel = (lags >= lo) & (lags <= hi)
    if sel.sum() < 2:
        sel = np.ones_like(lags, dtype=bool)
    slope, _ = np.polyfit(lags[sel], msd[sel], 1)
    good = sel & (msd > 0)
    slope_ll = (np.polyfit(np.log(lags[good]), np.log(msd[good]), 1)[0]
                if good.sum() >= 2 else np.nan)
    return slope / 4.0, slope_ll, (lags[sel][0], lags[sel][-1])


def trajectory(potential, r_c, tau, n_steps, B=1.0, seed=0, start=None,
               n_samples_per_arc=32, n_sub=None, collisions="fixed"):
    """A single walker's path: the list of contour arcs it drifts along.

    Each arc is sampled by propagating the *same* starting point for a range of
    elapsed times (the propagators accept an array of times), so the sampling
    costs one propagator call per arc and inherits its accuracy.  Consecutive
    arcs are separated by an impurity hop of length ``r_c``.
    """
    rng = np.random.default_rng(seed)
    if start is None:
        start = rng.random(2) * 2.0 * getattr(potential, "a", 1.0)
    x = np.array([float(start[0])])
    y = np.array([float(start[1])])
    arcs = []
    for _ in range(n_steps):
        dt = rng.exponential(tau) if collisions == "poisson" else tau
        ts = np.linspace(0.0, dt, n_samples_per_arc)
        xs, ys = potential.propagate(np.full(n_samples_per_arc, x[0]),
                                     np.full(n_samples_per_arc, y[0]), ts, B=B,
                                     n_sub=n_sub)
        arcs.append(np.stack([xs, ys]))
        x, y = xs[-1:].copy(), ys[-1:].copy()
        theta = rng.random() * 2.0 * np.pi
        x = x + r_c * np.cos(theta)
        y = y + r_c * np.sin(theta)
    return arcs
