"""Sanity checks on the propagators and on the diffusion measurement."""

import numpy as np
import pytest

from mrdiff import SquarePyramid, Sinusoid, simulate


def test_pyramid_contour_is_conserved():
    """The exact propagator must keep V (i.e. the contour) fixed."""
    pot = SquarePyramid()
    rng = np.random.default_rng(1)
    x, y = rng.random(2000) * 6 - 3, rng.random(2000) * 6 - 3
    v0 = pot.value(x, y)
    for _ in range(50):
        x, y = pot.propagate(x, y, 0.37)
    assert np.max(np.abs(pot.value(x, y) - v0)) < 1e-12


def test_pyramid_orbits_are_periodic():
    """After one perimeter's worth of arclength the walker returns exactly."""
    pot = SquarePyramid()
    rng = np.random.default_rng(2)
    x, y = rng.random(500) * 4, rng.random(500) * 4
    _, _, X, Y, _, ell = pot._cell(x, y)
    period = 8 * ell / pot.drift_speed_factor          # B = 1
    # propagate each walker for its own orbital period, one at a time
    for k in range(x.size):
        xn, yn = pot.propagate(x[k:k+1], y[k:k+1], period[k])
        assert abs(xn[0] - x[k]) < 1e-10 and abs(yn[0] - y[k]) < 1e-10


def test_pyramid_exact_matches_generic_rk4():
    """The analytic map agrees with brute-force RK4 on the same landscape."""
    pot = SquarePyramid()
    rng = np.random.default_rng(3)
    # Stay away from the pyramid ridges (the cell diagonals), where the
    # gradient jumps by 90 deg and no finite-difference scheme can follow the
    # corner.  These starts sit on the bottom face of the l = 0.3 contour, at
    # least 0.15 from a ridge, and travel an arclength of only 2 * 0.05.
    x = rng.random(200) * 0.3 + 0.35
    y = np.full(200, 0.2)
    xe, ye = pot.propagate(x, y, 0.05)
    xg, yg = super(SquarePyramid, pot).propagate(x, y, 0.05, n_sub=400)
    assert np.max(np.hypot(xe - xg, ye - yg)) < 1e-6


def test_sinusoid_rk4_conserves_potential():
    pot = Sinusoid()
    rng = np.random.default_rng(4)
    x, y = rng.random(300) * 2, rng.random(300) * 2
    v0 = pot.value(x, y)
    for _ in range(20):
        x, y = pot.propagate(x, y, 0.5, n_sub=64)
    assert np.max(np.abs(pot.value(x, y) - v0)) < 1e-10


def test_free_random_walk_limit():
    """With no potential the walk is a pure r_c random walk: D = r_c^2/(4 tau)."""
    flat = Sinusoid(V0=0.0)
    r_c, tau = 0.3, 0.7
    res = simulate(flat, r_c=r_c, tau=tau, n_steps=400, n_walkers=4000, seed=5)
    expected = r_c ** 2 / (4 * tau)
    assert res.D == pytest.approx(expected, rel=0.05)
    assert res.fit_slope_loglog == pytest.approx(1.0, abs=0.05)


def test_diffusive_regime_reached():
    """On the pyramid landscape the late-time MSD grows linearly in t."""
    pot = SquarePyramid()
    res = simulate(pot, r_c=0.5, tau=1.0, n_steps=3000, n_walkers=4096, seed=6)
    assert res.fit_slope_loglog == pytest.approx(1.0, abs=0.1)
    assert res.D > 0
