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


def test_no_net_drift():
    """Hills circulate one way and valleys the other, in equal numbers, so the
    checkerboard carries no net current: <r(t) - r(0)> stays ~0 while the
    spread grows like sqrt(4Dt)."""
    pot = SquarePyramid()
    res = simulate(pot, r_c=0.3, tau=1.0, n_steps=2000, n_walkers=4096, seed=8)
    x, y = res.positions
    n = x.size
    spread = np.sqrt(4 * res.D * res.n_steps * res.tau)
    # the mean of n displacements of typical size `spread` is itself ~
    # spread/sqrt(n); allow 4 sigma
    assert np.hypot(x.mean() - 1.0, y.mean() - 1.0) < 4 * spread / np.sqrt(n)


def test_random_field_statistics():
    """The mode sum really is a Gaussian field of correlation length xi0."""
    from mrdiff import GaussianRandomField
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-60, 60, (2, 8000))
    rs = np.array([0.5, 1.0, 2.0])
    acc, var, grad2 = [], [], []
    for s in range(10):
        f = GaussianRandomField(xi0=1.0, Gamma=1.0, n_modes=128, seed=s)
        v = f.value(x, y)
        acc.append([np.mean(v * f.value(x + r, y)) for r in rs])
        var.append(np.mean(v ** 2))
        gx, gy = f.grad(x, y)
        grad2.append(np.mean(gx ** 2 + gy ** 2))
    # <V^2> = Gamma^2, <|grad V|^2> = 2 Gamma^2 / xi0^2
    assert np.mean(var) == pytest.approx(1.0, rel=0.05)
    assert np.mean(grad2) == pytest.approx(2.0, rel=0.05)
    # <V(0)V(r)> = exp(-r^2 / 2 xi0^2)
    assert np.allclose(np.mean(acc, axis=0), np.exp(-rs ** 2 / 2), atol=0.03)


def test_random_field_contour_is_conserved():
    """RK4 + projection keeps a walker on its contour at the step size used."""
    from mrdiff import GaussianRandomField
    f = GaussianRandomField(xi0=1.0, Gamma=1 / np.sqrt(2), n_modes=64, seed=3)
    rng = np.random.default_rng(1)
    x, y = rng.uniform(-15, 15, (2, 400))
    v0 = f.value(x, y)
    for _ in range(5):                    # tau = 2 at h = 0.125
        x, y = f.propagate(x, y, 2.0, n_sub=16)
    dv = np.abs(f.value(x, y) - v0)
    # the level must stay negligible against the potential scale (Gamma = 0.71);
    # a coarser step (h = 1) instead lets it drift by ~0.2, which doubles D
    assert np.max(dv) < 1e-6
    assert np.median(dv) < 1e-10


def test_poisson_collisions_match_free_walk():
    """Exponential drift times must not change the potential-free limit."""
    flat = Sinusoid(V0=0.0)
    r_c, tau = 0.3, 0.7
    res = simulate(flat, r_c=r_c, tau=tau, n_steps=400, n_walkers=4000, seed=5,
                   collisions="poisson")
    assert res.D == pytest.approx(r_c ** 2 / (4 * tau), rel=0.05)
