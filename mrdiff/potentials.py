"""Potential landscapes and guiding-centre (equipotential) propagators.

Physics
-------
In a strong magnetic field B = B z-hat the fast cyclotron motion averages out and
the guiding centre drifts with the E x B velocity

    v_d = (E x B) / B^2 = (z-hat x grad V) / B ,          (E = -grad V)

which is everywhere perpendicular to grad V: the guiding centre follows an
*equipotential contour* of V, at speed |grad V| / B.  Sign conventions (electron
charge, the sign of B) only set the direction of circulation, not the geometry
of the orbits or any diffusion coefficient, so we use the form above throughout.
"""

from __future__ import annotations

import numpy as np


class Potential:
    """Base class: a 2-D potential plus a guiding-centre contour propagator."""

    def value(self, x, y):
        raise NotImplementedError

    def grad(self, x, y):
        """Return (dV/dx, dV/dy)."""
        raise NotImplementedError

    def drift(self, x, y, B=1.0):
        """E x B drift velocity, v = (z-hat x grad V) / B."""
        gx, gy = self.grad(x, y)
        return -gy / B, gx / B

    def value_grad(self, x, y):
        """Return (V, dV/dx, dV/dy).  Subclasses that share work between the
        value and the gradient should override this."""
        gx, gy = self.grad(x, y)
        return self.value(x, y), gx, gy

    def _rk4_project(self, x, y, dt, B):
        """One RK4 step of the drift, then a Newton projection back onto the
        contour the walker started this step on."""
        level = self.value(x, y)
        k1x, k1y = self.drift(x, y, B)
        k2x, k2y = self.drift(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, B)
        k3x, k3y = self.drift(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, B)
        k4x, k4y = self.drift(x + dt * k3x, y + dt * k3y, B)
        x = x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
        y = y + (dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)
        # The correction is capped at one substep of arclength so that a walker
        # passing close to a saddle or an extremum, where |grad V| is small and
        # the Newton step is ill-conditioned, cannot be thrown across the
        # landscape.
        v, gx, gy = self.value_grad(x, y)
        g2 = gx * gx + gy * gy
        with np.errstate(divide="ignore", invalid="ignore"):
            fac = np.where(g2 > 1e-30, (level - v) / g2, 0.0)
        ddx, ddy = fac * gx, fac * gy
        step = np.hypot(ddx, ddy)
        cap = dt * np.sqrt(g2) / B
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(step > cap, np.where(step > 0, cap / step, 0.0), 1.0)
        return x + scale * ddx, y + scale * ddy

    def _integrate_time(self, x, y, t_total, B, h):
        """Advance every walker along its own contour by its own time.

        A *fixed* step ``h`` is used for all walkers, with the number of steps
        set per walker, rather than a fixed number of steps of size t/n.  That
        matters once the drift times are exponentially distributed: a walker
        that happens to draw t = 5 tau must not be integrated with a step five
        times coarser than the one the scheme was validated at.  Walkers that
        finish drop out of the working set, so the cost follows the total time
        actually integrated rather than the largest time in the batch.
        """
        x = np.array(x, dtype=float, copy=True)
        y = np.array(y, dtype=float, copy=True)
        rem = np.array(np.broadcast_to(t_total, x.shape), dtype=float, copy=True)
        while True:
            ia = np.flatnonzero(rem > 1e-13)
            if ia.size == 0:
                return x, y
            dt = np.minimum(h, rem[ia])
            xn, yn = self._rk4_project(x[ia], y[ia], dt, B)
            x[ia], y[ia] = xn, yn
            rem[ia] -= dt

    def orbit_period(self, x, y, B=1.0, h=0.125, max_time=np.inf,
                     r_in=1.5, r_out=6.0):
        """Time for each guiding centre to come back round its closed contour.

        The walker is integrated from its starting point until it has left a
        neighbourhood of radius ``r_out * v * h`` and then returned to within
        ``r_in * v * h`` -- both scaled by the arclength the walker covers in one
        step, so that the test works equally for a tiny orbit round an extremum
        and a large one near the percolating level.  The crossing time is then
        refined by projecting the residual offset onto the local drift velocity,
        which removes the O(h) detection lag.

        Returns ``(period, closed, x_end, y_end)``.  Walkers that have not come
        back within ``max_time`` get ``inf`` and ``False``, and their end point
        is their position at exactly ``max_time`` -- so passing ``max_time=tau``
        means a contour longer than the drift itself costs one integration, not
        two.
        """
        x = np.array(np.atleast_1d(x), dtype=float, copy=True)
        y = np.array(np.atleast_1d(y), dtype=float, copy=True)
        rx, ry = x.copy(), y.copy()
        gx, gy = self.grad(x, y)
        v0 = np.hypot(gx, gy) / B
        rin = np.maximum(r_in * v0 * h, 1e-14)
        rout = r_out * v0 * h
        limit = np.array(np.broadcast_to(max_time, x.shape), dtype=float)
        elapsed = np.zeros(x.shape)
        left = np.zeros(x.shape, dtype=bool)
        period = np.full(x.shape, np.inf)
        done = np.zeros(x.shape, dtype=bool)
        while True:
            ia = np.flatnonzero(~done & (elapsed < limit - 1e-13))
            if ia.size == 0:
                return period, done, x, y
            dt = np.minimum(h, limit[ia] - elapsed[ia])
            xn, yn = self._rk4_project(x[ia], y[ia], dt, B)
            x[ia], y[ia] = xn, yn
            elapsed[ia] += dt
            d = np.hypot(xn - rx[ia], yn - ry[ia])
            left[ia] |= d > rout[ia]
            close = left[ia] & (d < rin[ia])
            if close.any():
                ic = ia[close]
                gxc, gyc = self.grad(x[ic], y[ic])
                vx, vy = -gyc / B, gxc / B
                v2 = vx * vx + vy * vy
                with np.errstate(divide="ignore", invalid="ignore"):
                    corr = np.where(v2 > 0, ((x[ic] - rx[ic]) * vx
                                             + (y[ic] - ry[ic]) * vy) / v2, 0.0)
                period[ic] = elapsed[ic] - corr
                done[ic] = True

    def propagate(self, x, y, tau, B=1.0, n_sub=None, h=None,
                  loop_detect=False):
        """Advance guiding centres along their contours for a time ``tau``.

        RK4 with a Newton projection back onto the starting contour after every
        substep, so that V is conserved to round-off.  Potentials with an
        analytic solution (e.g. :class:`SquarePyramid`) override this.

        With ``loop_detect=True`` the closed orbit is timed first and the drift
        is reduced modulo that period.  On a contour of period P a drift of
        time tau then costs at most ~2P of integration instead of tau, which is
        what makes tau >> P affordable.  Once the walker has gone round many
        times its phase on the orbit is set by the dwell-time measure anyway, so
        replacing tau by tau mod P changes nothing statistical.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        tau_a = np.array(np.broadcast_to(np.asarray(tau, dtype=float), x.shape),
                         dtype=float)
        if h is None:
            h = (np.max(tau_a) / n_sub) if n_sub else 0.125
        if loop_detect:
            period, closed, xe, ye = self.orbit_period(x, y, B=B, h=h,
                                                       max_time=tau_a)
            # A walker whose orbit never closed inside tau has just been
            # integrated for exactly tau: keep that end point.  One that closed
            # is restarted from its true starting point for the leftover time.
            resid = np.where(closed, np.mod(tau_a, period), 0.0)
            xr, yr = self._integrate_time(x, y, resid, B, h)
            return np.where(closed, xr, xe), np.where(closed, yr, ye)
        return self._integrate_time(x, y, tau_a, B, h)


class SquarePyramid(Potential):
    """A checkerboard of square pyramids -- a "pointy" sin(pi x) sin(pi y).

    The plane is tiled by unit cells ``[i, i+1] x [j, j+1]``.  Each cell carries a
    pyramid whose apex sits at the cell centre and whose value falls linearly to
    zero on the cell edges, with the apex sign alternating like a checkerboard:

        V(x, y) = V0 * (-1)^(i+j) * (1 - 2 max(|x - cx|, |y - cy|))

    Every equipotential is therefore an axis-aligned *square* centred on a cell
    centre, and |grad V| = 2 V0 everywhere (the drift speed is uniform).  The
    zero level is the full grid of cell edges -- the percolating network that
    joins hills to valleys through the saddle points at the cell corners.

    Because the contours are squares traversed at constant speed, the guiding
    centre motion is integrated *exactly*: it is uniform motion along the
    perimeter of a square, with no discretisation error at the pyramid ridges.
    """

    def __init__(self, V0=1.0, a=1.0):
        self.V0 = float(V0)          # apex height
        self.a = float(a)            # cell (pyramid) size; lattice period is 2a

    # -- geometry helpers -------------------------------------------------
    def _cell(self, x, y):
        """Cell-centre coordinates (X, Y), checkerboard sign s, and level l."""
        a = self.a
        i = np.floor(x / a)
        j = np.floor(y / a)
        cx = (i + 0.5) * a
        cy = (j + 0.5) * a
        s = np.where((np.abs(i + j) % 2) == 0, 1.0, -1.0)
        X = x - cx
        Y = y - cy
        ell = np.maximum(np.abs(X), np.abs(Y))     # in [0, a/2]
        return cx, cy, X, Y, s, ell

    def value(self, x, y):
        _, _, _, _, s, ell = self._cell(np.asarray(x, float), np.asarray(y, float))
        return self.V0 * s * (1.0 - 2.0 * ell / self.a)

    def grad(self, x, y):
        _, _, X, Y, s, _ = self._cell(np.asarray(x, float), np.asarray(y, float))
        k = 2.0 * self.V0 / self.a
        on_x = np.abs(X) >= np.abs(Y)              # on a left/right face
        gx = np.where(on_x, -k * s * np.sign(X), 0.0)
        gy = np.where(on_x, 0.0, -k * s * np.sign(Y))
        return gx, gy

    @property
    def drift_speed_factor(self):
        """|grad V|, so that the drift speed is this divided by B."""
        return 2.0 * self.V0 / self.a

    # -- exact contour propagation ---------------------------------------
    def propagate(self, x, y, tau, B=1.0, n_sub=None, h=None,
                  loop_detect=False):
        """Exact motion along the square equipotentials for a time ``tau``.

        ``n_sub``, ``h`` and ``loop_detect`` are accepted and ignored: this map
        is closed-form and already exact for any ``tau``.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        cx, cy, X, Y, s, ell = self._cell(x, y)

        speed = self.drift_speed_factor / B          # |grad V| / B, uniform
        perim = 8.0 * ell
        # s = +1 (a hill) circulates clockwise for B > 0, s = -1 counter-clockwise
        arc = self._arclength(X, Y, ell)
        arc = arc - s * speed * tau                  # sign sets the sense of rotation
        with np.errstate(divide="ignore", invalid="ignore"):
            arc = np.where(perim > 0, np.mod(arc, np.where(perim > 0, perim, 1.0)), 0.0)
        Xn, Yn = self._position(arc, ell)
        # guiding centres sitting exactly on an apex are stationary
        tiny = perim <= 1e-14
        Xn = np.where(tiny, X, Xn)
        Yn = np.where(tiny, Y, Yn)
        return cx + Xn, cy + Yn

    @staticmethod
    def _arclength(X, Y, ell):
        """Arclength along the square |.|_inf = ell, measured counter-clockwise
        from the corner (ell, -ell)."""
        on_x = np.abs(X) >= np.abs(Y)
        right = on_x & (X > 0)
        left = on_x & ~(X > 0)
        top = ~on_x & (Y > 0)
        arc = np.where(
            right, Y + ell,
            np.where(top, 3.0 * ell - X,
                     np.where(left, 5.0 * ell - Y, 7.0 * ell + X)),
        )
        return arc

    @staticmethod
    def _position(arc, ell):
        """Inverse of :meth:`_arclength`."""
        with np.errstate(divide="ignore", invalid="ignore"):
            seg = np.where(ell > 0, np.floor(arc / np.where(ell > 0, 2.0 * ell, 1.0)), 0.0)
        seg = np.clip(seg, 0, 3)
        X = np.select(
            [seg == 0, seg == 1, seg == 2],
            [ell, 3.0 * ell - arc, -ell],
            default=arc - 7.0 * ell,
        )
        Y = np.select(
            [seg == 0, seg == 1, seg == 2],
            [arc - ell, ell, 5.0 * ell - arc],
            default=-ell,
        )
        return X, Y


class Sinusoid(Potential):
    """Smooth reference landscape V = V0 sin(pi x / a) sin(pi y / a).

    Same topology as :class:`SquarePyramid` (checkerboard of hills and valleys,
    saddles at the cell corners) but with rounded contours and a drift speed
    that vanishes at the extrema and the saddles.  Used to exercise the generic
    RK4 propagator.
    """

    def __init__(self, V0=1.0, a=1.0):
        self.V0 = float(V0)
        self.a = float(a)

    def value(self, x, y):
        k = np.pi / self.a
        return self.V0 * np.sin(k * x) * np.sin(k * y)

    def grad(self, x, y):
        k = np.pi / self.a
        return (self.V0 * k * np.cos(k * x) * np.sin(k * y),
                self.V0 * k * np.sin(k * x) * np.cos(k * y))


class GaussianRandomField(Potential):
    """An isotropic Gaussian random potential built from a sum of sine waves.

        V(r) = Gamma sqrt(2/N) sum_j cos(k_j . r + phi_j)

    with random phases, uniformly random directions, and wavenumbers drawn from
    the Rayleigh density p(k) = k xi0^2 exp(-k^2 xi0^2 / 2) -- the weights of the
    wavelengths are Gaussian in k.  That spectrum makes the correlation function
    exactly Gaussian,

        <V(0) V(r)> = Gamma^2 exp(-r^2 / 2 xi0^2),

    so ``xi0`` *is* the correlation length, and

        <V^2> = Gamma^2 ,   <|grad V|^2> = 2 Gamma^2 / xi0^2 .

    The field is statistically homogeneous and isotropic, and is not periodic:
    walkers never see an artificial lattice.  Contours are closed loops of every
    size, with the percolating level at V = 0.
    """

    def __init__(self, xi0=1.0, Gamma=1.0, n_modes=64, seed=0):
        self.xi0 = float(xi0)
        self.Gamma = float(Gamma)
        self.n_modes = int(n_modes)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)
        k = rng.rayleigh(scale=1.0 / self.xi0, size=self.n_modes)
        ang = rng.uniform(0.0, 2.0 * np.pi, self.n_modes)
        self.kx = k * np.cos(ang)
        self.ky = k * np.sin(ang)
        self.phase = rng.uniform(0.0, 2.0 * np.pi, self.n_modes)
        self.amp = self.Gamma * np.sqrt(2.0 / self.n_modes)
        # a length scale for callers that ask (used e.g. to seed walkers)
        self.a = self.xi0

    def _ph(self, x, y):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        return (self.kx[:, None] * x[None, :] + self.ky[:, None] * y[None, :]
                + self.phase[:, None])

    def value(self, x, y):
        return self.amp * np.cos(self._ph(x, y)).sum(axis=0)

    def grad(self, x, y):
        s = np.sin(self._ph(x, y))
        return -self.amp * (self.kx @ s), -self.amp * (self.ky @ s)

    def value_grad(self, x, y):
        ph = self._ph(x, y)
        s, c = np.sin(ph), np.cos(ph)
        return (self.amp * c.sum(axis=0),
                -self.amp * (self.kx @ s), -self.amp * (self.ky @ s))

    @property
    def rms_grad(self):
        """sqrt(<|grad V|^2>) = sqrt(2) Gamma / xi0 (exact as n_modes -> inf)."""
        return np.sqrt(2.0) * self.Gamma / self.xi0

    def correlation(self, r):
        """The exact ensemble correlation function <V(0) V(r)>."""
        return self.Gamma ** 2 * np.exp(-np.asarray(r, float) ** 2
                                        / (2.0 * self.xi0 ** 2))
