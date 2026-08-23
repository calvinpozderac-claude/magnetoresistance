"""Analytic estimates from the project notes (Pozderac & Skinner, Feb 2020).

Geometry and conventions of the notes
-------------------------------------
* ``xi`` is the pyramid *half*-width: a pyramid occupies a 2*xi x 2*xi cell, its
  apex is ``Gamma`` above the cell edges, adjacent apexes are 2*xi apart, and
  |grad V| = Gamma/xi.  (In :class:`mrdiff.potentials.SquarePyramid` language,
  ``a = 2*xi`` and ``V0 = Gamma``.)
* ``T0 = xi / v_d`` with the drift speed ``v_d = |grad V| / B = Gamma/(xi B)``:
  the time to drift one half-width.  The full orbit around the outermost square
  contour takes 8*T0.
* **The notes define D through ``<|dr|^2> = 2 D t``**, i.e. D is the *trace*
  Dxx + Dyy.  Everywhere else in this repo D is the standard 2-D coefficient
  ``<|dr|^2> = 4 D t`` = Dxx.  So ``D_note = 2 * D_standard``; use
  :func:`to_standard` / :func:`to_note` to move between them.  Every function in
  this module returns the *note* convention.

The three regimes of section 3 (the Isichenko re-derivation of section 6 is
deliberately not implemented).
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc


def to_standard(D_note):
    """Notes' D (<r^2> = 2Dt) -> standard 2-D D (<r^2> = 4Dt)."""
    return 0.5 * np.asarray(D_note, dtype=float)


def to_note(D_standard):
    return 2.0 * np.asarray(D_standard, dtype=float)


def T0(xi=1.0, Gamma=1.0, B=1.0):
    """Drift time across one half-width, xi / v_d."""
    return xi ** 2 * B / Gamma


def D_collision(tau, r_c):
    """tau << T0: collisions dominate, the drift never gets anywhere.

    Notes, regime 1:  D = r_c^2 / 2 tau.
    """
    return r_c ** 2 / (2.0 * tau)


def D_intermediate(tau, r_c, xi=1.0, t0=1.0):
    """tau ~ T0, notes regime 2 (asymptote of eq. 7):  D = 2 xi r_c / sqrt(pi T0 tau)."""
    return 2.0 * xi * r_c / np.sqrt(np.pi * t0 * tau)


def D_drift(tau, r_c, xi=1.0):
    """tau >> T0, notes regime 3:  D = 4 xi r_c / (pi tau).

    Only walkers within r_c of a cell edge can change pyramid; one that does
    is taken to have moved 2*xi, the centre-to-centre spacing.
    """
    return 4.0 * xi * r_c / (np.pi * tau)


def D_eq7(tau, r_c, xi=1.0, t0=1.0):
    """The unsimplified crossover formula, notes eq. (7).

    Note that its own tau >> T0 limit is 2 xi r_c / tau, which exceeds regime 3
    by pi/2 -- the mismatch the notes patch with the factor 2/pi in fig. 3.
    """
    tau = np.asarray(tau, dtype=float)
    return 2.0 * r_c * xi * ((1.0 - np.exp(-t0 / tau)) / np.sqrt(np.pi * t0 * tau)
                             + erfc(np.sqrt(t0 / tau)) / tau)


def D_piecewise(tau, r_c, xi=1.0, t0=1.0):
    """The notes' summary table: the three regimes, joined at their crossings.

    Crossings quoted in the notes: tau/T0 = pi r_c^2 / (16 xi^2) between regimes
    1 and 2, and tau/T0 = 4/pi between regimes 2 and 3.
    """
    d1 = D_collision(tau, r_c)
    d2 = D_intermediate(tau, r_c, xi, t0)
    d3 = D_drift(tau, r_c, xi)
    return np.maximum(d1, np.minimum(d2, d3))


def regime_boundaries(r_c, xi=1.0, t0=1.0):
    """(tau_12, tau_23) in absolute time units."""
    return np.pi * r_c ** 2 / (16.0 * xi ** 2) * t0, 4.0 / np.pi * t0
