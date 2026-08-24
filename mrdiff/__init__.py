"""Diffusion of guiding centres on equipotential contours in a magnetic field."""

from . import theory
from .potentials import (GaussianRandomField, PeriodicGaussianField,
                         Potential, Sinusoid, SquarePyramid)
from .walk import simulate, trajectory, WalkResult

__all__ = ["Potential", "SquarePyramid", "Sinusoid", "GaussianRandomField", "PeriodicGaussianField",
           "simulate", "trajectory", "WalkResult", "theory"]
