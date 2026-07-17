# -*- coding: utf-8 -*-
"""Structured-grid GPU solver for the 2D axisymmetric (dip=0) forward problem.

A JAX-based, matrix-free alternative to the NGSolve/Netgen pipeline: fixed
graded (r, z) grid, two interchangeable discretizations (finite-volume and
Q1 FEM), PCG solver, and vmap batching over measurement depths. NGSolve
remains the validation reference and the only path for dip != 0.
"""

import os

# This machine is a shared multi-user GPU server: without this, XLA grabs 75%
# of a card's VRAM at first use. Allocate on demand instead. Must be set
# before jax initializes its backends, so it lives at package import time;
# an explicit user setting in the environment always wins (setdefault).
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
