"""
Gaussian Hidden Markov Model — Built From Scratch
===================================================
2-state HMM for bull/bear market regime detection.
Uses Forward-Backward algorithm + EM (Baum-Welch) for parameter estimation.
"""

import numpy as np


class GaussianHMM:
    """
    2-state Gaussian HMM trained from scratch.

    States:
        0 → Bear (low-return, high-volatility)
        1 → Bull (high-return, low-volatility)

    Parameters learned via Baum-Welch (EM):
        - Transition matrix A  [n_states × n_states]
        - Emission means       [n_states]
        - Emission variances   [n_states]
        - Initial state dist   [n_states]
    """

    def __init__(self, n_states: int = 2, n_iter: int = 100, tol: float = 1e-4, random_state: int = 42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state

        # Will be set after fit()
        self.A = None        # Transition matrix
        self.means = None    # Emission means
        self.vars = None     # Emission variances
        self.pi = None       # Initial distribution
        self.log_likelihoods = []

    # ------------------------------------------------------------------ #
    #  Emission probability  p(x | state k)  — univariate Gaussian
    # ------------------------------------------------------------------ #
    def _emission_prob(self, x: float, k: int) -> float:
        mu, sigma2 = self.means[k], self.vars[k]
        return (1.0 / np.sqrt(2 * np.pi * sigma2)) * np.exp(-0.5 * (x - mu) ** 2 / sigma2)

