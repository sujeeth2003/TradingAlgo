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

    def _emission_matrix(self, obs: np.ndarray) -> np.ndarray:
        """B[t, k] = p(obs[t] | state=k)"""
        T = len(obs)
        B = np.zeros((T, self.n_states))
        for k in range(self.n_states):
            mu, sigma2 = self.means[k], self.vars[k]
            B[:, k] = (1.0 / np.sqrt(2 * np.pi * sigma2)) * np.exp(
                -0.5 * (obs - mu) ** 2 / sigma2
            )
        B = np.clip(B, 1e-300, None)
        return B

    # ------------------------------------------------------------------ #
    #  Forward pass  α[t, k] = P(o1…ot, qt=k | λ)
    # ------------------------------------------------------------------ #
    def _forward(self, obs: np.ndarray, B: np.ndarray):
        T = len(obs)
        alpha = np.zeros((T, self.n_states))
        scales = np.zeros(T)

        alpha[0] = self.pi * B[0]
        scales[0] = alpha[0].sum()
        alpha[0] /= scales[0] + 1e-300

        for t in range(1, T):
            alpha[t] = (alpha[t - 1] @ self.A) * B[t]
            scales[t] = alpha[t].sum()
            alpha[t] /= scales[t] + 1e-300

        log_likelihood = np.sum(np.log(scales + 1e-300))
        return alpha, scales, log_likelihood

    # ------------------------------------------------------------------ #
    #  Backward pass  β[t, k] = P(ot+1…oT | qt=k, λ)
    # ------------------------------------------------------------------ #
    def _backward(self, obs: np.ndarray, B: np.ndarray, scales: np.ndarray):
        T = len(obs)
        beta = np.zeros((T, self.n_states))
        beta[-1] = 1.0

