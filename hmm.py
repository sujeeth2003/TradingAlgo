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

        for t in range(T - 2, -1, -1):
            beta[t] = (self.A @ (B[t + 1] * beta[t + 1])) / (scales[t + 1] + 1e-300)

        return beta

    # ------------------------------------------------------------------ #
    #  E-step: compute γ and ξ
    # ------------------------------------------------------------------ #
    def _e_step(self, obs: np.ndarray, B: np.ndarray, alpha: np.ndarray, beta: np.ndarray):
        T = len(obs)

        # γ[t, k] = P(qt=k | O, λ)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

        # ξ[t, i, j] = P(qt=i, qt+1=j | O, λ)
        xi = np.zeros((T - 1, self.n_states, self.n_states))
        for t in range(T - 1):
            xi[t] = (
                alpha[t, :, None]
                * self.A
                * B[t + 1][None, :]
                * beta[t + 1][None, :]
            )
            xi[t] /= xi[t].sum() + 1e-300

        return gamma, xi

    # ------------------------------------------------------------------ #
    #  M-step: update parameters
    # ------------------------------------------------------------------ #
    def _m_step(self, obs: np.ndarray, gamma: np.ndarray, xi: np.ndarray):
        self.pi = gamma[0] / gamma[0].sum()
        self.A = xi.sum(axis=0) / xi.sum(axis=0).sum(axis=1, keepdims=True)

        gamma_sum = gamma.sum(axis=0)
        self.means = (gamma * obs[:, None]).sum(axis=0) / (gamma_sum + 1e-300)

        diff = obs[:, None] - self.means[None, :]
        self.vars = (gamma * diff ** 2).sum(axis=0) / (gamma_sum + 1e-300)
        self.vars = np.clip(self.vars, 1e-6, None)

    # ------------------------------------------------------------------ #
    #  Fit via Baum-Welch EM
    # ------------------------------------------------------------------ #
    def fit(self, obs: np.ndarray):
        rng = np.random.RandomState(self.random_state)
        obs = np.asarray(obs, dtype=float)

        # Initialise parameters
        self.pi = np.ones(self.n_states) / self.n_states
        self.A = rng.dirichlet(np.ones(self.n_states), size=self.n_states)
        sorted_idx = np.argsort(obs)
        split = len(obs) // self.n_states
        self.means = np.array([
            obs[sorted_idx[:split]].mean(),
            obs[sorted_idx[split:]].mean()
        ])
        self.vars = np.array([obs.var(), obs.var()])

        self.log_likelihoods = []
        prev_ll = -np.inf

        for _ in range(self.n_iter):
            B = self._emission_matrix(obs)
            alpha, scales, ll = self._forward(obs, B)
            beta = self._backward(obs, B, scales)
            gamma, xi = self._e_step(obs, B, alpha, beta)
            self._m_step(obs, gamma, xi)
            self.log_likelihoods.append(ll)

            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        # Ensure state 0 = bear (lower mean), state 1 = bull (higher mean)
        if self.means[0] > self.means[1]:
            self._swap_states()

        return self

    def _swap_states(self):
        self.means = self.means[::-1].copy()
        self.vars = self.vars[::-1].copy()
        self.pi = self.pi[::-1].copy()
        self.A = self.A[::-1, :][:, ::-1].copy()

    # ------------------------------------------------------------------ #
    #  Predict (Viterbi decoding)
    # ------------------------------------------------------------------ #
    def predict(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=float)
        T = len(obs)
        B = self._emission_matrix(obs)

        log_delta = np.log(self.pi + 1e-300) + np.log(B[0] + 1e-300)
        psi = np.zeros((T, self.n_states), dtype=int)

        log_A = np.log(self.A + 1e-300)

        for t in range(1, T):
            trans = log_delta[None, :] + log_A.T  # shape [n_states, n_states]
            psi[t] = trans.argmax(axis=1)
            log_delta = trans.max(axis=1) + np.log(B[t] + 1e-300)

        states = np.zeros(T, dtype=int)
        states[-1] = log_delta.argmax()
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]

        return states

    def predict_proba(self, obs: np.ndarray) -> np.ndarray:
        """Returns smoothed state probabilities γ[t, k]."""
        obs = np.asarray(obs, dtype=float)
        B = self._emission_matrix(obs)
        alpha, scales, _ = self._forward(obs, B)
        beta = self._backward(obs, B, scales)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300
        return gamma
