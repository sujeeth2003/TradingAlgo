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

