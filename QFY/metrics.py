import math
import numpy as np


# ==============================================================================
# Helper functions to use when smoothing distributions
# ==============================================================================

def calc_eps(n):
    return 1./(2*n)


def smooth_probs(p, eps):
    return (eps + p) / (eps * len(p) + 1)


# ==============================================================================
# Actual metrics start here
# ==============================================================================

def AE(p_true, p_hat):
    return np.sum(np.abs(p_true - p_hat))


def KLD(p_true, p_hat, eps=1e-08):
    if eps > 0.0:
        p_true = smooth_probs(p_true, eps)
        p_hat = smooth_probs(p_hat, eps)

    return sum(p_true * np.log2(p_true / p_hat))


def NKLD(p_true, p_hat, eps=1e-08):
    exp_kld = math.exp(KLD(p_true, p_hat, eps=eps))
    return max(0., 2 * exp_kld / (1 + exp_kld) - 1)


def RAE(p_true, p_hat, eps=1e-08):
    if eps > 0.0:
        p_true = smooth_probs(p_true, eps)
        p_hat = smooth_probs(p_hat, eps)
    return (abs(p_true - p_hat) / p_true).mean(axis=-1)
