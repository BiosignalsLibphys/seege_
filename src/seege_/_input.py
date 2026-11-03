import numpy as np


def as_NCT(x, name="data"):
    """Accept (N, T) or (N, C, T); return (N, C, T) float array."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        return x[:, None, :]
    if x.ndim == 3:
        return x
    raise ValueError(f"{name}: expected (N, T) or (N, C, T), got shape {x.shape}.")


def check_pair(real, synth):
    """Both sets as (N, C, T); C and T must match, N may differ."""
    R, S = as_NCT(real, "real_data"), as_NCT(synth, "synthetic_data")
    if R.shape[1:] != S.shape[1:]:
        raise ValueError(f"real {R.shape} and synthetic {S.shape} must share channels and length.")
    return R, S


def as_signals(X, max_signals=None, seed=0):
    """(N, C, T) -> (N*C, T): every channel of every sample is one signal.
    max_signals: random subset (fixed seed) to bound the cost of pairwise metrics."""
    S = X.reshape(-1, X.shape[-1])
    if max_signals is not None and len(S) > max_signals:
        S = S[np.random.default_rng(seed).choice(len(S), max_signals, replace=False)]
    return S
