"""
Input handling shared by all seege_ modules.

Every public function accepts real and synthetic data in one of two formats:

    (N, T)     N single-channel segments of T samples
    (N, C, T)  N segments with C channels of T samples

N may differ between the real and synthetic sets; C and T must match.
"""
import numpy as np


def as_NCT(x, name="data"):
    """Accept (N, T) or (N, C, T); return a float (N, C, T) array (no copy if already float)."""
    x = np.asarray(x)
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(float)
    if x.ndim == 2:
        return x[:, None, :]
    if x.ndim == 3:
        return x
    raise ValueError(f"{name}: expected (N, T) or (N, C, T), got shape {x.shape}. "
                     "For a single segment use x[None, :] (single-channel) or x[None] (multichannel).")


def check_pair(real, synth):
    """Both sets as (N, C, T); C and T must match, N may differ."""
    R, S = as_NCT(real, "real_data"), as_NCT(synth, "synthetic_data")
    if R.shape[1:] != S.shape[1:]:
        raise ValueError(f"real {R.shape} and synthetic {S.shape} must have the same number of "
                         "channels and the same segment length.")
    return R, S


def as_signals(X, max_signals=None, seed=0):
    """
    (N, C, T) -> (N*C, T): every channel of every segment becomes one signal.

    max_signals : optional cap (random subset, fixed seed) to bound the cost of
                  pairwise metrics, whose run time grows with the square of the
                  number of signals.
    """
    X = as_NCT(X)
    S = X.reshape(-1, X.shape[-1])
    if max_signals is not None and len(S) > max_signals:
        idx = np.sort(np.random.default_rng(seed).choice(len(S), int(max_signals), replace=False))
        S = S[idx]
    return S
