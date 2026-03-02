import numpy as np
from numpy.linalg import norm

class SpatialFidelity:
    """
    Spatial fidelity based on channel-wise correlation matrix similarity.

    This metric compares inter-channel dependency structure by computing a
    correlation matrix per sample and measuring Frobenius distances between:
      - RR: real vs real (baseline variability)
      - SS: synthetic vs synthetic (synthetic variability)
      - RS: real vs synthetic (cross-domain similarity)

    Example Usage:
    --------------
    # N samples, C channels, T timepoints
    real_data = np.random.randn(10, 20, 1500)       # (N_real=10, C=20, T=1500)
    synthetic_data = np.random.randn(10, 20, 1500)  # (N_synth=10, C=20, T=1500)

    spatial_eval = SpatialFidelity()
    results = spatial_eval.evaluate(real_data, synthetic_data)

    Notes:
    ------
    - Inputs can also be a single sample with shape (C, T); it will be expanded to (1, C, T).
    """

    def __init__(self):
        pass

    def _as_NCT(self, x, name="data", *, expected_C=None, expected_T=None):
        x = np.asarray(x, dtype=float)

        if x.ndim == 3:
            # (N,C,T)
            N, C, T = x.shape
            if C < 2:
                raise ValueError(f"{name}: need at least 2 channels (C>=2). Got {x.shape}.")
            if expected_C is not None and C != expected_C:
                raise ValueError(f"{name}: expected C={expected_C}, got {C} in shape {x.shape}.")
            if expected_T is not None and T != expected_T:
                raise ValueError(f"{name}: expected T={expected_T}, got {T} in shape {x.shape}.")
            return x

        if x.ndim == 2:
            # Ambiguous: could be (C,T) or (N,T)
            a, b = x.shape

            # If user provides expected_C / expected_T, we can disambiguate safely
            if expected_C is not None and expected_T is not None:
                if (a, b) == (expected_C, expected_T):
                    pass  # it's (C,T)
                elif (a, b) == (expected_T, expected_C):
                    x = x.T
                else:
                    raise ValueError(
                        f"{name}: expected either (C,T)=({expected_C},{expected_T}) or "
                        f"(T,C)=({expected_T},{expected_C}), got {x.shape}."
                    )
            else:
                # Heuristic: if it looks like (N,T) with N>>C, reject
                # Example: (21200,1536) should not be treated as (C,T)
                if a > 256 and (expected_T is None or b == expected_T):
                    raise ValueError(
                        f"{name}: got 2D array {x.shape}. This looks like (N,T) single-channel segments, "
                        "not (C,T). Spatial fidelity requires multi-channel samples shaped (N,C,T)."
                    )

                # Otherwise treat as (C,T) but fix orientation if needed
                if a > b:
                    x = x.T

            if x.shape[0] < 2:
                raise ValueError(f"{name}: need at least 2 channels (C>=2). Got {x.shape}.")

            return x[np.newaxis, ...]  # (1,C,T)

        raise ValueError(f"{name}: expected shape (N,C,T) or (C,T). Got {x.shape}.")

    def _compute_corr_matrix(self, data):
        x = np.asarray(data, dtype=float)

        # MUST be 2D: (C, T)
        if x.ndim != 2:
            raise ValueError(f"Expected a 2D array (C,T). Got shape {x.shape}")

        # Force (C, T) orientation
        if x.shape[0] > x.shape[1]:  # e.g. (1536, 20)
            x = x.T  # -> (20, 1536)

        if x.shape[0] < 2:
            raise ValueError(f"Need at least 2 channels to compute correlation matrix. Got {x.shape}.")

        # guard against NaNs/infs and constant channels
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        std = x.std(axis=1)
        if np.any(std == 0):
            # If a channel is constant, correlation is undefined; add tiny noise
            rng = np.random.default_rng(0)
            x = x + (1e-12 * rng.normal(size=x.shape))

        C = np.corrcoef(x)
        if C.shape != (x.shape[0], x.shape[0]):
            raise RuntimeError(f"corrcoef returned {C.shape} for input {x.shape}")

        return C

    def _frobenius_distance(self, A, B):
        return norm(A - B, ord="fro")

    def evaluate(self, real_data, synthetic_data, mode="all_vs_all", max_pairs_rr=20000, max_pairs_ss=20000, max_pairs_rs=40000, seed=0):
        """
        Evaluate spatial correlation fidelity between real and synthetic datasets.

        Parameters
        ----------
        real_data : array-like
            Real data with shape (N_real, C, T) or single sample (C, T).
        synthetic_data : array-like
            Synthetic data with shape (N_synth, C, T) or single sample (C, T).
        mode : str, optional
            Currently kept for API consistency (default "all_vs_all").
            This implementation always computes:
              - RR: all unique real-real pairs
              - SS: all unique synthetic-synthetic pairs
              - RS: all real-synthetic pairs

        Returns
        -------
        dict
            {"RR_mean": float, "RR_sd": float,
              "SS_mean": float, "SS_sd": float,
              "RS_mean": float, "RS_sd": float,
              "z_SC": float,
              "F_spatial": float }
        """

        R = self._as_NCT(real_data, name="real_data", expected_C=None, expected_T=None)
        S = self._as_NCT(synthetic_data, name="synthetic_data", expected_C=R.shape[1], expected_T=R.shape[2])

        nR, nS = R.shape[0], S.shape[0]

        if nR < 2:
            raise ValueError(f"Need at least 2 real samples for RR. Got nR={nR}.")
        if nS < 2:
            raise ValueError(f"Need at least 2 synthetic samples for SS. Got nS={nS}.")

        # Compute correlation matrices
        corr_R = [self._compute_corr_matrix(r) for r in R]
        corr_S = [self._compute_corr_matrix(s) for s in S]

        rng = np.random.default_rng(seed)

        # --- helper: sample unique pairs (i<j)
        def sample_pairs(n, max_pairs):
            total = n * (n - 1) // 2
            if max_pairs is None or max_pairs >= total:
                return [(i, j) for i in range(n) for j in range(i + 1, n)]
            pairs = set()
            while len(pairs) < max_pairs:
                i = rng.integers(0, n)
                j = rng.integers(0, n)
                if i == j:
                    continue
                a, b = (i, j) if i < j else (j, i)
                pairs.add((a, b))
            return list(pairs)

        # --- helper: sample cross pairs (i in [0,nA), j in [0,nB))
        def sample_cross_pairs(nA, nB, max_pairs):
            total = nA * nB
            if max_pairs is None or max_pairs >= total:
                return [(i, j) for i in range(nA) for j in range(nB)]
            pairs = set()
            while len(pairs) < max_pairs:
                i = int(rng.integers(0, nA))
                j = int(rng.integers(0, nB))
                pairs.add((i, j))
            return list(pairs)

        rr_pairs = sample_pairs(nR, max_pairs_rr)
        ss_pairs = sample_pairs(nS, max_pairs_ss)
        rs_pairs = sample_cross_pairs(nR, nS, max_pairs_rs)

        RR = np.asarray([self._frobenius_distance(corr_R[i], corr_R[j]) for i, j in rr_pairs], dtype=float)
        SS = np.asarray([self._frobenius_distance(corr_S[i], corr_S[j]) for i, j in ss_pairs], dtype=float)
        RS = np.asarray([self._frobenius_distance(corr_R[i], corr_S[j]) for i, j in rs_pairs], dtype=float)

        RR_mean = float(np.nanmean(RR)) if RR.size else np.nan
        RR_sd = float(np.nanstd(RR, ddof=0)) if RR.size else np.nan

        SS_mean = float(np.nanmean(SS)) if SS.size else np.nan
        SS_sd = float(np.nanstd(SS, ddof=0)) if SS.size else np.nan

        RS_mean = float(np.nanmean(RS)) if RS.size else np.nan
        RS_sd = float(np.nanstd(RS, ddof=0)) if RS.size else np.nan

        if np.isfinite(RR_sd) and RR_sd > 0 and np.isfinite(RR_mean) and np.isfinite(RS_mean):
            z_SC = (RS_mean - RR_mean) / RR_sd
        else:
            z_SC = np.nan

        F_spatial = 1 / (1 + abs(z_SC)) if np.isfinite(z_SC) else np.nan# --- compute distances for RR

        print(f"Spatial Fidelity Score = {F_spatial:.3f}\n")
        print(f"RR  | Distance = {RR_mean:.3f} ± {RR_sd:.3f}")
        print(f"SS  | Distance = {SS_mean:.3f} ± {SS_sd:.3f}")
        print(f"RS  | Distance = {RS_mean:.3f} ± {RS_sd:.3f}")
        print(f"Standardised z = {z_SC:.3f}")

        return {
            "RR_mean": RR_mean, "RR_sd": RR_sd,
            "SS_mean": SS_mean, "SS_sd": SS_sd,
            "RS_mean": RS_mean, "RS_sd": RS_sd,
            "z_SC": z_SC,
            "F_spatial": F_spatial
        }

