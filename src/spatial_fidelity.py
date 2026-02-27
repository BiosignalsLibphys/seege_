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

    def _compute_corr_matrix(self, data):
        x = np.asarray(data, dtype=float)

        # MUST be 2D: (C, T)
        if x.ndim != 2:
            raise ValueError(f"Expected a 2D array (C,T). Got shape {x.shape}")

        # Force (C, T) orientation
        if x.shape[0] > x.shape[1]:  # e.g. (1536, 20)
            x = x.T  # -> (20, 1536)

        # guard against NaNs/infs and constant channels
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        std = x.std(axis=1)
        if np.any(std == 0):
            # If a channel is constant, correlation is undefined; add tiny noise
            x = x + (1e-12 * np.random.default_rng(0).normal(size=x.shape))

        C = np.corrcoef(x)
        if C.shape != (x.shape[0], x.shape[0]):
            raise RuntimeError(f"corrcoef returned {C.shape} for input {x.shape}")

        return C

    def _frobenius_distance(self, A, B):
        return norm(A - B, ord="fro")

    def evaluate(self, real_data, synthetic_data, mode="all_vs_all"):
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
        R = np.asarray(real_data)
        S = np.asarray(synthetic_data)

        if R.ndim == 2:
            R = R[np.newaxis, ...]
        if S.ndim == 2:
            S = S[np.newaxis, ...]

        nR, nS = R.shape[0], S.shape[0]

        # Compute correlation matrices
        corr_R = [self._compute_corr_matrix(r) for r in R]
        corr_S = [self._compute_corr_matrix(s) for s in S]

        r0 = real_data[0]
        print("single sample shape:", r0.shape)  # should be (20, 1536)

        C0 = np.corrcoef(r0)
        print("corr matrix shape:", C0.shape)  # MUST be (20, 20)
        print("any NaNs in corr?:", np.isnan(C0).any())

        # RR distances
        RR = []
        for i in range(nR):
            for j in range(i + 1, nR):
                RR.append(self._frobenius_distance(corr_R[i], corr_R[j]))

        # SS distances
        SS = []
        for i in range(nS):
            for j in range(i + 1, nS):
                SS.append(self._frobenius_distance(corr_S[i], corr_S[j]))

        # RS distances
        RS = []
        for i in range(nR):
            for j in range(nS):
                RS.append(self._frobenius_distance(corr_R[i], corr_S[j]))

        RR = np.asarray(RR, dtype=float)
        SS = np.asarray(SS, dtype=float)
        RS = np.asarray(RS, dtype=float)

        RR_mean = np.nanmean(RR) if RR.size > 0 else np.nan
        RR_sd = np.nanstd(RR, ddof=1) if RR.size > 1 else np.nan

        SS_mean = np.nanmean(SS) if SS.size > 0 else np.nan
        SS_sd = np.nanstd(SS, ddof=1) if SS.size > 1 else np.nan

        RS_mean = np.nanmean(RS) if RS.size > 0 else np.nan
        RS_sd = np.nanstd(RS, ddof=1) if RS.size > 1 else np.nan

        #  Standardised score
        if RR_sd is not None and RR_sd > 0:
            z_SC = (RS_mean - RR_mean) / RR_sd
        else:
            z_SC = np.nan

        F_spatial = 1 / (1 + abs(z_SC)) if np.isfinite(z_SC) else np.nan

        # Print (kept inside the function, l
        print(f"Spatial Fidelity Score = {F_spatial:.3f}\n")
        print(f"RR  | Distance = {RR_mean:.3f} ± {RR_sd:.3f}")
        print(f"SS  | Distance = {SS_mean:.3f} ± {SS_sd:.3f}")
        print(f"RS  | Distance = {RS_mean:.3f} ± {RS_sd:.3f}")
        print(f"Standardised z = {z_SC:.3f}")

        return {
            "RR_mean": RR_mean,
            "RR_sd": RR_sd,
            "SS_mean": SS_mean,
            "SS_sd": SS_sd,
            "RS_mean": RS_mean,
            "RS_sd": RS_sd,
            "z_SC": z_SC,
            "F_spatial": F_spatial
        }