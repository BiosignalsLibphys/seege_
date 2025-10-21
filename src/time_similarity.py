import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import wasserstein_distance
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Set Arial font globally
mpl.rcParams['font.family'] = 'Arial'

class TimeSimilarity:
    """
    A class to compute Hjorth parameter-based and entropy-complexity similarity metrics between real and synthetic EEG signals.

    Implemented Metrics
    -------------------
    1. Hjorth Parameters (Activity, Mobility, Complexity)
    2. Entropy/Complexity Metrics: Sample Entropy, Permutation Entropy, Lempel-Ziv Complexity

    Example usage:
    --------------
    real_data = np.random.randn(100, 128)
    synthetic_data = real_data + np.random.normal(0, 0.1, real_data.shape)

    sim = TimeSimilarity()
    hjorth_results = sim.compute_hjorth_metrics(real_data, synthetic_data, verbose=True)
    hjorth_hist = sim.plot_hjorth_histograms(real_data, synthetic_data)
    hjorth_3d = sim.plot_hjorth_3d(real_data, synthetic_data)
    ec_results = sim.compute_entropy_complexity_metrics(real_data, synthetic_data)

    Notes:
    ------
    Hjorth parameters are useful descriptors of time-domain signal dynamics, but they are translation-invariant
    and should be complemented with additional metrics, namely entropy and complexity.
    """

    @staticmethod
    def hjorth_parameters(signal):
        """
        Compute Hjorth parameters for a 1D signal.
        Returns activity, mobility, and complexity.
        """
        first_deriv = np.diff(signal)
        second_deriv = np.diff(first_deriv)
        mobility = np.std(first_deriv) / (np.std(signal) + 1e-9)
        complexity = (np.std(second_deriv) / (np.std(first_deriv) + 1e-9)) / (mobility + 1e-9)
        activity = np.var(signal)
        return activity, mobility, complexity

    def compute_hjorth_metrics(self, real_data, synthetic_data, verbose=True):
        """

        Computes hjorth metrics.

        Parameters
        ----------
        real_data : array
            1D (T,) or 2D (N x T) real signals.
        synthetic_data : array
            1D (T,) or 2D (N x T) synthetic signals.

        """
        real_hjorth = np.array([self.hjorth_parameters(sig) for sig in real_data])
        syn_hjorth = np.array([self.hjorth_parameters(sig) for sig in synthetic_data])

        real_mean = real_hjorth.mean(axis=0)
        syn_mean = syn_hjorth.mean(axis=0)

        # Compute distances
        ws_hjorth_all = [
            wasserstein_distance(real_hjorth[:, i], syn_hjorth[:, i])
            for i in range(3)
        ]
        ws_hjorth_avg = np.mean(ws_hjorth_all)

        cov = np.cov(np.vstack((real_hjorth, syn_hjorth)).T)
        inv_cov = inv(cov + np.eye(cov.shape[0]) * 1e-6)
        hjorth_mahalanobis = mahalanobis(real_mean, syn_mean, inv_cov)

        if verbose:
            print("=== Mean Hjorth Parameters ===")
            print(
                f"Real Signals: Activity={real_mean[0]:.4f}, Mobility={real_mean[1]:.4f}, Complexity={real_mean[2]:.4f}")
            print(
                f"Synthetic Signals: Activity={syn_mean[0]:.4f}, Mobility={syn_mean[1]:.4f}, Complexity={syn_mean[2]:.4f}")
            print("=== Hjorth Parameters Summary ===")
            for i, label in enumerate(["Activity", "Mobility", "Complexity"]):
                print(f"{label} - Wasserstein Distance: {ws_hjorth_all[i]:.4f}")
            print(f"Average Wasserstein Distance: {ws_hjorth_avg:.4f}")
            print(f"Mahalanobis Distance: {hjorth_mahalanobis:.4f}\n")

        return {
            "Avg_WD": ws_hjorth_avg,
            "WD_Activity": ws_hjorth_all[0],
            "WD_Mobility": ws_hjorth_all[1],
            "WD_Complexity": ws_hjorth_all[2],
            "Mahalanobis": hjorth_mahalanobis,
            "Real_Activity": real_mean[0],
            "Real_Mobility": real_mean[1],
            "Real_Complexity": real_mean[2],
            "Synthetic_Activity": syn_mean[0],
            "Synthetic_Mobility": syn_mean[1],
            "Synthetic_Complexity": syn_mean[2],
        }

    def plot_hjorth_histograms(self, real_signals, synthetic_signals):
        """
        Plot histogram comparison for each Hjorth parameter.

        Parameters
        ----------
        real_data : array
            1D (T,) or 2D (N x T) real signals.
        synthetic_data : array
            1D (T,) or 2D (N x T) synthetic signals.

        """
        real_hjorth = np.array([self.hjorth_parameters(sig) for sig in real_signals])
        synthetic_hjorth = np.array([self.hjorth_parameters(sig) for sig in synthetic_signals])

        param_names = ["Activity", "Mobility", "Complexity"]
        for i in range(3):
            plt.figure(figsize=(8, 4))
            sns.histplot(real_hjorth[:, i], label='Real', kde=True, stat="density", color='limegreen')
            sns.histplot(synthetic_hjorth[:, i], label='Synthetic', kde=True, stat="density", color='lightskyblue')
            plt.title(f"Hjorth {param_names[i]} Histogram", fontsize=20)
            plt.xlabel(param_names[i], fontsize = 15)
            plt.ylabel("Density", fontsize= 15)
            plt.legend(fontsize = 15)
            plt.tight_layout()
            plt.show()

    def plot_hjorth_3d(self, real_signals, synthetic_signals):
        """
        3D scatter plot of Activity, Mobility, and Complexity.

        Parameters
        ----------
        real_data : array
            1D (T,) or 2D (N x T) real signals.
        synthetic_data : array
            1D (T,) or 2D (N x T) synthetic signals.
        """

        real_hjorth = np.array([self.hjorth_parameters(sig) for sig in real_signals])
        synthetic_hjorth = np.array([self.hjorth_parameters(sig) for sig in synthetic_signals])

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Ensure real_hjorth and synthetic_hjorth are 2D arrays with three columns
        if real_hjorth.shape[1] != 3 or synthetic_hjorth.shape[1] != 3:
            raise ValueError("Input arrays must have three columns: Activity, Mobility, Complexity.")

        ax.scatter(real_hjorth[:, 0], real_hjorth[:, 1], real_hjorth[:, 2],
                   c='limegreen', label='Real', alpha=0.7, edgecolor='black')
        ax.scatter(synthetic_hjorth[:, 0], synthetic_hjorth[:, 1], synthetic_hjorth[:, 2],
                   c='lightskyblue', label='Synthetic', alpha=0.7, edgecolor='black')

        ax.set_title("3D Hjorth Parameters", fontsize=20)
        ax.set_xlabel("Activity", fontsize = 15)
        ax.set_ylabel("Mobility", fontsize = 15)
        ax.set_zlabel("Complexity", fontsize = 15)
        ax.legend(fontsize = 15)
        plt.tight_layout()
        plt.show()

  
    # Entropy/complexity helpers

    @staticmethod
    def _sample_entropy(x: np.ndarray, m: int = 2, r: float | None = None) -> float:
        """
        Sample Entropy (SampEn). r defaults to 0.2*std(x) if None.
        """
        x = np.asarray(x, dtype=float).ravel()
        n = x.size
        if n <= m + 1:
            return np.nan
        if r is None:
            r = 0.2 * (np.std(x) + 1e-12)

        def _phi(mm):
            # build embedded vectors of length mm
            N = n - mm + 1
            if N <= 1:
                return 0.0
            emb = np.lib.stride_tricks.sliding_window_view(x, mm)
            # Chebyshev distance (max norm)
            # Count matches within r (exclude self-matches by subtracting N later)
            d = np.max(np.abs(emb[:, None, :] - emb[None, :, :]), axis=2)
            C = np.sum(d <= r, axis=1) - 1  # remove self match
            return np.sum(C) / (N - 1) / N

        A = _phi(m + 1)
        B = _phi(m)
        if A <= 0 or B <= 0:
            return np.inf  # convention when zero matches at m+1
        return -np.log(A / B)

    @staticmethod
    def _permutation_entropy(x: np.ndarray, m: int = 3, tau: int = 1, normalize: bool = True) -> float:
        """
        Permutation Entropy (Bandt & Pompe). m=3..7 reasonable; tau>=1.
        """
        x = np.asarray(x, dtype=float).ravel()
        n = x.size
        L = n - (m - 1) * tau
        if L <= 0:
            return np.nan
        # build delayed embedding
        Y = np.vstack([x[i:i + L] for i in range(0, m * tau, tau)]).T  # shape (L, m)
        # rank-order patterns
        # argsort twice to get ranks; ties broken by stable kind
        ranks = np.argsort(np.argsort(Y, axis=1), axis=1)
        # hash patterns as tuples
        import numpy as _np
        from collections import Counter
        pats = [tuple(row) for row in ranks]
        counts = Counter(pats)
        p = _np.array(list(counts.values()), dtype=float)
        p /= p.sum()
        H = -np.sum(p * np.log(p + 1e-12))
        if normalize:
            H /= np.log(np.math.factorial(m))
        return float(H)

    @staticmethod
    def _lz_complexity(x: np.ndarray, threshold: float | None = None) -> float:
        """
        Lempel–Ziv 76 complexity (Kaspar–Schuster) on a binarized sequence.
        Normalized as c * log2(n) / n, ~1 for random Bernoulli(0.5).
        """
        import numpy as np
        b = (np.asarray(x, float).ravel() >
             (np.median(x) if threshold is None else float(threshold))).astype(np.uint8)
        n = b.size
        if n == 0:
            return np.nan

        i = 0
        c = 1
        l = 1
        k = 1
        k_max = 1
        while True:
            if i + k >= n or l + k >= n:
                c += 1
                break
            if b[i + k] == b[l + k]:
                k += 1
                if k > k_max:
                    k_max = k
            else:
                i += 1
                if i == l:
                    c += 1
                    l += k_max
                    if l >= n:
                        break
                    i = 0
                    k = 1
                    k_max = 1
                else:
                    k = 1

        # normalization
        return float(c * (np.log2(n + 1e-12) / (n + 1e-12)))

  
    # Surrogate generator (for nonlinearity check)
    
    @staticmethod
    def _fft_phase_randomized_surrogate(x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """
        Amplitude-adjusted phase randomization preserving power spectrum.
        """
        x = np.asarray(x, dtype=float).ravel()
        N = x.size
        rng = rng or np.random.default_rng()
        X = np.fft.rfft(x)
        # randomize phases except DC and Nyquist
        phases = rng.uniform(0, 2 * np.pi, size=X.size)
        phases[0] = 0.0
        if (N % 2) == 0:
            phases[-1] = 0.0
        Y = np.abs(X) * np.exp(1j * phases)
        y = np.fft.irfft(Y, n=N)
        # optional rescale to match x variance/mean
        y = (y - y.mean()) / (y.std() + 1e-12) * (x.std() + 1e-12) + x.mean()
        return y

    # Entropy/Complexity similarity + optional nonlinearity
    
    def compute_entropy_complexity_metrics(
            self,
            real_data: np.ndarray,
            synthetic_data: np.ndarray,
            *,
            sampen_m: int = 2,
            sampen_r: float | None = None,  # default = 0.2*std(signal)
            permen_m: int = 3,
            permen_tau: int = 1,
            lzc_threshold: float | None = None,  # default = median(signal)
            n_surrogates: int = 0,  # set >0 to run surrogate nonlinearity check
            verbose: bool = True
    ):
        """
        Computes SampEn, PermEn, LZC on each signal; compares real vs synthetic
        distributions with Wasserstein Distance (WD). Optionally estimates a
        nonlinearity index via surrogate testing (z-scores).
        """
        from scipy.stats import wasserstein_distance as WD
        R = np.asarray(real_data, dtype=float)
        S = np.asarray(synthetic_data, dtype=float)
        if R.ndim == 1: R = R[None, :]
        if S.ndim == 1: S = S[None, :]

        def _features(arr):
            se, pe, lz = [], [], []
            for sig in arr:
                se.append(self._sample_entropy(sig, m=sampen_m, r=sampen_r))
                pe.append(self._permutation_entropy(sig, m=permen_m, tau=permen_tau, normalize=True))
                lz.append(self._lz_complexity(sig, threshold=lzc_threshold))
            return np.array(se, float), np.array(pe, float), np.array(lz, float)

        se_R, pe_R, lz_R = _features(R)
        se_S, pe_S, lz_S = _features(S)

        out = {
            "WD_SampEn": float(WD(se_R[~np.isinf(se_R) & ~np.isnan(se_R)],
                                  se_S[~np.isinf(se_S) & ~np.isnan(se_S)])) if np.isfinite(se_R).any() and np.isfinite(
                se_S).any() else np.nan,
            "WD_PermEn": float(WD(pe_R[~np.isnan(pe_R)], pe_S[~np.isnan(pe_S)])) if np.isfinite(
                pe_R).any() and np.isfinite(pe_S).any() else np.nan,
            "WD_LZC": float(WD(lz_R[~np.isnan(lz_R)], lz_S[~np.isnan(lz_S)])) if np.isfinite(
                lz_R).any() and np.isfinite(lz_S).any() else np.nan,
            "Real_SampEn_mean": float(np.nanmean(se_R)), "Synth_SampEn_mean": float(np.nanmean(se_S)),
            "Real_PermEn_mean": float(np.nanmean(pe_R)), "Synth_PermEn_mean": float(np.nanmean(pe_S)),
            "Real_LZC_mean": float(np.nanmean(lz_R)), "Synth_LZC_mean": float(np.nanmean(lz_S)),
        }

        # --- Optional surrogate nonlinearity check (z-score of real vs surrogate)
        if n_surrogates and n_surrogates > 0:
            rng = np.random.default_rng(0)

            def _nz(arr, func):
                # per-signal z-scores relative to its own surrogates
                zs = []
                for sig in arr:
                    vals = []
                    for _ in range(n_surrogates):
                        y = self._fft_phase_randomized_surrogate(sig, rng=rng)
                        vals.append(func(y))
                    vals = np.asarray(vals, float)
                    mu, sd = np.mean(vals), np.std(vals) + 1e-12
                    zs.append((func(sig) - mu) / sd)
                return np.array(zs, float)

            z_se = _nz(R, lambda y: self._sample_entropy(y, m=sampen_m, r=sampen_r))
            z_pe = _nz(R, lambda y: self._permutation_entropy(y, m=permen_m, tau=permen_tau, normalize=True))
            z_lz = _nz(R, lambda y: self._lz_complexity(y, threshold=lzc_threshold))
            out.update({
                "NonlinearityZ_SampEn_mean": float(np.nanmean(z_se)),
                "NonlinearityZ_PermEn_mean": float(np.nanmean(z_pe)),
                "NonlinearityZ_LZC_mean": float(np.nanmean(z_lz)),
            })

        if verbose:
            def _fmt(x):
                return "nan" if not np.isfinite(x) else f"{x:.4g}"

            print("=== Entropy/Complexity Similarity ===")
            print("Wasserstein Distances (lower = more similar):")
            print(f"  SampEn WD: {_fmt(out['WD_SampEn'])}")
            print(f"  PermEn WD: {_fmt(out['WD_PermEn'])}")
            print(f"  LZC   WD: {_fmt(out['WD_LZC'])}")
            print("Means (Real vs Synthetic):")
            print(f"  SampEn: {_fmt(out['Real_SampEn_mean'])} vs {_fmt(out['Synth_SampEn_mean'])}")
            print(f"  PermEn: {_fmt(out['Real_PermEn_mean'])} vs {_fmt(out['Synth_PermEn_mean'])}")
            print(f"  LZC   : {_fmt(out['Real_LZC_mean'])} vs {_fmt(out['Synth_LZC_mean'])}")
            if n_surrogates and n_surrogates > 0:
                print(
                    "Nonlinearity z-scores (real vs phase-randomized surrogates; higher magnitude suggests nonlinearity):")
                print(f"  SampEn z̄: {_fmt(out['NonlinearityZ_SampEn_mean'])}")
                print(f"  PermEn z̄: {_fmt(out['NonlinearityZ_PermEn_mean'])}")
                print(f"  LZC   z̄: {_fmt(out['NonlinearityZ_LZC_mean'])}")
            print()

        return out
