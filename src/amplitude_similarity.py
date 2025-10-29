from matplotlib import rcParams
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import seaborn as sns
import pandas as pd

# Set Arial as the default font
rcParams['font.family'] = 'Arial'

class AmplitudeSimilarity:
    """
    A class for evaluating amplitude similarity between real and synthetic signals
    using Feature Selective Validation (FSV) metrics in the frequency domain.

    Metrics computed:
        - Amplitude Difference Measure (ADM)
        - Feature Difference Measure (FDM)
        - Global Difference Measure (GDM)
        - Similarity Score

    Parameters:
    ----------
    fs : int
        Sampling frequency of the signals.

    Example Usage:
    --------------
    real_data = np.random.randn(10, 2048)  # 10 real signals, each 2048 samples
    synthetic_data = np.random.randn(10, 2048) # 10 synthetic signals, each 2048 samples

    asim = AmplitudeSimilarity(fs=2048)

    metrics_dataset = asim.compute_fsv(real_data, synthetic_data)
    asim.plot_fsv(metrics_dataset)

    metrics_sample = asim.compute_fsv(real_data[0], synthetic_data[0])
    asim.plot_fsv(metrics_sample)

    References:
    ----------
    [1] IEEE Std 1597.1-2008, "Feature Selective Validation (FSV) for validation of computational electromagnetics", DOI: 10.1109/IEEESTD.2008.4661914.
    """

    def __init__(self, fs):
        """Initialize the AmplitudeSimilarity class with sampling frequency."""
        self.fs = fs

    def compute_fsv(self, real_data, synthetic_data, *,
                        mode: str = "all_vs_all",  # "zip" or "all_vs_all"
                        nperseg: int = 256, return_per_pair: bool = False,
                        return_details: bool = False):
        """
        Frequency-domain FSV metrics (ADM, FDM, GDM, Similarity) with zip/all-vs-all pairing.
        Follows the original method: PSD via Welch and index-based gradients.

        Parameters
        ----------
        real_data, synthetic_data : array
            1D (T,) or 2D (N x T) real and synthetic signals

        mode : {"zip","all_vs_all"}, default "zip"
            Pairing strategy for batches.
        nperseg : int, default 256
            Welch segment length (internally clipped to the shortest signal).

        Returns
        -------
        dict
            Single pair: {"ADM","FDM","GDM","Similarity"}.
            Batches: mean metrics; optionally per-pair list, mode and pair count.
        """

        R = np.asarray(real_data, dtype=float)
        S = np.asarray(synthetic_data, dtype=float)

        #  Single pair
        if R.ndim == 1 and S.ndim == 1:
            eff_nperseg = max(8, min(nperseg, len(R), len(S)))
            f_r, P_r = welch(R, fs=self.fs, nperseg=eff_nperseg)
            f_s, P_s = welch(S, fs=self.fs, nperseg=eff_nperseg)
            # align (should already match with same fs/nperseg)
            if f_r.shape != f_s.shape or not np.allclose(f_r, f_s):
                P_s = np.interp(f_r, f_s, P_s)

            # ORIGINAL FSV: index-based gradients
            dPr = np.gradient(P_r)
            dPs = np.gradient(P_s)

            eps = 1e-8
            adm = float(np.mean(np.abs(P_r - P_s) / (0.5 * (P_r + P_s) + eps)))
            fdm = float(np.mean(np.abs(dPr - dPs) / (0.5 * (np.abs(dPr) + np.abs(dPs)) + eps)))
            gdm = float(np.sqrt(adm ** 2 + fdm ** 2))
            sim = float(np.exp(-gdm))

            print(f"ADM: {adm:.3f}, FDM: {fdm:.3f}, GDM: {gdm:.3f}, Similarity: {sim:.3f}  | mode: single_pair")

            return {"ADM": adm, "FDM": fdm, "GDM": gdm, "Similarity": sim}



        # Batches
        if R.ndim != 2 or S.ndim != 2:
            raise ValueError("For batches, both inputs must be 2D (N x T).")

        nR, nS = R.shape[0], S.shape[0]
        if nR == 0 or nS == 0:
            raise ValueError("Need at least one real and one synthetic signal.")

        # Use a common nperseg so all PSDs share the same frequency grid
        eff_nperseg = max(8, min(nperseg, R.shape[1], S.shape[1]))

        # Precompute PSDs (shared grid expected) and index-gradients
        f_ref = None
        R_psd, R_grad = [], []
        for i in range(nR):
            f_i, P_i = welch(R[i], fs=self.fs, nperseg=eff_nperseg)
            if f_ref is None:
                f_ref = f_i
            R_psd.append(P_i)
            R_grad.append(np.gradient(P_i))

        S_psd, S_grad = [], []
        for j in range(nS):
            f_j, P_j = welch(S[j], fs=self.fs, nperseg=eff_nperseg)
            if f_ref.shape != f_j.shape or not np.allclose(f_ref, f_j):
                P_j = np.interp(f_ref, f_j, P_j)
            S_psd.append(P_j)
            S_grad.append(np.gradient(P_j))

        eps = 1e-8

        def _pair_metrics(i, j):
            Pr, dPr = R_psd[i], R_grad[i]
            Ps, dPs = S_psd[j], S_grad[j]
            adm = float(np.mean(np.abs(Pr - Ps) / (0.5 * (Pr + Ps) + eps)))
            fdm = float(np.mean(np.abs(dPr - dPs) / (0.5 * (np.abs(dPr) + np.abs(dPs)) + eps)))
            gdm = float(np.sqrt(adm ** 2 + fdm ** 2))
            sim = float(np.exp(-gdm))
            return {"ADM": adm, "FDM": fdm, "GDM": gdm, "Similarity": sim}

        # Pairing
        per_pair = []
        if mode == "zip":
            N = min(nR, nS)
            for i in range(N):
                per_pair.append(_pair_metrics(i, i))
        elif mode == "all_vs_all":
            for i in range(nR):
                for j in range(nS):
                    per_pair.append(_pair_metrics(i, j))
        else:
            raise ValueError("mode must be 'zip' or 'all_vs_all'")

        # Aggregate means
        ADM_mean = float(np.mean([m["ADM"] for m in per_pair])) if per_pair else float('nan')
        FDM_mean = float(np.mean([m["FDM"] for m in per_pair])) if per_pair else float('nan')
        GDM_mean = float(np.mean([m["GDM"] for m in per_pair])) if per_pair else float('nan')
        SIM_mean = float(np.mean([m["Similarity"] for m in per_pair])) if per_pair else float('nan')

        print(f"ADM: {ADM_mean:.3f}, FDM: {FDM_mean:.3f}, GDM: {GDM_mean:.3f}, "
                  f"Similarity: {SIM_mean:.3f}  | mode:{mode}, pairs:{len(per_pair)}")

        amplitude_metrics = {"ADM": ADM_mean, "FDM": FDM_mean, "GDM": GDM_mean, "Similarity": SIM_mean}

        if return_details:
            amplitude_metrics.update({
                "Pairs": len(per_pair),
                "Mode": mode,
                "nperseg_used": max(8, min(nperseg, real_data.shape[1], synthetic_data.shape[1]))
            })
        if return_per_pair:
            amplitude_metrics["Per-pair"] = per_pair  # list of {"ADM","FDM","GDM","Similarity"}

        return amplitude_metrics

    def plot_fsv(self, metrics):
        """
        Plot the computed Feature Selective Validation metrics in a bar chart.

        Parameters
        ----------
        metrics : dict
            A dictionary containing the computed metrics.
        """
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Metric', y='Value', hue='Metric',data=metrics_df, palette=["lightskyblue", "limegreen", "black", "grey"],legend=False)
        plt.title("Feature Selective Validation Metrics", fontsize=20, fontname='Arial')
        plt.xlabel("Metric", fontsize=16, fontname='Arial')
        plt.ylabel("Value", fontsize=16, fontname='Arial')
        plt.ylim(0, 5)
        plt.xticks(fontsize=15)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

