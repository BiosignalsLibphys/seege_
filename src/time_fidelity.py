import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import wasserstein_distance
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.stats import chi2

# Set Arial font globally
mpl.rcParams['font.family'] = 'Arial'

class TimeFidelity:
    """
    A class to compute Hjorth parameter-based metrics between real and synthetic EEG signals.

    Implemented Metrics
    -------------------
    1. Hjorth Parameters (Activity, Mobility, Complexity)
    2. Wasserstein Distance for each Hjorth parameter
        - Wasserstein Distance normalized by real-data standard deviation
        - Average Wasserstein Distance
    3. Mahalanobis Distance in Hjorth parameter space

    Example usage:
    --------------
    real_data = np.random.randn(100, 128)
    synthetic_data = real_data + np.random.normal(0, 0.1, real_data.shape)

    tf = TimeFidelity()
    hjorth_results = tf.compute_hjorth_metrics(real_data, synthetic_data, verbose=True)
    hjorth_figures = tf.plot_hjorth_metrics(real_data, synthetic_data)

    Notes:
    ------
    Hjorth parameters are useful descriptors of time-domain signal dynamics, but they are translation-invariant
    and should be complemented with additional metrics, namely complexity metrics.
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

        # Normalize per parameter using real-data scale before combining.
        #~0.1 SD → essentially identical; ~0.5 SD → small difference; ~1 SD → moderate difference; 1.5–2 SD → large difference
        real_std = real_hjorth.std(axis=0)  # shape (3,)

        ws_hjorth_std_norm = [
            ws_hjorth_all[i] / (real_std[i] + 1e-9)
            for i in range(3)
        ]
        ws_hjorth_effective_avg = np.mean(ws_hjorth_std_norm)

        cov = np.cov(np.vstack((real_hjorth, syn_hjorth)).T)
        inv_cov = inv(cov + np.eye(cov.shape[0]) * 1e-6)
        hjorth_mahalanobis = mahalanobis(real_mean, syn_mean, inv_cov)

        # Compute p-value from Mahalanobis distance
        md2 = hjorth_mahalanobis ** 2
        p_value = 1.0 - chi2.cdf(md2, df=3)

        # Compute Hjorth parameters relative differences
        rel_diff = (syn_mean - real_mean) / (real_mean + 1e-9)

        if verbose:
            print("=== Mean Hjorth Parameters ===")
            print(
                f"Real Signals: Activity={real_mean[0]:.4f}, Mobility={real_mean[1]:.4f}, Complexity={real_mean[2]:.4f}")
            print(
                f"Synthetic Signals: Activity={syn_mean[0]:.4f}, Mobility={syn_mean[1]:.4f}, Complexity={syn_mean[2]:.4f}")
            print("=== Hjorth Parameters Summary ===")
            for i, label in enumerate(["Activity", "Mobility", "Complexity"]):
                print(f"{label} - Wasserstein Distance (R-S): {ws_hjorth_all[i]:.4f}")
            for i, label in enumerate(["Activity", "Mobility", "Complexity"]):
                print(f"{label} - Wasserstein Distance Normalized (R-S): {ws_hjorth_std_norm[i]:.4f}")
            for name, rd in zip(["Activity", "Mobility", "Complexity"], rel_diff):
                print(f"{name} mean relative diff: {rd * 100:.1f}% (synthetic vs real)")
            print(f"Average Wasserstein Distance Normalized: {ws_hjorth_effective_avg:.4f}")
            print(f"Average Wasserstein Distance: {ws_hjorth_avg:.4f}")
            print(f"Mahalanobis Distance: {hjorth_mahalanobis:.4f} Approx. p-value (χ², df=3): {p_value:.4f}\n")

        return {
            "Avg_WD": ws_hjorth_avg,
            "Avg_WD_normSD": ws_hjorth_effective_avg,
            "WD_Activity": ws_hjorth_all[0],
            "WD_Mobility": ws_hjorth_all[1],
            "WD_Complexity": ws_hjorth_all[2],
            "WD_Activity_normSD": ws_hjorth_std_norm[0],
            "WD_Mobility_normSD": ws_hjorth_std_norm[1],
            "WD_Complexity_normSD": ws_hjorth_std_norm[2],
            "Mahalanobis": hjorth_mahalanobis,
            "Real_Activity": real_mean[0],
            "Real_Mobility": real_mean[1],
            "Real_Complexity": real_mean[2],
            "Synthetic_Activity": syn_mean[0],
            "Synthetic_Mobility": syn_mean[1],
            "Synthetic_Complexity": syn_mean[2],
        }

    def plot_hjorth_metrics(self, real_signals, synthetic_signals):
        """
        Plot Hjorth parameter distributions and 3D scatter in a single 2x2 figure.

        Parameters
        ----------
        real_signals : array-like
            1D (T,) or 2D (N x T) real signals.
        synthetic_signals : array-like
            1D (T,) or 2D (N x T) synthetic signals.
        """
        # Compute Hjorth parameters once
        real_hjorth = np.array([self.hjorth_parameters(sig) for sig in real_signals])
        synthetic_hjorth = np.array([self.hjorth_parameters(sig) for sig in synthetic_signals])

        # Sanity check: expect 3 parameters (Activity, Mobility, Complexity)
        if real_hjorth.shape[1] != 3 or synthetic_hjorth.shape[1] != 3:
            raise ValueError("Hjorth arrays must have shape (N, 3): Activity, Mobility, Complexity.")

        param_names = ["Activity", "Mobility", "Complexity"]

        # Create 2x2 figure
        fig = plt.figure(figsize=(14, 10))

        # Histograms: Activity, Mobility, Complexity (3 subplots)
        for i, name in enumerate(param_names):
            ax = fig.add_subplot(2, 2, i + 1)
            sns.histplot(real_hjorth[:, i],
                         label="Real",
                         kde=True,
                         stat="density",
                         color="limegreen",
                         ax=ax)
            sns.histplot(synthetic_hjorth[:, i],
                         label="Synthetic",
                         kde=True,
                         stat="density",
                         color="lightskyblue",
                         ax=ax)
            ax.set_title(f"Hjorth {name} Histogram", fontsize=20)
            ax.set_xlabel(name, fontsize=15)
            ax.set_ylabel("Density", fontsize=15)
            if i == 0:  # legend only on first histogram to avoid repetition
                ax.legend(fontsize=12)

        # 3D scatter on bottom-right subplot
        ax3d = fig.add_subplot(2, 2, 4, projection="3d")

        ax3d.scatter(
            real_hjorth[:, 0], real_hjorth[:, 1], real_hjorth[:, 2],
            c="limegreen", label="Real", alpha=0.7, edgecolor="black"
        )
        ax3d.scatter(
            synthetic_hjorth[:, 0], synthetic_hjorth[:, 1], synthetic_hjorth[:, 2],
            c="lightskyblue", label="Synthetic", alpha=0.7, edgecolor="black"
        )

        ax3d.set_title("3D Hjorth Parameters", fontsize=20)
        ax3d.set_xlabel("Activity", fontsize=15)
        ax3d.set_ylabel("Mobility", fontsize=15)
        ax3d.set_zlabel("Complexity", fontsize=15)
        ax3d.legend(fontsize=12)

        plt.tight_layout()
        plt.show()

