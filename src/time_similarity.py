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
    A class to compute Hjorth parameter-based similarity metrics between real and synthetic EEG signals.

    Implemented Metrics
    -------------------
    1. Hjorth Parameters (Activity, Mobility, Complexity)
    2. Wasserstein Distance (per parameter and averaged)
    3. Mahalanobis Distance between parameter distributions

    Example usage:
    --------------
    real_data = np.random.randn(100, 128)
    synthetic_data = real_data + np.random.normal(0, 0.1, real_data.shape)

    sim = TimeSimilarity()
    results = sim.compute_time_metrics(real_data, synthetic_data, verbose=True)
    hjorth_hist = sim.plot_hjorth_histograms(real_data, synthetic_data)
    hjorth_3d = sim.plot_hjorth_3d(real_data, synthetic_data)

    Notes:
    ------
    Hjorth parameters are useful descriptors of time-domain signal dynamics, but they are translation-invariant
    and should be complemented with additional metrics if mean shifts or frequency differences are relevant.
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

    def compute_time_metrics(self, real_data, synthetic_data, verbose=True):
        """

        Computes time metrics.

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
