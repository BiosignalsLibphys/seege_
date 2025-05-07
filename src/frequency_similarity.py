import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, savgol_filter, coherence
from scipy.integrate import simps
from scipy.stats import shapiro, ttest_rel, wilcoxon
import ot

##############################################################################
#                           FrequencySimilarity Class
##############################################################################
class FrequencySimilarity:
    """
    A class for evaluating frequency similarity between real and synthetic signals
    using Power Spectral Density (PSD) analysis and statistical tests.

    This class computes relative power in different frequency bands,
    dominant frequencies, and performs statistical comparisons (normality,
    paired t-test / Wilcoxon). It can also compute coherence and approximate
    a Wasserstein distance in the frequency domain.

    Example Usage:
    --------------
    real_data = np.random.randn(10, 2048)  # 10 real signals, each 2048 samples
    synthetic_data = np.random.randn(10, 2048) # 10 synthetic signals, each 2048 samples

    frequency_analysis = FrequencySimilarity(fs=2048)
    frequency_analysis.compare_relative_power(real_data, synthetic_data)
    frequency_analysis.spectral_coherence(real_data, synthetic_data)
    frequency_analysis.plot_psd(real_data, synthetic_data, scale="linear")

    References:
    -----------
    [1] https://www.scitepress.org/Papers/2023/119908/119908.pdf
    [2] https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1219133/full
    [3] https://pmc.ncbi.nlm.nih.gov/articles/PMC11573898/
    """

    def __init__(self, fs=2048):
        """
        Initialize the FrequencySimilarity class with a given sampling frequency.

        Parameters
        ----------
        fs : int, optional
            Sampling frequency of the signals, by default 2048.
        """
        self.fs = fs

    def compute_relative_power(self, data):
        """
        Compute the relative power for different frequency bands
        (Delta, Theta, Alpha, Beta, Gamma) and determine the dominant frequency.

        Returns:
        -------
        tuple:
            - freqs (list of np.ndarray): Frequency arrays for each signal.
            - psd (list of np.ndarray): PSD arrays for each signal.
            - rel_power (dict): Relative power for each frequency band.
            - dominant_freq (list): Dominant frequencies for each signal.
        """
        bands = [0.5, 4, 8, 13, 30]
        band_names = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
        freqs, psd, rel_power = [], [], {name: [] for name in band_names}
        dominant_freq = []

        if isinstance(data, np.ndarray) and data.ndim == 1:
            data = [data]

        for sig in data:
            # Use a segment length (nperseg) to compute PSD:
            win = min(len(sig), 4 * self.fs)
            freqs_, psd_ = welch(sig, self.fs, nperseg=win, window='boxcar')
            freq_res = freqs_[1] - freqs_[0]
            total_power = simps(psd_, dx=freq_res)

            for i, name in enumerate(band_names[:-1]):
                band_idx = np.where((freqs_ >= bands[i] - 0.5) & (freqs_ <= bands[i + 1] + 0.5))[0]
                band_power = simps(psd_[band_idx], dx=freq_res)
                if band_power > total_power:
                    band_power = total_power
                rel_power[name].append(band_power / total_power if total_power else 0.0)

            # Gamma band
            gamma_power = simps(psd_[freqs_ >= bands[-1]], dx=freq_res)
            rel_power["Gamma"].append(gamma_power / total_power if total_power else 0.0)

            dominant_freq.append(freqs_[np.argmax(psd_)])
            freqs.append(freqs_)
            psd.append(psd_)

        return freqs, psd, rel_power, dominant_freq

    def test_normality(self, real_power, synthetic_power):
        """
        Test the normality of the differences between real and synthetic relative power.
        """
        results = {}
        for band in real_power.keys():
            differences = np.array(real_power[band]) - np.array(synthetic_power[band])
            if len(differences) < 3:
                results[band] = {"statistic": None, "p-value": None}
            else:
                stat, p_value = shapiro(differences)
                results[band] = {"statistic": stat, "p-value": p_value}
        return results

    def perform_statistical_tests(self, real_power, synthetic_power, normality_results):
        """
        Perform statistical tests for each frequency band based on normality results.
        """
        test_results = {}

        for band in real_power.keys():
            real_vals = np.array(real_power[band])
            synthetic_vals = np.array(synthetic_power[band])

            if len(real_vals) < 2 or len(synthetic_vals) < 2:
                continue

            # If normal => paired t-test, else => Wilcoxon
            if normality_results[band]["p-value"] is not None and \
                    normality_results[band]["p-value"] > 0.05:
                stat, p_value = ttest_rel(real_vals, synthetic_vals)
                test_type = "Paired t-test"
            else:
                stat, p_value = wilcoxon(real_vals, synthetic_vals)
                test_type = "Wilcoxon signed-rank test"

            test_results[band] = {
                "test": test_type,
                "statistic": f"{stat:.3f}",
                "p-value": f"{p_value:.3f}",
            }

        return test_results

    def compare_relative_power(self, real_data, synthetic_data):
        """
        Computes relative power for real and synthetic data, tests normality,
        and performs statistical tests (only if data are 2D).
        """
        # 1D => single sample => no stats
        if real_data.ndim == 1 or synthetic_data.ndim == 1:
            _, _, real_power, _ = self.compute_relative_power(real_data)
            _, _, synthetic_power, _ = self.compute_relative_power(synthetic_data)

            print("Relative Bands Power (Sample):")
            for band in ["Delta", "Theta", "Alpha", "Beta", "Gamma"]:
                real_mean = np.mean(real_power[band]) * 100
                synthetic_mean = np.mean(synthetic_power[band]) * 100
                diff = abs(real_mean - synthetic_mean)
                print(f"  {band}: Real: {real_mean:.2f}%, Synthetic: {synthetic_mean:.2f}%, Diff: {diff:.2f}%")
            print("⚠ Insufficient data to perform Statistical Analysis!\n")
            return {}

        # 2D => multiple signals => do full analysis
        _, _, real_power, _ = self.compute_relative_power(real_data)
        _, _, synthetic_power, _ = self.compute_relative_power(synthetic_data)

        print("Relative Bands Power (Dataset):")
        for band in ["Delta", "Theta", "Alpha", "Beta", "Gamma"]:
            real_mean = np.mean(real_power[band]) * 100
            synthetic_mean = np.mean(synthetic_power[band]) * 100
            diff = abs(real_mean - synthetic_mean)
            print(f"  {band}: Real: {real_mean:.2f}%, Synthetic: {synthetic_mean:.2f}%, Diff: {diff:.2f}%")

        normality_results = self.test_normality(real_power, synthetic_power)
        test_results = self.perform_statistical_tests(real_power, synthetic_power, normality_results)

        print("Statistical Test Results (Dataset):")
        for band, result in test_results.items():
            print(f"  {band}: Test={result['test']}, Stat={result['statistic']}, p-value={result['p-value']}")
        print()
        return test_results

    def spectral_coherence(self, real_data, synthetic_data):
        """
        Compute the mean coherence between real and synthetic signals.
        Returns the average (median-based) coherence across signals.
        """
        coherence_values = []
        # Wrap single 1D signals as a list
        if isinstance(real_data, np.ndarray) and real_data.ndim == 1:
            real_data = [real_data]
        if isinstance(synthetic_data, np.ndarray) and synthetic_data.ndim == 1:
            synthetic_data = [synthetic_data]

        # Pair up signals one-to-one
        for real_sig, synth_sig in zip(real_data, synthetic_data):
            win = min(len(real_sig) // 8, self.fs // 2)
            f, Cxy = coherence(real_sig, synth_sig, fs=self.fs, nperseg=win, window='boxcar')
            valid_freqs = (f >= 0.5) & (f <= 100)
            coherence_values.append(np.median(Cxy[valid_freqs]))

        avg_coh = np.mean(coherence_values)
        print(f"Spectral coherence: {avg_coh:.3f}")
        return avg_coh

    def spectral_wasserstein_distance(self, real_data, synthetic_data):
        """
        Compute a Wasserstein distance in the frequency domain based on PSD differences.
        """
        # Compute PSD for real vs synthetic
        _, real_psd, _, _ = self.compute_relative_power(real_data)
        _, synthetic_psd, _, _ = self.compute_relative_power(synthetic_data)

        real_psd = np.array(real_psd)
        synthetic_psd = np.array(synthetic_psd)

        # Align number of frequency bins
        min_len = min(real_psd.shape[1], synthetic_psd.shape[1])
        real_psd = real_psd[:, :min_len]
        synthetic_psd = synthetic_psd[:, :min_len]

        real_psd = real_psd.reshape(-1, min_len)
        synthetic_psd = synthetic_psd.reshape(-1, min_len)

        # Cost matrix in PSD space (Euclidean)
        cost_matrix_psd = ot.dist(real_psd, synthetic_psd, metric='euclidean')
        # Uniform distributions
        a = np.ones(real_psd.shape[0]) / real_psd.shape[0]
        b = np.ones(synthetic_psd.shape[0]) / synthetic_psd.shape[0]

        # Sinkhorn regularized Wasserstein distance
        wd_psd = ot.sinkhorn2(a, b, cost_matrix_psd, reg=0.01)
        print(f"Multivariate Wasserstein Distance (Frequency Domain): {wd_psd:.4f}")
        return wd_psd

    def plot_psd(self, real_data, synthetic_data, scale="linear", smooth=False, window_length=11, polyorder=2):
        """
        Plots the power spectral density (PSD) for real and synthetic data on a specified scale.

        Parameters
        ----------
        real_data : list or np.ndarray
            Real signals or a single signal.
        synthetic_data : list or np.ndarray
            Synthetic signals or a single signal.
        scale : str, optional
            Plot scale, "linear" or "log". Defaults to "linear".
        smooth : bool, optional
            Whether to smooth the PSD using Savitzky-Golay filter. Defaults to True.
        window_length : int, optional
            Window length for smoothing (must be odd). Defaults to 11.
        polyorder : int, optional
            Polynomial order for smoothing. Defaults to 2.
        """
        if scale not in ["linear", "log"]:
            raise ValueError("Invalid scale. Accepted values are 'linear' or 'log'.")

        # Infer analysis type
        analysis_type = "sample" if isinstance(real_data, np.ndarray) and real_data.ndim == 1 else "dataset"

        # Wrap single signals into lists
        real_data = [real_data] if isinstance(real_data, np.ndarray) and real_data.ndim == 1 else real_data
        synthetic_data = [synthetic_data] if isinstance(synthetic_data,
                                                        np.ndarray) and synthetic_data.ndim == 1 else synthetic_data

        # Compute PSD for real and synthetic data
        freqs_r, psd_r, _, _ = self.compute_relative_power(real_data)
        freqs_s, psd_s, _, _ = self.compute_relative_power(synthetic_data)

        # Compute mean PSD and frequencies
        real_psd = np.mean(psd_r, axis=0) if len(psd_r) > 1 else psd_r[0]
        synthetic_psd = np.mean(psd_s, axis=0) if len(psd_s) > 1 else psd_s[0]
        real_freqs = freqs_r[0]  # Frequencies are the same for all signals
        synthetic_freqs = freqs_s[0]  # Frequencies are the same for all signals

        # Apply smoothing if enabled
        if smooth:
            real_psd = savgol_filter(real_psd, window_length=window_length, polyorder=polyorder)
            synthetic_psd = savgol_filter(synthetic_psd, window_length=window_length, polyorder=polyorder)

        # Find common y-limits
        min_y = min(real_psd.min(), synthetic_psd.min())
        max_y = max(real_psd.max(), synthetic_psd.max())

        # Ensure y-axis starts at 0.00
        y_min = 0.00
        y_max = max_y * 1.1  # Add some headroom

        plt.figure(figsize=(12, 6))

        if scale == "linear":
            # Real data PSD (linear scale)
            plt.subplot(1, 2, 1)
            plt.plot(real_freqs, real_psd, color="blue", label="real data")
            plt.xlabel("Frequency (Hz)", fontsize=15)
            plt.ylabel("PSD ($\mu V^2$/Hz)", fontsize=15)
            plt.xlim(0, 100)
            plt.ylim(y_min, y_max)
            plt.title(f"Real data PSD ({analysis_type}) - linear scale", fontsize=16)
            plt.grid()
            plt.legend()

            # Synthetic data PSD (linear scale)
            plt.subplot(1, 2, 2)
            plt.plot(synthetic_freqs, synthetic_psd, color="grey", label="synthetic data")
            plt.xlabel("Frequency (Hz)", fontsize=15)
            plt.ylabel("PSD ($\mu V^2$/Hz)", fontsize=15)
            plt.xlim(0, 100)
            plt.ylim(y_min, y_max)
            plt.title(f"Synthetic data PSD ({analysis_type}) - linear scale", fontsize=16)
            plt.grid()
            plt.legend()

        elif scale == "log":
            # Real data PSD (log scale)
            plt.subplot(1, 2, 1)
            plt.semilogy(real_freqs, real_psd, color="lightgreen", label="real data")
            plt.xlabel("Frequency (Hz)", fontsize=15)
            plt.ylabel("PSD ($\mu V^2$/Hz, Log Scale)", fontsize=15)
            plt.xlim(0, 100)
            # plt.ylim(max(min_y, 1e-6), y_max)  # Prevent log(0)
            plt.title(f"Real data PSD ({analysis_type}) - log scale", fontsize=16)
            plt.grid()
            plt.legend()

            # Synthetic data PSD (log scale)
            plt.subplot(1, 2, 2)
            plt.semilogy(synthetic_freqs, synthetic_psd, color="grey", label="synthetic data")
            plt.xlabel("Frequency (Hz)", fontsize=15)
            plt.ylabel("PSD ($\mu V^2$/Hz, Log Scale)", fontsize=15)
            plt.xlim(0, 100)
            # plt.ylim(max(min_y, 1e-6), y_max)  # Prevent log(0)
            plt.title(f"Synthetic data PSD ({analysis_type}) - log scale", fontsize=16)
            plt.grid()
            plt.legend()

        plt.tight_layout()
        plt.show()