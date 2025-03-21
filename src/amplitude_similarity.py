import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import seaborn as sns
import pandas as pd

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
    ```python
    real_signal = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 2048))
    synthetic_signal = real_signal + 0.1 * np.random.randn(2048)

    asim = AmplitudeSimilarity(fs=2048)

    metrics = asim.compute_metrics(real_signal, synthetic_signal)
    asim.plot_metrics(metrics)
    ```

    References:
    ----------
    [1] IEEE Std 1597.1-2008, "Feature Selective Validation (FSV) for validation of computational electromagnetics", DOI: 10.1109/IEEESTD.2008.4661914.
    """

    def __init__(self, fs):
        """Initialize the AmplitudeSimilarity class with sampling frequency."""
        self.fs = fs

    def compute_metrics(self, real_data, synthetic_data):
        """
        Compute ADM, FDM, GDM, and similarity score between real and synthetic signals in frequency domain.

        Parameters:
        ----------
        real_data : np.ndarray
            Real signal(s): 1D (samples) or 2D (batch x samples).
        synthetic_data : np.ndarray
            Synthetic signal(s): same shape as real_data.

        Returns:
        -------
        dict
            Dictionary containing computed metrics.
        """

        # Ensure numpy arrays
        real_data = np.array(real_data)
        synthetic_data = np.array(synthetic_data)

        # Handle batch processing
        if real_data.ndim == 2 and synthetic_data.ndim == 2:
            metrics_list = [self.compute_metrics(r, s) for r, s in zip(real_data, synthetic_data)]

            # Average each metric across batch
            avg_metrics = {
                key: np.mean([m[key] for m in metrics_list])
                for key in metrics_list[0]
            }

            print(f"ADM: {avg_metrics['ADM']:.3f}, FDM: {avg_metrics['FDM']:.3f}, "
                  f"GDM: {avg_metrics['GDM']:.3f}, Similarity: {avg_metrics['Similarity']:.3f}")

            return avg_metrics

        # 1D processing (single pair of signals)
        f_real, psd_real = welch(real_data, fs=self.fs, nperseg=256)
        f_synth, psd_synth = welch(synthetic_data, fs=self.fs, nperseg=256)

        min_len = min(len(psd_real), len(psd_synth))
        psd_real = psd_real[:min_len]
        psd_synth = psd_synth[:min_len]

        adm_freq = np.abs(psd_real - psd_synth) / (0.5 * (psd_real + psd_synth) + 1e-8)
        adm = np.mean(adm_freq)

        grad_real = np.gradient(psd_real)
        grad_synth = np.gradient(psd_synth)
        fdm_freq = np.abs(grad_real - grad_synth) / (0.5 * (np.abs(grad_real) + np.abs(grad_synth)) + 1e-8)
        fdm = np.mean(fdm_freq)

        gdm = np.sqrt(adm ** 2 + fdm ** 2)
        similarity = np.exp(-gdm)

        print(f"ADM: {adm:.3f}, FDM: {fdm:.3f}, GDM: {gdm:.3f}, Similarity: {similarity:.3f}")

        return {"ADM": adm, "FDM": fdm, "GDM": gdm, "Similarity": similarity}

    def plot_metrics(self, metrics):
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
        plt.xlabel("Metric", fontsize=15, fontname='Arial')
        plt.ylabel("Value", fontsize=15, fontname='Arial')
        plt.ylim(0, 5)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
