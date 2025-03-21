import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cwt, morlet2
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import mean_squared_error, structural_similarity as ssim

class ScalogramSimilarity:
    """
    A class to compute scalogram representations of signals and measure similarity between real and synthetic signals
    using various metrics (SSIM, RMSE, and Cosine Similarity).

    The scalogram is computed using **Continuous Wavelet Transform (CWT)** with **Morlet wavelets**.

    Parameters:
    ----------
    fs : int, optional
    Sampling frequency of the signals. Default is **2048 Hz**.
    frequencies : np.array, optional
    Array of frequencies for wavelet transformation. Default is **30 frequencies from 1 Hz to 30 Hz**.

    Example Usage:
    --------------
    ```python
    real_signals = np.random.randn(5, 2048)  # 5 real signals of length 2048
    synthetic_signals = np.random.randn(5, 2048)  # 5 synthetic signals of length 2048

    scalogram_analysis = ScalogramSimilarity(fs=2048)

    # Compute similarity metrics
    similarity_results = scalogram_analysis.compute_scalogram_similarity_metrics(real_signals, synthetic_signals)

    # Plot a scalogram for visual comparison
    scalogram_analysis.plot_scalogram(real_signals, synthetic_signals, signal_index_real=0, signal_index_synth=5)
    ```

    References:
    ----------
    [1] https://scikit-image.org/docs/stable/api/skimage.metrics.html
    [2] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
    [3] https://arxiv.org/html/2405.08431v3?utm_source=chatgpt.com
    """
    def __init__(self, fs=2048, frequencies=np.linspace(1, 30, 30)):
        """
        Initializes the ScalogramSimilarity class for computing scalogram representations
        of signals and evaluating their similarity using various metrics.

        Parameters:
        ----------
        fs : int, optional
            Sampling frequency of the signals in Hz (default is **2048 Hz**).
        frequencies : np.array, optional
            Array of frequencies for the wavelet transformation, used to generate the scalogram
            (default is **30 frequencies ranging from 1 Hz to 30 Hz**).
        """
        self.fs = fs
        self.frequencies = frequencies

    def _convert_to_rgb(self, image, colormap='terrain'):
        """
        Convert a single-channel grayscale image to an RGB image using a colormap.

        Parameters:
        ----------
        image : np.ndarray
            Input grayscale image.
        colormap : str, optional
            Colormap to use for conversion (default is 'terrain').

        Returns:
        -------
        np.ndarray
            RGB image with values scaled between 0 and 255.
        """
        if np.max(image) == 0:
            return np.zeros((*image.shape, 3), dtype=np.uint8)
        cmap = plt.get_cmap(colormap)
        rgb_image = cmap(image / np.max(image))[:, :, :3]
        return (rgb_image * 255).astype(np.uint8)

    def _compute_scalogram(self, signal):
        """
        Compute the scalogram using Continuous Wavelet Transform (CWT) with Morlet wavelets.

        Parameters:
        ----------
        signal : np.ndarray
            Input signal (1D array (sample)).

        Returns:
        -------
        np.ndarray
            Scalogram representation of the input signal.
        """
        signal = np.asarray(signal)
        if signal.ndim != 1:
            raise ValueError("Signal must be a 1D array.")
        widths = self.fs / self.frequencies
        return np.abs(cwt(signal, morlet2, widths, w=5.0))

    def plot_scalogram(self, real_signals, synthetic_signals, signal_index_real=0, signal_index_synth=0):
        """
        Compute and plot the scalogram for the provided real and synthetic signals.

        Parameters:
        ----------
        real_signals : np.ndarray
            Array containing real signals.
        synthetic_signals : np.ndarray
            Array containing synthetic signals.
        signal_index_real : int, optional
            Index of the real signal to plot (default is 0).
        signal_index_synth : int, optional
            Index of the synthetic signal to plot (default is 0).
        """
        real_signals = np.asarray(real_signals)
        synthetic_signals = np.asarray(synthetic_signals)

        if real_signals.ndim > 1:
            real_signal = real_signals[signal_index_real]
            synthetic_signal = synthetic_signals[signal_index_synth]
        else:
            real_signal = real_signals
            synthetic_signal = synthetic_signals

        real_scalogram = self._compute_scalogram(real_signal)
        synthetic_scalogram = self._compute_scalogram(synthetic_signal)

        vmin = min(real_scalogram.min(), synthetic_scalogram.min())
        vmax = max(real_scalogram.max(), synthetic_scalogram.max())
        total_duration = len(real_signal) / self.fs

        # Ensure step size is at least 1 to avoid ZeroDivisionError
        step_size = max(1, int(total_duration / 6))
        x_ticks = np.arange(0, int(total_duration) + 1, step=step_size)

        fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
        axs[0].imshow(real_scalogram, aspect='auto', origin='lower', extent=[0, total_duration, 1, 30], cmap='terrain',
                      vmin=vmin, vmax=vmax)
        axs[0].set_title(f'Real signal {signal_index_real}', fontsize=20, fontname='Arial')
        axs[0].set_xlabel('Time (s)', fontsize=15, fontname='Arial')
        axs[0].set_ylabel('Frequency (Hz)', fontsize=15, fontname='Arial')
        axs[0].set_xticks(x_ticks)

        axs[1].imshow(synthetic_scalogram, aspect='auto', origin='lower', extent=[0, total_duration, 1, 30],
                      cmap='terrain', vmin=vmin, vmax=vmax)
        axs[1].set_title(f'Synthetic Signal {signal_index_synth}', fontsize=20, fontname='Arial')
        axs[1].set_xlabel('Time (s)', fontsize=15)
        axs[1].set_xticks(x_ticks)

        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(axs[0].images[0], cax=cbar_ax)
        cbar.set_label('Intensity', fontsize=14, fontname='Arial')
        cbar_ax.yaxis.label.set_fontsize(15)

        fig.subplots_adjust(right=0.85)
        #plt.savefig('scalogram_arial.pdf')
        plt.show()

    def compute_scalogram_similarity_metrics(self, real_signals, synthetic_signals):
        """
        Compute similarity metrics between real and synthetic scalograms.

        Parameters
        ----------
        real_signals : np.ndarray
            Array of real signals (1D (sample) or 2D (dataset)).

        synthetic_signals : np.ndarray
            Array of synthetic signals (1D (sample) or 2D (dataset)).

        Returns:
        -------
        dict
            Dictionary containing SSIM, RMSE, and Cosine Similarity metrics.
        """
        real_signals = np.asarray(real_signals)
        synthetic_signals = np.asarray(synthetic_signals)

        # Ensure inputs are at least 2D (handle case of single signals)
        if real_signals.ndim == 1:
            real_signals = real_signals[np.newaxis, :]
        if synthetic_signals.ndim == 1:
            synthetic_signals = synthetic_signals[np.newaxis, :]

        analysis_type = "sample" if real_signals.shape[0] == 1 else "dataset"

        num_real = real_signals.shape[0]
        num_synth = synthetic_signals.shape[0]

        ssim_values, rmse_values, cosine_sim_values = [], [], []

        # Compare each real signal with each synthetic signal
        for i in range(num_real):
            for j in range(num_synth):
                real_scalogram = self._compute_scalogram(real_signals[i])
                synthetic_scalogram = self._compute_scalogram(synthetic_signals[j])

                real_rgb = self._convert_to_rgb(real_scalogram)
                synthetic_rgb = self._convert_to_rgb(synthetic_scalogram)

                ssim_values.append(ssim(real_rgb, synthetic_rgb, channel_axis=2, data_range=255))
                rmse_values.append(np.sqrt(mean_squared_error(real_scalogram, synthetic_scalogram)))
                cosine_sim_values.append(
                    cosine_similarity(real_scalogram.flatten().reshape(1, -1),
                                      synthetic_scalogram.flatten().reshape(1, -1))[0, 0])

        print(f"SSIM: {np.mean(ssim_values):.3f}, MSE: {np.mean(rmse_values):.3f}, Cosine Similarity: {np.mean(cosine_sim_values):.3f} ({analysis_type.capitalize()})")

        return {
            "Mean SSIM": np.mean(ssim_values),
            "Mean RMSE": np.mean(rmse_values),
            "Mean Cosine Similarity": np.mean(cosine_sim_values)
        }

