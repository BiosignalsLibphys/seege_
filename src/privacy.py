import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon


class Privacy:
    """
    A class for evaluating privacy-preserving properties of synthetic data by measuring
    statistical distances between real and synthetic distributions using:

    - **Wasserstein Distance (WD)**: Measures the minimum cost to transform one distribution into another.
    - **Euclidean Distance (ED)**: Measures the straight-line distance between two data points.
    - **Jensen-Shannon Divergence (JSD)**: Measures the similarity between two probability distributions.

    The class applies histogram-based distance computations to approximate distributional differences.

    Parameters:
    ----------
    num_bins : int, optional
        Number of bins used to construct histograms for metric computation (default is **30**).
    range_bins : tuple, optional
        Range of bin edges (default is **(0,1)**).

    Example Usage:
    --------------
    ```python
    real_data = [np.random.rand(1000) for _ in range(5)]  # Simulated real samples
    synthetic_data = [np.random.rand(1000) for _ in range(5)]  # Simulated synthetic samples

    privacy_evaluator = Privacy(num_bins=30)

    # Compute all privacy metrics
    metrics = privacy_evaluator.compute_privacy_metrics(real_data, synthetic_data)
    ```

    References:
    ----------
    [1] https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10568134
    [2] https://www.semanticscholar.org/reader/190169b88f0803e2a6eeb311703ea421461cf3fe
    [3] https://arxiv.org/abs/2404.06787?utm_source=chatgpt.com
    """

    def __init__(self, num_bins=30, range_bins=(0, 1)):
        """
        Initializes the Privacy class with parameters for histogram-based distance computations.

        Parameters:
        ----------
        num_bins : int, optional
            The number of bins used to construct histograms for metric computation (default is 30).
        range_bins : tuple, optional
            The range of bin edges for histogram construction, specified as (min, max) (default is (0,1)).
        """
        self.num_bins = num_bins
        self.range_bins = range_bins

    def compute_distance_metric(self, real_data, synthetic_data, metric_function):
        """
        Generalized function to compute a statistical distance metric and find the minimum distance.

        Parameters
        ----------
        real_data : np.ndarray
            Array of real signals (2D (dataset)).

        synthetic_data: np.ndarray
            Array of synthetic signals (2D (dataset)).

        metric_function : function
            Function that computes the distance between two signals.

        Returns:
        -------
        tuple:
            - mean_distance (float): Mean distance across all pairs of real and synthetic signals.
            - min_distance (float): Minimum distance found.
            - min_real_index (int): Index of the real sample that produced the min distance.
            - min_synthetic_index (int): Index of the synthetic sample that produced the min distance.
        """
        real_data = [np.asarray(signal).flatten() for signal in real_data]
        synthetic_data = [np.asarray(signal).flatten() for signal in synthetic_data]

        distances = []
        min_distance = float("inf")
        min_real_index, min_synthetic_index = -1, -1

        for i, real_signal in enumerate(real_data):
            for j, synthetic_signal in enumerate(synthetic_data):
                distance = metric_function(real_signal, synthetic_signal)
                distances.append(distance)

                # Track minimum distance
                if distance < min_distance:
                    min_distance = distance
                    min_real_index, min_synthetic_index = i, j

        return np.mean(distances), min_distance, min_real_index, min_synthetic_index

    def compute_wasserstein_distance(self, real_data, synthetic_data):
        """
        Compute the **Wasserstein Distance (WD)** between real and synthetic data distributions.

        Returns:
        -------
        tuple:
            - mean_distance (float): Average Wasserstein distance.
            - min_distance (float): Minimum Wasserstein distance found.
            - min_real_index (int): Index of real sample with minimum WD.
            - min_synthetic_index (int): Index of synthetic sample with minimum WD.
        """

        def wasserstein_func(real_signal, synthetic_signal):
            hist_1, _ = np.histogram(real_signal, bins=self.num_bins, range=self.range_bins, density=True)
            hist_2, _ = np.histogram(synthetic_signal, bins=self.num_bins, range=self.range_bins, density=True)
            bin_midpoints = np.linspace(self.range_bins[0], self.range_bins[1], self.num_bins)
            return wasserstein_distance(bin_midpoints, bin_midpoints, u_weights=hist_1, v_weights=hist_2)

        return self.compute_distance_metric(real_data, synthetic_data, wasserstein_func)

    def compute_euclidean_distance(self, real_data, synthetic_data):
        """
        Compute the **Euclidean Distance (ED)** between real and synthetic distributions
        using histogram representations.

        Returns:
        -------
        tuple:
            - mean_distance (float): Average Euclidean distance.
            - min_distance (float): Minimum Euclidean distance found.
            - min_real_index (int): Index of real sample with minimum ED.
            - min_synthetic_index (int): Index of synthetic sample with minimum ED.
        """

        def euclidean_func(real_signal, synthetic_signal):
            # Create histograms
            hist_1, _ = np.histogram(real_signal, bins=self.num_bins, range=self.range_bins, density=True)
            hist_2, _ = np.histogram(synthetic_signal, bins=self.num_bins, range=self.range_bins, density=True)

            # Normalize histograms (convert them into probability distributions)
            hist_1 = hist_1 / np.sum(hist_1)
            hist_2 = hist_2 / np.sum(hist_2)

            # Compute Euclidean distance and normalize by max possible distance
            distance = euclidean(hist_1, hist_2)

            # Normalize to ensure values stay within 0-1 range
            max_possible_distance = np.sqrt(2)  # Max L2 norm distance for probability distributions
            normalized_distance = distance / max_possible_distance

            return min(1.0, normalized_distance)  # Clip to [0,1]

        return self.compute_distance_metric(real_data, synthetic_data, euclidean_func)

    def compute_js_divergence(self, real_data, synthetic_data):
        """
        Compute the **Jensen-Shannon Divergence (JSD)** between real and synthetic distributions.

        Returns:
        -------
        tuple:
            - mean_distance (float): Average Jensen-Shannon divergence.
            - min_distance (float): Minimum Jensen-Shannon divergence found.
            - min_real_index (int): Index of real sample with minimum JSD.
            - min_synthetic_index (int): Index of synthetic sample with minimum JSD.
        """

        def jsd_func(real_signal, synthetic_signal):
            hist_1, _ = np.histogram(real_signal, bins=self.num_bins, range=self.range_bins, density=True)
            hist_2, _ = np.histogram(synthetic_signal, bins=self.num_bins, range=self.range_bins, density=True)
            hist_1 += 1e-10  # Avoid zero probabilities
            hist_2 += 1e-10
            hist_1 /= np.sum(hist_1)
            hist_2 /= np.sum(hist_2)
            return jensenshannon(hist_1, hist_2)

        return self.compute_distance_metric(real_data, synthetic_data, jsd_func)

    def compute_privacy_metrics(self, real_data, synthetic_data):
        """
        Compute all privacy metrics: **Wasserstein Distance (WD), Euclidean Distance (ED),
        and Jensen-Shannon Divergence (JSD)**.

        Parameters:
        ----------
        real_data : list or np.ndarray
            List or array of real signals.
        synthetic_data : list or np.ndarray
            List or array of synthetic signals.

        Returns:
        -------
        dict
            Dictionary containing individual metric values along with min distances and corresponding samples.
        """
        wd, wd_min, wd_real_idx, wd_synth_idx = self.compute_wasserstein_distance(real_data, synthetic_data)
        ed, ed_min, ed_real_idx, ed_synth_idx = self.compute_euclidean_distance(real_data, synthetic_data)
        jsd, jsd_min, jsd_real_idx, jsd_synth_idx = self.compute_js_divergence(real_data, synthetic_data)

        print(
            f"Wasserstein Distance: {wd:.4f} (Min: {wd_min:.4f}, Real Index: {wd_real_idx}, Synth Index: {wd_synth_idx})")
        print(
            f"Euclidean Distance: {ed:.4f} (Min: {ed_min:.4f}, Real Index: {ed_real_idx}, Synth Index: {ed_synth_idx})")
        print(
            f"Jensen-Shannon Divergence: {jsd:.4f} (Min: {jsd_min:.4f}, Real Index: {jsd_real_idx}, Synth Index: {jsd_synth_idx})")

        return {
            "wd": wd, "wd_min": wd_min, "wd_real_idx": wd_real_idx, "wd_synth_idx": wd_synth_idx,
            "ed": ed, "ed_min": ed_min, "ed_real_idx": ed_real_idx, "ed_synth_idx": ed_synth_idx,
            "jsd": jsd, "jsd_min": jsd_min, "jsd_real_idx": jsd_real_idx, "jsd_synth_idx": jsd_synth_idx
        }