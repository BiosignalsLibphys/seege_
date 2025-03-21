import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from umap import UMAP
from sklearn.metrics import silhouette_score

# For distribution-level metrics:
from scipy.stats import entropy  # for KL divergence
from scipy.stats import wasserstein_distance  # for 1D Earth Mover's
from sklearn.metrics import pairwise_kernels  # to help with MMD

class Diversity:
    """
    Extended class for evaluating diversity between real and synthetic datasets using:

      1) **Compactness (PCA)**:        How closely synthetic data mimics real data in PCA space.
      2) **Local Separation (t-SNE)**: How distinctly real vs. synthetic cluster in t-SNE.
      3) **Global Separation (UMAP)**: Global separation of real vs. synthetic in UMAP.
      4) **Coverage (original space)**: Fraction of real points covered by synthetic.
      5) **Outliers (original space)**: Fraction of synthetic points lying outside real distribution.

    New **distribution-level metrics**:
      6) **MMD (Maximum Mean Discrepancy)**: Kernel-based measure of distribution difference.
      7) **KL Divergence**: Compares distribution histograms in each dimension, then averages.
      8) **Wasserstein (Earth Mover’s) Distance**: 1D measure of how far mass must move to transform real -> syn.

    Parameters
    ----------
    n_components : int
        Number of components for PCA, t-SNE, UMAP (default=2).
    perplexity : int
        t-SNE perplexity (default=30).
    n_neighbors : int
        UMAP n_neighbors (default=15).
    min_dist : float
        UMAP min_dist (default=0.1).

    random_state : int or None
        Random seed for t-SNE/UMAP (default=None => nondeterministic).

    Example Usage
    -------------
    ```python
    real_data = np.random.randn(100, 5)
    synthetic_data = np.random.randn(100, 5)

    div = Diversity()
    results = div.compute_all_metrics(real_data, synthetic_data)
    print(results)  # dictionary of 8 metrics

    # Plot embeddings if desired
    div.plot_pca()
    div.plot_tsne()
    div.plot_umap()
    ```

    References
    ----------
    [1] https://github.com/stefan-jansen/synthetic-data-for-finance
    """

    def __init__(self, n_components=2, perplexity=30, n_neighbors=15, min_dist=0.1, random_state=None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state

        # For optional plotting
        self.pca_real = None
        self.pca_synthetic = None
        self.tsne_real = None
        self.tsne_synthetic = None
        self.umap_real = None
        self.umap_synthetic = None

    # ---------------------------------------------------------
    # Primary entry point: compute ALL metrics
    # ---------------------------------------------------------
    def compute_all_metrics(self, real_data, synthetic_data):
        """
        Returns a dictionary of:
          {
            'Compactness': float,
            'LocalSeparation': float,
            'GlobalSeparation': float,
            'Coverage': float,
            'Outliers': float,
            'MMD': float,
            'KL': float,
            'Wasserstein': float,
          }
        """

        # 1) Our original 5 metrics
        (compactness, local_sep, global_sep, coverage, outliers) = self._compute_original_metrics(
            real_data, synthetic_data
        )

        # 2) Distribution-level metrics
        mmd_score = self._compute_mmd(real_data, synthetic_data)
        kl_score = self._compute_kl_divergence(real_data, synthetic_data, bins=20)
        wass_score = self._compute_wasserstein(real_data, synthetic_data)

        return {
            "Compactness": compactness,
            "LocalSeparation": local_sep,
            "GlobalSeparation": global_sep,
            "Coverage": coverage,
            "Outliers": outliers,
            "MMD": mmd_score,
            "KL": kl_score,
            "Wasserstein": wass_score
        }

    # ---------------------------------------------------------
    # Original 5 metrics (PCA, t-SNE, UMAP, Coverage, Outliers)
    # ---------------------------------------------------------
    def _compute_original_metrics(self, real_data, synthetic_data):
        """
        Returns 5 metrics:
          (compactness, local_sep, global_sep, coverage, outliers)
        """

        # Basic checks
        real_data = np.atleast_2d(real_data)
        synthetic_data = np.atleast_2d(synthetic_data)

        combined_data = np.vstack((real_data, synthetic_data))
        n_real = len(real_data)

        # ~~~~~~~~~ 1) PCA => Compactness
        pca = PCA(n_components=self.n_components)
        self.pca_real = pca.fit_transform(real_data)
        self.pca_synthetic = pca.transform(synthetic_data)

        dist_real_pca = pairwise_distances(self.pca_real).mean()
        dist_synth_pca = pairwise_distances(self.pca_synthetic).mean()
        dist_combined_pca = pairwise_distances(
            np.vstack((self.pca_real, self.pca_synthetic))
        ).mean()

        diff = abs(dist_real_pca - dist_synth_pca)
        compactness = 1.0 - (diff / max(dist_combined_pca, 1e-9))
        compactness = np.clip(compactness, 0.0, 1.0)

        # ~~~~~~~~~ 2) t-SNE => Local Separation
        tsne = TSNE(
            n_components=self.n_components,
            perplexity=min(self.perplexity, len(combined_data) - 1),
            random_state=self.random_state
        )
        tsne_results = tsne.fit_transform(combined_data)
        self.tsne_real = tsne_results[:n_real]
        self.tsne_synthetic = tsne_results[n_real:]

        # After computing t-SNE embeddings:
        tsne_all = np.vstack([self.tsne_real, self.tsne_synthetic])
        labels_tsne = np.hstack([
            np.zeros(len(self.tsne_real)),
            np.ones(len(self.tsne_synthetic))
        ])

        sil_tsne = silhouette_score(tsne_all, labels_tsne)
        local_sep = (sil_tsne + 1) / 2  # optional transform to [0,1]

        #inter_dist_tsne = pairwise_distances(self.tsne_real, self.tsne_synthetic).mean()
        #max_tsne_dist = pairwise_distances(tsne_results).max()
        #local_sep = inter_dist_tsne / max(max_tsne_dist, 1e-9)
        #local_sep = np.clip(local_sep, 0.0, 1.0)

        # ~~~~~~~~~ 3) UMAP => Global Separation
        umap_ = UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state
        )
        umap_results = umap_.fit_transform(combined_data)
        self.umap_real = umap_results[:n_real]
        self.umap_synthetic = umap_results[n_real:]

        #inter_dist_umap = pairwise_distances(self.umap_real, self.umap_synthetic).mean()
        #max_umap_dist = pairwise_distances(umap_results).max()
        #global_sep = inter_dist_umap / max(max_umap_dist, 1e-9)
        #global_sep = np.clip(global_sep, 0.0, 1.0)

        umap_all = np.vstack([self.umap_real, self.umap_synthetic])
        labels_umap = np.hstack([
            np.zeros(len(self.umap_real)),
            np.ones(len(self.umap_synthetic))
        ])

        sil_umap = silhouette_score(umap_all, labels_umap)
        global_sep = (sil_umap + 1) / 2  # or keep it in [-1,1]

        # ~~~~~~~~~ 4) Coverage & 5) Outliers
        rr_distances = pairwise_distances(real_data).flatten()
        threshold = np.percentile(rr_distances, 90)  # 90th percentile => "typical max"

        dist_real_to_syn = pairwise_distances(real_data, synthetic_data)
        min_dist_per_real = dist_real_to_syn.min(axis=1)
        coverage = np.mean(min_dist_per_real <= threshold)
        coverage = np.clip(coverage, 0.0, 1.0)

        min_dist_per_syn = dist_real_to_syn.min(axis=0)
        outliers = np.mean(min_dist_per_syn >= threshold)
        outliers = np.clip(outliers, 0.0, 1.0)

        return compactness, local_sep, global_sep, coverage, outliers

    # ---------------------------------------------------------
    # 6) MMD
    # ---------------------------------------------------------
    def _compute_mmd(self, real_data, synthetic_data, kernel='rbf', gamma=None):
        """
        Computes Maximum Mean Discrepancy (MMD) for real vs. synthetic data.

        MMD ~ 0 => similar distributions
        MMD ~ large => dissimilar

        We'll do a simple pairwise kernel approach. Typically "rbf" kernel is used.
        """
        # data must be 2D
        real_data = np.atleast_2d(real_data)
        synthetic_data = np.atleast_2d(synthetic_data)

        if gamma is None:
            # a quick heuristic for gamma
            gamma = 1.0 / (real_data.shape[1] * real_data.var())

        # compute kernel matrices
        Kxx = pairwise_kernels(real_data, real_data, metric=kernel, gamma=gamma)
        Kyy = pairwise_kernels(synthetic_data, synthetic_data, metric=kernel, gamma=gamma)
        Kxy = pairwise_kernels(real_data, synthetic_data, metric=kernel, gamma=gamma)

        m = len(real_data)
        n = len(synthetic_data)

        # MMD^2 = (1/m^2)*sum(Kxx) + (1/n^2)*sum(Kyy) - (2/(m*n))*sum(Kxy)
        mmd_sq = (Kxx.sum() / (m**2)) + (Kyy.sum() / (n**2)) - 2 * (Kxy.sum() / (m*n))
        # we often take sqrt, but returning MMD^2 is also common
        mmd = np.sqrt(max(mmd_sq, 0.0))
        return mmd

    # ---------------------------------------------------------
    # 7) KL Divergence
    # ---------------------------------------------------------
    def _compute_kl_divergence(self, real_data, synthetic_data, bins=20):
        """
        Approximates KL divergence by binning each dimension (1D hist),
        summing average KL across dimensions.

        NOTE: This is a crude approach for multi-D data. It's purely for demonstration.
        """
        real_data = np.atleast_2d(real_data)
        synthetic_data = np.atleast_2d(synthetic_data)
        d = real_data.shape[1]  # number of features

        kl_list = []
        for dim in range(d):
            # 1D hist for real
            real_vals = real_data[:, dim]
            syn_vals = synthetic_data[:, dim]

            min_v = min(real_vals.min(), syn_vals.min())
            max_v = max(real_vals.max(), syn_vals.max())

            # build hist
            r_hist, edges = np.histogram(real_vals, bins=bins, range=(min_v, max_v), density=True)
            s_hist, _ = np.histogram(syn_vals, bins=bins, range=(min_v, max_v), density=True)

            # convert to float
            r_hist = r_hist.astype(np.float64) + 1e-12  # to avoid zero
            s_hist = s_hist.astype(np.float64) + 1e-12

            # KL for this dimension
            kl_dim = entropy(r_hist, s_hist)  # KL(real||synthetic)
            kl_list.append(kl_dim)

        kl_avg = np.mean(kl_list)
        return kl_avg

    # ---------------------------------------------------------
    # 8) Wasserstein Distance
    # ---------------------------------------------------------
    def _compute_wasserstein(self, real_data, synthetic_data):
        """
        Computes the average 1D Wasserstein (Earth Mover's) distance
        across each dimension, then returns the mean.

        NOTE: This is a simplistic approach for multi-D data. Typically
        multi-dimensional EMD is more complex to compute. We just average
        the 1D distances dimension by dimension.
        """
        real_data = np.atleast_2d(real_data)
        synthetic_data = np.atleast_2d(synthetic_data)
        d = real_data.shape[1]

        distances = []
        for dim in range(d):
            wdist = wasserstein_distance(real_data[:, dim], synthetic_data[:, dim])
            distances.append(wdist)

        return np.mean(distances)

    # ---------------------------------------------------------
    # Plot Helpers
    # ---------------------------------------------------------
    def _plot_embedding(self, real_emb, synth_emb, title, filename, save_fig):
        df = pd.DataFrame(np.vstack([real_emb, synth_emb]), columns=["Dim1", "Dim2"])
        df["Type"] = ["Real"] * len(real_emb) + ["Synthetic"] * len(synth_emb)

        plt.figure(figsize=(6,5))
        sns.scatterplot(data=df, x="Dim1", y="Dim2", hue="Type",
                        palette={"Real":"limegreen","Synthetic":"lightskyblue"},
                        s=60, edgecolor="black")
        plt.title(title, fontsize=14)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.tight_layout()

        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)

        if save_fig:
            plt.savefig(filename, dpi=150)
        plt.show()

    def plot_pca(self, save_fig=False):
        """
        Plots the PCA embeddings for Real vs. Synthetic.
        """
        if self.pca_real is None or self.pca_synthetic is None:
            print("PCA not computed yet.")
            return
        self._plot_embedding(self.pca_real, self.pca_synthetic, "PCA Embedding", "pca.pdf", save_fig)

    def plot_tsne(self, save_fig=False):
        """
        Plots the t-SNE embeddings for Real vs. Synthetic.
        """
        if self.tsne_real is None or self.tsne_synthetic is None:
            print("t-SNE not computed yet.")
            return
        self._plot_embedding(self.tsne_real, self.tsne_synthetic, "t-SNE Embedding", "tsne.pdf", save_fig)

    def plot_umap(self, save_fig=False):
        """
        Plots the UMAP embeddings for Real vs. Synthetic.
        """
        if self.umap_real is None or self.umap_synthetic is None:
            print("UMAP not computed yet.")
            return
        self._plot_embedding(self.umap_real, self.umap_synthetic, "UMAP Embedding", "umap.pdf", save_fig)