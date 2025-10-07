import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_score, silhouette_samples
from umap import UMAP


class Diversity:
    """
    A class to evaluate diversity between real and synthetic EEG datasets.

    Implemented Metrics:
    -------------------
    1. Gaussian-weighted Coverage Rate in Original Feature Space
    2. Gaussian-weighted Outlier Score in Original Feature Space
    3. Projection-based metrics using PCA and UMAP:
        - Centroid Distance
        - Silhouette Score

    Parameters:
    ----------
    n_components : int, optional
        Number of components for dimensionality reduction (default is 2).
    n_neighbors : int, optional
        Number of neighbors for UMAP (default is 15).
    min_dist : float, optional
        Minimum distance between points in UMAP (default is 0.1).
    random_state : int, optional
        Seed for reproducibility (default is 42).

    Example Usage:
    ------------
    real_data = np.random.randn(10, 2048)  # 10 real signals, each 2048 samples
    synthetic_data = np.random.randn(10, 2048) # 10 synthetic signals, each 2048 samples
    div = Diversity()
    results = div.compute_metrics(real_data, synthetic_data)
    div.plot_embeddings("PCA", results)

    """

    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state

    def compute_metrics(self, real_data, synthetic_data):
        """
        Compute diversity metrics between real and synthetic datasets.

        Parameters
        ----------
        real_data : array_like, shape (n_samples, n_features)
            Real dataset.
        synthetic_data : array_like, shape (n_samples, n_features)
            Synthetic dataset.

        Returns
        -------
        dict
            Dictionary with the following keys:
            - 'Coverage': Gaussian-weighted coverage score
            - 'Outliers': Gaussian-weighted outlier score
            - 'PCA_Embedding': 2D PCA projection of combined data
            - 'UMAP_Embedding': 2D UMAP projection of combined data
            - 'PCA_Compactness': class-wise silhouette avg in [0,1]
            - 'UMAP_Compactness': class-wise silhouette avg in [0,1]
            - 'PCA_Separation': normalized centroid distance in [0,1]
            - 'UMAP_Separation': normalized centroid distance in [0,1]
        """
        n_real = len(real_data)
        n_syn = len(synthetic_data)
        combined = np.vstack([real_data, synthetic_data])
        labels = np.array([0] * n_real + [1] * n_syn)

        # Coverage and Outlier metrics
        D_rs = pairwise_distances(real_data, synthetic_data, metric="euclidean")
        D_rr = pairwise_distances(real_data, metric="euclidean")
        # median of non-zero distances; fallback to 1.0
        nz = D_rr[D_rr > 0]
        sigma = float(np.median(nz)) if nz.size else 1.0
        eps = 1e-12
        sigma = max(sigma, eps)

        # Gaussian similarity to the nearest point of the other set
        # For coverage: how close each real point is to *some* synthetic point (higher = better)
        min_r2s = D_rs.min(axis=1)
        coverage = float(np.mean(np.exp(-(min_r2s ** 2) / (2 * sigma ** 2))))
        # For outliers-goodness: how close each synthetic point is to *some* real point (higher = better)
        min_s2r = D_rs.min(axis=0)
        outliers = float(np.mean(np.exp(-(min_s2r ** 2) / (2 * sigma ** 2))))  # ∈ [0,1], 1 best

        # PCA
        pca = PCA(n_components=self.n_components, svd_solver='randomized', random_state=self.random_state)
        pca_emb = pca.fit_transform(combined)

        mu_r = pca_emb[:n_real].mean(axis=0)
        mu_s = pca_emb[n_real:].mean(axis=0)
        delta = float(np.linalg.norm(mu_r - mu_s))
        denom_norm = float(np.linalg.norm(mu_r) + np.linalg.norm(mu_s)) + eps
        pca_sep = delta / denom_norm  # normalized (text version) ∈ [0,1]

        r_r = float(np.mean(np.linalg.norm(pca_emb[:n_real] - mu_r, axis=1)))
        r_s = float(np.mean(np.linalg.norm(pca_emb[n_real:] - mu_s, axis=1)))
        pca_sep_spread = delta / (delta + r_r + r_s + eps)  # spread-normalized ∈ [0,1]
        pca_sep_good = 1.0 - pca_sep_spread  # higher = better

        s_pca = silhouette_samples(pca_emb, labels)  # per-sample in [-1,1]
        pca_c = float(((s_pca[:n_real].mean() + s_pca[n_real:].mean()) / 2.0 + 1.0) / 2.0)  # -> [0,1]

        # UMAP
        umap_ = UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state
        ).fit_transform(combined)

        mu_r_u = umap_[:n_real].mean(axis=0)
        mu_s_u = umap_[n_real:].mean(axis=0)
        delta_u = float(np.linalg.norm(mu_r_u - mu_s_u))
        denom_norm_u = float(np.linalg.norm(mu_r_u) + np.linalg.norm(mu_s_u)) + eps
        umap_sep = delta_u / denom_norm_u  # normalized (text) ∈ [0,1]

        r_r_u = float(np.mean(np.linalg.norm(umap_[:n_real] - mu_r_u, axis=1)))
        r_s_u = float(np.mean(np.linalg.norm(umap_[n_real:] - mu_s_u, axis=1)))
        umap_sep_spread = delta_u / (delta_u + r_r_u + r_s_u + eps)  # spread-normalized ∈ [0,1]
        umap_sep_good = 1.0 - umap_sep_spread # higher = better

        s_umap = silhouette_samples(umap_, labels)
        umap_c = float(((s_umap[:n_real].mean() + s_umap[n_real:].mean()) / 2.0 + 1.0) / 2.0)

        # Prints

        # print all metrics
        print(f"Coverage: {coverage:.3f}")
        print(f"Outliers: {outliers:.3f}")
        print(f"PCA Compactness: {pca_c:.3f}, UMAP Compactness: {umap_c:.3f}")
        print(f"PCA Separation: {pca_sep_good:.3f}, UMAP Separation: {umap_sep_good:.3f}")

        return {
            'Coverage': coverage,
            'Outliers': outliers,
            'PCA_Embedding': pca_emb,
            'UMAP_Embedding': umap_,
            'PCA_Compactness': pca_c,
            'UMAP_Compactness': umap_c,
            'PCA_Separation': pca_sep_good,
            'UMAP_Separation': umap_sep_good
        }

    @staticmethod
    def plot_embeddings(projection_name, results, n_real=None, title=None):
        """
        Plot a 2D projection of real vs synthetic data using PCA or UMAP.

        Parameters
        ----------
        projection_name : str
            The projection to visualize. Options: 'PCA', 'UMAP'.
        results : dict
            The dictionary returned by `compute_metrics()`.
        n_real : int, optional
            Number of real samples (needed if labels are not passed). If None, assumes half are real.
        title : str, optional
            Plot title. If None, uses "<projection_name> Projection".
        """
        projection_key = f"{projection_name.upper()}_Embedding"
        if projection_key not in results:
            raise ValueError(f"Projection '{projection_name}' not found. Choose from: 'PCA', 'UMAP'.")

        emb = results[projection_key]

        # --- Guard: must be a 2D array, not an estimator ---
        if not isinstance(emb, np.ndarray):
            # Common mistake: storing the PCA/UMAP object instead of its transform
            raise TypeError(
                f"'{projection_key}' must be a 2D numpy array, got {type(emb).__name__}. "
                "Ensure compute_metrics() stores the embedding array from .fit_transform(...), "
                "e.g., 'PCA_Embedding': pca.fit_transform(combined)."
            )

        if emb.ndim != 2:
            raise ValueError(f"'{projection_key}' must be 2D, got shape {emb.shape}.")

        # If more than 2 dims were returned, use the first two for plotting
        if emb.shape[1] < 2:
            raise ValueError(f"'{projection_key}' must have at least 2 columns, got shape {emb.shape}.")
        if emb.shape[1] > 2:
            emb = emb[:, :2]

        n_total = emb.shape[0]
        if n_real is None:
            n_real = n_total // 2  # fallback assumption
        if not (0 <= n_real <= n_total):
            raise ValueError(f"n_real={n_real} is out of bounds for total n={n_total}.")

        if title is None:
            title = f"{projection_name.upper()}"

        labels = ["Real"] * n_real + ["Synthetic"] * (n_total - n_real)

        df = pd.DataFrame(emb, columns=['dim1', 'dim2'])
        df['Type'] = labels

        plt.figure(figsize=(6, 5))
        sns.scatterplot(
            data=df, x='dim1', y='dim2', hue='Type',
            palette={'Real': 'limegreen', 'Synthetic': 'lightskyblue'},
            edgecolor='black', s=80, alpha=0.8, legend=True
        )
        plt.title(title, fontsize=20)
        plt.xticks([], fontsize=15);
        plt.yticks([], fontsize=15)
        plt.xlabel('', fontsize=15);
        plt.ylabel('', fontsize=15)
        plt.grid(False)
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
        plt.legend(title='', fontsize=15)
        plt.tight_layout()
        plt.savefig(f"{projection_name.lower()}_embedding.png", dpi=300)
        plt.show()


real_data = np.random.randn(50, 2048*10)  # 10 real signals, each 2048 samples
synthetic_data = np.random.randn(50, 2048*10) # 10 synthetic signals, each 2048 samples
div = Diversity()
results = div.compute_metrics(real_data, synthetic_data)
div.plot_embeddings("PCA", results)
