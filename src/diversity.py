import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_samples
from umap import UMAP


class Diversity:
    """
    Diversity evaluation for real vs. synthetic EEG (time-series) datasets.

    Implemented Domains & Metrics
    -----------------------------
    1) Manifold Coverage (original feature space)
       • Gaussian-weighted Coverage (real → nearest synthetic)
       • Gaussian-weighted Outlier Goodness (synthetic → nearest real)

    2) Geometric (Structural) Diversity in low-dimensional embeddings
       • Compactness (class-wise silhouette, PCA & UMAP) ∈ [0, 1]
       • Separation/Overlap (spread-normalized centroid alignment, PCA & UMAP) ∈ [0, 1]
         (interpreted as “overlap”: 1≈well-aligned, 0≈disjoint)

    3) Intrinsic (Sample-level) Diversity within the synthetic set
       • Uniqueness (mean nearest-neighbour distance ratio, syn/real)
       • Global Diversity (mean pairwise distance ratio, syn/real)
       • Local Diversity (10th–50th percentile NN distance ratio, syn/real)

    Parameters
    ----------
    n_components : int, optional
        Number of components for PCA/UMAP visualization (default=2).
    n_neighbors : int, optional
        Number of neighbours for UMAP (default=15).
    min_dist : float, optional
        UMAP minimum distance (default=0.1).
    random_state : int, optional
        Random seed for reproducibility (default=42).
    max_pairs : int, optional
        Max random pairs when estimating mean pairwise distances (O(n^2) guard).
        Default=200_000.

    Example
    -------
    real_data = np.random.randn(50, 2048)       # 50 real signals, 2048 features
    synthetic_data = np.random.randn(50, 2048)      # 50 synthetic signals
    div = Diversity()
    cov_out = div.compute_coverage_diversity(real_data, synthetic_data)
    geom = div.compute_geometric_diversity(real_data, synthetic_data)
    intr = div.compute_intrinsic_diversity(real_data, synthetic_data)
    div.plot_embeddings("UMAP", geom, save="results/umap_diversity.png")
    """

    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42, max_pairs=200_000):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        self.max_pairs = max_pairs


    # 1) Manifold Coverage: Coverage & Outlier Goodness (original space)

    def compute_coverage_diversity(self, real_data: np.ndarray, synthetic_data: np.ndarray):
        """
        Compute manifold coverage metrics in the original feature space.

        Metrics
        -------
        Coverage : float in [0, 1]
            Gaussian-weighted proximity of each real sample to its nearest synthetic neighbour.
            Higher is better: 1.0 ≈ full coverage of real manifold.
        Outliers : float in [0, 1]
            Gaussian-weighted proximity of each synthetic sample to its nearest real neighbour
            (an “outlier-goodness” score). Higher is better: 1.0 ≈ few synthetic outliers.

        Notes
        -----
        We use a data-driven bandwidth σ = median(non-zero pairwise real-real distances),
        and weight nearest-neighbour distances d via w = exp(−d² / (2σ²)).

        Returns
        -------
        dict with keys:
            'Coverage', 'Outliers', 'Sigma'
        """
        real = np.asarray(real_data)
        synth = np.asarray(synthetic_data)

        # Pairwise distances
        D_rs = pairwise_distances(real, synth, metric="euclidean")
        D_rr = pairwise_distances(real, metric="euclidean")

        # Bandwidth σ from real-real distances (non-zero median)
        nz = D_rr[D_rr > 0]
        sigma = float(np.median(nz)) if nz.size else 1.0
        eps = 1e-12
        sigma = max(sigma, eps)

        # Coverage: real → nearest synthetic
        min_r2s = D_rs.min(axis=1)
        coverage = float(np.mean(np.exp(-(min_r2s ** 2) / (2 * sigma ** 2))))

        # Outlier Goodness: synthetic → nearest real
        min_s2r = D_rs.min(axis=0)
        outliers = float(np.mean(np.exp(-(min_s2r ** 2) / (2 * sigma ** 2))))

        print(f"[Manifold Coverage] Sigma: {sigma:.4f}")
        print(f"[Manifold Coverage] Coverage (real→synth): {coverage:.3f}  (↑ better)")
        print(f"[Manifold Coverage] Outlier Goodness (synth→real): {outliers:.3f}  (↑ better)")

        return {
            "Coverage": coverage,
            "Outliers": outliers,
            "Sigma": sigma,
        }


    # 2) Geometric Diversity: Compactness & Separation in PCA/UMAP spaces

    def compute_geometric_diversity(self, real_data: np.ndarray, synthetic_data: np.ndarray):
        """
        Compute structural diversity in PCA/UMAP spaces.

        Metrics
        -------
        PCA_Compactness, UMAP_Compactness : float in [0, 1]
            Rescaled class-wise silhouette (per class in {real, synthetic}, averaged and mapped from [−1, 1] → [0, 1]).
            Higher ≈ tighter class cohesion while respecting separation from the other class.
        PCA_Separation, UMAP_Separation : float in [0, 1]
            Spread-normalized centroid alignment interpreted as “overlap”:
              overlap = 1 − delta / (delta + r_real + r_synth + ε).
            Higher ≈ better alignment / overlap of real vs. synthetic centroids w.r.t. class spreads.

        Returns
        -------
        dict with keys:
            'PCA_Embedding', 'UMAP_Embedding',
            'PCA_Compactness', 'UMAP_Compactness',
            'PCA_Separation', 'UMAP_Separation'
        (Embeddings are 2D arrays for direct plotting.)
        """
        real = np.asarray(real_data)
        synth = np.asarray(synthetic_data)
        n_real = len(real)
        n_syn = len(synth)
        combined = np.vstack([real, synth])
        labels = np.array([0] * n_real + [1] * n_syn)
        eps = 1e-12

        # PCA
        pca = PCA(n_components=self.n_components, svd_solver='randomized', random_state=self.random_state)
        pca_emb = pca.fit_transform(combined)

        mu_r = pca_emb[:n_real].mean(axis=0)
        mu_s = pca_emb[n_real:].mean(axis=0)
        delta = float(np.linalg.norm(mu_r - mu_s))
        r_r = float(np.mean(np.linalg.norm(pca_emb[:n_real] - mu_r, axis=1)))
        r_s = float(np.mean(np.linalg.norm(pca_emb[n_real:] - mu_s, axis=1)))
        pca_overlap = 1.0 - (delta / (delta + r_r + r_s + eps))  # ∈ [0,1]; “Separation” interpreted as overlap

        s_pca = silhouette_samples(pca_emb, labels)  # per-sample ∈ [−1,1]
        pca_compact = float(((s_pca[:n_real].mean() + s_pca[n_real:].mean()) / 2.0 + 1.0) / 2.0)

        # ----- UMAP -----
        umap_emb = UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state
        ).fit_transform(combined)

        mu_r_u = umap_emb[:n_real].mean(axis=0)
        mu_s_u = umap_emb[n_real:].mean(axis=0)
        delta_u = float(np.linalg.norm(mu_r_u - mu_s_u))
        r_r_u = float(np.mean(np.linalg.norm(umap_emb[:n_real] - mu_r_u, axis=1)))
        r_s_u = float(np.mean(np.linalg.norm(umap_emb[n_real:] - mu_s_u, axis=1)))
        umap_overlap = 1.0 - (delta_u / (delta_u + r_r_u + r_s_u + eps))

        s_umap = silhouette_samples(umap_emb, labels)
        umap_compact = float(((s_umap[:n_real].mean() + s_umap[n_real:].mean()) / 2.0 + 1.0) / 2.0)

        print("[Geometric Diversity] PCA  -> Compactness: "
              f"{pca_compact:.3f}  | Overlap (Separation↑): {pca_overlap:.3f}")
        print("[Geometric Diversity] UMAP -> Compactness: "
              f"{umap_compact:.3f}  | Overlap (Separation↑): {umap_overlap:.3f}")

        return {
            "PCA_Embedding": pca_emb,
            "UMAP_Embedding": umap_emb,
            "PCA_Compactness": pca_compact,
            "UMAP_Compactness": umap_compact,
            "PCA_Separation": pca_overlap,
            "UMAP_Separation": umap_overlap,
        }


    # 3) Intrinsic Diversity: Uniqueness & Local/Global Diversity (syn/real)

    def compute_intrinsic_diversity(self, real_data: np.ndarray, synthetic_data: np.ndarray):
        """
        Compute intrinsic (within-synthetic) diversity relative to real.

        Metrics (ratios; syn/real)
        --------------------------
        Uniqueness_NN : float (≈1 ideal; <1 collapse; >1 over-dispersion)
            Ratio of mean nearest-neighbour distance within synthetic vs. within real.
        Global_Diversity : float (≈1 ideal)
            Ratio of mean pairwise distance within synthetic vs. within real
            (estimated from random pairs to avoid O(n^2) explosion).
        Local_Diversity_P10 / P50 : float (≈1 ideal)
            Ratios of the 10th and 50th percentiles of NN distances (syn/real),
            probing local variability beyond the mean.

        Returns
        -------
        dict with keys:
            'Uniqueness_NN', 'Global_Diversity', 'Local_Diversity_P10', 'Local_Diversity_P50'
        """
        real = np.asarray(real_data)
        synth = np.asarray(synthetic_data)
        eps = 1e-12

        # Helper: mean NN distance & percentiles within a set ----
        def _nn_stats(X, *, jitter=0.0, rng=None):
            """
            Returns (mean_nn, p10_nn, p50_nn) with NaN-safe handling.
            - Masks self-distances with NaN and uses nanmin.
            - Optionally adds tiny jitter to break exact duplicates (keeps NN >= 0).
            """
            X = np.asarray(X, dtype=float)
            n = X.shape[0]
            if n < 2:
                # No NN notion for singleton sets
                return np.nan, np.nan, np.nan

            if jitter and rng is not None:
                X = X + jitter * rng.standard_normal(X.shape)

            D = pairwise_distances(X, metric="euclidean")
            # Mask self-distances instead of adding +inf
            np.fill_diagonal(D, np.nan)

            # Row-wise NN with NaN-safe min
            nn = np.nanmin(D, axis=1)

            # If any row had only NaNs (pathological), replace with 0 so stats remain finite
            if np.isnan(nn).any():
                nn = np.where(np.isfinite(nn), nn, 0.0)

            return float(np.mean(nn)), float(np.percentile(nn, 10)), float(np.percentile(nn, 50))

        # Helper: mean pairwise distance with random subsampling ----
        def _mean_pairwise(X, max_pairs=200_000):
            """
            Computes mean pairwise distance within X.
            Uses random subsampling for efficiency and guards against NaNs/infs.
            """
            X = np.asarray(X, dtype=float)
            n = X.shape[0]
            if n < 2:
                return np.nan  # undefined for singleton

            # Full or sampled distance computation
            total_pairs = n * (n - 1) // 2
            if total_pairs <= max_pairs:
                D = pairwise_distances(X, metric="euclidean")
                iu = np.triu_indices(n, k=1)
                vals = D[iu]
            else:
                rng = np.random.default_rng()
                idx_i = rng.integers(0, n, size=max_pairs)
                idx_j = rng.integers(0, n, size=max_pairs)
                mask = idx_i != idx_j
                vals = np.linalg.norm(X[idx_i[mask]] - X[idx_j[mask]], axis=1)

            # Clean invalid entries
            vals = vals[np.isfinite(vals)]
            return float(np.mean(vals)) if vals.size else np.nan

        # Compute stats
        rng = np.random.default_rng(self.random_state)

        nn_real_mean, nn_real_p10, nn_real_p50 = _nn_stats(real, jitter=0.0, rng=rng)
        nn_syn_mean, nn_syn_p10, nn_syn_p50 = _nn_stats(synth, jitter=0.0, rng=rng)

        pw_real = _mean_pairwise(real, self.max_pairs)
        pw_syn  = _mean_pairwise(synth, self.max_pairs)

        eps = 1e-12

        def safe_ratio(num, den):
            return float(num) / float(den + eps) if np.isfinite(num) and np.isfinite(den) else np.nan

        uniqueness_nn = safe_ratio(nn_syn_mean, nn_real_mean)
        global_div = safe_ratio(pw_syn, pw_real)
        local_div_p10 = safe_ratio(nn_syn_p10, nn_real_p10)
        local_div_p50 = safe_ratio(nn_syn_p50, nn_real_p50)

        print("[Intrinsic Diversity] Uniqueness (NN ratio, syn/real): "
              f"{uniqueness_nn:.3f}  (~1 ideal; <1 collapse; >1 over-dispersion)")
        print("[Intrinsic Diversity] Global Diversity (pairwise ratio, syn/real): "
              f"{global_div:.3f}")
        print("[Intrinsic Diversity] Local Diversity P10 / P50 (NN ratio): "
              f"{local_div_p10:.3f} / {local_div_p50:.3f}")

        return {
            "Uniqueness_NN": uniqueness_nn,
            "Global_Diversity": global_div,
            "Local_Diversity_P10": local_div_p10,
            "Local_Diversity_P50": local_div_p50,
        }


    # 4) Plotting: same as before (expects *_Embedding in results dict)

    @staticmethod
    def plot_embeddings(projection_name: str, geom: dict, save: str = None):
        """
        Plot a 2D projection of real vs synthetic data using PCA or UMAP.

        Parameters
        ----------
        projection_name : {'PCA', 'UMAP'}
            Which embedding to visualize.
        geom : dict
            The dictionary returned by `compute_geometric_diversity`, containing
            '<PROJECTION>_Embedding' arrays (e.g., 'PCA_Embedding', 'UMAP_Embedding').
        save : str, optional
            Path to save the generated figure (e.g., 'results/pca_diversity.png').
            If None, the figure is only displayed and not saved.

        Notes
        -----
        - Assumes half of the points are real and half synthetic.
        - Returns the matplotlib Figure object.
        """
        key = f"{projection_name.upper()}_Embedding"
        if key not in geom:
            raise ValueError(f"Embedding '{key}' not found in geom dict. "
                             "Call compute_geometric_diversity(...) first.")

        emb = geom[key]
        if not isinstance(emb, np.ndarray):
            raise TypeError(f"'{key}' must be a 2D numpy array, got {type(emb).__name__}.")
        if emb.ndim != 2:
            raise ValueError(f"'{key}' must be 2D, got shape {emb.shape}.")
        if emb.shape[1] < 2:
            raise ValueError(f"'{key}' must have at least 2 columns, got shape {emb.shape}.")
        if emb.shape[1] > 2:
            emb = emb[:, :2]  # use first two dims if more were returned

        n_total = emb.shape[0]
        n_real = n_total // 2  # assume half are real
        if not (0 <= n_real <= n_total):
            raise ValueError(f"Inferred n_real={n_real} is out of bounds for total n={n_total}.")

        title = projection_name.upper()

        labels = ["Real"] * n_real + ["Synthetic"] * (n_total - n_real)
        df = pd.DataFrame(emb, columns=['dim1', 'dim2'])
        df['Type'] = labels

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(
            data=df, x='dim1', y='dim2', hue='Type',
            palette={'Real': 'limegreen', 'Synthetic': 'lightskyblue'},
            edgecolor='black', s=80, alpha=0.8, legend=True, ax=ax
        )

        ax.set_title(title, fontsize=20)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.grid(False)

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.legend(title='', fontsize=15)
        fig.tight_layout()

        # Save if path is provided
        if save:
            fig.savefig(save, dpi=300)

        plt.show()
        return fig

