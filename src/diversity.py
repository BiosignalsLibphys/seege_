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
       • LabelMixingScore (class-wise silhouette, PCA & UMAP) ∈ [0, 1]
       • Mahalanobis-based Overlap (pooled-covariance; PCA & UMAP) ∈ (0, 1]
         (higher = closer centroids after normalizing by joint scatter)
       • Covariance Shape Similarity (Frobenius-based; PCA & UMAP) ∈ [0, 1]

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
    div.plot_embeddings("UMAP", geom)
    """

    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42, max_pairs=200_000):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state
        self.max_pairs = max_pairs


    # 1) Manifold Coverage: Coverage & Outlier Goodness (original space)

    def compute_coverage_diversity(self, real_data: np.ndarray, synthetic_data: np.ndarray,
                                   k_sigma: float = 2.0):
        """
        Compute manifold coverage metrics in the original feature space.

        Metrics (threshold-based)
        -------------------------
        Coverage : float in [0, 1]
            Fraction of real samples whose nearest synthetic neighbour lies
            within R = k_sigma * sigma.
            Higher is better: 1.0 ≈ all real modes are covered by synthetic data.

        Outliers : float in [0, 1]
            Fraction of synthetic samples whose nearest real neighbour lies
            within R = k_sigma * sigma.
            Higher is better: 1.0 ≈ few synthetic samples lie far from the
            real manifold.

        Notes
        -----
        - sigma is a data-driven scale estimated from real–real nearest-neighbour
          distances (NOT all pairwise distances).
        - k_sigma controls how strict the notion of “near” is (default = 2.0).

        Returns
        -------
        dict with keys:
            'Coverage', 'Outliers', 'Sigma', 'Radius'
        """
        real = np.asarray(real_data, dtype=float)
        synth = np.asarray(synthetic_data, dtype=float)

        # Pairwise distances
        D_rs = pairwise_distances(real, synth, metric="euclidean")  # real x synth
        D_rr = pairwise_distances(real, real, metric="euclidean")  # real x real

        # --- 1) Bandwidth sigma from *nearest-neighbour* real-real distances ---
        # Ignore self-distances by setting diagonal to +inf, then take row-wise min
        np.fill_diagonal(D_rr, np.inf)
        nn_real = D_rr.min(axis=1)  # nearest neighbour distance for each real point

        # Robust scale: median NN distance (guard against pathological cases)
        sigma = float(np.median(nn_real)) if nn_real.size else 1.0
        eps = 1e-12
        sigma = max(sigma, eps)

        # Define radius R = k_sigma * sigma
        R = k_sigma * sigma

        # --- 2) Coverage: real -> synthetic ---
        min_r2s = D_rs.min(axis=1)  # NN distance real -> synthetic
        coverage = float(np.mean(min_r2s <= R))  # fraction within radius

        # --- 3) Outlier "goodness": synthetic -> real ---
        min_s2r = D_rs.min(axis=0)  # NN distance synthetic -> real
        outliers = float(np.mean(min_s2r <= R))  # fraction within radius

        print(f"[Coverage Diversity] sigma (NN): {sigma:.4f}")
        print(f"[Coverage Diversity] Radius R = k_sigma * sigma = {R:.4f} (k_sigma={k_sigma})")
        print(f"[Coverage Diversity] Coverage (real→synth): {coverage:.3f}  (↑ better)")
        print(f"[Coverage Diversity] Outlier Goodness (synth→real): {outliers:.3f}  (↑ better)")

        return {
            "Coverage": coverage,
            "Outliers": outliers,
            "Sigma": sigma,
            "Radius": R,
        }

    # 2) Geometric Diversity: Label Mixing Score, overlap and covariance shape in PCA/UMAP spaces

    def compute_geometric_diversity(self, real_data: np.ndarray, synthetic_data: np.ndarray):
        """
        Compute structural diversity in PCA/UMAP spaces.

        Metrics
        -------
        PCA_LabelMixingScore, UMAP_LabelMixingScore : float in [0, 1]
            1 - |mean silhouette per label|, where silhouette is computed using
            labels {real, synthetic}. Values close to 1 mean that real and
            synthetic are well mixed (indistinguishable by label); values near 0
            mean they form clearly separated clusters.

        PCA_OverlapMahalanobis, UMAP_OverlapMahalanobis : float in (0, 1]
            Mahalanobis-based overlap using pooled covariance (higher = closer
            centroids after normalizing by joint scatter).

        PCA_CovShape, UMAP_CovShape : float in [0, 1]
            Similarity of covariance structure between real and synthetic:
            1 - ||Σ_real - Σ_syn||_F / (||Σ_real||_F + ||Σ_syn||_F).
            Values close to 1 mean similar shape/anisotropy; values near 0 mean
            strong mismatch in spread/shape.

        Returns
        -------
        dict with keys:
            'PCA_Embedding', 'UMAP_Embedding',
            'PCA_LabelMixingScore', 'UMAP_LabelMixingScore',
            'PCA_OverlapMahalanobis', 'UMAP_OverlapMahalanobis',
            'PCA_CovShape', 'UMAP_CovShape'
        """
        real = np.asarray(real_data)
        synth = np.asarray(synthetic_data)
        n_real = len(real)
        n_syn = len(synth)
        combined = np.vstack([real, synth])
        labels = np.array([0] * n_real + [1] * n_syn)
        eps = 1e-12

        # ---------------- PCA ----------------
        pca = PCA(n_components=self.n_components,
                  svd_solver='randomized',
                  random_state=self.random_state)
        pca_emb = pca.fit_transform(combined)

        mu_r = pca_emb[:n_real].mean(axis=0)
        mu_s = pca_emb[n_real:].mean(axis=0)

        Xr = pca_emb[:n_real]
        Xs = pca_emb[n_real:]

        Sr = np.cov(Xr.T)
        Ss = np.cov(Xs.T)
        Sp = Sr + Ss + 1e-6 * np.eye(Sr.shape[0])

        # Mahalanobis centroid overlap
        delta_vec = (mu_r - mu_s).reshape(-1, 1)
        d2_pca = float((delta_vec.T @ np.linalg.inv(Sp) @ delta_vec).squeeze())
        pca_overlap_maha = float(np.exp(-0.5 * d2_pca))

        # Label mixing score (label mixing)
        s_pca = silhouette_samples(pca_emb, labels)
        mean_s_pca = 0.5 * (s_pca[:n_real].mean() + s_pca[n_real:].mean())
        pca_labelmixingscore = float(1.0 - abs(mean_s_pca))  # already in [0, 1]

        # Covariance shape similarity (Frobenius-based)
        num_pca = np.linalg.norm(Sr - Ss, ord="fro")
        den_pca = np.linalg.norm(Sr, ord="fro") + np.linalg.norm(Ss, ord="fro") + eps
        pca_covshape = float(np.clip(1.0 - num_pca / den_pca, 0.0, 1.0))

        # ---------------- UMAP ----------------
        umap_emb = UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=self.random_state
        ).fit_transform(combined)

        mu_r_u = umap_emb[:n_real].mean(axis=0)
        mu_s_u = umap_emb[n_real:].mean(axis=0)

        Xr_u = umap_emb[:n_real]
        Xs_u = umap_emb[n_real:]

        Sr_u = np.cov(Xr_u.T)
        Ss_u = np.cov(Xs_u.T)
        Sp_u = Sr_u + Ss_u + 1e-6 * np.eye(Sr_u.shape[0])

        # Mahalanobis centroid overlap (UMAP)
        delta_vec_u = (mu_r_u - mu_s_u).reshape(-1, 1)
        d2_umap = float((delta_vec_u.T @ np.linalg.inv(Sp_u) @ delta_vec_u).squeeze())
        umap_overlap_maha = float(np.exp(-0.5 * d2_umap))

        # Label mixing score (UMAP)
        s_umap = silhouette_samples(umap_emb, labels)
        mean_s_umap = 0.5 * (s_umap[:n_real].mean() + s_umap[n_real:].mean())
        umap_labelmixingscore = float(1.0 - abs(mean_s_umap))

        # Covariance shape similarity (UMAP)
        num_umap = np.linalg.norm(Sr_u - Ss_u, ord="fro")
        den_umap = np.linalg.norm(Sr_u, ord="fro") + np.linalg.norm(Ss_u, ord="fro") + eps
        umap_covshape = float(np.clip(1.0 - num_umap / den_umap, 0.0, 1.0))

        print("[Geometric Diversity] PCA  -> Label Mixing Score:", f"{pca_labelmixingscore:.3f}",
              "| Mahalanobis Overlap:", f"{pca_overlap_maha:.3f}",
              "| CovShape:", f"{pca_covshape:.3f}")
        print("[Geometric Diversity] UMAP -> Label Mixing Score:", f"{umap_labelmixingscore:.3f}",
              "| Mahalanobis Overlap:", f"{umap_overlap_maha:.3f}",
              "| CovShape:", f"{umap_covshape:.3f}")

        return {
            "PCA_Embedding": pca_emb,
            "UMAP_Embedding": umap_emb,
            "PCA_LabelMixingScore": pca_labelmixingscore,
            "UMAP_LabelMixingScore": umap_labelmixingscore,
            "PCA_OverlapMahalanobis": pca_overlap_maha,
            "UMAP_OverlapMahalanobis": umap_overlap_maha,
            "PCA_CovShape": pca_covshape,
            "UMAP_CovShape": umap_covshape,
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
              f"{global_div:.3f} (~1 ideal; <1 collapse; >1 over-dispersion)")
        print("[Intrinsic Diversity] Local Diversity P10/P50 (NN ratio): "
              f"{local_div_p10:.3f} / {local_div_p50:.3f} (~1 ideal; <1 collapse; >1 over-dispersion)")

        return {
            "Uniqueness_NN": uniqueness_nn,
            "Global_Diversity": global_div,
            "Local_Diversity_P10": local_div_p10,
            "Local_Diversity_P50": local_div_p50,
        }


    # 4) Plotting: same as before (expects *_Embedding in results dict)

    @staticmethod
    def plot_embeddings(projection_name: str, geom: dict, save: str = None, plot_title: str = None, *,
                        kind: str = 'scatter', scatter_kws: dict | None = None, kde_kws: dict | None = None):
        """
        Plot a 2D projection of real vs synthetic data using PCA or UMAP.

        Argmuments:
        - kind: {'scatter','kde'} choose plotting style. 'scatter' uses points (default).
                'kde' shows 2D kernel density estimate contours/filled densities for each class.
        - scatter_kws: dict of keyword args forwarded to seaborn.scatterplot (when kind='scatter').
        - kde_kws: dict of keyword args forwarded to seaborn.kdeplot (when kind='kde').
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

        # plotting defaults
        scatter_kws = {} if scatter_kws is None else dict(scatter_kws)
        kde_kws = {} if kde_kws is None else dict(kde_kws)

        fig, ax = plt.subplots(figsize=(5, 4))

        if kind not in {'scatter', 'kde'}:
            raise ValueError("kind must be either 'scatter' or 'kde'.")

        if kind == 'scatter':
            # default scatter kwargs, can be overridden
            skw = dict(edgecolor='black', s=80, alpha=0.8)
            skw.update(scatter_kws)
            sns.scatterplot(
                data=df, x='dim1', y='dim2', hue='Type',
                palette={'Real': 'limegreen', 'Synthetic': 'lightskyblue'},
                legend=True, ax=ax, **skw
            )

        else:  # kind == 'kde'
            # For KDE we draw filled contours for each class separately
            # Default kde kwargs
            default_kde = dict(levels=6, fill=True, thresh=0.05, alpha=0.45, bw_method='scott')
            default_kde.update(kde_kws)

            # Plot KDE for Real
            real_df = df[df['Type'] == 'Real']
            synth_df = df[df['Type'] == 'Synthetic']

            # If too few points for KDE, fall back to scatter for that class
            def _safe_kde_plot(dfi, color, label):
                if dfi.shape[0] < 3 or np.isfinite(dfi[['dim1','dim2']].values).sum() < 3:
                    # fallback scatter
                    sns.scatterplot(data=dfi, x='dim1', y='dim2', color=color, edgecolor='black', s=60, alpha=0.7, ax=ax, label=label)
                    return False
                try:
                    sns.kdeplot(data=dfi, x='dim1', y='dim2', color=color, ax=ax, **default_kde)
                    return True
                except Exception:
                    # any KDE failure -> fallback scatter
                    sns.scatterplot(data=dfi, x='dim1', y='dim2', color=color, edgecolor='black', s=60, alpha=0.7, ax=ax, label=label)
                    return False

            r_ok = _safe_kde_plot(real_df, 'limegreen', 'Real')
            s_ok = _safe_kde_plot(synth_df, 'lightskyblue', 'Synthetic')

            # Build legend manually because kdeplot doesn't always add nice handles
            from matplotlib.patches import Patch
            handles = []
            if r_ok:
                handles.append(Patch(facecolor='limegreen', edgecolor='k', alpha=0.45, label='Real'))
            else:
                handles.append(Patch(facecolor='limegreen', edgecolor='k', alpha=0.9, label='Real'))
            if s_ok:
                handles.append(Patch(facecolor='lightskyblue', edgecolor='k', alpha=0.45, label='Synthetic'))
            else:
                handles.append(Patch(facecolor='lightskyblue', edgecolor='k', alpha=0.9, label='Synthetic'))
            ax.legend(handles=handles, title='', fontsize=12)

        final_title = plot_title if plot_title is not None else title
        ax.set_title(final_title, fontsize=20)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.grid(False)

        for spine in ax.spines.values():
            spine.set_visible(False)

        if kind == 'scatter':
            ax.legend(title='', fontsize=15)

        fig.tight_layout()

        # Save if path is provided
        if save:
            fig.savefig(save, dpi=300)

        plt.show()
        return fig
