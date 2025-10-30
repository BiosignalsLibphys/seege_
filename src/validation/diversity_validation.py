
from diversity import *


# Test dataset generator

def create_test_datasets(n_real=200, n_feat=128, seed=1234):
    """
    Creates:
      - real_data: N(0,1) baseline (n_real x n_feat)
      - test_datasets: dict of 5 synthetic sets designed to stress coverage, geometry, and intrinsic diversity

    Scenarios
    ---------
    1) Identical (Baseline): Y = X
    2) Shifted (+3): geometric centroid shift (overlap -> 0)
    3) Core-only subset (under-coverage): synth lives in the central mass (misses tails)
    4) Tail-contamination (5% far): good coverage, synthetic has a few far outliers
    5) Rank-deficient (subspace): collapse onto low-rank subspace (intrinsic < 1)
    """
    rng = np.random.default_rng(seed)
    real_data = rng.normal(0, 1, size=(n_real, n_feat))

    # Helpers
    def _radius(x):
        # Euclidean norm per sample
        return np.linalg.norm(x, axis=1)

    def _resample_rows(X, n, jitter_std=0.01):
        # Resample with replacement to size n and apply tiny jitter to avoid duplicates
        idx = rng.integers(0, X.shape[0], size=n)
        Y = X[idx].copy()
        if jitter_std > 0:
            Y += rng.normal(0, jitter_std, size=Y.shape)
        return Y

    # 1) Identical
    synth_identical = real_data.copy()

    # 2) Shifted (+3)
    synth_shifted = real_data + 3.0

    # 3) Core-only subset (under-coverage): keep central ~60% by radius, then resample to n_real
    radii = _radius(real_data)
    cut = np.percentile(radii, 60.0)
    core = real_data[radii <= cut]
    # ensure non-empty (in pathological small sets)
    if core.shape[0] == 0:
        core = real_data[: max(1, n_real // 3)]
    synth_core_only = _resample_rows(core, n_real, jitter_std=0.01)

    # 4) Tail-contamination (5% far)
    base = real_data + rng.normal(0, 0.05, size=real_data.shape)  # near the real set (good coverage)
    k = max(1, n_real // 20)  # 5%
    far = rng.normal(0, 1, size=(k, n_feat)) + 8.0  # push far away
    synth_tail = base.copy()
    synth_tail[:k] = far  # contaminate a small slice with far outliers

    # 5) Rank-deficient (subspace collapse)
    # Project onto k-dim subspace (k << d) via random orthonormal basis U, then back-project
    k_dim = max(4, n_feat // 8)  # e.g., 16 for 128-D
    Q, _ = np.linalg.qr(rng.normal(size=(n_feat, k_dim)))
    proj = real_data @ Q @ Q.T
    synth_lowrank = proj + rng.normal(0, 0.01, size=proj.shape)  # tiny noise

    test_datasets = {
        "Identical (Baseline)": synth_identical,
        "Shifted (+3)": synth_shifted,
        "Core-only subset (under-coverage)": synth_core_only,
        "Tail-contamination (5% far)": synth_tail,
        "Rank-deficient (subspace)": synth_lowrank,
    }

    return {"real_data": real_data, "test_datasets": test_datasets}



# Validation runner

def validate_diversity(save_figs=False, outdir="div_figs"):
    """
    Runs all scenarios through:
      - compute_coverage_diversity()
      - compute_geometric_diversity()
      - compute_intrinsic_diversity()
    Prints a summary DataFrame and optionally saves PCA/UMAP plots.
    """
    pkg = create_test_datasets()
    real = pkg["real_data"]
    tests = pkg["test_datasets"]

    if save_figs:
        os.makedirs(outdir, exist_ok=True)

    div = Diversity()

    rows = []
    for name, syn in tests.items():
        print(f"\n===== {name} =====")

        # 1) Coverage diversity (original space)
        cov = div.compute_coverage_diversity(real, syn)
        # keys: Coverage, Outliers, Sigma

        # 2) Geometric diversity (embeddings + compactness & Mahalanobis overlap)
        geom = div.compute_geometric_diversity(real, syn)
        # keys: PCA_Embedding, UMAP_Embedding, PCA_Compactness, UMAP_Compactness, PCA_OverlapMahalanobis, UMAP_OverlapMahalanobis

        # 3) Intrinsic diversity (uniqueness + local/global diversity)
        intr = div.compute_intrinsic_diversity(real, syn)
        # keys: Uniqueness_NN, Global_Diversity, Local_Diversity_P10, Local_Diversity_P50

        # Optional plots
        if save_figs:
            div.plot_embeddings("PCA",  geom, save=os.path.join(outdir, f"{name.replace(' ','_')}_pca.png"))
            div.plot_embeddings("UMAP", geom, save=os.path.join(outdir, f"{name.replace(' ','_')}_umap.png"))
        else:
            div.plot_embeddings("PCA",  geom)
            div.plot_embeddings("UMAP", geom)

        # Collect a summary row
        rows.append({
            "Scenario": name,
            # Coverage domain
            "Coverage": round(cov["Coverage"], 3),
            "Anti-Outlier": round(cov["Outliers"], 3),
            # Geometric domain (higher is better for both)
            "PCA_Compactness": round(geom["PCA_Compactness"], 3),
            "UMAP_Compactness": round(geom["UMAP_Compactness"], 3),
            "PCA_OverlapScore": round(geom["PCA_OverlapMahalanobis"], 3),
            "UMAP_OverlapScore": round(geom["UMAP_OverlapMahalanobis"], 3),
            # Intrinsic domain
            "Uniqueness": round(intr["Uniqueness_NN"], 3),
            "LocalDiversity_P10": round(intr["Local_Diversity_P10"], 3),
            "LocalDiversity_P50": round(intr["Local_Diversity_P50"], 3),
            "GlobalDiversity": round(intr["Global_Diversity"], 3),
        })

    df = pd.DataFrame(rows)
    # Pretty print ordered columns
    cols = [
        "Scenario",
        "Coverage", "Anti-Outlier",
        "PCA_Compactness", "UMAP_Compactness",
        "PCA_OverlapScore", "UMAP_OverlapScore",
        "Uniqueness", "LocalDiversity_P10", "LocalDiversity_P50", "GlobalDiversity"
    ]
    df = df[cols]

    print("\n=========== DIVERSITY SUMMARY ===========")
    print(df.to_string(index=False))

    return df



# Heuristic expectations

def quick_checks(df: pd.DataFrame):
    """
    Optional sanity checks that the scenarios behave as expected.
    These aren’t hard assertions—just quick signals while iterating.
    """
    def val(scn, col):
        return float(df.loc[df["Scenario"] == scn, col].values[0])

    try:
        # Identical baseline: everything ~1
        assert val("Identical (Baseline)", "Coverage") > 0.95
        assert val("Identical (Baseline)", "Anti-Outlier") > 0.95
        assert val("Identical (Baseline)", "PCA_OverlapScore") > 0.95
        assert 0.8 <= val("Identical (Baseline)", "Uniqueness") <= 1.2
        assert 0.8 <= val("Identical (Baseline)", "GlobalDiversity") <= 1.2

        # Shifted: very low overlap & low coverage/outlier
        assert val("Shifted (+3)", "PCA_OverlapScore") < 0.1
        assert val("Shifted (+3)", "Coverage") < 0.5
        assert val("Shifted (+3)", "Anti-Outlier") < 0.5

        # Core-only: under-coverage, decent outlier goodness, overlap high-ish
        assert val("Core-only subset (under-coverage)", "Coverage") < 0.8
        assert val("Core-only subset (under-coverage)", "Anti-Outlier") > 0.7
        assert val("Core-only subset (under-coverage)", "PCA_OverlapScore") > 0.7

        # Tail-contamination: good coverage, poorer outliers, overlap high-ish
        assert val("Tail-contamination (5% far)", "Coverage") > 0.8
        assert val("Tail-contamination (5% far)", "Anti-Outlier") < 0.8
        assert val("Tail-contamination (5% far)", "PCA_OverlapScore") > 0.7

        # Rank-deficient: intrinsic collapse (ratios < 1)
        assert val("Rank-deficient (subspace)", "Uniqueness") < 1.0
        assert val("Rank-deficient (subspace)", "GlobalDiversity") < 1.0
        assert val("Rank-deficient (subspace)", "LocalDiversity_P10") < 1.0

        print("\nQuick checks: ✅ passed")
    except AssertionError:
        print("\nQuick checks: ⚠️ some expectations were not met (this can happen with randomness).")


if __name__ == "__main__":
    # Set save_figs=True to write PNGs to disk
    summary_df = validate_diversity(save_figs=False, outdir="div_figs")
    quick_checks(summary_df)
