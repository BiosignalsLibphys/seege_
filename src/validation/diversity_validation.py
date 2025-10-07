from diversity import *

def create_test_datasets():
    """
    Creates:
      - real_data: a base EEG-like dataset (200 signals x 128 samples)
      - test_datasets: dict of 5 synthetic sets for diversity evaluation
    """
    np.random.seed(1234)
    real_data = np.random.normal(0, 1, size=(200, 128))

    return {
        "real_data": real_data,
        "test_datasets": {
            "100% Equal": real_data.copy(),
            "50% Overlap": np.vstack([real_data[:100], np.random.normal(0.5, 1, (100, 128))]),
            "Random": np.random.uniform(-2, 2, size=(200, 128)),
            "Nonlinear": (real_data ** 2) + np.random.normal(0, 0.2, real_data.shape),
            "Shifted": real_data + 3
        }
    }

def validate_diversity():
    data_pkg = create_test_datasets()
    real_data = data_pkg["real_data"]
    test_datasets = data_pkg["test_datasets"]

    div = Diversity()
    summary = []

    for name, syn_data in test_datasets.items():
        print(f"\n===== {name} =====")
        metrics = div.compute_metrics(real_data, syn_data)

        labels = ["Real"] * len(real_data) + ["Synthetic"] * len(syn_data)

        # Plot 2D embeddings
        div.plot_embeddings("PCA", metrics, len(real_data), f"PCA: {name}")
        div.plot_embeddings("UMAP", metrics, len(real_data), f"UMAP: {name}")

        summary.append({
            "Scenario": name,
            "Coverage": round(metrics['Coverage'], 3),
            "Outliers": round(metrics['Outliers'], 3),
            "PCA_Compactness": round(metrics['PCA_Compactness'], 3),
            "UMAP_Compactness": round(metrics['UMAP_Compactness'], 3),
            "PCA_Separation": round(metrics['PCA_Separation'], 3),
            "UMAP_Separation": round(metrics['UMAP_Separation'], 3),
        })

    df_summary = pd.DataFrame(summary)
    print("\n=========== SUMMARY ===========")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    validate_diversity()
