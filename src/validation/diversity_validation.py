from src.diversity import *

def create_test_datasets():
    """
    Creates 1 real dataset (200 x 10) and a dictionary of 7 synthetic sets:
      1) 100% Equal      (exact copy of real_data)
      2) 50% Overlap     (partial shift)
      3) 30% Overlap     (bigger shift)
      4) Random Data     (uniform distribution)

      [NEW SCENARIOS to stress-test Local vs. Global manifold differences]:
      5) Multi-Modal
         - Real data is combined with an added "second mode" offset,
           making the synthetic distribution strongly multi-modal.
      6) Correlated Synthetic
         - Applies a random linear transform to real_data to induce
           correlations, possibly changing global structure.
      7) Nonlinear (Polynomial) Synthetic
         - Squares each feature of real_data + small noise, creating
           a nonlinear shape that differs from the original manifold.

    Returns:
    --------
    {
      "real_data": real_data,         # shape: (200,10)
      "test_datasets": {
          "100% Equal": ...,
          "50% Overlap": ...,
          "30% Overlap": ...,
          "Random Data": ...,
          "Multi-Modal": ...,
          "Correlated": ...,
          "Nonlinear": ...
      }
    }
    """
    np.random.seed(1234)  # Reproducibility

    # ---------------------------
    # Base Real Data
    # ---------------------------
    real_data = np.random.normal(0, 1, size=(200, 10))

    # ---------------------------
    # Original 4 Scenarios
    # ---------------------------
    # 1) 100% Equal => exact copy
    data_equal = real_data.copy()

    # 2) 50% Overlap => partial shift
    data_50_overlap = np.vstack([
        real_data[:100],
        np.random.normal(loc=2.0, scale=1.2, size=(100, 10))
    ])

    # 3) 30% Overlap => bigger shift
    data_30_overlap = np.vstack([
        real_data[:60],
        np.random.normal(loc=5, scale=1.5, size=(140, 10))
    ])

    # 4) Random => uniform distribution
    data_random = np.random.uniform(low=-10, high=10, size=(200, 10))

    # ---------------------------
    # NEW SCENARIO 5: Multi-Modal
    # ---------------------------
    data_multimodal = np.vstack([
        real_data[:100] + np.random.normal(loc=0, scale=1, size=(100, 10)),
        real_data[100:] + np.random.normal(loc=4, scale=1.2, size=(100, 10))
    ])

    # ---------------------------
    # NEW SCENARIO 6: Correlated Synthetic
    # ---------------------------
    A = np.random.normal(loc=0, scale=1, size=(10, 10))
    data_correlated = real_data @ A  # matrix multiplication

    # ---------------------------
    # NEW SCENARIO 7: Nonlinear (Polynomial)
    # ---------------------------
    data_poly = (real_data ** 2) + np.random.normal(loc=0, scale=0.5, size=real_data.shape)

    return {
        "real_data": real_data,
        "test_datasets": {
            "100% Equal":    data_equal,
            "50% Overlap":   data_50_overlap,
            "30% Overlap":   data_30_overlap,
            "Random Data":   data_random,
            "Multi-Modal":   data_multimodal,
            "Correlated":    data_correlated,
            "Nonlinear":     data_poly
        }
    }

import numpy as np
import pandas as pd

def validate_diversity_assessment():
    """
    1) Creates test datasets (7 scenarios total).
    2) Computes all metrics (Compactness, LocalSep, GlobalSep, Coverage, Outliers, MMD, KL, Wasserstein).
    3) Prints a stacked table comparing Actual vs. Expected.

    Each scenario will have 2 lines:
      - Actual
      - Expected
    """
    # -----------------------------------------------------
    # Step 1: Generate Datasets
    # -----------------------------------------------------
    data_pkg = create_test_datasets()
    real_data = data_pkg["real_data"]
    test_dict = data_pkg["test_datasets"]

    # We'll define approximate expected ranges for each scenario
    # (illustrative guesses)
    expected_info = {
        "100% Equal": {
            "Compactness": "≈1.0", "LocalSep": "≈0.0", "GlobalSep": "≈0.0",
            "Coverage": "≈1.0", "Outliers": "≈0.0", "MMD": "≈0.0", "KL": "≈0.0", "Wasserstein": "≈0.0"
        },
        "50% Overlap": {
            "Compactness": "0.5–0.7", "LocalSep": "0.3–0.6", "GlobalSep": "0.3–0.6",
            "Coverage": "0.5–0.7", "Outliers": "0.3–0.5", "MMD": "low–med", "KL": "low–med", "Wasserstein": "low–med"
        },
        "30% Overlap": {
            "Compactness": "0.3–0.5", "LocalSep": "0.4–0.7", "GlobalSep": "0.4–0.7",
            "Coverage": "0.3–0.4", "Outliers": "0.5–0.7", "MMD": "med", "KL": "med", "Wasserstein": "med–high"
        },
        "Random Data": {
            "Compactness": "≈0.0", "LocalSep": "≈1.0", "GlobalSep": "≈1.0",
            "Coverage": "≈0.0", "Outliers": "≈1.0", "MMD": "high", "KL": "high", "Wasserstein": "high"
        },
        "Multi-Modal": {
            "Compactness": "0.2–0.6", "LocalSep": "0.5–0.9", "GlobalSep": "0.5–0.9",
            "Coverage": "varies", "Outliers": "varies", "MMD": "med–high", "KL": "med–high", "Wasserstein": "med–high"
        },
        "Correlated": {
            "Compactness": "0.3–0.7", "LocalSep": "0.3–0.8", "GlobalSep": "0.3–0.8",
            "Coverage": "0.3–0.5", "Outliers": "0.3–0.6", "MMD": "med", "KL": "med", "Wasserstein": "med"
        },
        "Nonlinear": {
            "Compactness": "0.1–0.5", "LocalSep": "0.5–0.9", "GlobalSep": "0.5–0.9",
            "Coverage": "0.1–0.4", "Outliers": "0.4–0.8", "MMD": "med–high", "KL": "med–high", "Wasserstein": "med–high"
        }
    }

    # -----------------------------------------------------
    # Step 2: Instantiate Diversity
    # -----------------------------------------------------
    # You may have replaced local/global separation with a silhouette-based approach
    # or you can keep your original ratio method. We'll assume the "all_metrics" approach.
    div = Diversity(n_components=2, perplexity=30, n_neighbors=15, min_dist=0.1, random_state=42)

    # -----------------------------------------------------
    # Step 3: Compute Metrics
    # -----------------------------------------------------
    results_list = []
    for scenario_name, syn_data in test_dict.items():
        # get all metrics from your "compute_all_metrics()" or similar
        metrics_dict = div.compute_all_metrics(real_data, syn_data)

        # We'll store them in a row for final summarizing
        row_actual = {
            "Scenario": scenario_name,
            "Type": "Actual",
            "Compactness": f"{metrics_dict['Compactness']:.3f}",
            "LocalSep": f"{metrics_dict['LocalSeparation']:.3f}",
            "GlobalSep": f"{metrics_dict['GlobalSeparation']:.3f}",
            "Coverage": f"{metrics_dict['Coverage']:.3f}",
            "Outliers": f"{metrics_dict['Outliers']:.3f}",
            "MMD": f"{metrics_dict['MMD']:.3f}",
            "KL": f"{metrics_dict['KL']:.3f}",
            "Wasserstein": f"{metrics_dict['Wasserstein']:.3f}"
        }

        # Look up expected ranges for that scenario
        scenario_expected = expected_info.get(scenario_name, {
            "Compactness": "???", "LocalSep": "???", "GlobalSep": "???",
            "Coverage": "???", "Outliers": "???", "MMD": "???", "KL": "???", "Wasserstein": "???"
        })

        row_expected = {
            "Scenario": "",
            "Type": "Expected",
            "Compactness": scenario_expected["Compactness"],
            "LocalSep": scenario_expected["LocalSep"],
            "GlobalSep": scenario_expected["GlobalSep"],
            "Coverage": scenario_expected["Coverage"],
            "Outliers": scenario_expected["Outliers"],
            "MMD": scenario_expected["MMD"],
            "KL": scenario_expected["KL"],
            "Wasserstein": scenario_expected["Wasserstein"]
        }

        results_list.append(row_actual)
        results_list.append(row_expected)

    # -----------------------------------------------------
    # Step 4: Present Results
    # -----------------------------------------------------
    df_results = pd.DataFrame(results_list)

    print("\n=================== DIVERSITY VALIDATION RESULTS ===================\n")
    # We can display the table with "Scenario, Type" as row groups
    print(df_results.to_string(index=False))

    print("\n(Values in [0,1] for some metrics; MMD/KL/Wasserstein may vary. "
          "Expected ranges are approximate guidance.)")


if __name__ == "__main__":
    validate_diversity_assessment()
