import numpy as np
import pandas as pd
from scipy.signal import square
from time_similarity import *

def generate_identical_sine(freq=10, length=1000, fs=250, n=10):
    t = np.linspace(0, length / fs, length)
    return np.array([np.sin(2 * np.pi * freq * t) for _ in range(n)])

def generate_freq_shifted_sine(freq1=10, freq2=12, length=1000, fs=250, n=10):
    t = np.linspace(0, length / fs, length)
    return np.array([np.sin(2 * np.pi * freq2 * t) for _ in range(n)])

def generate_square_wave(freq=10, length=1000, fs=250, n=10):
    t = np.linspace(0, length / fs, length)
    return np.array([square(2 * np.pi * freq * t) for _ in range(n)])

def generate_lorenz_like_noise(length=1000, n=10):
    return np.random.randn(n, length) * np.linspace(1, 2, length)

def validate_time_similarity(verbose=True, *,  # >>> NEW: args for the new metric
                             ec_sampen_m=2, ec_sampen_r=None,
                             ec_permen_m=3, ec_permen_tau=1,
                             ec_lzc_threshold=None,
                             ec_n_surrogates=0):
    """
    Validates Hjorth-based similarity (existing) and Entropy/Complexity (NEW)
    across curated pairs.
    """
    sim = TimeSimilarity()
    pairs = {
        "Sine vs Sine (identical)": (generate_identical_sine(), generate_identical_sine()),
        "Sine vs Freq-Shifted Sine": (generate_identical_sine(), generate_freq_shifted_sine()),
        "Sine vs Square": (generate_identical_sine(), generate_square_wave()),
        "Sine vs Lorenz-like Noise": (generate_identical_sine(), generate_lorenz_like_noise())
    }

    results_summary = {}
    for name, (real, synth) in pairs.items():
        # --- Hjorth block (existing) ---
        # NOTE: your class method is named compute_time_metrics (not compute_hjorth_metrics)
        hj = sim.compute_hjorth_metrics(real, synth, verbose=False)

        # --- Entropy/Complexity block (NEW) ---
        ec = sim.compute_entropy_complexity_metrics(
            real, synth,
            sampen_m=ec_sampen_m, sampen_r=ec_sampen_r,
            permen_m=ec_permen_m, permen_tau=ec_permen_tau,
            lzc_threshold=ec_lzc_threshold,
            n_surrogates=ec_n_surrogates,
            verbose=False
        )

        # Collect in one row
        results_summary[name] = {
            # Hjorth
            "Avg_WD": hj["Avg_WD"],
            "WD_Activity": hj["WD_Activity"],
            "WD_Mobility": hj["WD_Mobility"],
            "WD_Complexity": hj["WD_Complexity"],
            "Mahalanobis": hj["Mahalanobis"],
            # Entropy/Complexity (WDs + means)
            "WD_SampEn": ec["WD_SampEn"],
            "WD_PermEn": ec["WD_PermEn"],
            "WD_LZC": ec["WD_LZC"],
            "Mean_SampEn_R": ec["Real_SampEn_mean"],
            "Mean_SampEn_S": ec["Synth_SampEn_mean"],
            "Mean_PermEn_R": ec["Real_PermEn_mean"],
            "Mean_PermEn_S": ec["Synth_PermEn_mean"],
            "Mean_LZC_R": ec["Real_LZC_mean"],
            "Mean_LZC_S": ec["Synth_LZC_mean"],
        }

        if verbose:
            print(f"\n--- {name} ---")
            print("[Hjorth]")
            print(f"  Avg WD      : {hj['Avg_WD']:.4f}")
            print(f"    - Activity: {hj['WD_Activity']:.4f}")
            print(f"    - Mobility: {hj['WD_Mobility']:.4f}")
            print(f"    - Complexity: {hj['WD_Complexity']:.4f}")
            print(f"  Mahalanobis : {hj['Mahalanobis']:.4f}")
            print("[Entropy/Complexity]")
            # WD entries might be tuples if you adopted valid-count reporting; handle both cases:
            def _fmt_wd(x):
                if isinstance(x, (tuple, list)):
                    val = x[0]
                else:
                    val = x
                return "nan" if not np.isfinite(val) else f"{val:.4g}"
            print(f"  SampEn WD   : {_fmt_wd(ec['WD_SampEn'])}   (R̄={ec['Real_SampEn_mean']:.4f}, S̄={ec['Synth_SampEn_mean']:.4f})")
            print(f"  PermEn WD   : {_fmt_wd(ec['WD_PermEn'])}   (R̄={ec['Real_PermEn_mean']:.4f}, S̄={ec['Synth_PermEn_mean']:.4f})")
            print(f"  LZC   WD    : {_fmt_wd(ec['WD_LZC'])}   (R̄={ec['Real_LZC_mean']:.4f}, S̄={ec['Synth_LZC_mean']:.4f})")
            if ec_n_surrogates > 0:
                # If you added per-set z-scores, print them (guard keys if absent)
                z_keys = [k for k in ec.keys() if k.startswith("NonlinearityZ_")]
                if z_keys:
                    print("  Surrogate nonlinearity (z̄):")
                    for k in z_keys:
                        print(f"    - {k}: {ec[k]:.4f}")

    return pd.DataFrame.from_dict(results_summary, orient="index")

if __name__ == "__main__":
    # Verbose run; you can set ec_n_surrogates>0 to add nonlinearity checks
    df_results = validate_time_similarity(
        verbose=True,
        ec_sampen_m=2, ec_sampen_r=None,
        ec_permen_m=3, ec_permen_tau=1,
        ec_lzc_threshold=None,
        ec_n_surrogates=0   # set to 20 for z-scores
    )

    print("\n==== FINAL VALIDATION SUMMARY (Hjorth + Entropy/Complexity) ====\n")
    cols = [
        "Avg_WD", "WD_Activity", "WD_Mobility", "WD_Complexity", "Mahalanobis",
        "WD_SampEn", "WD_PermEn", "WD_LZC",
        "Mean_SampEn_R", "Mean_SampEn_S",
        "Mean_PermEn_R", "Mean_PermEn_S",
        "Mean_LZC_R", "Mean_LZC_S",
    ]
    print(df_results[cols])
