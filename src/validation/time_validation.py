import pandas as pd
from scipy.signal import square
from time_fidelity import *

# Synthetic data generators
def _time_axis(length=1000, fs=250):
    # Use endpoint=False so we get exactly `length` samples
    return np.linspace(0, length / fs, length, endpoint=False)

def generate_identical_sine(freq=10, length=1000, fs=250, n=10):
    t = _time_axis(length, fs)
    x = np.sin(2 * np.pi * freq * t)
    return np.tile(x, (n, 1))

def generate_freq_shifted_pair(freq_real=10, freq_syn=12, length=1000, fs=250, n=10):
    t = _time_axis(length, fs)
    real = np.tile(np.sin(2 * np.pi * freq_real * t), (n, 1))
    syn  = np.tile(np.sin(2 * np.pi * freq_syn  * t), (n, 1))
    return real, syn

def generate_square_wave(freq=10, length=1000, fs=250, n=10):
    t = _time_axis(length, fs)
    x = square(2 * np.pi * freq * t)
    return np.tile(x, (n, 1))

def generate_lorenz_like_noise(length=1000, n=10, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, length))
    # amplitude slowly increases across time
    ramp = np.linspace(1.0, 2.0, length, endpoint=True)
    return base * ramp

# Validation (Hjorth-only to match TimeFidelity)
def validate_time_fidelity(verbose=True):
    """
    Validates Hjorth-based similarity (Activity, Mobility, Complexity)
    with curated pairs using TimeFidelity.compute_hjorth_metrics.
    """
    sim = TimeFidelity()

    # Curated pairs
    pairs = {
        "Sine vs Sine (identical)": (
            generate_identical_sine(),
            generate_identical_sine()
        ),
        "Sine vs Freq-Shifted Sine (10Hz vs 12Hz)": (
            generate_freq_shifted_pair(10, 12)
        ),
        "Sine vs Square (10Hz)": (
            generate_identical_sine(),
            generate_square_wave()
        ),
        "Sine vs Lorenz-like Noise": (
            generate_identical_sine(),
            generate_lorenz_like_noise()
        ),
    }

    results_summary = {}
    for name, (real, synth) in pairs.items():
        hj = sim.compute_hjorth_metrics(real, synth, verbose=verbose)

        results_summary[name] = {
            "Avg_WD": hj["Avg_WD"],
            "WD_Activity": hj["WD_Activity"],
            "WD_Mobility": hj["WD_Mobility"],
            "WD_Complexity": hj["WD_Complexity"],
            "Mahalanobis": hj["Mahalanobis"],
            "Real_Activity": hj["Real_Activity"],
            "Real_Mobility": hj["Real_Mobility"],
            "Real_Complexity": hj["Real_Complexity"],
            "Synthetic_Activity": hj["Synthetic_Activity"],
            "Synthetic_Mobility": hj["Synthetic_Mobility"],
            "Synthetic_Complexity": hj["Synthetic_Complexity"],
        }

        if verbose:
            print(f"\n--- {name} ---")
            print(f"  Avg WD           : {hj['Avg_WD']:.4f}")
            print(f"    - Activity     : {hj['WD_Activity']:.4f}")
            print(f"    - Mobility     : {hj['WD_Mobility']:.4f}")
            print(f"    - Complexity   : {hj['WD_Complexity']:.4f}")
            print(f"  Mahalanobis      : {hj['Mahalanobis']:.4f}")
            print("  Means (Real)     : "
                  f"A={hj['Real_Activity']:.4f}, M={hj['Real_Mobility']:.4f}, C={hj['Real_Complexity']:.4f}")
            print("  Means (Synthetic): "
                  f"A={hj['Synthetic_Activity']:.4f}, M={hj['Synthetic_Mobility']:.4f}, C={hj['Synthetic_Complexity']:.4f}")

    return pd.DataFrame.from_dict(results_summary, orient="index")

if __name__ == "__main__":
    df_results = validate_time_fidelity(verbose=True)

    print("\n==== FINAL VALIDATION SUMMARY ====\n")
    cols = [
        "Avg_WD", "WD_Activity", "WD_Mobility", "WD_Complexity", "Mahalanobis",
        "Real_Activity", "Real_Mobility", "Real_Complexity",
        "Synthetic_Activity", "Synthetic_Mobility", "Synthetic_Complexity",
    ]
    df_pretty = df_results[cols].round(6)
    print(df_pretty.to_string())
