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


def validate_time_similarity(verbose=True):
    sim = TimeSimilarity()
    pairs = {
        "Sine vs Sine (identical)": (generate_identical_sine(), generate_identical_sine()),
        "Sine vs Freq-Shifted Sine": (generate_identical_sine(), generate_freq_shifted_sine()),
        "Sine vs Square": (generate_identical_sine(), generate_square_wave()),
        "Sine vs Lorenz-like Noise": (generate_identical_sine(), generate_lorenz_like_noise())
    }

    results_summary = {}
    for name, (real, synth) in pairs.items():
        result_dict = sim.compute_time_metrics(real, synth)
        results_summary[name] = result_dict
        if verbose:
            print(f"\n--- {name} ---")
            print(f"Avg Wasserstein Distance: {result_dict['Avg_WD']:.4f}")
            print(f"  - Activity  : {result_dict['WD_Activity']:.4f}")
            print(f"  - Mobility  : {result_dict['WD_Mobility']:.4f}")
            print(f"  - Complexity: {result_dict['WD_Complexity']:.4f}")
            print(f"Mahalanobis Distance     : {result_dict['Mahalanobis']:.4f}")
    return pd.DataFrame.from_dict(results_summary, orient="index")



if __name__ == "__main__":
    df_results = validate_time_similarity()
    print("\n==== FINAL VALIDATION SUMMARY ====\n")
    print(df_results[["Avg_WD", "WD_Activity", "WD_Mobility", "WD_Complexity", "Mahalanobis"]])
