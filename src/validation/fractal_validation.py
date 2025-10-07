from fractal_similarity import *

def generate_white_noise(length=10000, num_signals=10):
    """Generate white noise signals (H≈0.5)."""
    return [np.random.randn(length) for _ in range(num_signals)]

def generate_pink_noise(length=10000, num_signals=10):
    """Generate pink noise signals (H≈1.0)."""
    signals = []
    for _ in range(num_signals):
        X_white = np.fft.rfft(np.random.randn(length))
        freqs = np.fft.rfftfreq(length)
        # Avoid division by zero
        freqs[0] = freqs[1]
        X_pink = X_white / np.sqrt(freqs)
        pink = np.fft.irfft(X_pink, length)
        # Normalize
        pink = (pink - np.mean(pink)) / np.std(pink)
        signals.append(pink)
    return signals

def generate_brown_noise(length=10000, num_signals=10):
    """Generate brown noise signals (H≈1.5) via random walk + detrending."""
    signals = []
    for _ in range(num_signals):
        white = np.random.randn(length)
        brown = np.cumsum(white)
        brown = detrend(brown)
        brown = (brown - np.mean(brown)) / np.std(brown)
        signals.append(brown)
    return signals

def validate_fractal_properties():
    """
    Test the FractalSimilarity class with different types of signals,
    then compare to theoretical or literature-based values for each method.
    """
    # 1) Generate signals
    signal_length = 1000
    num_signals = 5

    print("Generating signals...")
    signals = {
        "White Noise": generate_white_noise(signal_length, num_signals),
        "Pink Noise": generate_pink_noise(signal_length, num_signals),
        "Brown Noise": generate_brown_noise(signal_length, num_signals)
    }

    # 2) Plot example signals (just to visualize the first 500 points of each)
    plt.figure(figsize=(15, 10))
    for i, (name, data) in enumerate(signals.items()):
        plt.subplot(5, 1, i + 1)
        plt.plot(data[0][:500])
        plt.title(f"{name} (first 500 points)")
        plt.tight_layout()
    #plt.savefig("example_signals.pdf")
    plt.show()

    # 3) We'll store results in a dictionary for each method
    results = {}
    signal_types = list(signals.keys())
    methods = ["DCCA", "MFDFA", "MFDCCA"]

    for method in methods:
        print(f"\n==== Testing {method} method ====\n")
        method_results = {}

        # (A) MFDFA => single-series analysis
        if method == "MFDFA":
            for s_type in signal_types:
                print(f"Analyzing {s_type} signals with {method}...")
                fs = FractalSimilarity(signals[s_type], signals[s_type], method=method)
                fs.compute_fractal_metrics()

                key = f"{s_type}"
                # MFDFA returns 2 categories [real, synthetic], but here we are using
                # the same dataset for both => the "real" index is 0 in fs.means
                method_results[key] = {
                    "mean": fs.means[0],
                    "std": fs.stds[0],
                }

        # (B) DCCA / MFDCCA => pairwise analysis
        else:
            for i, type1 in enumerate(signal_types):
                for j, type2 in enumerate(signal_types):
                    if j < i:
                        continue
                    pair_label = f"{type1} vs {type2}"
                    print(f"Analyzing {pair_label} with {method}...")

                    try:
                        fs = FractalSimilarity(signals[type1], signals[type2], method=method)
                        fs.compute_fractal_metrics()
                        fs.plot_metrics()

                        if i == j:
                            # same dataset => typically index=0 => "real vs real"
                            method_results[pair_label] = {
                                "hurst_mean": fs.means[0],
                                "hurst_std": fs.stds[0],
                            }
                            # If DCCA -> store rho
                            if method == "DCCA" and fs.rho_means is not None:
                                method_results[pair_label]["rho_mean"] = fs.rho_means[0]
                                method_results[pair_label]["rho_std"] = fs.rho_stds[0]
                            # If MFDCCA -> store Fq, deltaAlpha, p(q) if present
                            if method == "MFDCCA":
                                if fs.Fq_means is not None:
                                    method_results[pair_label]["Fq_mean"] = fs.Fq_means[0]
                                    method_results[pair_label]["Fq_std"]  = fs.Fq_stds[0]
                                if fs.deltaAlpha_means is not None:
                                    method_results[pair_label]["deltaAlpha_mean"] = fs.deltaAlpha_means[0]
                                    method_results[pair_label]["deltaAlpha_std"]  = fs.deltaAlpha_stds[0]
                                if fs.p_means is not None:
                                    method_results[pair_label]["p_mean"] = fs.p_means[0]
                                    method_results[pair_label]["p_std"]  = fs.p_stds[0]

                        else:
                            # real vs synthetic => index=1
                            method_results[pair_label] = {
                                "hurst_mean": fs.means[1],
                                "hurst_std": fs.stds[1],
                            }
                            if method == "DCCA" and fs.rho_means is not None:
                                method_results[pair_label]["rho_mean"] = fs.rho_means[1]
                                method_results[pair_label]["rho_std"]  = fs.rho_stds[1]
                            if method == "MFDCCA":
                                if fs.Fq_means is not None:
                                    method_results[pair_label]["Fq_mean"] = fs.Fq_means[1]
                                    method_results[pair_label]["Fq_std"]  = fs.Fq_stds[1]
                                if fs.deltaAlpha_means is not None:
                                    method_results[pair_label]["deltaAlpha_mean"] = fs.deltaAlpha_means[1]
                                    method_results[pair_label]["deltaAlpha_std"]  = fs.deltaAlpha_stds[1]
                                if fs.p_means is not None:
                                    method_results[pair_label]["p_mean"] = fs.p_means[1]
                                    method_results[pair_label]["p_std"]  = fs.p_stds[1]

                    except Exception as e:
                        print(f"Error in {method} analysis for {type1} vs {type2}: {str(e)}")
                        method_results[pair_label] = {"error": str(e)}

        results[method] = method_results

    # 4) Print summary
    print("\n======= FRACTAL ANALYSIS RESULTS SUMMARY =======\n")

    # Theoretical references for single-series MFDFA
    theoretical_values = {
        "White Noise": 0.5,
        "Pink Noise": 1.0,
        "Brown Noise": 1.5
    }

    # DCCA references for same-signal pairs
    dcca_self_refs = {
        "White Noise vs White Noise": {"H": 0.5, "rho": 0.0},
        "Pink Noise vs Pink Noise":   {"H": 1.0, "rho": 0.9},
        "Brown Noise vs Brown Noise": {"H": 1.5, "rho": 0.9}
    }

    # MFDCCA references for same-signal pairs (approx placeholders)
    mfdcca_self_refs = {
        "White Noise vs White Noise": {"H": 0.5, "Fq": 0.5, "deltaAlpha": 0.5},
        "Pink Noise vs Pink Noise":   {"H": 1.0, "Fq": 1.0, "deltaAlpha": 1.0},
        "Brown Noise vs Brown Noise": {"H": 1.5, "Fq": 1.5, "deltaAlpha": 1.5}
    }

    # Summaries for MFDFA
    if "MFDFA" in results:
        print("\n--- MFDFA Hurst Exponents vs. Theoretical ---")
        mfdfa_res = results["MFDFA"]
        for s_type, val in mfdfa_res.items():
            if s_type in theoretical_values:
                calc_h = val["mean"]
                theo_h = theoretical_values[s_type]
                print(f"  {s_type}: H_calc={calc_h:.3f}, H_theo={theo_h:.2f}")

    # Summaries for DCCA
    if "DCCA" in results:
        print("\n--- DCCA Self-Correlation Validation ---")
        dcca_res = results["DCCA"]
        for pair, val in dcca_res.items():
            if pair in dcca_self_refs:
                # Compare H
                if "hurst_mean" in val:
                    calc_h = val["hurst_mean"]
                    ref_h = dcca_self_refs[pair]["H"]
                    print(f"  {pair}: H_calc={calc_h:.3f}, H_ref={ref_h:.2f}")
                # Compare rho
                if "rho_mean" in val:
                    calc_rho = val["rho_mean"]
                    ref_rho = dcca_self_refs[pair]["rho"]
                    print(f"  {pair}: rho_calc={calc_rho:.3f}, rho_ref={ref_rho:.2f}")

    # Summaries for MFDCCA
    if "MFDCCA" in results:
        print("\n--- MFDCCA Self-Correlation Validation ---")
        mfdcca_res = results["MFDCCA"]
        for pair, val in mfdcca_res.items():
            if pair in mfdcca_self_refs:
                # Compare H
                if "hurst_mean" in val and "H" in mfdcca_self_refs[pair]:
                    calc_h = val["hurst_mean"]
                    ref_h = mfdcca_self_refs[pair]["H"]
                    print(f"  {pair}: H_calc={calc_h:.3f}, H_ref={ref_h:.2f}")
                # Compare Fq
                if "Fq_mean" in val and "Fq" in mfdcca_self_refs[pair]:
                    calc_fq = val["Fq_mean"]
                    ref_fq = mfdcca_self_refs[pair]["Fq"]
                    print(f"  {pair}: Fq_calc={calc_fq:.3f}, Fq_ref={ref_fq:.2f}")
                # Compare Δα
                if "deltaAlpha_mean" in val and "deltaAlpha" in mfdcca_self_refs[pair]:
                    calc_da = val["deltaAlpha_mean"]
                    ref_da = mfdcca_self_refs[pair]["deltaAlpha"]
                    print(f"  {pair}: Δα_calc={calc_da:.3f}, Δα_ref={ref_da:.2f}")

                # If you also want to show p_mean (the cross-correlation ratio):
                if "p_mean" in val:
                    print(f"  {pair}: p_mean={val['p_mean']:.3f}")

    return results


# VALIDATION ON EEG-vs-EEG (REAL vs. SYNTH)                    

def validate_eeg_data():
    """
    Test using real EEG vs synthetic EEG signals, referencing known literature
    values for EEG fractal exponents.

    Literature references (e.g. Podobnik & Stanley, 2007) often report:
      - Single-channel EEG H ~ 0.8–0.9
      - Cross-correlation exponent ~ 0.84 for multi-channel EEG
      - Possibly p(q) ~ 0.6–0.8 range depending on channel overlap

    For intraoperative ECoG (ioECoG) or other invasive recordings under deep
    anesthesia or slow-wave states, exponents can be higher (H ~ 1.2–1.5).
    E.g.:
      - Le Van Quyen et al. (2001, 2005) in epilepsy iEEG
      - Freeman et al. (2003) in cat ECoG
      - Destexhe & Rudy (2007) modeling LFP/ECoG scaling
    """
    def load_pickle(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)

    # Adjust the file paths to your environment
    real_data = load_pickle("/Users/is/PycharmProjects/seege_/data/good1A_injured_processed.pkl")
    synthetic_data = load_pickle("/Users/is/PycharmProjects/seege_/data/generated_signals_EcogGAN_1A_injured_new_trimm_60s.pkl")

    # Example: take first 10 signals for demonstration
    real_data = np.array(real_data[:10])
    synthetic_data = np.array(synthetic_data[:10]).squeeze()

    print("EEG (Real) shape:", real_data.shape)
    print("EEG (Synth) shape:", synthetic_data.shape)

    # MFDCCA Example (multifractal cross-correlation) 
    print("\n-- EEG vs EEG: MFDCCA --")
    fs_mfdcca = FractalSimilarity(real_data, synthetic_data, method='MFDCCA')
    fs_mfdcca.compute_fractal_metrics()

    # We can interpret these results relative to typical EEG or ECoG values:
    print("\n=== EEG/ECoG MFDCCA: Literature References ===")
    print(" - Podobnik & Stanley (2007): scalp EEG often 0.8–1.0, cross-exponent ~0.84.")
    print(" - Freeman, Le Van Quyen, Destexhe (2000s+): iEEG/ECoG can reach 1.2–1.5 under deep sedation.")
    print("Check the above results to see if your data matches these ranges.\n")

    # Show the core MFDCCA metrics:
    print("Measured cross-H exponents (H(q) averaged):       ", fs_mfdcca.means)
    print("Measured fluctuation function F_q means:         ", fs_mfdcca.Fq_means)
    print("Measured singularity widths Δα:                  ", fs_mfdcca.deltaAlpha_means)
    print("Measured cross-correlation ratio p(q) (avg):     ", fs_mfdcca.p_means)


if __name__=="__main__":
    # 1) Validate known noise signals
    results = validate_fractal_properties()
    print("\nFINAL NOISE VALIDATION RESULTS:\n", results)

    # 2) Validate real EEG vs synthetic EEG
    validate_eeg_data()
