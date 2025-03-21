from src.frequency_similarity import *

def main_validation():
    fs = 2048
    t = np.linspace(0, 1, fs)

    # Generate simple sine waves in various bands
    alpha = 10
    beta = 20
    delta = 1
    theta = 6
    gamma = 40

    alpha_test = np.sin(2 * np.pi * alpha * t)
    beta_test = np.sin(2 * np.pi * beta * t)
    delta_test = np.sin(2 * np.pi * delta * t)
    theta_test = np.sin(2 * np.pi * theta * t)
    gamma_test = np.sin(2 * np.pi * gamma * t)

    # Powerline noise
    powerline_50hz = np.sin(2 * np.pi * 50 * t)
    powerline_60hz = np.sin(2 * np.pi * 60 * t)

    # List of comparisons
    scenarios = [
        ("Alpha vs Alpha", alpha_test, alpha_test, 10.0, 1.0, 0.0),
        ("Alpha vs Beta", alpha_test, beta_test, (10.0, 20.0), 0.0, "Large"),
        ("Beta vs Delta", beta_test, delta_test, (20.0, 1.0), 0.0, "Large"),
        ("Delta vs Theta", delta_test, theta_test, (1.0, 6.0), 0.0, "Large"),
        ("Theta vs Gamma", theta_test, gamma_test, (6.0, 40.0), 0.0, "Large"),
        ("50Hz vs 60Hz", powerline_50hz, powerline_60hz, (50.0, 60.0), 0.0, "Large"),
    ]
    """
      For each scenario we store:
        - scenario name
        - real_signal
        - synthetic_signal
        - expected_dominant_frequency (or pair)
        - expected_coherence
        - expected_wdistance (descriptive, e.g. 0.0 or "Large")
    """

    freq_analyzer = FrequencySimilarity(fs=fs)

    # We'll collect results in a list of dicts for building a final table
    all_results = []

    # Create one figure with 2 rows (time domain + PSD) and 6 columns
    # so that all comparisons appear in a single figure
    fig, axes = plt.subplots(nrows=2, ncols=len(scenarios), figsize=(4 * len(scenarios), 6))
    fig.suptitle("Validation of Multiple Comparisons (Time-Domain & PSD)", fontsize=14)

    for i, (name, real_sig, synth_sig, exp_dom, exp_coh, exp_wd) in enumerate(scenarios):
        # Time-Domain Plot (top row)
        ax_td = axes[0, i]
        ax_td.plot(t, real_sig, label="Real")
        ax_td.plot(t, synth_sig, label="Synthetic")
        ax_td.set_title(f"{name}\nTime Domain")
        ax_td.set_xlabel("Time (s)")
        ax_td.set_ylabel("Amplitude")
        ax_td.grid(True)
        if i == 0:  # show legend only in the first column
            ax_td.legend()

        # PSD Plot (bottom row)
        ax_psd = axes[1, i]
        # Compute PSD for each signal
        freqs_r, psd_r, _, _ = freq_analyzer.compute_relative_power(real_sig)
        freqs_s, psd_s, _, _ = freq_analyzer.compute_relative_power(synth_sig)
        fr = freqs_r[0]
        fsd = freqs_s[0]
        # Single-sample => psd_r, psd_s are lists with a single entry
        ax_psd.plot(fr, psd_r[0], label="Real PSD")
        ax_psd.plot(fsd, psd_s[0], label="Synthetic PSD")
        ax_psd.set_title(f"{name}\nPSD")
        ax_psd.set_xlabel("Frequency (Hz)")
        ax_psd.set_ylabel("PSD")
        ax_psd.set_xlim(0, 60)  # up to 60 for clarity
        ax_psd.grid(True)
        if i == 0:  # legend only in the first column
            ax_psd.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # room for main title
    plt.show()

    # Now run the actual numeric comparisons and fill up our table
    for (name, real_sig, synth_sig, exp_dom, exp_coh, exp_wd) in scenarios:
        print(f"\n=== {name} ===")
        # 1) Coherence
        actual_coh = freq_analyzer.spectral_coherence(real_sig, synth_sig)

        # 2) Dominant Frequencies
        _, _, _, dom_real = freq_analyzer.compute_relative_power(real_sig)
        _, _, _, dom_synth = freq_analyzer.compute_relative_power(synth_sig)
        # They are single signals => each a list of one freq
        actual_dom_real = dom_real[0]
        actual_dom_synth = dom_synth[0]
        dom_diff = abs(actual_dom_real - actual_dom_synth)

        # 3) Wasserstein Distance
        actual_wd = freq_analyzer.spectral_wasserstein_distance(real_sig, synth_sig)

        # Store in results
        all_results.append({
            "scenario": name,
            "metric": "Coherence",
            "expected_value": exp_coh,
            "actual_value": round(actual_coh, 3)
        })
        all_results.append({
            "scenario": name,
            "metric": "DominantFreqDiff(Hz)",
            "expected_value": (
                0.0 if isinstance(exp_dom, float)
                else abs(exp_dom[0] - exp_dom[1])  # For pairs
            ),
            "actual_value": round(dom_diff, 3)
        })
        all_results.append({
            "scenario": name,
            "metric": "WassersteinDist",
            "expected_value": exp_wd,
            "actual_value": round(float(actual_wd), 3)
        })

    # Finally, print a table with columns: metric, expected value, actual value
    print("\n======================================")
    print("Summary Table (Metric | Expected | Actual)")
    print("======================================")
    for row in all_results:
        print(f"{row['scenario']} - {row['metric']}: "
              f"Expected={row['expected_value']}, "
              f"Actual={row['actual_value']}")

if __name__ == "__main__":
    main_validation()