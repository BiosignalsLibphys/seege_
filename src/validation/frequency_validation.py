
from frequency_fidelity import *
import numpy as np
import matplotlib.pyplot as plt

def _expected_dom_diff(exp_dom):
    """
    Accepts either a scalar (float/int) or a pair-like (a, b).
    - Scalar: expected dominant difference = 0.0
    - Pair:   absolute difference between the two values
    """
    if np.isscalar(exp_dom):
        return 0.0
    try:
        a, b = exp_dom
        return abs(float(a) - float(b))
    except Exception:
        return float('nan')


def main_validation():
    fs = 2048
    t = np.linspace(0, 1, fs, endpoint=False)  # 1 second

    # Base band sinusoids
    alpha = 10
    beta  = 20
    delta = 1
    theta = 6
    gamma = 40

    # HFOs
    ripple_freq      = 200   # 150–250 Hz
    fast_ripple_freq = 320   # 250–500 Hz

    # Signals
    alpha_test = np.sin(2 * np.pi * alpha * t)
    beta_test  = np.sin(2 * np.pi * beta  * t)
    delta_test = np.sin(2 * np.pi * delta * t)
    theta_test = np.sin(2 * np.pi * theta * t)
    gamma_test = np.sin(2 * np.pi * gamma * t)

    ripple_test      = np.sin(2 * np.pi * ripple_freq * t)
    fast_ripple_test = np.sin(2 * np.pi * fast_ripple_freq * t)

    # Powerline
    powerline_50hz = np.sin(2 * np.pi * 50 * t)
    powerline_60hz = np.sin(2 * np.pi * 60 * t)

    # Each tuple: (name, real, synth, expected_dominant (scalar or pair), expected_coh_note, expected_WD_note)
    scenarios = [
        ("Alpha vs Alpha", alpha_test, alpha_test, 10.0, 1.0, 0.0),
        ("Alpha vs Beta",  alpha_test, beta_test,  (10.0, 20.0), 0.0, "Large"),
        ("Beta vs Delta",  beta_test,  delta_test, (20.0, 1.0),  0.0, "Large"),
        ("Delta vs Theta", delta_test, theta_test, (1.0, 6.0),   0.0, "Large"),
        ("Theta vs Gamma", theta_test, gamma_test, (6.0, 40.0),  0.0, "Large"),
        ("50Hz vs 60Hz",   powerline_50hz, powerline_60hz, (50.0, 60.0), 0.0, "Large"),
        ("Ripple vs Ripple", ripple_test, ripple_test, ripple_freq, 1.0, 0.0),
        ("Ripple vs Beta",   ripple_test, beta_test,   (ripple_freq, 20.0), 0.0, "Large"),
        ("FastRipple vs FastRipple", fast_ripple_test, fast_ripple_test, fast_ripple_freq, 1.0, 0.0),
        ("FastRipple vs Ripple",     fast_ripple_test, ripple_test,       (fast_ripple_freq, ripple_freq), 0.0, "Large"),
    ]

    freq_analyzer = FrequencyFidelity(fs=fs)
    all_results = []

    # Figure with 2 rows (time + PSD) and one column per scenario
    fig, axes = plt.subplots(nrows=2, ncols=len(scenarios), figsize=(4 * len(scenarios), 6))
    fig.suptitle("Validation (Time-Domain & PSD)", fontsize=14)

    for i, (name, real_sig, synth_sig, exp_dom, exp_coh_note, exp_wd_note) in enumerate(scenarios):
        # Time domain
        ax_td = axes[0, i]
        ax_td.plot(t, real_sig, label="Real")
        ax_td.plot(t, synth_sig, label="Synthetic")
        ax_td.set_title(f"{name}\nTime Domain")
        ax_td.set_xlabel("Time (s)")
        ax_td.set_ylabel("Amplitude")
        ax_td.grid(True)
        if i == 0:
            ax_td.legend()

        # Decide PSD band: 0–100 by default; expand to 0–500 for Ripple/FastRipple scenarios
        use_hfo_band = ("Ripple" in name)
        PSD_BAND = (0.0, 500.0) if use_hfo_band else (0.0, 100.0)

        # PSD via compute_relative_power (returns lists of freqs & psd)
        ax_psd = axes[1, i]
        freqs_r, psd_r, _, _ = freq_analyzer.compute_relative_power(
            real_sig,
            analysis_band=PSD_BAND,
            win_seconds=None, window=None, detrend=None, overlap=None
        )
        freqs_s, psd_s, _, _ = freq_analyzer.compute_relative_power(
            synth_sig,
            analysis_band=PSD_BAND,
            win_seconds=None, window=None, detrend=None, overlap=None
        )

        fr = freqs_r[0]
        fsd = freqs_s[0]
        ax_psd.plot(fr, psd_r[0], label="Real PSD")
        ax_psd.plot(fsd, psd_s[0], label="Synthetic PSD")
        ax_psd.set_title(f"{name}\nPSD ({int(PSD_BAND[0])}–{int(PSD_BAND[1])} Hz)")
        ax_psd.set_xlabel("Frequency (Hz)")
        ax_psd.set_ylabel("PSD")
        ax_psd.set_xlim(*PSD_BAND)
        ax_psd.grid(True)
        if i == 0:
            ax_psd.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

    # Numeric comparisons
    for (name, real_sig, synth_sig, exp_dom, exp_coh_note, exp_wd_note) in scenarios:
        print(f"\n=== {name} ===")

        # Coherence over the class default global band (0.5–500, clipped by Nyquist)
        coh_res = freq_analyzer.spectral_coherence(
            real_sig, synth_sig,
            mode="zip",          # pair the two
            per_band=False,      # global only for this check
            analysis_band=None,  # use class default (0.5–500)
            win_seconds=None, window=None, detrend=None, overlap=None
        )
        actual_coh = coh_res["RS Summary"]["Global Mean"]

        # Dominant frequencies using class default dominance band (0.5–500)
        _, _, _, dom_real = freq_analyzer.compute_relative_power(
            real_sig, analysis_band=None, win_seconds=None, window=None, detrend=None, overlap=None
        )
        _, _, _, dom_synth = freq_analyzer.compute_relative_power(
            synth_sig, analysis_band=None, win_seconds=None, window=None, detrend=None, overlap=None
        )
        actual_dom_real = float(dom_real[0])
        actual_dom_synth = float(dom_synth[0])
        dom_diff = abs(actual_dom_real - actual_dom_synth)

        # Spectral Wasserstein distance (0.5–500 by default)
        actual_wd = freq_analyzer.spectral_wasserstein_distance(
            real_sig, synth_sig, fmin=0.5, fmax=500, mode="pairmean", per_band=False
        )

        # Collect
        all_results.append({
            "scenario": name,
            "metric": "Coherence (RS mean)",
            "expected_value": exp_coh_note,
            "actual_value": None if actual_coh is None or np.isnan(actual_coh) else round(float(actual_coh), 3),
        })
        all_results.append({
            "scenario": name,
            "metric": "DominantFreqDiff(Hz)",
            "expected_value": _expected_dom_diff(exp_dom),
            "actual_value": round(dom_diff, 3),
        })
        all_results.append({
            "scenario": name,
            "metric": "WassersteinDist (Hz)",
            "expected_value": exp_wd_note,
            "actual_value": None if actual_wd is None or np.isnan(actual_wd) else round(float(actual_wd), 3),
        })

    # Summary table
    print("\n======================================")
    print("Summary Table (Metric | Expected | Actual)")
    print("======================================")
    for row in all_results:
        print(f"{row['scenario']} - {row['metric']}: "
              f"Expected={row['expected_value']}, Actual={row['actual_value']}")

if __name__ == "__main__":
    main_validation()
