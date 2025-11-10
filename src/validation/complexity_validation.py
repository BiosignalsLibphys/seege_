
from complexity_fidelity import *

# Synthetic noise generators
def generate_white_noise(length=10000, num_signals=10):
    """White noise (H≈0.5)."""
    return [np.random.randn(length) for _ in range(num_signals)]

def generate_pink_noise(length=10000, num_signals=10):
    """Pink noise (H≈1.0) via 1/sqrt(f) shaping."""
    signals = []
    for _ in range(num_signals):
        X_white = np.fft.rfft(np.random.randn(length))
        freqs = np.fft.rfftfreq(length)
        freqs[0] = freqs[1]  # avoid division by 0 at DC
        X_pink = X_white / np.sqrt(freqs)
        pink = np.fft.irfft(X_pink, length)
        pink = (pink - np.mean(pink)) / (np.std(pink) + 1e-12)
        signals.append(pink)
    return signals

def generate_brown_noise(length=10000, num_signals=10):
    """Brown noise (H≈1.5) via cum-sum(white) + detrend + z-score."""
    signals = []
    for _ in range(num_signals):
        white = np.random.randn(length)
        brown = np.cumsum(white)
        brown = detrend(brown)
        brown = (brown - np.mean(brown)) / (np.std(brown) + 1e-12)
        signals.append(brown)
    return signals

# Noise validation (Fractal + Entropy)
def validate_fractal_properties(show_plots=False):
    """
    Validate ComplexityFidelity on white/pink/brown noise.
    Runs DCCA, MFDFA, MFDCCA, and entropy metrics (SampEn, PermEn, LZC).
    """
    signal_length = 10000
    num_signals = 10

    print("Generating noise signals...")
    signals = {
        "White Noise": generate_white_noise(signal_length, num_signals),
        "Pink Noise":  generate_pink_noise(signal_length,  num_signals),
        "Brown Noise": generate_brown_noise(signal_length, num_signals),
    }

    if show_plots:
        plt.figure(figsize=(12, 6))
        for k, (name, data) in enumerate(signals.items(), start=1):
            plt.subplot(3, 1, k)
            plt.plot(data[0][:500])
            plt.title(f"{name} (first 500 samples)")
            plt.tight_layout()
        plt.show()

    results = {"DCCA": {}, "MFDFA": {}, "MFDCCA": {}, "ENTROPY": {}}
    signal_types = list(signals.keys())

    # MFDFA (single-series)
    print("\n==== Testing MFDFA (single-series) ====\n")
    for s_type in signal_types:
        cf = ComplexityFidelity(signals[s_type], signals[s_type], method="MFDFA")
        cf.compute_fractal_metrics()
        results["MFDFA"][s_type] = {
            "H_mean_real": cf.means[0], "H_std_real": cf.stds[0],
            "H_mean_syn":  cf.means[1], "H_std_syn":  cf.stds[1],
        }

    # DCCA / MFDCCA (pairwise)
    for method in ["DCCA", "MFDCCA"]:
        print(f"\n==== Testing {method} (pairwise) ====\n")
        for i, type1 in enumerate(signal_types):
            for j, type2 in enumerate(signal_types):
                if j < i:
                    continue
                pair = f"{type1} vs {type2}"
                cf = ComplexityFidelity(signals[type1], signals[type2], method=method)
                try:
                    cf.compute_fractal_metrics()
                    out = {
                        "H_mean_rr": cf.means[0], "H_std_rr": cf.stds[0],
                        "H_mean_rs": cf.means[1], "H_std_rs": cf.stds[1],
                        "H_mean_ss": cf.means[2], "H_std_ss": cf.stds[2],
                    }
                    if method == "DCCA":
                        out.update({
                            "rho_mean_rr": cf.rho_means[0], "rho_std_rr": cf.rho_stds[0],
                            "rho_mean_rs": cf.rho_means[1], "rho_std_rs": cf.rho_stds[1],
                            "rho_mean_ss": cf.rho_means[2], "rho_std_ss": cf.rho_stds[2],
                        })
                    if method == "MFDCCA":
                        out.update({
                            "Fq_mean_rr": cf.Fq_means[0], "Fq_std_rr": cf.Fq_stds[0],
                            "Fq_mean_rs": cf.Fq_means[1], "Fq_std_rs": cf.Fq_stds[1],
                            "Fq_mean_ss": cf.Fq_means[2], "Fq_std_ss": cf.Fq_stds[2],
                            "deltaAlpha_rr": cf.deltaAlpha_means[0], "deltaAlpha_std_rr": cf.deltaAlpha_stds[0],
                            "deltaAlpha_rs": cf.deltaAlpha_means[1], "deltaAlpha_std_rs": cf.deltaAlpha_stds[1],
                            "deltaAlpha_ss": cf.deltaAlpha_means[2], "deltaAlpha_std_ss": cf.deltaAlpha_stds[2],
                            "p_mean_rr": cf.p_means[0], "p_std_rr": cf.p_stds[0],
                            "p_mean_rs": cf.p_means[1], "p_std_rs": cf.p_stds[1],
                            "p_mean_ss": cf.p_means[2], "p_std_ss": cf.p_stds[2],
                        })
                    results[method][pair] = out
                except Exception as e:
                    print(f"  {method} error on {pair}: {e}")
                    results[method][pair] = {"error": str(e)}

    # Entropy/Complexity WD for all pairwise combinations
    print("\n==== Testing Entropy/Complexity (SampEn, PermEn, LZC) ====\n")
    for i, type1 in enumerate(signal_types):
        for j, type2 in enumerate(signal_types):
            if j < i:
                continue
            pair = f"{type1} vs {type2}"
            cf = ComplexityFidelity(signals[type1], signals[type2], method="DCCA")
            e_out = cf.compute_entropy_complexity_metrics(
                signals[type1], signals[type2],
                sampen_m=2, sampen_r=None,     # r defaults to 0.2*std
                permen_m=3, permen_tau=1,
                lzc_threshold=None,            # threshold = median
                n_surrogates=0,                # set >0 for nonlinearity z-scores
                verbose=True
            )
            results["ENTROPY"][pair] = e_out

    # Summaries
    print("\n======= FRACTAL ANALYSIS RESULTS SUMMARY =======")
    theoretical = {"White Noise": 0.5, "Pink Noise": 1.0, "Brown Noise": 1.5}

    print("\n--- MFDFA Hurst Exponents vs. Theoretical ---")
    for s_type, val in results["MFDFA"].items():
        h_real = val["H_mean_real"]; h_syn = val["H_mean_syn"]
        h_ref = theoretical.get(s_type, np.nan)
        print(f"  {s_type}: H_real={h_real:.3f}, H_syn={h_syn:.3f}, H_ref≈{h_ref:.2f}")

    if "DCCA" in results:
        print("\n--- DCCA Self-Correlation (same-type pairs) ---")
        for name in ["White Noise vs White Noise", "Pink Noise vs Pink Noise", "Brown Noise vs Brown Noise"]:
            if name in results["DCCA"]:
                v = results["DCCA"][name]
                print(f"  {name}: H_rr={v['H_mean_rr']:.3f}, H_ss={v['H_mean_ss']:.3f}, H_rs={v['H_mean_rs']:.3f}")
                if "rho_mean_rr" in v:
                    print(f"           rho_rr={v['rho_mean_rr']:.3f}, rho_ss={v['rho_mean_ss']:.3f}, rho_rs={v['rho_mean_rs']:.3f}")

    if "MFDCCA" in results:
        print("\n--- MFDCCA Self-Correlation (same-type pairs) ---")
        for name in ["White Noise vs White Noise", "Pink Noise vs Pink Noise", "Brown Noise vs Brown Noise"]:
            if name in results["MFDCCA"]:
                v = results["MFDCCA"][name]
                print(f"  {name}: H_rr={v['H_mean_rr']:.3f}, H_ss={v['H_mean_ss']:.3f}, H_rs={v['H_mean_rs']:.3f}")
                print(f"           Δα_rr={v['deltaAlpha_rr']:.3f}, Δα_ss={v['deltaAlpha_ss']:.3f}, Δα_rs={v['deltaAlpha_rs']:.3f}")
                print(f"           ⟨p(q)⟩_rr={v['p_mean_rr']:.3f}, ⟨p(q)⟩_ss={v['p_mean_ss']:.3f}, ⟨p(q)⟩_rs={v['p_mean_rs']:.3f}")

    print("\n--- Entropy Wasserstein Distances (lower = more similar) ---")
    for name, e in results["ENTROPY"].items():
        print(f"  {name}: WD_SampEn={e['WD_SampEn']:.4g}, WD_PermEn={e['WD_PermEn']:.4g}, WD_LZC={e['WD_LZC']:.4g}")

    return results

# Extra Entropy/Complexity validation scenarios
import numpy as np
from complexity_fidelity import ComplexityFidelity

def _make_sine(n=10, length=5000, fs=250, freq=10.0, noise_std=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(length) / fs
    X = []
    for i in range(n):
        x = np.sin(2*np.pi*freq*t)
        if noise_std > 0:
            x = x + rng.normal(0, noise_std, size=length)
        X.append((x - x.mean()) / (x.std() + 1e-12))
    return X

def _make_white(n=10, length=5000, seed=0):
    rng = np.random.default_rng(seed)
    return [(rng.standard_normal(length)) for _ in range(n)]

def _make_AR1(n=10, length=5000, rho=0.1, sigma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    X = []
    for i in range(n):
        x = np.zeros(length, dtype=float)
        e = rng.normal(0, sigma, size=length)
        for t in range(1, length):
            x[t] = rho * x[t-1] + e[t]
        X.append((x - x.mean()) / (x.std() + 1e-12))
    return X

def _make_logistic(n=10, length=5000, r=4.0, x0=0.123456):
    # chaotic logistic map in (0,1); rescale to zero mean / unit var
    X = []
    for i in range(n):
        x = np.empty(length, dtype=float)
        x[0] = (x0 + i*0.01) % 1.0
        for t in range(1, length):
            x[t] = r * x[t-1] * (1.0 - x[t-1])
        x = (x - np.mean(x)) / (np.std(x) + 1e-12)
        X.append(x)
    return X

def _phase_surrogates_batch(X, cf_dummy=None, seed=0):
    # Use your class's FFT phase-randomized surrogate for consistency
    if cf_dummy is None:
        cf_dummy = ComplexityFidelity([X[0]], [X[0]], method="DCCA")
    rng = np.random.default_rng(seed)
    Y = []
    for x in X:
        y = cf_dummy._fft_phase_randomized_surrogate(np.asarray(x), rng=rng)
        y = (y - y.mean()) / (y.std() + 1e-12)
        Y.append(y)
    return Y

def _block_shuffle(X, block=50, seed=0):
    # Break temporal order but preserve amplitude distribution roughly
    rng = np.random.default_rng(seed)
    Y = []
    for x in X:
        x = np.asarray(x)
        L = len(x)
        idx = np.arange(0, L, block)
        blocks = [x[i:i+block] for i in idx]
        rng.shuffle(blocks)
        y = np.concatenate(blocks)
        y = (y - y.mean()) / (y.std() + 1e-12)
        Y.append(y)
    return Y

def _entropy_compare(real, synth, **kwargs):
    # method is irrelevant here; we call the entropy method directly
    cf = ComplexityFidelity(real, synth, method="DCCA")
    return cf.compute_entropy_complexity_metrics(real, synth, verbose=False, **kwargs)

def validate_entropy_scenarios(
    length=5000, n=10, fs=250, verbose=True,
    sampen_m=2, sampen_r=None, permen_m=3, permen_tau=1, lzc_threshold=None,
    n_surrogates=0
):
    """
    Runs targeted entropy/complexity scenarios and prints WD + means.
    Returns a dict of results per scenario.
    """
    results = {}

    # 1) Identical periodic: sine vs sine
    A = _make_sine(n=n, length=length, fs=fs, freq=10.0, noise_std=0.0, seed=1)
    B = _make_sine(n=n, length=length, fs=fs, freq=10.0, noise_std=0.0, seed=2)
    res = _entropy_compare(A, B, sampen_m=sampen_m, sampen_r=sampen_r,
                           permen_m=permen_m, permen_tau=permen_tau,
                           lzc_threshold=lzc_threshold, n_surrogates=n_surrogates)
    results["Sine vs Sine (identical dynamics)"] = res

    # 2) Sine vs White Noise
    A = _make_sine(n=n, length=length, fs=fs, freq=10.0, noise_std=0.0, seed=3)
    B = _make_white(n=n, length=length, seed=3)
    res = _entropy_compare(A, B, sampen_m=sampen_m, sampen_r=sampen_r,
                           permen_m=permen_m, permen_tau=permen_tau,
                           lzc_threshold=lzc_threshold, n_surrogates=n_surrogates)
    results["Sine vs White Noise"] = res

    # 3) Clean Sine vs Noisy Sine
    A = _make_sine(n=n, length=length, fs=fs, freq=10.0, noise_std=0.0, seed=4)
    B = _make_sine(n=n, length=length, fs=fs, freq=10.0, noise_std=0.3, seed=4)
    res = _entropy_compare(A, B, sampen_m=sampen_m, sampen_r=sampen_r,
                           permen_m=permen_m, permen_tau=permen_tau,
                           lzc_threshold=lzc_threshold, n_surrogates=n_surrogates)
    results["Clean Sine vs Noisy Sine"] = res

    # 4) Logistic map (chaotic) vs phase-randomized surrogate
    A = _make_logistic(n=n, length=length, r=4.0, x0=0.1234)
    B = _phase_surrogates_batch(A, seed=5)
    res = _entropy_compare(A, B, sampen_m=sampen_m, sampen_r=sampen_r,
                           permen_m=permen_m, permen_tau=permen_tau,
                           lzc_threshold=lzc_threshold, n_surrogates=n_surrogates)
    results["Logistic (chaos) vs Phase-Surrogate"] = res

    # 5) White Noise vs Block-Shuffled White Noise (destroys local order)
    A = _make_white(n=n, length=length, seed=6)
    B = _block_shuffle(A, block=50, seed=6)
    res = _entropy_compare(A, B, sampen_m=sampen_m, sampen_r=sampen_r,
                           permen_m=permen_m, permen_tau=permen_tau,
                           lzc_threshold=lzc_threshold, n_surrogates=n_surrogates)
    results["White Noise vs Block-Shuffled"] = res

    # 6) AR(1): low vs high persistence (ρ=0.1 vs 0.9)
    A = _make_AR1(n=n, length=length, rho=0.1, sigma=1.0, seed=7)
    B = _make_AR1(n=n, length=length, rho=0.9, sigma=1.0, seed=7)
    res = _entropy_compare(A, B, sampen_m=sampen_m, sampen_r=sampen_r,
                           permen_m=permen_m, permen_tau=permen_tau,
                           lzc_threshold=lzc_threshold, n_surrogates=n_surrogates)
    results["AR(1) ρ=0.1 vs ρ=0.9"] = res

    if verbose:
        print("\n==== ENTROPY/COMPLEXITY VALIDATION SCENARIOS ====\n")
        for name, e in results.items():
            def fmt(x): return "nan" if not np.isfinite(x) else f"{x:.4g}"
            print(f"[{name}]")
            print(f"  WD_SampEn = {fmt(e['WD_SampEn'])} | WD_PermEn = {fmt(e['WD_PermEn'])} | WD_LZC = {fmt(e['WD_LZC'])}")
            print(f"  Means (Real vs Synth): "
                  f"SampEn {fmt(e['Real_SampEn_mean'])}/{fmt(e['Synth_SampEn_mean'])}, "
                  f"PermEn {fmt(e['Real_PermEn_mean'])}/{fmt(e['Synth_PermEn_mean'])}, "
                  f"LZC {fmt(e['Real_LZC_mean'])}/{fmt(e['Synth_LZC_mean'])}")
            print()

    return results


if __name__ == "__main__":
    # 1) Fractal + baseline entropy on white/pink/brown
    noise_results = validate_fractal_properties(show_plots=False)

    # 2) Targeted entropy/complexity stress tests
    entropy_results = validate_entropy_scenarios(
        length=5000, n=10, fs=250,
        sampen_m=2, sampen_r=None,    # r defaults to 0.2*std
        permen_m=3, permen_tau=1,
        lzc_threshold=None,
        n_surrogates=0,               # set >0 to compute nonlinearity z-scores for scenario #4 (real-only)
        verbose=True
    )
