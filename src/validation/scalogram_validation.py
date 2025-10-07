from scalogram_similarity import *
from typing import Dict, List, Tuple


# Signal generation 

def generate_signals(fs: int = 2048,
                     duration: float = 2.0,
                     seed: int = 42,
                     powerline_freq: int = 50) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Generate a library of validation signals.

    Parameters
    ----------
    fs : int
        Sampling frequency [Hz].
    duration : float
        Duration [s].
    seed : int
        RNG seed.
    powerline_freq : int
        50 or 60 Hz.

    Returns
    -------
    signals : dict[str, np.ndarray]
        Named signals (all length T = fs*duration).
    t : np.ndarray
        Time vector.
    """
    if powerline_freq not in (50, 60):
        raise ValueError("powerline_freq must be 50 or 60.")

    rng = np.random.default_rng(seed)
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    f = 10.0  # base frequency

    signals: Dict[str, np.ndarray] = {}

    # 1) Identical
    base = np.sin(2 * np.pi * f * t)
    signals["identical_1"] = base.copy()
    signals["identical_2"] = base.copy()

    # 2) Sine + small noise
    signals["sine_clean"] = base
    signals["sine_small_noise"] = base + 0.01 * rng.standard_normal(len(t))

    # 3) Sine + larger noise
    signals["sine_large_noise"] = base + 0.5 * rng.standard_normal(len(t))

    # 4) Square vs. brown noise
    signals["square_wave"] = signal.square(2 * np.pi * f * t)
    brown = np.cumsum(0.01 * rng.standard_normal(len(t)))
    brown /= np.max(np.abs(brown)) or 1.0
    signals["brown_noise"] = brown

    # 5) Powerline (with harmonics)
    pl = (np.sin(2 * np.pi * powerline_freq * t)
          + 0.5 * np.sin(2 * np.pi * 2 * powerline_freq * t)
          + 0.25 * np.sin(2 * np.pi * 3 * powerline_freq * t))
    pl /= np.max(np.abs(pl)) or 1.0
    signals["powerline_noise"] = pl

    # 6) Chirp
    signals["chirp"] = signal.chirp(t, f0=1, f1=30, t1=duration, method='linear')

    # 7) Bursts (deterministic placement via RNG for reproducibility)
    burst = np.zeros_like(t)
    burst_len = int(0.05 * fs)  # 50 ms
    # pick 5 non-overlapping start indices
    starts = rng.choice(np.arange(0, len(t) - burst_len, burst_len), size=5, replace=False)
    osc = np.sin(2 * np.pi * 20 * np.linspace(0, burst_len / fs, burst_len, endpoint=False))
    for s in starts:
        burst[s:s + burst_len] = osc
    signals["burst"] = burst

    # 8) Triangle vs sine
    signals["triangle_wave"] = signal.sawtooth(2 * np.pi * f * t, 0.5)

    # "Different but related" extras
    signals["am_sine"] = (1 + 0.5 * np.sin(2 * np.pi * 2 * t)) * base
    signals["sine_5hz"] = np.sin(2 * np.pi * 5 * t)
    signals["sine_15hz"] = np.sin(2 * np.pi * 15 * t)

    # Highpass-noisy sine (HP at ~0.2 * Nyquist = 0.2 * (fs/2) -> Wn=0.2 in [0..1])
    noisy = base + 0.3 * rng.standard_normal(len(t))
    b, a = signal.butter(5, 0.2, btype='highpass')
    signals["sine_highpass_noise"] = signal.filtfilt(b, a, noisy)

    return signals, t


# Pairwise scalogram validation 

def validate_scalogram_similarity():
    """
    Validate scalogram similarity metrics on curated signal pairs.
    Produces individual scalogram plots and a summary table.
    """
    fs = 2048
    duration = 2.0

    signals, t = generate_signals(fs=fs, duration=duration, powerline_freq=50)
    analyzer = ScalogramSimilarity(fs=fs, freq_min=0.5, freq_max=100, num_freqs=100)

    # Plot the generated signals (grid)
    num = len(signals)
    rows = (num + 3) // 4
    plt.figure(figsize=(16, 3 * rows))
    for i, (name, sig) in enumerate(signals.items()):
        ax = plt.subplot(rows, 4, i + 1)
        ax.plot(t, sig, linewidth=1)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amp')
    plt.tight_layout()
    plt.savefig('generated_signals.png', dpi=160)
    plt.show()

    # Define signal pairs
    pairs: List[Tuple[str, np.ndarray, np.ndarray]] = [
        ("Identical Signals", signals["identical_1"], signals["identical_2"]),
        ("Sine vs Small Noise", signals["sine_clean"], signals["sine_small_noise"]),
        ("Sine vs Large Noise", signals["sine_clean"], signals["sine_large_noise"]),
        ("Sine vs Triangle", signals["sine_clean"], signals["triangle_wave"]),
        ("Square vs Brown", signals["square_wave"], signals["brown_noise"]),
        ("Sine vs Powerline", signals["sine_clean"], signals["powerline_noise"]),
        ("Sine vs Chirp", signals["sine_clean"], signals["chirp"]),
        ("Sine vs Burst", signals["sine_clean"], signals["burst"]),
        ("Sine vs AM Sine", signals["sine_clean"], signals["am_sine"]),
        ("5 Hz vs 15 Hz", signals["sine_5hz"], signals["sine_15hz"]),
        ("Clean vs HP-Noisy", signals["sine_clean"], signals["sine_highpass_noise"]),
    ]

    results = []
    for title, x, y in pairs:
        print(f"\nAnalyzing {title}:")
        analyzer.plot_scalograms(x, y)
        metrics = analyzer.compute_scalogram_similarity_metrics(x, y)  # CHANGED: now returns dict
        results.append({
            "Pair": title,
            "SSIM": metrics["Mean SSIM"],
            "NRMSE": metrics["Mean NRMSE"],
            "Cosine": metrics["Mean Cosine Similarity"]
        })

    # Pretty summary
    print("\nSummary of Scalogram Similarity Metrics")
    print("-" * 86)
    print(f"{'Signal Pair':<28} | {'SSIM':>7} | {'NRMSE':>7} | {'Cosine':>7}")
    print("-" * 86)
    for r in results:
        print(f"{r['Pair']:<28} | {r['SSIM']:7.4f} | {r['NRMSE']:7.4f} | {r['Cosine']:7.4f}")

    print("\nGuideline bins (tune empirically for your datasets):")
    print("1) Identical:           SSIM > 0.99,  Cosine > 0.999, NRMSE < 0.01")
    print("2) Very similar:        0.93–0.99,    > 0.99,         0.01–0.10")
    print("3) Moderately similar:  0.50–0.93,    0.95–0.99,      0.10–0.50")
    print("4) Different:           0.10–0.50,    0.20–0.95,      0.50–1.50")
    print("5) Completely different:< 0.10,       < 0.20,         > 1.50")

# Mean-scalogram validation (set-level comparison)

def validate_mean_scalogram_similarity():
    """
    Validate the new mean-scalogram workflow on *sets* of signals.

    It builds two small sets that should be close (clean sine vs small-noise sine),
    and two sets that should be farther (clean sine vs powerline mix),
    then:
      - plots the mean scalograms side-by-side
      - reports SSIM / NRMSE / Cosine between those *mean* scalograms
    """
    fs = 2048
    duration = 2.0
    signals, _ = generate_signals(fs=fs, duration=duration, powerline_freq=50)
    analyzer = ScalogramSimilarity(fs=fs, freq_min=0.5, freq_max=100, num_freqs=100)

    # Build small cohorts (same length signals)
    rng = np.random.default_rng(123)

    # A) Close cohorts: clean sine vs small-noise variations
    clean_set = []
    noise_small_set = []
    for _ in range(8):
        clean_set.append(signals["sine_clean"])
        noise_small_set.append(signals["sine_clean"] + 0.01 * rng.standard_normal(len(signals["sine_clean"])))

    # B) Farther cohorts: clean sine vs powerline-contaminated
    powerline_set = []
    for _ in range(8):
        # add mild amplitude jitter to prevent trivial exact duplicates
        amp = 0.8 + 0.4 * rng.random()
        powerline_set.append(amp * signals["powerline_noise"])

    # Case 1: close cohorts (expect high similarity)
    print("\n[Mean-Scalogram Validation] Close cohorts: Clean vs Small-Noise")
    analyzer.plot_mean_scalograms(clean_set, noise_small_set,
                                  titles=("Mean real (clean)", "Mean synth (small noise)"),
                                  save="mean_scalo_close.png")
    m_close_real = analyzer.compute_mean_scalogram(clean_set)
    m_close_synth = analyzer.compute_mean_scalogram(noise_small_set)
    close_metrics = _compare_two_scalograms(analyzer, m_close_real, m_close_synth)
    _print_mean_scalo_metrics(close_metrics)

    # Case 2: farther cohorts (expect lower similarity)
    print("\n[Mean-Scalogram Validation] Farther cohorts: Clean vs Powerline")
    analyzer.plot_mean_scalograms(clean_set, powerline_set,
                                  titles=("Mean real (clean)", "Mean synth (powerline)"),
                                  save="mean_scalo_far.png")
    m_far_real = analyzer.compute_mean_scalogram(clean_set)
    m_far_synth = analyzer.compute_mean_scalogram(powerline_set)
    far_metrics = _compare_two_scalograms(analyzer, m_far_real, m_far_synth)
    _print_mean_scalo_metrics(far_metrics)


# Helpers for mean-scalogram validation 
def _compare_two_scalograms(analyzer: ScalogramSimilarity,
                            A: np.ndarray,
                            B: np.ndarray) -> Dict[str, float]:
    """
    Use the same metric stack as pairwise comparison but on two already-computed scalograms.
    Returns a dict with Mean SSIM / Mean NRMSE / Mean Cosine Similarity.
    """
    # Reuse analyzer internals for consistent scaling
    vmin = float(min(A.min(), B.min()))
    vmax = float(max(A.max(), B.max()))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    # Convert to RGB for SSIM (consistent with your pipeline)
    A_rgb = analyzer._convert_to_rgb(A, vmin=vmin, vmax=vmax)
    B_rgb = analyzer._convert_to_rgb(B, vmin=vmin, vmax=vmax)

    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import mean_squared_error

    ssim_val = ssim(A_rgb, B_rgb, channel_axis=2, data_range=255)
    rmse = np.sqrt(mean_squared_error(A, B))
    nrmse = rmse / (vmax - vmin)
    from sklearn.metrics.pairwise import cosine_similarity
    cos = float(cosine_similarity(A.flatten().reshape(1, -1), B.flatten().reshape(1, -1))[0, 0])

    return {"Mean SSIM": float(ssim_val), "Mean NRMSE": float(nrmse), "Mean Cosine Similarity": cos}


def _print_mean_scalo_metrics(metrics: Dict[str, float]) -> None:
    print("Mean-Scalogram Metrics:")
    print(f"  SSIM   : {metrics['Mean SSIM']:.4f}")
    print(f"  NRMSE  : {metrics['Mean NRMSE']:.4f}")
    print(f"  Cosine : {metrics['Mean Cosine Similarity']:.4f}")

# Main execution

if __name__ == "__main__":
    # Pairwise validation (plots + table)
    validate_scalogram_similarity()

    # NEW: mean-scalogram validation (plots + metrics)
    validate_mean_scalogram_similarity()
