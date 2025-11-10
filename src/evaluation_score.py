
from amplitude_fidelity import *
from time_fidelity import *
from frequency_fidelity import *
from time_frequency_fidelity import *
from complexity_fidelity import *
from diversity import *
from privacy import *

def _sim(a, b, eps=1e-12):
    """Normalised absolute-difference similarity in [0,1]."""
    return 1.0 - np.abs(a - b) / (np.abs(a) + np.abs(b) + eps)

def compute_amplitude_fidelity_score(real_data, synthetic_data, fs, *,
                                       mode: str = "all_vs_all",
                                       nperseg: int = 256):
    """
    Computes the Feature Selective Validation (FSV) similarity score
    between real and synthetic signals using the AmplitudeFidelity class.

    Parameters
    ----------
    real_data : np.ndarray
        Real signal data. Can be 1D (single signal) or 2D (multiple signals).
    synthetic_data : np.ndarray
        Synthetic signal data. Can be 1D (single signal) or 2D (multiple signals).
    fs : int
        Sampling frequency of the signals.
    mode : {"zip", "all_vs_all"}, default="all_vs_all"
        Pairing strategy used when both inputs are 2D.
    nperseg : int, default=256
        Segment length used for Welch PSD computation.


    Example Usage:
    --------------
    real_data = np.random.randn(10, 2048)  # 10 real signals, each 2048 samples
    synthetic_data = np.random.randn(10, 2048) # 10 synthetic signals, each 2048 samples

    evaluation_score.compute_amplitude_fidelity_score(real_data, synthetic_data, fs=2048)
    evaluation_score.compute_amplitude_fidelity_score(real_data[0], synthetic_data[0], fs=2048)

    Returns
    -------
    float
        Mean FSV fidelity score.
    """
    fsv = AmplitudeFidelity(fs)
    metrics = fsv.compute_amplitude_metrics(real_data, synthetic_data,mode=mode, nperseg=nperseg)

    print("Amplitude Fidelity Score: {:.2f}".format(metrics["Similarity"]))

    return metrics["Similarity"]

def compute_time_fidelity_score(real_data, synthetic_data, weights=None):
    """
    Compute a time-domain fidelity score between real and synthetic signals
    using Hjorth parameter statistics.

    Integrated components (each mapped as S = 1/(1 + distance)):
        1. Hjorth: Activity (WD)
        2. Hjorth: Mobility (WD)
        3. Hjorth: Complexity (WD)
        4. Hjorth: Mahalanobis distance (means)

    Parameters
    ----------
    real_data : np.ndarray or list
        Real signals of shape [n_signals, n_samples].
    synthetic_data : np.ndarray or list
        Synthetic signals with the same shape as real_data.
    weights : dict, optional
        Weights for the components. Keys:
        {'activity','mobility','complexity','mahalanobis'}.
        Default: equal weights across all provided components 0.25 each.

    Returns
    -------
    float
        Composite time-domain fidelity score in [0, 1].
    """
    # Defaults (auto-balance across 4 components)
    if weights is None:
        w = 1.0 / 4.0
        weights = {
            'activity': w, 'mobility': w, 'complexity': w, 'mahalanobis': w
        }
    # Ensure all keys exist; missing ones default to 0 (excluded from sum)
    for k in ('activity','mobility','complexity','mahalanobis'):
        weights.setdefault(k, 0.0)

    # Compute base Hjorth metrics
    ts = TimeFidelity()
    hj = ts.compute_hjorth_metrics(real_data, synthetic_data, verbose=False)

    # Helper: distance -> similarity with NaN/Inf safety
    import numpy as _np
    def _sim(d):
        d = float(d)
        if not _np.isfinite(d) or d < 0:
            return 0.0
        return 1.0 / (1.0 + d)

    # Hjorth similarities
    activity_score    = _sim(hj['WD_Activity'])
    mobility_score    = _sim(hj['WD_Mobility'])
    complexity_score  = _sim(hj['WD_Complexity'])
    mahalanobis_score = _sim(hj['Mahalanobis'])


    # Weighted combination
    time_fidelity_score = (
        weights['activity']    * activity_score   +
        weights['mobility']    * mobility_score   +
        weights['complexity']  * complexity_score +
        weights['mahalanobis'] * mahalanobis_score
    )

    # Print components
    print(f"Time Fidelity Score  : {time_fidelity_score:.3f}")
    print("Time Fidelity Components:")
    print(f"  Activity: {activity_score:.3f}")
    print(f"  Mobility: {mobility_score:.3f}")
    print(f"  Complexity: {complexity_score:.3f}")
    print(f"  Mahalanobis: {mahalanobis_score:.3f}")

    return time_fidelity_score

def compute_frequency_fidelity_score(real_data, synthetic_data, fs, weights=None):
    """
    Compute a similarity score between real and synthetic signals that blends
    spectral-band power, dominant frequency, coherence and a PSD–Wasserstein
    similarity term.

    Parameters
    ----------
    real_data : list | np.ndarray
        List/array of real signals (shape: [n_signals, n_samples] or 1-D).
    synthetic_data : list | np.ndarray
        List/array of synthetic signals (shape: [n_signals, n_samples] or 1-D).
    fs : int
        Sampling frequency (Hz).
    weights : dict, optional
        Weights for the four sub-scores.  Must contain the keys:
            {'relative', 'dom_freq', 'psd_coherence', 'wasserstein'}
        Defaults to: {'relative': 0.40,
                      'dom_freq': 0.20,
                      'psd_coherence': 0.20,
                      'wasserstein': 0.20}

    Example usage:
    --------------

    real_data = np.random.randn(10, 2048)  # 10 real signals, each 2048 samples
    synthetic_data = np.random.randn(10, 2048) # 10 synthetic signals, each 2048 samples

    evaluation_score.compute_frequency_fidelity_score(real_data, synthetic_data, fs=2048)

    Returns
    -------
    float
        Composite frequency-domain fidelity score in the range (0, 1].
    """
    # Default values definition

    analysis_band = (0.5, 100.0)  # Hz
    win_seconds = 4.0  # Welch window length (s)
    window = 'hann'
    detrend = 'constant'
    overlap = 0.5  # 50%

    # Initialize FrequencyFidelity class
    frequency_fidelity = FrequencyFidelity(fs,
                 analysis_band=analysis_band,
                 win_seconds=win_seconds,
                 window=window,
                 detrend=detrend,
                 overlap=overlap)

    # Weights
    if weights is None:
        weights = {'relative': 0.40,
                   'dom_freq': 0.20,
                   'psd_coherence': 0.20,
                   'wasserstein': 0.20}

    # Fallback – add any missing keys with zero weight
    for k in ('relative', 'dom_freq', 'psd_coherence', 'wasserstein'):
        weights.setdefault(k, 0.0)

    # Shape handling
    if isinstance(real_data, np.ndarray) and real_data.ndim == 1:
        real_data = [real_data]
    if isinstance(synthetic_data, np.ndarray) and synthetic_data.ndim == 1:
        synthetic_data = [synthetic_data]

    # 1) Relative power
    freqs_r, psd_r, rel_power_r, dominant_freq_r = \
        frequency_fidelity.compute_relative_power(real_data, analysis_band=analysis_band,
                 win_seconds=win_seconds,
                 window=window,
                 detrend=detrend,
                 overlap=overlap)
    freqs_s, psd_s, rel_power_s, dominant_freq_s = \
        frequency_fidelity.compute_relative_power(synthetic_data,analysis_band=analysis_band,
                 win_seconds=win_seconds,
                 window=window,
                 detrend=detrend,
                 overlap=overlap)

    band_names = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    real_mean_rel_power = np.array([np.nanmean(rel_power_r[b]) for b in band_names])
    synth_mean_rel_power = np.array([np.nanmean(rel_power_s[b]) for b in band_names])
    mean_band_diff = np.nanmean(np.abs(real_mean_rel_power - synth_mean_rel_power) * 100)  # %
    # Map diff → similarity (clipped ≥ 0.2 to avoid 0 in extreme cases)
    relative_power_score = max(0.2, 1.0 - mean_band_diff / 20.0)

    # 2) Dominant frequency
    freq_diff = abs(np.nanmean(dominant_freq_r) - np.nanmean(dominant_freq_s))
    dominant_freq_score = max(0.2, 1.0 - freq_diff / 3.0)

    # 3) Coherence
    coh = frequency_fidelity.spectral_coherence(
        real_data, synthetic_data,
        mode="all_vs_all", per_band=False,
        analysis_band=analysis_band, win_seconds=2.0,
        window=window, detrend=detrend, overlap=overlap
    )
    mean_coherence = float(coh["RS Summary"]["Global Mean"]) if np.isfinite(coh["RS Summary"]["Global Mean"]) else 0.0
    mean_coherence = float(np.clip(mean_coherence, 0.0, 1.0))

    # 4) Spectral Wasserstein distance
    wd_psd = frequency_fidelity.spectral_wasserstein_distance(
        real_data, synthetic_data,
        fmin=analysis_band[0],
        fmax=analysis_band[1],
        mode="pairmean",
        per_band=False
    )
    wasserstein_score = 1.0 / (1.0 + wd_psd) if np.isfinite(wd_psd) else 0.2

    # Composite score
    frequency_fidelity_score = (
        weights['relative']      * relative_power_score  +
        weights['dom_freq']      * dominant_freq_score   +
        weights['psd_coherence'] * mean_coherence        +
        weights['wasserstein']   * wasserstein_score
    )

    print(f"Frequency Fidelity Score: {frequency_fidelity_score:.2f}")
    return frequency_fidelity_score


def compute_time_frequency_fidelity_score(real_data, synthetic_data, fs, *, weights=None, mode: str = "auto",
    pad: bool = True, rr_zip_strategy: str = "consecutive", ss_zip_strategy: str = "consecutive",return_sd: bool = False,
    return_per_pair: bool = False, verbose: bool = True):
    """
    Composite time frequnecy similarity for 1-D or 2-D inputs.
    Reuses `compute_scalogram_similarity_metrics` and combines RS per-pair metrics:

        score_i = w_cssim * SSIM_i
                  + w_rmse * (1 / (1 + nRMSE_i))
                  + w_cos   * Cosine_i

    Parameters
    ----------
    real_data, synthetic_data : array_like
        1-D (T,) or 2-D (N, T). Only these are required.
    weights : dict, optional
        {'color_ssim':0.4, 'rmse':0.3, 'cosine_similarity':0.3} by default.
    mode : {"auto","zip","all_vs_all"}, default "auto"
        - "auto": use "zip" if both inputs are 1-D or both are 2-D with the same N; else "all_vs_all".
        - "zip": index-wise pairing.
        - "all_vs_all": every real vs every synthetic.
    pad : bool, default True
        Right-pad signals to a common length before CWT.
    rr_zip_strategy, ss_zip_strategy : {"consecutive","halves"}
        Zip strategies for within-set RR/SS in the metrics method.
    return_sd : bool, default False
        Also return SD of per-pair scores.
    return_per_pair : bool, default False
        Also return list of per-pair scores.
    verbose : bool, default True
        Print a brief summary.

    Example usage:
    --------------

    real_data = np.random.randn(10, 2048)  # 10 real signals, each 2048 samples
    synthetic_data = np.random.randn(10, 2048) # 10 synthetic signals, each 2048 samples

    evaluation_score.compute_scalogram_fidelity_score(real_data, synthetic_data, fs=2048)

    Returns
    -------
    mean_score : float
    (sd_score) : float, optional if return_sd=True
    (per_pair_scores) : list[float], optional if return_per_pair=True
    """

    if weights is None:
        weights = {'color_ssim': 0.4, 'rmse': 0.3, 'cosine_similarity': 0.3}
    w_cssim = float(weights.get('color_ssim', 0.4))
    w_rmse  = float(weights.get('rmse', 0.3))
    w_cos   = float(weights.get('cosine_similarity', 0.3))

    # Normalize inputs: auto-wrap 1-D → (1, T)
    R = np.asarray(real_data, dtype=float)
    S = np.asarray(synthetic_data, dtype=float)

    if R.ndim == 1:
        R = R[None, :]  # shape (1, T)
    if S.ndim == 1:
        S = S[None, :]  # shape (1, T)

    # Auto-select pairing if requested

    if mode == "auto":
        if R.shape[0] == S.shape[0]:
            eff_mode = "zip"  # pairwise 1–1
        else:
            eff_mode = "all_vs_all"  # every real vs every synthetic
    else:
        eff_mode = mode

    scalo = TimeFrequencyFidelity(fs=fs)

    # Reuse your metrics method (does RS, RR, SS under the same mode/strategies)
    metrics = scalo.compute_scalogram_similarity_metrics(
        R, S,
        mode=eff_mode, pad=pad,
        rr_zip_strategy=rr_zip_strategy, ss_zip_strategy=ss_zip_strategy,
    )

    # RS per-pair lists → composite per-pair scores
    ssim_list  = metrics.get("Per-pair SSIM (RS)", [])
    nrmse_list = metrics.get("Per-pair NRMSE (RS)", [])
    cos_list   = metrics.get("Per-pair Cosine (RS)", [])

    scores = []
    for ssim_i, nrmse_i, cos_i in zip(ssim_list, nrmse_list, cos_list):
        cos_i = max(-1.0, min(1.0, float(cos_i)))  # safety
        sim_rmse = 1.0 / (1.0 + float(nrmse_i))    # maps to (0,1]
        score_i = w_cssim*float(ssim_i) + w_rmse*sim_rmse + w_cos*cos_i
        scores.append(float(score_i))

    if not scores:
        mean_score = float('nan'); sd_score = float('nan')
    else:
        arr = np.asarray(scores, dtype=float)
        mean_score = float(np.mean(arr))
        sd_score = float(np.std(arr, ddof=1)) if arr.size > 1 else float('nan')

    if verbose:
        label = eff_mode if eff_mode != "zip" else f"zip"
        print(f"Time-frequency Fidelity Score: {mean_score:.3f} | mode: {label}, pairs: {len(scores)}")

    out = (mean_score,)
    if return_sd:
        out += (sd_score,)
    if return_per_pair:
        out += (scores,)
    return out if len(out) > 1 else out[0]


def compute_complexity_fidelity_score(
        real_data,
        synthetic_data,
        q_range=np.arange(-5, 5, 0.1),
        weights=None
    ):
    """
    Complexity fidelity score combining fractal and entropy/complexity similarities.

    This function computes six subscores that reflect how closely synthetic signals
    reproduce the complexity properties of real signals. Each subscore is mapped to
    [0,1] (higher = more similar), and the final score is the weighted average of
    available subscores (ignoring any NaNs).

    Integrated components
    ---------------------
    Fractal (multifractal / cross-fractal), mapped via range-aware similarities:
      1) DCCA   — cross Hurst exponent similarity   (S_H_dcca)
      2) MFDFA  — single-series H + H(q) curve      (S_H_mfdfa & S_Hq → F_MFDFA)
      3) MFDCCA — cross H(q) + Δα spectrum width    (S_H_mfdcca & S_Dalpha → F_MFDCCA)

    Entropy/complexity (distribution similarity via WD), mapped as S = 1 / (1 + WD):
      4) Sample Entropy           (WD_SampEn → S_SampEn)
      5) Permutation Entropy      (WD_PermEn → S_PermEn)
      6) Lempel–Ziv Complexity    (WD_LZC    → S_LZC)

    Parameters
    ----------
    real_data, synthetic_data : list[np.ndarray] or np.ndarray
        1-D signals. A single 1-D array is auto-wrapped into a list. Signals should be
        pre-normalized to comparable ranges when possible. NaN/inf signals are skipped.
    q_range : np.ndarray, optional
        q-orders for multifractal analysis (used in MFDFA/MFDCCA). Default: np.arange(-5,5,0.1).
    weights : dict, optional
        Weights per subscore. If None, all six subscores share equal weight (=1/6):
        {
            'dcca': 1/6, 'mfdfa': 1/6, 'mfdcca': 1/6,
            'sampen': 1/6, 'permen': 1/6, 'lzc': 1/6
        }
        Any NaN subscore is dropped and weights are renormalized over the remaining ones.

    Returns
    -------
    float
        Complexity fidelity in [0,1] (higher ⇒ closer match).

    Example
    -------
    real_data = np.random.randn(10, 2048)      # 10 real signals
    synthetic_data = np.random.randn(10, 2048) # 10 synthetic signals
    complexity_score = compute_complexity_fidelity_score(real_data, synthetic_data)
    """

    # Default: equal weights across ALL subscores (6 components)
    if weights is None:
        weights = {
            'dcca': 1/6, 'mfdfa': 1/6, 'mfdcca': 1/6,
            'sampen': 1/6, 'permen': 1/6, 'lzc': 1/6
        }

    # Wrap 1D arrays
    if isinstance(real_data, np.ndarray) and real_data.ndim == 1:
        real_data = [real_data]
    if isinstance(synthetic_data, np.ndarray) and synthetic_data.ndim == 1:
        synthetic_data = [synthetic_data]

    # Helpers
    def _sim_range(a, b, lo, hi):
        # Range-aware similarity in [0,1]
        d = min(abs(float(a) - float(b)), hi - lo)
        return 1.0 - d / (hi - lo)

    def _mean_Hq(signals, q_range, get_scales):
        # Average single-series H(q) curves over a set of signals
        Hqs = []
        for x in signals:
            scales = get_scales(len(x))
            if len(scales) < 4:
                continue
            _, info = nk.fractal_dfa(x, scale=scales, multifractal=True, q=q_range, show=False)
            Hq = np.asarray(info["H"])
            if np.all(np.isfinite(Hq)):
                Hqs.append(Hq)
        if not Hqs:
            return None
        return np.nanmean(np.vstack(Hqs), axis=0)

    # DCCA (use Hxy only)
    fs_dcca = ComplexityFidelity(real_data, synthetic_data, method='DCCA', q_range=q_range)
    fs_dcca.compute_fractal_metrics()
    H_rr, H_rs = fs_dcca.means[:2]
    S_H_dcca = _sim_range(H_rr, H_rs, lo=0.3, hi=1.2)
    F_DCCA = S_H_dcca  # rho dropped from scoring to avoid inflation

    #  MFDFA: H level + H(q) curve shape
    fs_mfdfa = ComplexityFidelity(real_data, synthetic_data, method='MFDFA', q_range=q_range)
    fs_mfdfa.compute_fractal_metrics()
    H_r, H_s = fs_mfdfa.means
    S_H_mfdfa = _sim_range(H_r, H_s, lo=0.3, hi=1.2)

    dummy = ComplexityFidelity(real_data, synthetic_data, method='MFDFA', q_range=q_range)
    Hq_r = _mean_Hq(real_data, q_range, dummy._get_win_sizes)
    Hq_s = _mean_Hq(synthetic_data, q_range, dummy._get_win_sizes)
    if Hq_r is not None and Hq_s is not None:
        rmse = float(np.sqrt(np.nanmean((Hq_r - Hq_s) ** 2)))
        tau = 0.15
        S_Hq = float(np.exp(-rmse / tau))
    else:
        S_Hq = np.nan

    F_MFDFA = S_H_mfdfa if np.isnan(S_Hq) else 0.5 * S_H_mfdfa + 0.5 * S_Hq

    #  MFDCCA: cross H(q) & Δα
    fs_mfdcca = ComplexityFidelity(real_data, synthetic_data, method='MFDCCA', q_range=q_range)
    fs_mfdcca.compute_fractal_metrics()
    Hc_rr, Hc_rs = fs_mfdcca.means[:2]
    Da_rr, Da_rs = fs_mfdcca.deltaAlpha_means[:2]

    S_H_mfdcca = _sim_range(Hc_rr, Hc_rs, lo=0.3, hi=1.2)
    S_Dalpha = _sim_range(Da_rr, Da_rs, lo=0.0, hi=0.5)
    F_MFDCCA = 0.7 * S_H_mfdcca + 0.3 * S_Dalpha

    # Entropy/complexity (WD → similarity)
    # Use the class method to compute WDs; then map S = 1/(1+WD)
    cf_entropy = ComplexityFidelity(real_data, synthetic_data, method='MFDFA', q_range=q_range)
    e = cf_entropy.compute_entropy_complexity_metrics(
        real_data, synthetic_data,
        sampen_m=2, sampen_r=None,  # r defaults to 0.2*std
        permen_m=3, permen_tau=1,
        lzc_threshold=None,
        n_surrogates=0,             # set >0 if you want surrogate z-scores (not used in score)
        verbose=False
    )

    def _to_sim(wd):
        return np.nan if not np.isfinite(wd) else 1.0 / (1.0 + float(wd))

    S_SampEn = _to_sim(e.get("WD_SampEn", np.nan))
    S_PermEn = _to_sim(e.get("WD_PermEn", np.nan))
    S_LZC    = _to_sim(e.get("WD_LZC",    np.nan))

    # Aggregate
    subscores = {
        'dcca':   F_DCCA,
        'mfdfa':  F_MFDFA,
        'mfdcca': F_MFDCCA,
        'sampen': S_SampEn,
        'permen': S_PermEn,
        'lzc':    S_LZC
    }

    valid_vals = [(k, v) for k, v in subscores.items() if np.isfinite(v)]
    if not valid_vals:
        raise RuntimeError("Complexity fidelity score could not be computed (all subscores NaN).")

    numer = sum(weights[k] * subscores[k] for k, v in valid_vals)
    denom = sum(weights[k] for k, v in valid_vals)
    score = numer / denom

    # Pretty print
    print(f"Complexity Fidelity Score: {score:0.2f}")
    for k in ['dcca','mfdfa','mfdcca','sampen','permen','lzc']:
        v = subscores[k]
        vs = "nan" if not np.isfinite(v) else f"{v:0.2f}"
        print(f"  {k.upper():7s}: {vs}")

    return score



def compute_fidelity_score(real_data, synthetic_data, fs):
    """
    Computes an overall fidelity score by averaging amplitude, time, frequency, time-frequency,
    and complexity fidelity scores.

    Parameters
    ----------
    real_data : list or np.ndarray
        List of real signals.
    synthetic_data : list or np.ndarray
        List of synthetic signals.
    fs : int
        Sampling frequency of the signals.

    Returns
    -------
    float
        The computed fidelity score.

    Example usage:
    --------------

    real_data = np.random.randn(10, 2048)  # 10 real signals, each 2048 samples
    synthetic_data = np.random.randn(10, 2048) # 10 synthetic signals, each 2048 samples

    evaluation_score.compute_fidelity_score(real_data, synthetic_data, fs=2048)
    """
    amp_sim = compute_amplitude_fidelity_score(real_data, synthetic_data, fs)
    time_sim = compute_time_fidelity_score(real_data, synthetic_data)
    freq_sim = compute_frequency_fidelity_score(real_data, synthetic_data, fs)
    scalogram_sim = compute_time_frequency_fidelity_score(real_data, synthetic_data, fs)
    fractal_sim = compute_complexity_fidelity_score(real_data, synthetic_data)

    fidelity_score = (amp_sim + time_sim + freq_sim + scalogram_sim + fractal_sim) / 5
    print(f"Fidelity Score: {fidelity_score:.2f}")
    print(f"  - Amplitude Similarity Score: {amp_sim:.2f}")
    print(f"  - Time Similarity Score: {time_sim:.2f}")
    print(f"  - Frequency Similarity Score: {freq_sim:.2f}")
    print(f"  - Scalogram Similarity Score: {scalogram_sim:.2f}")
    print(f"  - Fractal Similarity Score: {fractal_sim:.2f}")
    return fidelity_score

def compute_diversity_score(real_data, synthetic_data, weights=None, n_components=2):
    """
    Computes a composite diversity score between real and synthetic EEG datasets.

    Nine normalised sub-metrics are averaged (or linearly combined via ``weights``):

        1.  C_cov        – Gaussian-weighted coverage                    ↑ good
        2.  O_out        – Gaussian-weighted outlier goodness            ↑ good
        3.  S_PCA        – silhouette coefficient in PCA space           ↑ good
        4.  S_UM         – silhouette coefficient in UMAP space          ↑ good
        5.  D_PCA        – centroid overlap score in PCA space           ↑ good
        6.  D_UM         – centroid overlap score in UMAP space          ↑ good
        7.  U_NN         – Uniqueness (NN ratio, syn/real, normalized)   ↑ good
        8.  L_loc        – Local diversity (P50 NN ratio, normalized)    ↑ good
        9.  G_glob       – Global diversity (pairwise ratio, normalized) ↑ good

    Parameters
    ----------
    real_data : np.ndarray
        Shape (n_samples, n_features).
    synthetic_data : np.ndarray
        Shape (n_samples, n_features).
    weights : dict, optional
        Dictionary with keys:
        ['coverage','outliers','pca_compactness','umap_compactness',
         'pca_overlap','umap_overlap','uniqueness','local_div','global_div'].
        Defaults to equal weights (1/9 each).
    n_components : int, optional
        Dimensionality of PCA / UMAP (default = 2).

    Example Usage:
    --------------
    real_data = np.random.randn(10, 2048)  # 10 real signals, each 2048 samples
    synthetic_data = np.random.randn(10, 2048) # 10 synthetic signals, each 2048 samples

    evaluation_score.compute_diversity_score(real_data, synthetic_data)

    Returns
    -------
    float
        Diversity score ∈ [0, 1] (higher ⇒ greater similarity/diversity quality).
    """

    # Helper for symmetric normalization of ratio metrics
    def normalize_ratio(ratio):
        """Map ratio>0 to [0,1], with 1 as ideal (ratio=1)."""
        if not np.isfinite(ratio) or ratio <= 0:
            return np.nan
        return float(np.exp(-abs(np.log(ratio))))

    # Compute raw metrics from the Diversity class
    div = Diversity(n_components=n_components)
    m_cov = div.compute_coverage_diversity(real_data, synthetic_data)
    m_geom = div.compute_geometric_diversity(real_data, synthetic_data)
    m_intr = div.compute_intrinsic_diversity(real_data, synthetic_data)

    # Transform each metric to a [0,1] similarity scale (higher=better)
    C_cov = m_cov['Coverage']           # already [0,1]
    O_out = m_cov['Outliers']           # already [0,1]
    S_PCA = m_geom['PCA_Compactness']    # already [0,1]
    S_UM  = m_geom['UMAP_Compactness']   # already [0,1]
    D_PCA = m_geom['PCA_OverlapMahalanobis']     # already [0,1] (overlap score)
    D_UM  = m_geom['UMAP_OverlapMahalanobis']    # already [0,1]

    # Normalize intrinsic diversity ratios (symmetrically centered at 1)
    U_NN   = normalize_ratio(m_intr['Uniqueness_NN'])
    L_loc  = normalize_ratio(m_intr['Local_Diversity_P50'])
    G_glob = normalize_ratio(m_intr['Global_Diversity'])

    # Default weights (equal importance)
    if weights is None:
        weights = {
            'coverage'         : 1/9,
            'outliers'         : 1/9,
            'pca_compactness'  : 1/9,
            'umap_compactness' : 1/9,
            'pca_overlap'      : 1/9,
            'umap_overlap'     : 1/9,
            'uniqueness'       : 1/9,
            'local_div'        : 1/9,
            'global_div'       : 1/9,
        }

    # Weighted composite score
    diversity_score = (
        weights['coverage']        * C_cov +
        weights['outliers']        * O_out +
        weights['pca_compactness'] * S_PCA +
        weights['umap_compactness']* S_UM  +
        weights['pca_overlap']     * D_PCA +
        weights['umap_overlap']    * D_UM  +
        weights['uniqueness']      * U_NN  +
        weights['local_div']       * L_loc +
        weights['global_div']      * G_glob
    )

    # Metrics printout
    print(f"Diversity Score: {diversity_score:.3f}")
    print(f"Coverage: {C_cov:.3f}")
    print(f"Outlier Goodness: {O_out:.3f}")
    print(f"PCA Compactness: {S_PCA:.3f}")
    print(f"UMAP Compactness: {S_UM:.3f}")
    print(f"PCA Overlap: {D_PCA:.3f}")
    print(f"UMAP Overlap: {D_UM:.3f}")
    print(f"Uniqueness: {U_NN:.3f}")
    print(f"Local Diversity: {L_loc:.3f}")
    print(f"Global Diversity: {G_glob:.3f}")


    return diversity_score



def compute_privacy_score(real_data, synthetic_data, weights: dict | None = None):
    """
    Computes a composite **privacy score** between real and synthetic EEG datasets.

    Four sub-metrics are averaged (or linearly combined via `weights`):

        1.  WD / (1 + WD)   – Wasserstein distance (↑ distance ⇒ ↑ privacy)
        2.  ED / (1 + ED)   – Euclidean histogram distance (↑ distance ⇒ ↑ privacy)
        3.  JSD             – Jensen–Shannon divergence (↑ divergence ⇒ ↑ privacy)
        4.  1 − MIR_acc     – Membership inference attack accuracy inverted (↑ privacy)

    Parameters
    ----------
    real_data : np.ndarray | list
        Real signals; each sample will be flattened then histogram-binned.
    synthetic_data : np.ndarray | list
        Synthetic signals processed in the same way.
    weights : dict, optional
        Custom weights for the metrics; keys must include
        'wd', 'ed', 'jsd', 'mir'.
        Values are automatically re-normalised to sum 1.

    Returns
    -------
    float
        Privacy score in [0, 1]; higher = safer.
    """

    pr = Privacy()

    # Calculate global distances
    dists = pr.compute_privacy_metrics(real_data, synthetic_data)

    # Convert to [0,1] scores (larger = safer)
    scores = {
        "wd": dists["wd"] / (1.0 + dists["wd"]),
        "ed": dists["ed"] / (1.0 + dists["ed"]),
        "jsd": dists["jsd"],  # already in [0,1]
    }

    # Compute MIR metric
    X_real = np.stack(real_data)
    X_synth = np.stack(synthetic_data)
    num_samples = len(X_real)
    half = num_samples // 2
    y_real = np.array([0] * half + [1] * (num_samples - half))
    np.random.shuffle(y_real)

    mir_metrics = pr.compute_privacy_metrics(
        real_data, synthetic_data,
        X_real=X_real, y_real=y_real, X_synth=X_synth,
        use_new_mir=False
    )
    # Try a set of keys from both legacy and black-box styles
    candidate_keys = [
        "confidence_attack_acc", "mir_conf_acc",
        "attack_acc", "mia_acc",  # generic fallbacks if you ever add them
    ]
    attack_acc = None
    for k in candidate_keys:
        if k in mir_metrics and mir_metrics[k] is not None:
            attack_acc = float(mir_metrics[k])
            break

    if attack_acc is None or np.isnan(attack_acc):
        # Prefer to fail loudly rather than pretend privacy is perfect
        available = ", ".join(mir_metrics.keys())
        raise ValueError(
            f"MIR attack accuracy not found. Looked for {candidate_keys}. "
            f"Available keys: {available}"
        )

    # Map accuracy → privacy score:
    #   0.5 (random) → 1.0 (best privacy)
    #   1.0 (perfect attack) → 0.0 (worst privacy)
    #   <0.5 treated as random (cap at 0.5)
    attack_acc = max(0.5, min(1.0, attack_acc))
    scores["mir"] = 2.0 * (1.0 - attack_acc)

    # 3) Weights + aggregate
    base_w = {"wd": 0.25, "ed": 0.25, "jsd": 0.25, "mir": 0.25}
    if weights is not None:
        base_w.update(weights)
    tot = sum(base_w.values())
    base_w = {k: v / tot for k, v in base_w.items()}

    privacy_score = sum(base_w[k] * scores[k] for k in scores)

    print(f"Privacy Score: {privacy_score:.2f}")
    print(f"  - WD Score:  {scores['wd']:.2f}")
    print(f"  - ED Score:  {scores['ed']:.2f}")
    print(f"  - JSD Score: {scores['jsd']:.2f}")
    print(f"  - MIR Score: {scores['mir']:.2f}")

    return privacy_score
