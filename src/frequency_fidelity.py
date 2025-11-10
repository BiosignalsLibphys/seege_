from matplotlib import rcParams
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, savgol_filter, coherence
from scipy.integrate import simps
from scipy.stats import shapiro, ttest_rel, wilcoxon
from scipy.stats import wasserstein_distance

# Set Arial as the default font
rcParams['font.family'] = 'Arial'

class FrequencyFidelity:
    """
    A class for evaluating frequency similarity between real and synthetic signals
    using Power Spectral Density (PSD) analysis and statistical tests.

    This class computes relative power in different frequency bands including HFO,
    dominant frequencies, and performs statistical comparisons (normality,
    paired t-test / Wilcoxon). It can also compute coherence and approximate
    a Wasserstein distance in the frequency domain.

    Example Usage:
    --------------
    real_data = np.random.randn(10, 2048)  # 10 real signals, each 2048 samples
    synthetic_data = np.random.randn(10, 2048) # 10 synthetic signals, each 2048 samples

    frequency_analysis = FrequencyFidelity(fs=512)
    frequency_analysis.compare_relative_power(real_data, synthetic_data)
    frequency_analysis.spectral_coherence(real_data, synthetic_data)
    frequency_analysis.spectral_wasserstein_distance(real_data, synthetic_data)
    frequency_analysis.plot_psd(real_data, synthetic_data, scale="linear")

    References:
    -----------
    [1] https://www.scitepress.org/Papers/2023/119908/119908.pdf
    [2] https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1219133/full
    [3] https://pmc.ncbi.nlm.nih.gov/articles/PMC11573898/
    """

    # Default values definition
    analysis_band = (0.5, 500.0)  # Hz
    win_seconds = 4.0  # Welch window length (s)
    window = 'hann'
    detrend = 'constant'
    overlap = 0.5  # 50%

    # Canonical EEG band edges (Hz)
    band_edges = [0.5, 4, 8, 13, 30, 80, 250, 500.0]
    band_names = ["Delta", "Theta", "Alpha", "Beta", "Gamma", "Ripple", "Fast Ripple"]

    if len(band_edges) != len(band_names) + 1:
        raise ValueError("band_edges must have len = len(band_names) + 1.")

    def __init__(self, fs,
                 analysis_band=analysis_band,
                 win_seconds=win_seconds,
                 window=window,
                 detrend=detrend,
                 overlap=overlap):
        """
        Initialize the FrequencyFidelity class with a given sampling frequency.

        Parameters
        ----------
        fs : int, optional
            Sampling frequency of the signals, by default 2048.
        """
        self.fs = fs
        nyq = fs / 2.0
        lo, hi = analysis_band
        self.fmin, self.fmax = float(lo), float(min(hi, nyq))
        self.win_seconds = float(win_seconds)
        self.window = window
        self.detrend = detrend
        self.overlap = float(overlap)

    def _format_p(self, p):
        """
        Helper that prints small p-values in scientific notation.
        """
        if p is None:
            return "—"
        if np.isnan(p):
            return "nan"
        if p < 1e-3:
            return f"< 0.001 (exact={p:.2e})"
        return f"{p:.3f}"

    def compute_cohens_d(self, real_vals, synthetic_vals):
        """
        Compute Cohen's d for paired samples (effect size).
        """
        differences = np.array(real_vals) - np.array(synthetic_vals)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        return mean_diff / std_diff if std_diff != 0 else np.nan

    def _resolve_params(self, analysis_band=None, win_seconds=None, window=None, detrend=None, overlap=None):
        """
        Per-call overrides merged with instance defaults.
        """
        fmin = self.fmin if analysis_band is None else float(analysis_band[0])
        fmax = self.fmax if analysis_band is None else float(analysis_band[1])
        win_s = self.win_seconds if win_seconds is None else float(win_seconds)
        wnd = self.window if window is None else window
        dtr = self.detrend if detrend is None else detrend
        ovl = self.overlap if overlap is None else float(overlap)
        return fmin, fmax, win_s, wnd, dtr, ovl

    def _compute_psd(self, sig,
                     analysis_band=None, win_seconds=None, window=None, detrend=None, overlap=None):
        """
        Unified Welch PSD. Returns: (f_full, pxx_full, f_band, pxx_band).
        """

        fmin, fmax, win_s, wnd, dtr, ovl = self._resolve_params(
            analysis_band, win_seconds, window, detrend, overlap
        )
        fs = self.fs
        L = len(sig)
        f_low = max(fmin, 1e-6)  # avoid divide-by-zero

        # Auto nperseg selection
        auto = (win_seconds is None) or (isinstance(win_seconds, str) and win_seconds.lower() == "auto")
        if auto:
            K_cycles = 10  # cycles of the lowest freq to include
            nperseg_target = int(np.ceil(K_cycles * fs / f_low))
            nperseg = min(L, max(8, nperseg_target))
        else:
            nperseg = min(L, int(float(win_s) * fs))
            nperseg = max(nperseg, 8)

        # Overlap
        noverlap = int(nperseg * (self.overlap if overlap is None else float(overlap)))
        if noverlap >= nperseg:
            noverlap = max(0, nperseg - 1)

        # Ensure enough segments (shrink nperseg if needed)
        def n_segments(L_, nseg_, ovlp_):
            step = max(1, nseg_ - ovlp_)
            return 1 + max(0, (L_ - nseg_) // step)

        min_segments = 4
        for shrink in (1.0, 0.75, 0.5, 0.33, 0.25):
            n_try = max(8, int(nperseg * shrink))
            o_try = min(int(n_try * (self.overlap if overlap is None else float(overlap))), n_try - 1)
            if n_segments(L, n_try, o_try) >= min_segments:
                nperseg, noverlap = n_try, o_try
                break
        # --------------------------------

        f, pxx = welch(sig, fs=fs, nperseg=nperseg, noverlap=noverlap,
                       window=(self.window if window is None else window),
                       detrend=(self.detrend if detrend is None else detrend),
                       scaling='density')

        mask = (f >= fmin) & (f <= fmax)
        f_sel, p_sel = f[mask], pxx[mask]
        return f, pxx, f_sel, p_sel

    def _band_mask(self, f_sel, lo, hi, right_closed=False):
        return ((f_sel >= lo) & (f_sel <= hi)) if right_closed else ((f_sel >= lo) & (f_sel < hi))

    def compute_relative_power(self, data, analysis_band, win_seconds, window, detrend,
                                                            overlap):
        """
        Compute the relative power for different frequency bands
        (Delta, Theta, Alpha, Beta, Gamma, Ripple and Fast Ripple) and determine the dominant frequency.

        Returns:
        -------
        tuple:
            - freqs (list of np.ndarray): Frequency arrays for each signal.
            - psd (list of np.ndarray): PSD arrays for each signal.
            - rel_power (dict): Relative power for each frequency band.
            - dominant_freq (list): Dominant frequencies for each signal.
        """
        freqs, psd = [], []
        rel_power = {name: [] for name in self.band_names}
        dominant_freq = []

        # ensure iterable
        if isinstance(data, np.ndarray) and data.ndim == 1:
            data = [data]

        for sig in data:
            f, pxx, f_sel, p_sel = self._compute_psd(sig, analysis_band, win_seconds, window, detrend, overlap)
            total = simps(p_sel, f_sel) if f_sel.size > 1 else 0.0

            for i, name in enumerate(self.band_names):
                lo, hi = self.band_edges[i], self.band_edges[i + 1]
                m = self._band_mask(f_sel, lo, hi, right_closed=(i == len(self.band_names) - 1))
                bp = simps(p_sel[m], f_sel[m]) if np.any(m) else 0.0
                rel_power[name].append(bp / total if total > 0 else 0.0)

            dom = float(f_sel[np.argmax(p_sel)]) if p_sel.size else float('nan')
            dominant_freq.append(dom)
            freqs.append(f)
            psd.append(pxx)

        return freqs, psd, rel_power, dominant_freq

    def test_normality(self, real_power, synthetic_power):
        """
        Test the normality of the differences between real and synthetic relative power.
        """
        results = {}
        for band in real_power.keys():
            differences = np.array(real_power[band]) - np.array(synthetic_power[band])
            if len(differences) < 3:
                results[band] = {"statistic": None, "p-value": None}
            else:
                stat, p_value = shapiro(differences)
                results[band] = {"statistic": stat, "p-value": p_value}
        return results

    def perform_statistical_tests(self, real_power, synthetic_power, normality_results):
        """
        Perform statistical tests for each frequency band based on normality results.
        """
        test_results = {}

        for band in real_power.keys():
            real_vals = np.array(real_power[band])
            synthetic_vals = np.array(synthetic_power[band])

            if len(real_vals) < 2 or len(synthetic_vals) < 2:
                continue

            # If normal => paired t-test, else => Wilcoxon
            if normality_results[band]["p-value"] is not None and \
                    normality_results[band]["p-value"] > 0.05:
                stat, p_value = ttest_rel(real_vals, synthetic_vals)
                test_type = "Paired t-test"
            else:
                stat, p_value = wilcoxon(real_vals, synthetic_vals)
                test_type = "Wilcoxon signed-rank test"

            # Compute effect size
            d = self.compute_cohens_d(real_vals, synthetic_vals)

            test_results[band] = {
                "test": test_type,
                "statistic": f"{stat:.3f}",
                "p-value": self._format_p(p_value),
                "cohens_d": d,
            }

        return test_results

    def compare_relative_power(self, real_data, synthetic_data,
                                   analysis_band=None, win_seconds=None, window=None, detrend=None, overlap=None):
        """
        Computes relative power for real and synthetic data, tests normality,
        and performs statistical tests (only if data are 2D).
        """
        # 1D => single sample => no stats
        if (isinstance(real_data, np.ndarray) and real_data.ndim == 1) or \
                (isinstance(synthetic_data, np.ndarray) and synthetic_data.ndim == 1):
            _, _, real_power, _ = self.compute_relative_power(real_data, analysis_band, win_seconds, window,
                                                                detrend, overlap)
            _, _, synth_power, _ = self.compute_relative_power(synthetic_data, analysis_band, win_seconds, window,
                                                                detrend, overlap)
            print("Relative Bands Power (Sample):")
            for band in self.band_names:
                r = np.mean(real_power[band]) * 100
                s = np.mean(synth_power[band]) * 100
                print(f"  {band}: Real: {r:.2f}%, Synthetic: {s:.2f}%, Diff: {abs(r - s):.2f}%")
                print("⚠ Insufficient data to perform Statistical Analysis!\n")
            return {}

        # 2D => dataset
        _, _, real_power, _ = self.compute_relative_power(real_data, analysis_band, win_seconds, window, detrend,
                                                            overlap)
        _, _, synth_power, _ = self.compute_relative_power(synthetic_data, analysis_band, win_seconds, window,
                                                            detrend, overlap)

        print("Relative Bands Power (Dataset):")
        for band in self.band_names:
            r = np.mean(real_power[band]) * 100
            s = np.mean(synth_power[band]) * 100
            print(f"  {band}: Real: {r:.2f}%, Synthetic: {s:.2f}%, Diff: {abs(r - s):.2f}%")

        normality = self.test_normality(real_power, synth_power)
        statistical_tests = self.perform_statistical_tests(real_power, synth_power, normality)
        print("Statistical Test Results (Dataset):")
        for band, res in statistical_tests.items():
            print(
                f"  {band}: Test={res['test']}, Stat={res['statistic']}, p-value={res['p-value']}, Cohen's d={res['cohens_d']:.3f}")
        print()
        return statistical_tests


    def spectral_coherence(self, real_signals, synthetic_signals, *, mode: str = "all_vs_all", rr_zip_strategy: str = "consecutive",
            ss_zip_strategy: str = "consecutive", per_band: bool = True, analysis_band=None, win_seconds = None, window = None,
            detrend = None, overlap = None):
        """
        Compute spectral coherence between real and synthetic signals.

        RS, RR and SS comparisons all respect `mode`:
          - mode="all_vs_all": compare all unique pairs (i<j) within-set (RR/SS), and every real vs every synthetic (RS). Use when you don’t have direct pairing, and want a global dataset-level similarity.
          - mode="zip":        pair by index without repetition: Use when signals are already matched (e.g., synthetic was generated from the corresponding real signal).
                               * RS:        (real[i], synth[i]) up to min(nR, nS)
                               * RR/SS:     within the same set using `*_zip_strategy`:
                                   - "consecutive": (0,1), (2,3), (4,5), ...
                                   - "halves":      split set in half: (0,half), (1,half+1), ...
                               If a set has an odd number of items, the last one is dropped.

        Parameters
        ----------
        real_signals : array
            1D (T,) or 2D (N x T) real signals.
        synthetic_signals : array
            1D (T,) or 2D (N x T) synthetic signals.
        mode : {"all_vs_all","zip"}, default "all_vs_all"
            Pairing strategy applied to RS, RR and SS.
        rr_zip_strategy, ss_zip_strategy : {"consecutive","halves"}
            Strategy for within-set zip pairing.
        per_band : bool, default True
            Whether to compute and summarize per-band mean coherence (Delta to Gamma).
        analysis_band : tuple[float,float] | None
            Optional custom global band (lo, hi) in Hz; if None, uses the class default.
        win_seconds, window, detrend, overlap : overrides or None
            Per-call overrides merged with instance defaults via `_resolve_params`.

        Returns
        -------
        dict
            Means, SDs, and per-pair lists for RS; summaries for RR and SS.

        Notes
        -----
        - Signals in each pair are truncated to the common length before coherence.
        - Windowing and overlap come from `_resolve_params`, consistent with PSD methods.
        """

        # Normalize to 2D
        R = np.asarray(real_signals, dtype=float)
        S = np.asarray(synthetic_signals, dtype=float)
        if R.ndim == 1: R = R[np.newaxis, :]
        if S.ndim == 1: S = S[np.newaxis, :]

        analysis_type = "sample" if R.shape[0] == 1 and S.shape[0] == 1 else "dataset"
        nR, nS = R.shape[0], S.shape[0]
        if nR == 0 or nS == 0:
            raise ValueError("Both real_signals and synthetic_signals must contain at least one signal.")

        # Resolve per-call vs instance defaults
        fmin, fmax, win_s, wnd, dtr, ovl = self._resolve_params(
            analysis_band=analysis_band, win_seconds=win_seconds,
            window=window, detrend=detrend, overlap=overlap
        )

        # Build band dict from class edges/names (clipped to global band)
        bands_used = {}
        if per_band:
            for i, name in enumerate(self.band_names):
                lo, hi = self.band_edges[i], self.band_edges[i + 1]
                lo_c = max(lo, fmin)
                hi_c = min(hi, fmax)
                if hi_c > lo_c:
                    bands_used[name] = (lo_c, hi_c)

        # Helpers
        def _mean_sd(vals):
            vals = np.asarray(vals, dtype=float)
            if vals.size == 0:
                return np.nan, np.nan
            m = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan
            return m, sd

        def _pairs_within_set(n, zip_strategy):
            if zip_strategy not in {"consecutive", "halves"}:
                raise ValueError("zip_strategy must be 'consecutive' or 'halves'")
            pairs = []
            if zip_strategy == "consecutive":
                for k in range(0, n - 1, 2):
                    pairs.append((k, k + 1))
            else:  # halves
                half = n // 2
                for k in range(half):
                    pairs.append((k, k + half))
            return pairs

        def _coh_pair(x, y):
            # truncate to common length and zero-mean
            L = min(len(x), len(y))
            x = x[:L] - np.mean(x[:L])
            y = y[:L] - np.mean(y[:L])

            # choose nperseg to ensure at least 2 segments (ideally more)
            base_nperseg = max(8, int(self.fs * float(win_s)))
            # start from min(requested, L//2); then try to shrink to reach ≥2 segments
            local_nperseg = min(base_nperseg, max(8, L // 2))

            def _num_segments(L_, nseg_, ovlp_):
                step = max(1, nseg_ - ovlp_)
                return 1 + max(0, (L_ - nseg_) // step)

            ovlp_frac = float(ovl)
            local_noverlap = int(local_nperseg * ovlp_frac)
            if local_noverlap >= local_nperseg:
                local_noverlap = max(0, local_nperseg - 1)

            # try to get ≥3 segments when possible by shrinking nperseg progressively
            for shrink in (1.0, 0.75, 0.5, 0.33):
                nseg_try = max(8, int(local_nperseg * shrink))
                ovlp_try = min(int(nseg_try * ovlp_frac), nseg_try - 1)
                if _num_segments(L, nseg_try, ovlp_try) >= 2:
                    local_nperseg, local_noverlap = nseg_try, ovlp_try
                    break

            f, Cxy = coherence(
                x,
                y,
                fs=self.fs,
                nperseg=local_nperseg,
                noverlap=local_noverlap,
                window=wnd,
                detrend=dtr,
            )

            # Global (bandwidth-weighted mean over [fmin, fmax])
            gmask = (f >= fmin) & (f <= fmax)
            if np.any(gmask):
                width_g = (fmax - fmin)
                gval = float(simps(Cxy[gmask], f[gmask]) / width_g) if width_g > 0 else float(np.nan)
            else:
                gval = np.nan

            # Per-band (bandwidth-weighted)
            band_vals = {}
            if per_band and bands_used:
                for bname, (lo, hi) in bands_used.items():
                    bmask = (f >= lo) & (f < hi if hi < fmax else f <= hi)
                    if np.any(bmask):
                        width = (hi - lo)
                        band_vals[bname] = float(simps(Cxy[bmask], f[bmask]) / width) if width > 0 else np.nan
                    else:
                        band_vals[bname] = np.nan

            return gval, band_vals, np.nan

        def _accumulate_pairs(XA, XB, pairs_iter):
            globals_list = []
            bands_list = []
            for i, j in pairs_iter:
                g, bdict, _ = _coh_pair(XA[i], XB[j])
                globals_list.append(g)
                if per_band and bands_used:
                    bands_list.append(bdict)
            return globals_list, bands_list

        # RS pairs
        if mode == "all_vs_all":
            rs_pairs = ((i, j) for i in range(nR) for j in range(nS))
        elif mode == "zip":
            N = min(nR, nS)
            rs_pairs = ((i, i) for i in range(N))
        else:
            raise ValueError("mode must be 'all_vs_all' or 'zip'")
        rs_globals, rs_bands_dicts = _accumulate_pairs(R, S, rs_pairs)

        # RR pairs
        if mode == "all_vs_all":
            rr_pairs = ((i, j) for i in range(nR) for j in range(i + 1, nR))
        else:
            rr_pairs = _pairs_within_set(nR, rr_zip_strategy)
        rr_globals, rr_bands_dicts = _accumulate_pairs(R, R, rr_pairs)

        # SS pairs
        if mode == "all_vs_all":
            ss_pairs = ((i, j) for i in range(nS) for j in range(i + 1, nS))
        else:
            ss_pairs = _pairs_within_set(nS, ss_zip_strategy)
        ss_globals, ss_bands_dicts = _accumulate_pairs(S, S, ss_pairs)

        # Summaries
        rs_g_m, rs_g_sd = _mean_sd(rs_globals)
        rr_g_m, rr_g_sd = _mean_sd(rr_globals)
        ss_g_m, ss_g_sd = _mean_sd(ss_globals)

        def _summarize_bands(dicts_list):
            if not (per_band and bands_used) or len(dicts_list) == 0:
                return {}
            out = {}
            for k in bands_used.keys():
                vals = [d.get(k, np.nan) for d in dicts_list]
                v = np.asarray(vals, dtype=float)
                out[k] = float(np.nanmean(v)) if np.any(np.isfinite(v)) else np.nan
            return out

        rr_bands_mean = _summarize_bands(rr_bands_dicts)
        ss_bands_mean = _summarize_bands(ss_bands_dicts)
        rs_bands_mean = _summarize_bands(rs_bands_dicts)

        # Print (mirrors your concise style)
        def _fmt(x):
            return "nan" if not np.isfinite(x) else f"{x:.3f}"

        print(f"Mode: {mode} | RS spectral coherence={_fmt(rs_g_m)} (± {_fmt(rs_g_sd)}) ({analysis_type})")
        if mode == "zip":
            print(f"RR zip strategy: {rr_zip_strategy} | pairs={len(rr_globals)}")
            print(f"SS zip strategy: {ss_zip_strategy} | pairs={len(ss_globals)}")

        print("RR  | Spectral coherence =", f"{_fmt(rr_g_m)} ± {_fmt(rr_g_sd)}")
        if per_band and bands_used:
            print("RR  | Bands mean:", {k: _fmt(v) for k, v in rr_bands_mean.items()})

        print("SS  | Spectral coherence =", f"{_fmt(ss_g_m)} ± {_fmt(ss_g_sd)}")
        if per_band and bands_used:
            print("SS  | Bands mean:", {k: _fmt(v) for k, v in ss_bands_mean.items()})

        print("RS  | Spectral coherence =", f"{_fmt(rs_g_m)} ± {_fmt(rs_g_sd)}")
        if per_band and bands_used:
            print("RS  | Bands mean:", {k: _fmt(v) for k, v in rs_bands_mean.items()})

        # Return dict in your standard shape
        result = {
            "Analysis Type": analysis_type,
            "RS Mode": mode,
            "RR Zip Strategy": rr_zip_strategy if mode == "zip" else "all_vs_all",
            "SS Zip Strategy": ss_zip_strategy if mode == "zip" else "all_vs_all",
            "Global Band": (fmin, fmax),
            "Bands Used": bands_used if per_band else {},
            "Per-pair Coherence (RS)": rs_globals,
            "RR Summary": {
                "Pairs": len(rr_globals),
                "Global Mean": rr_g_m, "Global SD": rr_g_sd,
                "Bands Mean": rr_bands_mean if (per_band and bands_used) else {},
                "Custom Mean": np.nan,  # placeholder for symmetry with other methods
            },
            "SS Summary": {
                "Pairs": len(ss_globals),
                "Global Mean": ss_g_m, "Global SD": ss_g_sd,
                "Bands Mean": ss_bands_mean if (per_band and bands_used) else {},
                "Custom Mean": np.nan,
            },
            "RS Summary": {
                "Pairs": len(rs_globals),
                "Global Mean": rs_g_m, "Global SD": rs_g_sd,
                "Bands Mean": rs_bands_mean if (per_band and bands_used) else {},
                "Custom Mean": np.nan,
            },
        }
        return result

    def spectral_wasserstein_distance(self, real_data, synthetic_data, fmin=0.5, fmax=500, mode="pairmean", per_band=True):
        """
        Spectral Wasserstein distance (Earth Mover's Distance) computed ALONG the
        frequency axis between normalized PSDs. Phase-agnostic. Units: Hz.

        Parameters
        ----------
        real_data, synthetic_data : np.ndarray or list[np.ndarray]
            Arrays of signals. If 1D arrays are provided, they are treated as single signals.
        fmin, fmax : float
            Frequency range (Hz) over which to compute the distance.
        mode : {"pairmean", "meanpsd"}
            - "pairmean": average WD over zipped real[i] vs synth[i] pairs
            - "meanpsd" : WD between the mean normalized PSDs of each set
        per_band : bool
            If True, also computes WD per canonical EEG band and returns (band_dict, overall).

        Returns
        -------
        float or (dict, float)
            Overall WD in Hz, or (per-band dict, overall) if per_band=True.
        """

        # Wrap single 1D signals
        if isinstance(real_data, np.ndarray) and real_data.ndim == 1:
            real_data = [real_data]
        if isinstance(synthetic_data, np.ndarray) and synthetic_data.ndim == 1:
            synthetic_data = [synthetic_data]

        nyq = self.fs / 2.0
        fmax = min(fmax, nyq)

        # Helpers
        def _psd_norm(sig, lo, hi):
            """Normalized PSD density over [lo, hi]; integrates to 1."""
            nperseg = min(len(sig), 4 * self.fs)
            f, p = welch(sig, fs=self.fs, nperseg=nperseg,
                         window='hann', detrend='constant')
            m = (f >= lo) & (f <= hi)
            f, p = f[m], p[m]
            area = simps(p, f)
            if area <= 0 or not np.isfinite(area):
                # fallback uniform density over the support
                p = np.ones_like(p)
                area = simps(p, f)
            p = p / area
            return f, p

        def _wd_pair(r_sig, s_sig, lo, hi):
            fr, pr = _psd_norm(r_sig, lo, hi)
            fs_, ps = _psd_norm(s_sig, lo, hi)
            # align on a common (finer) grid
            f_common = fr if fr.size >= fs_.size else fs_
            pr_i = np.interp(f_common, fr, pr)
            ps_i = np.interp(f_common, fs_, ps)
            # weights (densities) must sum to 1 for scipy's wasserstein_distance
            pr_i = pr_i / pr_i.sum()
            ps_i = ps_i / ps_i.sum()
            return wasserstein_distance(f_common, f_common, pr_i, ps_i)  # in Hz

        def _wd_meanpsd(data, lo, hi):
            grids, dens = [], []
            for sig in data:
                f, p = _psd_norm(sig, lo, hi)
                grids.append(f);
                dens.append(p)
            # interpolate onto the densest grid
            base = max(grids, key=len)
            dens_i = [np.interp(base, fi, di) for fi, di in zip(grids, dens)]
            mean_d = np.mean(dens_i, axis=0)
            mean_d = mean_d / mean_d.sum()
            return base, mean_d

        # Overall over [fmin, fmax]
        if mode == "pairmean":
            vals = [_wd_pair(r, s, fmin, fmax) for r, s in zip(real_data, synthetic_data)]
            spectral_wasserstein_distance = float(np.mean(vals)) if len(vals) else float('nan')
        elif mode == "meanpsd":
            fr, pr = _wd_meanpsd(real_data, fmin, fmax)
            fs_, ps = _wd_meanpsd(synthetic_data, fmin, fmax)
            f_common = fr if fr.size >= fs_.size else fs_
            pr_i = np.interp(f_common, fr, pr);
            pr_i = pr_i / pr_i.sum()
            ps_i = np.interp(f_common, fs_, ps);
            ps_i = ps_i / ps_i.sum()
            spectral_wasserstein_distance = float(wasserstein_distance(f_common, f_common, pr_i, ps_i))
        else:
            raise ValueError("mode must be 'pairmean' or 'meanpsd'")

        if per_band:
            # build per-band dict from class edges/names, clipped to Nyquist
            band_dict = {}
            for i, name in enumerate(self.band_names):
                lo, hi = self.band_edges[i], self.band_edges[i + 1]
                lo_c, hi_c = max(lo, fmin), min(hi, nyq)
                if hi_c > lo_c:
                    band_dict[name] = (lo_c, hi_c)

            spectral_wasserstein_distance_bands = {}
            for name, (lo, hi) in band_dict.items():
                if mode == "pairmean":
                    vals = [_wd_pair(r, s, lo, hi) for r, s in zip(real_data, synthetic_data)]
                    spectral_wasserstein_distance_bands[name] = float(np.mean(vals)) if len(vals) else float('nan')
                else:
                    fr, pr = _wd_meanpsd(real_data, lo, hi)
                    fs_, ps = _wd_meanpsd(synthetic_data, lo, hi)
                    f_common = fr if fr.size >= fs_.size else fs_
                    pr_i = np.interp(f_common, fr, pr);
                    pr_i /= pr_i.sum()
                    ps_i = np.interp(f_common, fs_, ps);
                    ps_i /= ps_i.sum()
                    spectral_wasserstein_distance_bands[name] = float(
                        wasserstein_distance(f_common, f_common, pr_i, ps_i)
                    )

            print(f"Spectral Wasserstein distance (Hz) [{fmin}-{fmax}]: {spectral_wasserstein_distance:.4f}")
            for k, v in spectral_wasserstein_distance_bands.items():
                print(f"  {k}: {v:.4f} Hz")
            return spectral_wasserstein_distance_bands, spectral_wasserstein_distance

        # Print & return overall only
        print(f"Spectral Wasserstein distance (Hz) [{fmin}-{fmax}]: {spectral_wasserstein_distance:.4f}")
        return spectral_wasserstein_distance

    def plot_psd(self, real_data, synthetic_data, scale="linear",smooth=False, window_length=11, polyorder=2,xlim=None, ylim=None, analysis_band=None):
        """
        Plots the power spectral density (PSD) for real and synthetic data on a specified scale.

        Parameters
        ----------
        real_data : list or np.ndarray
            Real signals or a single signal.
        synthetic_data : list or np.ndarray
            Synthetic signals or a single signal.
        scale : str, optional
            Plot scale, "linear" or "log". Defaults to "linear".
        smooth : bool, optional
            Whether to smooth the PSD using Savitzky-Golay filter. Defaults to True.
        window_length : int, optional
            Window length for smoothing (must be odd). Defaults to 11.
        polyorder : int, optional
            Polynomial order for smoothing. Defaults to 2.
        """
        # Local PSD default band
        # PSD-specific default band: (0, 100) unless user passes one
        analysis_band_plot = (0.0, 100.0) if analysis_band is None else tuple(analysis_band)

        if scale not in ("linear", "log"):
            raise ValueError("scale must be 'linear' or 'log'.")

        # Wrap single signals
        if isinstance(real_data, np.ndarray) and real_data.ndim == 1:
            real_data = [real_data]
        if isinstance(synthetic_data, np.ndarray) and synthetic_data.ndim == 1:
            synthetic_data = [synthetic_data]

        analysis_type = "sample" if len(real_data) == 1 and len(synthetic_data) == 1 else "dataset"

        # Use instance defaults (or allow optional overrides if you add params)
        fmin, fmax, win_s, wnd, dtr, ovl = self._resolve_params(analysis_band=analysis_band_plot)

        # Helper: compute PSD and interpolate to a reference grid
        def _psd_stack(data):
            freqs_list, psd_list = [], []
            for sig in data:
                _, _, f_sel, p_sel = self._compute_psd(sig, (fmin, fmax), win_s, wnd, dtr, ovl)
                freqs_list.append(f_sel)
                psd_list.append(p_sel)
            # Choose densest grid as reference
            f_ref = max(freqs_list, key=len)
            psd_interp = [np.interp(f_ref, fi, pi) if fi is not f_ref else pi
                          for fi, pi in zip(freqs_list, psd_list)]
            psd_stack = np.vstack(psd_interp)  # shape: (N, F)
            return f_ref, np.vstack(psd_interp)

        fr, psd_r_stack = _psd_stack(real_data)
        fs, psd_s_stack = _psd_stack(synthetic_data)

        # If reference grids differ, align synthetic to real (or vice-versa)
        if not (fr.shape == fs.shape and np.allclose(fr, fs)):
            psd_s_stack = np.vstack([np.interp(fr, fs, row) for row in psd_s_stack])
            f_plot = fr
        else:
            f_plot = fr

        # Mean in linear domain
        real_psd = psd_r_stack.mean(axis=0)
        synthetic_psd = psd_s_stack.mean(axis=0)

        # Optional smoothing
        if smooth:
            # ensure valid params
            wl = min(window_length, len(real_psd) - (1 - len(real_psd) % 2))  # keep odd
            wl = wl if wl % 2 == 1 else max(3, wl - 1)
            wl = max(wl, polyorder + 2 + (polyorder % 2))  # basic safety
            real_psd = savgol_filter(real_psd, window_length=wl, polyorder=polyorder)
            synthetic_psd = savgol_filter(synthetic_psd, window_length=wl, polyorder=polyorder)

        # Clamp for log plotting
        eps = 1e-12
        real_psd_clamped = np.maximum(real_psd, eps)
        synthetic_psd_clamped = np.maximum(synthetic_psd, eps)

        # Common y-limits for linear scale
        if scale == "linear":
            if ylim is not None:
                y_min, y_max = ylim
            else:
                y_min = 0.0
                y_max = 1.1 * max(real_psd.max(), synthetic_psd.max())
        else:
            y_min, y_max = None, None

        plt.figure(figsize=(12, 6))

        # Real
        plt.subplot(1, 2, 1)
        if scale == "linear":
            plt.plot(f_plot, real_psd, color="blue", label="real data")
            if y_min is not None and y_max is not None:
                plt.ylim(y_min, y_max)
            plt.ylabel("PSD ($\\mu V^2$/Hz)", fontsize=15)
        else:
            plt.semilogy(f_plot, real_psd_clamped, color="lightgreen", label="real data")
            if ylim is not None:
                plt.ylim(ylim)
            plt.ylabel("PSD ($\\mu V^2$/Hz, log axis)", fontsize=15)
        plt.xlabel("Frequency (Hz)", fontsize=15)
        if xlim is not None:
            plt.xlim(xlim)
        else:
            plt.xlim(fmin, fmax)
        plt.title(f"Real data PSD ({analysis_type}) - {scale} scale", fontsize=16)
        plt.grid(True, which="both")
        plt.legend()

        # Synthetic
        plt.subplot(1, 2, 2)
        if scale == "linear":
            plt.plot(f_plot, synthetic_psd, color="grey", label="synthetic data")
            if y_min is not None and y_max is not None:
                plt.ylim(y_min, y_max)
            plt.ylabel("PSD ($\\mu V^2$/Hz)", fontsize=15)
        else:
            plt.semilogy(f_plot, synthetic_psd_clamped, color="grey", label="synthetic data")
            if ylim is not None:
                plt.ylim(ylim)
            plt.ylabel("PSD ($\\mu V^2$/Hz, log axis)", fontsize=15)
        plt.xlabel("Frequency (Hz)", fontsize=15)
        if xlim is not None:
            plt.xlim(xlim)
        else:
            plt.xlim(fmin, fmax)
        plt.title(f"Synthetic data PSD ({analysis_type}) - {scale} scale", fontsize=16)
        plt.grid(True, which="both")
        plt.legend()

        plt.tight_layout()
        plt.show()
