import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, next_fast_len
from skimage.metrics import structural_similarity as ssim, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import mannwhitneyu, wilcoxon

try:
    from tqdm import tqdm
except ImportError:
    # Minimal no-op fallback so the script still runs if tqdm isn't installed
    # (run `pip install tqdm` to get real progress bars + ETA).
    def tqdm(iterable=None, total=None, desc=None, **kwargs):
        if iterable is None:
            return iterable
        n = total if total is not None else (len(iterable) if hasattr(iterable, "__len__") else None)
        label = f"{desc}: " if desc else ""
        for i, item in enumerate(iterable, 1):
            if n:
                print(f"\r{label}{i}/{n}", end="", flush=True)
            yield item
        print()


class TimeFrequencyFidelity:
    """
    A class to measure time-frequency similarity between real and synthetic signals, by computing and comparing
    scalogram representations of signals, as well as bursts statistics.

    The scalogram is computed using a **batched, FFT-based Continuous Wavelet Transform (CWT)** with
    **analytic Morlet wavelets** (Torrence & Compo, 1998 formulation).

    Parameters:
    ----------
    fs : int
    Sampling frequency of the signals. Chosen by users.
    frequencies : np.array, optional
    Array of frequencies for wavelet transformation. Default is **30 frequencies from 1 Hz to 30 Hz**.

    Example Usage:
    --------------
    real_data = np.random.randn(5, 2048)  # 5 real signals of length 2048
    synthetic_data = np.random.randn(5, 2048)  # 5 synthetic signals of length 2048

    time_frequency_analysis = TimeFrequencyFidelity(fs=2048)

    # Compute similarity metrics
    similarity_results = time_frequency_analysis.compute_scalogram_similarity_metrics(real_data, synthetic_data)

    # Plot a scalogram for visual comparison
    time_frequency_analysis.plot_scalograms(real_data, synthetic_data, signal_index_real=0, signal_index_synth=4)

    # Plot mean scalograms for both datasets
    time_frequency_analysis.plot_mean_scalograms(real_data, synthetic_data, save=None)

    # Compute burst statistics within a frequency band (beta in this case)
    burst_results = time_frequency_analysis.compute_burst_statistics(real_data, synthetic_data, band=(13, 30),
    threshold="percentile", p=70.0, min_duration_ms=50.0, merge_gap_ms=50.0, smooth_ms=20.0)

    References:
    ----------
    [1] https://scikit-image.org/docs/stable/api/skimage.metrics.html
    [2] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
    [3] https://arxiv.org/html/2405.08431v3?utm_source=chatgpt.com
    [4] Torrence, C. and Compo, G. P. (1998), "A Practical Guide to Wavelet Analysis",
        Bull. Amer. Meteor. Soc., 79, 61-78. (analytic Morlet wavelet, FFT-based CWT)
    """
    def __init__(self, *, fs: int, freq_min=0.5, freq_max=100, num_freqs=100) -> None:
        """
        Initializes the TimeFrequencyFidelity class for computing scalogram representations
        of signals and evaluating their similarity using various metrics.

        Parameters:
        ----------
        fs : int, optional
            Sampling frequency of the signals in Hz
        freq_min : float
            Minimum frequency in Hz for scalogram.
        freq_max : float
            Maximum frequency in Hz for scalogram.
        num_freqs : int
            Number of frequency bins for wavelet transform. Increase it for finer resolution at higher frequencies.
        """
        if fs <= 0:
            raise ValueError("Sampling frequency `fs` must be positive.")
        self.fs = fs
        self._set_frequencies(freq_min, freq_max, num_freqs)

    def _set_frequencies(self, freq_min: float, freq_max: float, num_freqs: int):
        if freq_min <= 0 or freq_max <= freq_min:
            raise ValueError("Invalid frequency range.")
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.num_freqs = int(num_freqs)
        self.frequencies = np.linspace(freq_min, freq_max, num_freqs)

    def cliffs_delta(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        # delta = P(x>y) - P(x<y)
        gt = 0
        lt = 0
        for xi in x:
            gt += np.sum(xi > y)
            lt += np.sum(xi < y)
        return (gt - lt) / (len(x) * len(y))

    def _test_rr_vs_rs(self, rr, rs, *, mode: str):
        rr = np.asarray(rr, dtype=float)
        rs = np.asarray(rs, dtype=float)
        rr = rr[np.isfinite(rr)]
        rs = rs[np.isfinite(rs)]

        if len(rr) == 0 or len(rs) == 0:
            return {"p": np.nan, "effect": np.nan, "test": "NA"}

        if mode == "zip":
            m = min(len(rr), len(rs))
            stat = wilcoxon(rr[:m], rs[:m], alternative="two-sided")
            return {"p": float(stat.pvalue), "effect": np.nan, "test": "Wilcoxon"}
        else:
            stat = mannwhitneyu(rr, rs, alternative="two-sided")
            cd = self.cliffs_delta(rr, rs)
            return {"p": float(stat.pvalue), "effect": float(cd), "test": "MWU + CliffΔ"}

    def set_sampling_rate(self, fs: int) -> None:
        """Change the sampling frequency after the object has been created."""
        if fs <= 0:
            raise ValueError("Sampling frequency `fs` must be positive.")
        self.fs = fs

    # Freq grid + intensity

    def _build_frequency_grid(self, freq_scale: str, num_freqs: int | None = None) -> np.ndarray:
        """
        Build a frequency grid using current [freq_min, freq_max] and the requested scale.

        freq_scale: "linear" (default) or "log"
        """
        if freq_scale not in {"linear", "log"}:
            raise ValueError("freq_scale must be 'linear' or 'log'")
        N = int(num_freqs or getattr(self, "num_freqs", 100))
        if freq_scale == "linear":
            return np.linspace(self.freq_min, self.freq_max, N)
        return np.logspace(np.log10(self.freq_min), np.log10(self.freq_max), N)

    def _intensity_from_amp_single(self, amp: np.ndarray, *, intensity: str, db_ref, ref_value=None) -> np.ndarray:
        """
        Per-image conversion:
          - amplitude: |CWT|
          - power: |CWT|^2
          - db: 10*log10( power / ref )
        """
        eps = 1e-12
        if intensity == "amplitude":
            return amp
        power = amp**2
        if intensity == "power":
            return power
        if isinstance(db_ref, (int, float)):
            ref = float(db_ref) + eps
        elif db_ref in {"global_max", "per_image_max"}:
            ref = (ref_value if ref_value is not None else np.max(power) + eps)
        else:
            raise ValueError("Invalid db_ref.")
        return 10.0 * np.log10((power + eps) / ref)

    def _intensity_from_mean_amp(self, mean_amp: np.ndarray, *, intensity: str, db_ref, ref_value=None) -> np.ndarray:
        """
        Mean-then-transform conversion (used by plot_mean_scalograms).
        """
        eps = 1e-12
        if intensity == "amplitude":
            return mean_amp
        power_from_mean_amp = mean_amp**2
        if intensity == "power":
            return power_from_mean_amp
        if isinstance(db_ref, (int, float)):
            ref = float(db_ref) + eps
        elif db_ref in {"global_max", "per_image_max"}:
            ref = (ref_value if ref_value is not None else np.max(power_from_mean_amp) + eps)
        else:
            raise ValueError("Invalid db_ref.")
        return 10.0 * np.log10((power_from_mean_amp + eps) / ref)

    # ------------------------------------------------------------------------------------

    def _convert_to_rgb(self, image, *, colormap='terrain', vmin=None, vmax=None):
        """
        Convert a single-channel grayscale image to an RGB image using a colormap.
        """
        if vmin is None:
            vmin = image.min()
        if vmax is None:
            vmax = image.max()

        denom = (vmax - vmin) or 1.0
        norm = np.clip((image - vmin) / denom, 0, 1)

        # Cache the colormap object per name instead of calling plt.get_cmap() fresh every
        # call -- cheap win for any code path (e.g. plot_scalograms) that still calls this
        # repeatedly; does not change output values.
        if not hasattr(self, "_cmap_cache"):
            self._cmap_cache = {}
        if colormap not in self._cmap_cache:
            self._cmap_cache[colormap] = plt.get_cmap(colormap)
        cmap = self._cmap_cache[colormap]
        return (cmap(norm)[..., :3] * 255).astype(np.uint8)


    # Update: batched, FFT-based analytic Morlet CWT

    def _cwt_fft_batch(self, signals: np.ndarray, w: float, frequencies: np.ndarray):
        """
        Vectorized Continuous Wavelet Transform (analytic Morlet) for a *batch* of signals,
        computed entirely via FFT instead of scipy.signal.cwt's per-scale time-domain
        convolution loop.

        This replaces `num_freqs` sequential O(N*M) `np.convolve` calls (scipy.signal.cwt)
        with: one batched FFT of the signals, one broadcasted multiply against a
        precomputable wavelet-response matrix, and one batched inverse FFT. It scales to
        an arbitrary number of signals in a single call (the Python-level `for` loop over
        signals disappears), which is where most of the wall-clock time was going.

        Uses the standard analytic-Morlet Fourier-domain formulation (Torrence & Compo,
        1998, eq. 6):
            psi_hat(s*omega) = pi^(-1/4) * H(omega) * exp( -0.5 * (s*omega - w)^2 )
        with H the Heaviside step (wavelet is one-sided / analytic), s the wavelet scale,
        and w the nondimensional Morlet parameter (same role as scipy.signal.morlet2's `w`).

        The scale <-> frequency relation is kept identical to the original implementation
        (`scales = w / (2*pi*F*dt)`) so results are directly comparable / interchangeable
        with the previous `scipy.signal.cwt(..., morlet2)`-based scalograms.

        Edge handling: multiplying spectra and taking one IFFT performs *circular*
        convolution, whereas the original `scipy.signal.cwt` (`np.convolve(..., mode="same")`)
        performs *linear* convolution. Without correction, circular convolution lets energy
        from one end of the signal leak ("wrap around") into the other end -- worst for low
        frequencies, where the wavelet's temporal support is widest relative to the signal
        length. To avoid that, the signal is zero-padded on both sides by ~4 standard
        deviations of the widest (lowest-frequency) wavelet envelope before transforming,
        and the padding is cropped back off afterwards. This reproduces linear-convolution
        behavior (matching the original implementation's edge behavior) while keeping the
        full speed benefit of the batched FFT. Standard cone-of-influence caveats still apply
        near the very edges of the signal (as they did in the original implementation) --
        that is an inherent property of CWT edge effects, not something either implementation
        removes.

        Parameters
        ----------
        signals : np.ndarray
            2D array (n_signals, T) of real-valued signals. All rows must share the same length.
        w : float
            Nondimensional Morlet wavelet parameter (number of cycles), default 6.0 elsewhere.
        frequencies : np.ndarray
            1D array of target frequencies (Hz), length n_freqs.

        Returns
        -------
        amp : np.ndarray
            3D array (n_signals, n_freqs, T) of scalogram amplitudes |CWT|.
        """
        signals = np.atleast_2d(np.asarray(signals, dtype=float))
        n_signals, T = signals.shape
        dt = 1.0 / self.fs
        F = np.asarray(frequencies, dtype=float)
        scales = w / (2 * np.pi * F * dt)  # (n_freqs,) -- identical mapping to the original code

        # Zero-pad both ends to suppress FFT circular-convolution wraparound. Sized to ~4
        # standard deviations of the widest (lowest-frequency) wavelet's Gaussian envelope,
        # which is the same effective support scipy's time-domain "same"-mode convolution
        # relies on implicitly.
        pad = int(np.ceil(4 * np.max(scales)))
        padded = np.pad(signals, ((0, 0), (pad, pad)), mode="constant")
        T_padded = T + 2 * pad

        # Pad further to a fast FFT length for speed, then crop back after the inverse transform.
        n_fft = next_fast_len(T_padded)

        # Angular frequency per FFT bin, in radians/sample (consistent with `scales` being in
        # the same "sample" units as the original morlet2-based scale formula).
        freq_idx = np.fft.fftfreq(n_fft)  # cycles/sample
        omega = 2 * np.pi * freq_idx      # radians/sample, shape (n_fft,)

        # Signal spectrum, batched over all signals at once.
        X = fft(padded, n=n_fft, axis=1)  # (n_signals, n_fft)

        # Analytic Morlet wavelet response for every scale at once: (n_freqs, n_fft)
        heaviside = (omega > 0).astype(float)
        s_omega = scales[:, None] * omega[None, :]
        psi_hat = (np.pi ** -0.25) * heaviside[None, :] * np.exp(-0.5 * (s_omega - w) ** 2)

        # W_n(s) = IFFT( X_k * conj(psi_hat(s*omega_k)) ) * sqrt(scale) normalization,
        # batched over (signals x scales) simultaneously.
        prod = X[:, None, :] * np.conj(psi_hat)[None, :, :]          # (n_signals, n_freqs, n_fft)
        W = ifft(prod, axis=2) * np.sqrt(scales)[None, :, None]      # (n_signals, n_freqs, n_fft)
        W = W[:, :, pad:pad + T]                                     # drop FFT + edge padding, back to T

        amp = np.abs(W)
        return amp

    def _compute_scalogram(self, signal, w=6.0, *, frequencies: np.ndarray | None = None, return_freqs: bool = False):
        """
        Compute the scalogram using a batched FFT-based analytic Morlet CWT (single signal,
        batch size 1 under the hood). Public signature/behavior is unchanged.

        Parameters:
        ----------
        signal : np.ndarray
            Input signal (1D array (sample)).
        w: float
            Number of cycles in Morlet wavelet (default is 6.0).
        frequencies : np.ndarray | None
            Optional frequency grid to use instead of self.frequencies.
        return_freqs : bool
            If True, returns (amplitude, frequencies_used). If False, returns amplitude only.

        Returns:
        -------
        np.ndarray or (np.ndarray, np.ndarray)
            Scalogram representation (amplitude) of the input signal; optionally with the frequency grid used.
        """
        signal = np.asarray(signal)
        if signal.ndim != 1:
            raise ValueError("Signal must be a 1D array.")
        F = frequencies if frequencies is not None else self.frequencies
        amp_batch = self._cwt_fft_batch(signal[None, :], w=w, frequencies=F)
        amp = amp_batch[0]
        if return_freqs:
            return amp, F
        return amp

    def plot_scalograms(self, real_data, synthetic_data, *, signal_index_real=0, signal_index_synth=0, save=None,
                        freq_scale: str | None = None, intensity: str | None = None, db_ref: str | float | None = None):
        """
        Compute and plot the scalogram for the provided real and synthetic signals.
        """
        freq_scale = freq_scale or "linear"
        intensity = intensity or "amplitude"
        db_ref = "global_max" if db_ref is None else db_ref

        real_data = np.asarray(real_data)
        synthetic_data = np.asarray(synthetic_data)

        if real_data.ndim > 1:
            real_signal = real_data[signal_index_real]
            synthetic_signal = synthetic_data[signal_index_synth]
        else:
            real_signal = real_data
            synthetic_signal = synthetic_data

        F = self._build_frequency_grid(freq_scale, self.num_freqs)
        real_amp, F_used = self._compute_scalogram(real_signal, frequencies=F, return_freqs=True)
        synth_amp, _ = self._compute_scalogram(synthetic_signal, frequencies=F_used, return_freqs=True)

        eps = 1e-12
        ref = None
        if intensity == "db" and db_ref == "global_max":
            ref = max(np.max(real_amp**2), np.max(synth_amp**2)) + eps

        real_int = self._intensity_from_amp_single(real_amp, intensity=intensity, db_ref=db_ref, ref_value=ref)
        synth_int = self._intensity_from_amp_single(synth_amp, intensity=intensity, db_ref=db_ref, ref_value=ref)

        vmin = min(float(np.nanmin(real_int)), float(np.nanmin(synth_int)))
        vmax = max(float(np.nanmax(real_int)), float(np.nanmax(synth_int)))
        total_duration = len(real_signal) / self.fs
        t = np.linspace(0, total_duration, real_int.shape[1])

        fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharey=True, constrained_layout=True)

        pm0 = axs[0].pcolormesh(t, F_used, real_int, shading="auto", cmap="terrain", vmin=vmin, vmax=vmax)
        axs[0].set_title(f'Real signal {signal_index_real} scalogram', fontsize=20, fontname='Arial')
        axs[0].set_xlabel('Time (s)', fontsize=15, fontname='Arial')
        axs[0].set_ylabel('Frequency (Hz)', fontsize=15, fontname='Arial')
        axs[0].tick_params(axis='both', which='major', labelsize=18)
        if freq_scale == "log":
            axs[0].set_yscale("log")

        axs[1].pcolormesh(t, F_used, synth_int, shading="auto", cmap="terrain", vmin=vmin, vmax=vmax)
        axs[1].set_title(f'Synthetic signal {signal_index_synth} scalogram', fontsize=20, fontname='Arial')
        axs[1].set_xlabel('Time (s)', fontsize=15)
        axs[1].tick_params(axis='both', which='major', labelsize=18)
        if freq_scale == "log":
            axs[1].set_yscale("log")

        cbar = fig.colorbar(pm0, ax=axs, location='right')
        # Explicit units on the intensity colorbar (reviewer-requested: "define the scale
        # explicitly in the caption/legend for accurate interpretation of the scalograms").
        # Amplitude/power here are wavelet-coefficient magnitudes, not a physical unit, hence
        # "a.u." (arbitrary units); dB is already a defined, self-explanatory unit.
        label = {"amplitude": "Amplitude (a.u.)", "power": "Power (a.u.)", "db": "Power (dB)"}[intensity]
        cbar.set_label(label, fontsize=14)
        cbar.ax.tick_params(labelsize=18)

        if save:
            fig.savefig(save, bbox_inches="tight", dpi=200)
        plt.show()

        return fig

    def compute_scalogram_similarity_metrics(self, real_data, synthetic_data, *, mode: str = "all_vs_all",
                pad: bool = True, rr_zip_strategy: str = "consecutive", ss_zip_strategy: str = "consecutive"):
        """
        Compute similarity metrics between real and synthetic scalograms.
        (unchanged from original; scalograms are now produced by the batched FFT CWT)
        """
        R = np.asarray(real_data, dtype=float)
        S = np.asarray(synthetic_data, dtype=float)
        if R.ndim == 1: R = R[np.newaxis, :]
        if S.ndim == 1: S = S[np.newaxis, :]

        analysis_type = "sample" if R.shape[0] == 1 else "dataset"
        nR, nS = R.shape[0], S.shape[0]
        if nR == 0 or nS == 0:
            raise ValueError("Both real_signals and synthetic_signals must contain at least one signal.")

        if pad:
            L = int(max(R.shape[1], S.shape[1]))
            if R.shape[1] != L:
                R = np.pad(R, ((0, 0), (0, L - R.shape[1])))
            if S.shape[1] != L:
                S = np.pad(S, ((0, 0), (0, L - S.shape[1])))
        else:
            if R.shape[1] != S.shape[1]:
                raise ValueError("When pad=False, real and synthetic signals must have identical lengths.")

        # Batched scalogram computation (single FFT-CWT call per dataset instead of per-signal loop)
        R_sc_batch = self._cwt_fft_batch(R, w=6.0, frequencies=self.frequencies)
        S_sc_batch = self._cwt_fft_batch(S, w=6.0, frequencies=self.frequencies)
        R_sc = [R_sc_batch[i] for i in range(nR)]
        S_sc = [S_sc_batch[j] for j in range(nS)]

        def _pair_metrics(A, B):
            vmin = float(min(A.min(), B.min()))
            vmax = float(max(A.max(), B.max()))
            if vmax <= vmin:
                vmax = vmin + 1e-6
            # SSIM computed directly on the normalized grayscale scalograms (no RGB/colormap
            # detour): skimage's SSIM accepts single-channel data natively via `data_range`,
            # so the colormap conversion was pure overhead -- extra memory (3x) and pointless
            # colormap-interpolation compute -- with no effect on the metric itself, since
            # SSIM operates on structural/luminance patterns, not display color.
            An = (A - vmin) / (vmax - vmin)
            Bn = (B - vmin) / (vmax - vmin)
            ssim_val = ssim(An, Bn, data_range=1.0)
            rmse = np.sqrt(mean_squared_error(A, B))
            nrmse = rmse / (vmax - vmin)
            cos = cosine_similarity(A.reshape(1, -1), B.reshape(1, -1))[0, 0]
            return float(ssim_val), float(nrmse), float(cos)

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
            else:
                half = n // 2
                for k in range(half):
                    pairs.append((k, k + half))
            return pairs

        rs_ssim, rs_nrmse, rs_cos = [], [], []
        if mode == "all_vs_all":
            # This loop is O(nR*nS) calls to skimage's structural_similarity, the single
            # most expensive step per pair -- inherent to the SSIM metric itself (a
            # windowed comparison over the scalogram), not something that can be sped up
            # further without changing what SSIM measures. Combined with the ~N^2/2 RR and
            # SS pairs below, large datasets can take a long time here; if your real/
            # synthetic signals are already matched 1:1, use mode="zip" instead (O(n)).
            n_pairs_est = nR * nS + nR * (nR - 1) // 2 + nS * (nS - 1) // 2
            print(f"[compute_scalogram_similarity_metrics] mode='all_vs_all': "
                  f"{n_pairs_est:,} total scalogram-pair comparisons to run "
                  f"(RS={nR*nS:,}, RR={nR*(nR-1)//2:,}, SS={nS*(nS-1)//2:,}).")
            if n_pairs_est > 2_000:
                print("[compute_scalogram_similarity_metrics] WARNING: this could take a "
                      "while. If your real/synthetic signals are already matched 1:1, "
                      "consider mode='zip' instead.")
            rs_pairs = [(i, j) for i in range(nR) for j in range(nS)]
            for i, j in tqdm(rs_pairs, desc="RS pairs"):
                sj, rj, cj = _pair_metrics(R_sc[i], S_sc[j])
                rs_ssim.append(sj); rs_nrmse.append(rj); rs_cos.append(cj)
        elif mode == "zip":
            N = min(nR, nS)
            for i in tqdm(range(N), desc="RS pairs (zip)"):
                sj, rj, cj = _pair_metrics(R_sc[i], S_sc[i])
                rs_ssim.append(sj); rs_nrmse.append(rj); rs_cos.append(cj)
        else:
            raise ValueError("mode must be 'all_vs_all' or 'zip'")

        rs_ssim_m, rs_ssim_sd = _mean_sd(rs_ssim)
        rs_nrmse_m, rs_nrmse_sd = _mean_sd(rs_nrmse)
        rs_cos_m, rs_cos_sd = _mean_sd(rs_cos)

        rr_ssim, rr_nrmse, rr_cos = [], [], []
        if mode == "all_vs_all":
            rr_pairs = [(i, j) for i in range(nR) for j in range(i + 1, nR)]
            for i, j in tqdm(rr_pairs, desc="RR pairs"):
                sj, rj, cj = _pair_metrics(R_sc[i], R_sc[j])
                rr_ssim.append(sj); rr_nrmse.append(rj); rr_cos.append(cj)
        else:
            for (i, j) in tqdm(_pairs_within_set(nR, rr_zip_strategy), desc="RR pairs (zip)"):
                sj, rj, cj = _pair_metrics(R_sc[i], R_sc[j])
                rr_ssim.append(sj); rr_nrmse.append(rj); rr_cos.append(cj)

        rr_ssim_m, rr_ssim_sd = _mean_sd(rr_ssim)
        rr_nrmse_m, rr_nrmse_sd = _mean_sd(rr_nrmse)
        rr_cos_m, rr_cos_sd = _mean_sd(rr_cos)

        ss_ssim, ss_nrmse, ss_cos = [], [], []
        if mode == "all_vs_all":
            ss_pairs = [(i, j) for i in range(nS) for j in range(i + 1, nS)]
            for i, j in tqdm(ss_pairs, desc="SS pairs"):
                sj, rj, cj = _pair_metrics(S_sc[i], S_sc[j])
                ss_ssim.append(sj); ss_nrmse.append(rj); ss_cos.append(cj)
        else:
            for (i, j) in tqdm(_pairs_within_set(nS, ss_zip_strategy), desc="SS pairs (zip)"):
                sj, rj, cj = _pair_metrics(S_sc[i], S_sc[j])
                ss_ssim.append(sj); ss_nrmse.append(rj); ss_cos.append(cj)

        ss_ssim_m, ss_ssim_sd = _mean_sd(ss_ssim)
        ss_nrmse_m, ss_nrmse_sd = _mean_sd(ss_nrmse)
        ss_cos_m, ss_cos_sd = _mean_sd(ss_cos)

        tests = {
            "SSIM": self._test_rr_vs_rs(rr_ssim, rs_ssim, mode=mode),
            "NRMSE": self._test_rr_vs_rs(rr_nrmse, rs_nrmse, mode=mode),
            "Cosine": self._test_rr_vs_rs(rr_cos, rs_cos, mode=mode),
        }

        def _fmt(x):
            return "nan" if not np.isfinite(x) else f"{x:.3g}"

        print("RR  | SSIM =", f"{_fmt(rr_ssim_m)} ± {_fmt(rr_ssim_sd)}",
              "| NRMSE =", f"{_fmt(rr_nrmse_m)} ± {_fmt(rr_nrmse_sd)}",
              "| Cosine =", f"{_fmt(rr_cos_m)} ± {_fmt(rr_cos_sd)} | mode: {mode}, pairs: {len(rr_ssim)} ({analysis_type})")
        print("RS  | SSIM =", f"{_fmt(rs_ssim_m)} ± {_fmt(rs_ssim_sd)}",
              "| NRMSE =", f"{_fmt(rs_nrmse_m)} ± {_fmt(rs_nrmse_sd)}",
              "| Cosine =", f"{_fmt(rs_cos_m)} ± {_fmt(rs_cos_sd)} | mode: {mode}, pairs: {len(rs_ssim)} ({analysis_type})")
        print("SS  | SSIM =", f"{_fmt(ss_ssim_m)} ± {_fmt(ss_ssim_sd)}",
              "| NRMSE =", f"{_fmt(ss_nrmse_m)} ± {_fmt(ss_nrmse_sd)}",
              "| Cosine =", f"{_fmt(ss_cos_m)} ± {_fmt(ss_cos_sd)} | mode: {mode}, pairs: {len(ss_ssim)} ({analysis_type})")

        def _fmt_p(x):
            return "nan" if not np.isfinite(x) else f"{x:.30e}"

        def _fmt_eff(x):
            return "nan" if not np.isfinite(x) else f"{x:.3g}"

        print(
            f"RR vs RS tests SSIM: p={_fmt_p(tests['SSIM']['p'])}, "
            f"CliffΔ={_fmt_eff(tests['SSIM']['effect'])}"
        )
        print(
            f"RR vs RS tests NRMSE: p={_fmt_p(tests['NRMSE']['p'])}, "
            f"CliffΔ={_fmt_eff(tests['NRMSE']['effect'])}"
        )
        print(
            f"RR vs RS tests Cosine: p={_fmt_p(tests['Cosine']['p'])}, "
            f"CliffΔ={_fmt_eff(tests['Cosine']['effect'])}"
        )

        scalogram_similarity_metrics = {
            "Analysis type": analysis_type,
            "RS mode": mode,
            "RR zip strategy": rr_zip_strategy if mode == "zip" else "all_vs_all",
            "SS zip strategy": ss_zip_strategy if mode == "zip" else "all_vs_all",
            "Per-pair SSIM (RS)": rs_ssim,
            "Per-pair NRMSE (RS)": rs_nrmse,
            "Per-pair Cosine (RS)": rs_cos,
            # Raw per-pair RR/SS lists (not just their summaries below) -- needed for any
            # downstream omnibus test across the three pairing groups (e.g. Kruskal-Wallis +
            # eta-squared comparing RR vs RS vs SS), which can't be reconstructed from
            # mean/SD alone.
            "Per-pair SSIM (RR)": rr_ssim,
            "Per-pair NRMSE (RR)": rr_nrmse,
            "Per-pair Cosine (RR)": rr_cos,
            "Per-pair SSIM (SS)": ss_ssim,
            "Per-pair NRMSE (SS)": ss_nrmse,
            "Per-pair Cosine (SS)": ss_cos,
            "RR Summary": {
                "Pairs": len(rr_ssim),
                "SSIM mean": rr_ssim_m, "SSIM SD": rr_ssim_sd,
                "NRMSE mean": rr_nrmse_m, "NRMSE SD": rr_nrmse_sd,
                "Cosine mean": rr_cos_m, "Cosine SD": rr_cos_sd,
            },
            "SS Summary": {
                "Pairs": len(ss_ssim),
                "SSIM mean": ss_ssim_m, "SSIM SD": ss_ssim_sd,
                "NRMSE mean": ss_nrmse_m, "NRMSE SD": ss_nrmse_sd,
                "Cosine mean": ss_cos_m, "Cosine SD": ss_cos_sd,
            },
            "RS Summary": {
                "Pairs": len(rs_ssim),
                "SSIM mean": rs_ssim_m, "SSIM SD": rs_ssim_sd,
                "NRMSE mean": rs_nrmse_m, "NRMSE SD": rs_nrmse_sd,
                "Cosine mean": rs_cos_m, "Cosine SD": rs_cos_sd},

            "RR_vs_RS": tests,
        }

        return scalogram_similarity_metrics

    def compute_mean_scalogram(self, signals, *, frequencies: np.ndarray | None = None):
        """
        Compute the **mean scalogram** across a set of signals, using the batched FFT-based
        CWT so the whole set is transformed in a single vectorized call instead of looping
        over signals one at a time in Python.

        Accepts:
          - 1D array (single signal)
          - 2D array (N_signals × T)
          - list/tuple of 1D arrays (all with the same length)

        Returns
        -------
        (np.ndarray, np.ndarray)
            Mean scalogram (freqs × time), and the frequency grid used.
        """
        if signals is None:
            raise ValueError("`signals` cannot be None.")

        if isinstance(signals, (list, tuple)):
            arr_list = [np.asarray(x).reshape(-1) for x in signals]
            lengths = {a.shape[0] for a in arr_list}
            if len(lengths) > 1:
                raise ValueError("All signals must have the same length (scalogram shapes must match).")
            arr = np.stack(arr_list, axis=0)
        else:
            arr = np.asarray(signals)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            elif arr.ndim != 2:
                raise ValueError("`signals` must be 1D, 2D, or a list of 1D arrays.")

        if arr.shape[0] == 0:
            raise ValueError("`signals` is empty.")

        F = frequencies if frequencies is not None else self.frequencies
        amp_batch = self._cwt_fft_batch(arr, w=6.0, frequencies=F)  # (n_signals, n_freqs, T)
        mean_scalo = amp_batch.mean(axis=0)
        return mean_scalo, F

    def plot_mean_scalograms(self, real_data, synthetic_data, *, save: str | None = None,
                             titles=("Mean real", "Mean synthetic"),
                             freq_scale: str | None = None,
                             intensity: str | None = None,
                             db_ref: str | float | None = None):
        """
        Compute and plot the mean scalogram of real and synthetic sets, side-by-side.
        """
        freq_scale = freq_scale or "linear"
        intensity = intensity or "amplitude"
        db_ref = "global_max" if db_ref is None else db_ref

        F = self._build_frequency_grid(freq_scale, self.num_freqs)

        real_mean_amp, F_used = self.compute_mean_scalogram(real_data, frequencies=F)
        synth_mean_amp, _ = self.compute_mean_scalogram(synthetic_data, frequencies=F_used)

        eps = 1e-12
        ref = None
        if intensity == "db" and db_ref == "global_max":
            ref = max(np.max(real_mean_amp ** 2), np.max(synth_mean_amp ** 2)) + eps

        real_int = self._intensity_from_mean_amp(real_mean_amp, intensity=intensity, db_ref=db_ref, ref_value=ref)
        synth_int = self._intensity_from_mean_amp(synth_mean_amp, intensity=intensity, db_ref=db_ref, ref_value=ref)

        vmin = min(float(np.nanmin(real_int)), float(np.nanmin(synth_int)))
        vmax = max(float(np.nanmax(real_int)), float(np.nanmax(synth_int)))

        real = np.asarray(real_data)
        if real.ndim == 2:
            T = real.shape[1]
        elif real.ndim == 1:
            T = real.shape[0]
        else:
            T = np.asarray(real_data[0]).shape[0]
        total_duration = T / self.fs if T > 0 else 0.0
        t = np.linspace(0, total_duration, real_int.shape[1])

        fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharey=True, constrained_layout=True)
        pm0 = axs[0].pcolormesh(t, F_used, real_int, shading="auto", cmap="terrain", vmin=vmin, vmax=vmax)
        axs[0].set_title(f'{titles[0]} scalogram', fontsize=20, fontname='Arial')
        axs[0].set_xlabel('Time (s)', fontsize=15, fontname='Arial')
        axs[0].set_ylabel('Frequency (Hz)', fontsize=15, fontname='Arial')
        axs[0].tick_params(axis='both', which='major', labelsize=18)
        if freq_scale == "log":
            axs[0].set_yscale("log")

        axs[1].pcolormesh(t, F_used, synth_int, shading="auto", cmap="terrain", vmin=vmin, vmax=vmax)
        axs[1].set_title(f'{titles[1]} scalogram', fontsize=20, fontname='Arial')
        axs[1].set_xlabel('Time (s)', fontsize=15, fontname='Arial')
        axs[1].tick_params(axis='both', which='major', labelsize=18)
        if freq_scale == "log":
            axs[1].set_yscale("log")

        cbar = fig.colorbar(pm0, ax=axs, location='right')
        # Explicit units on the intensity colorbar (reviewer-requested: "define the scale
        # explicitly in the caption for accurate interpretation of the scalograms").
        label = {"amplitude": "Amplitude (a.u.)", "power": "Power (a.u.)", "db": "Power (dB)"}[intensity]
        cbar.set_label(label, fontsize=15, fontname='Arial')
        cbar.ax.tick_params(labelsize=18)

        if save:
            plt.savefig(save, bbox_inches='tight', dpi=200)
        plt.show()
        return fig

    # Burst statistics helpers
    def _band_mask(self, F: np.ndarray, band: tuple[float, float]) -> np.ndarray:
        f_lo, f_hi = float(band[0]), float(band[1])
        if f_lo <= 0 or f_hi <= f_lo:
            raise ValueError("Invalid band. Use (f_low, f_high) with 0 < f_low < f_high.")
        return (F >= f_lo) & (F <= f_hi)

    def _band_envelope_from_scalogram(self, amp: np.ndarray, F: np.ndarray, band: tuple[float, float],
                                      smooth_ms: float = 50.0, normalize: bool = True) -> np.ndarray:
        mask = self._band_mask(F, band)
        if not np.any(mask):
            raise ValueError(f"No scalogram rows fall inside the requested band {band}. "
                             "Consider increasing num_freqs or adjusting band.")
        env = np.mean(amp[mask, :], axis=0)

        if normalize:
            baseline = np.median(env)
            if baseline > 0:
                env = env / baseline

        if smooth_ms and smooth_ms > 0:
            win = max(1, int(round((smooth_ms / 1000.0) * self.fs)))
            if win > 1:
                kernel = np.ones(win, dtype=float) / float(win)
                env = np.convolve(env, kernel, mode="same")
        return env

    def _detect_bursts_from_envelope(self, envelope: np.ndarray, *,
                                     threshold: str = "std",
                                     p: float = 75.0,
                                     kappa: float | None = 2.0,
                                     min_duration_ms: float = 20.0,
                                     merge_gap_ms: float = 50.0):
        x = np.asarray(envelope, dtype=float)
        if x.ndim != 1:
            raise ValueError("Envelope must be a 1D array.")

        if threshold == "percentile":
            thr = np.percentile(x, float(p))
        elif threshold == "std":
            if kappa is None:
                kappa = 1.5
            thr = float(np.mean(x) + float(kappa) * np.std(x))
        else:
            raise ValueError("threshold must be 'percentile' or 'std'.")

        above = (x >= thr).astype(np.int8)

        d = np.diff(np.r_[0, above, 0])
        starts = np.flatnonzero(d == 1)
        ends   = np.flatnonzero(d == -1) - 1

        min_samples = int(round((min_duration_ms / 1000.0) * self.fs))
        keep = []
        for s, e in zip(starts, ends):
            if (e - s + 1) >= max(1, min_samples):
                keep.append((s, e))
        events = keep

        merged = []
        if events:
            gap_samples = int(round((merge_gap_ms / 1000.0) * self.fs))
            cur_s, cur_e = events[0]
            for s, e in events[1:]:
                if s - cur_e - 1 <= gap_samples:
                    cur_e = e
                else:
                    merged.append((cur_s, cur_e))
                    cur_s, cur_e = s, e
            merged.append((cur_s, cur_e))
        events = merged

        bursts = []
        for s, e in events:
            seg = x[s:e+1]
            peak_amp = float(np.max(seg))
            peak_idx = int(s + np.argmax(seg))
            duration_s = float((e - s + 1) / self.fs)
            bursts.append({
                "start": int(s),
                "end": int(e),
                "peak_idx": int(peak_idx),
                "peak_amp": peak_amp,
                "duration_s": duration_s
            })

        ibis = []
        for (s1, e1), (s2, e2) in zip(events, events[1:]):
            ibis.append(float((s2 - e1 - 1) / self.fs))

        total_dur_s = float(len(x) / self.fs) if len(x) else 0.0
        total_burst_time_s = float(np.sum([(b["end"] - b["start"] + 1) for b in bursts]) / self.fs) if bursts else 0.0
        duty_cycle = (total_burst_time_s / total_dur_s) if total_dur_s > 0 else np.nan
        rate_hz = (len(bursts) / total_dur_s) if total_dur_s > 0 else np.nan

        summary = {
            "n_bursts": int(len(bursts)),
            "rate_hz": float(rate_hz),
            "mean_duration_s": float(np.mean([b["duration_s"] for b in bursts])) if bursts else 0.0,
            "median_duration_s": float(np.median([b["duration_s"] for b in bursts])) if bursts else 0.0,
            "mean_peak_amp": float(np.mean([b["peak_amp"] for b in bursts])) if bursts else 0.0,
            "median_peak_amp": float(np.median([b["peak_amp"] for b in bursts])) if bursts else 0.0,
            "mean_ibi_s": float(np.mean(ibis)) if len(ibis) > 0 else np.nan,
            "median_ibi_s": float(np.median(ibis)) if len(ibis) > 0 else np.nan,
            "duty_cycle": float(duty_cycle)
        }
        return bursts, summary, thr

    def compute_burst_statistics(self, real_data, synthetic_data, *,
                                 band=(13.0, 30.0),
                                 threshold="std",
                                 p=75.0,
                                 kappa=2.0,
                                 min_duration_ms=20.0,
                                 merge_gap_ms=50.0,
                                 freq_scale: str | None = None,
                                 smooth_ms: float = 20.0,
                                 verbose: bool = True):
        from scipy.stats import wasserstein_distance as WD

        R = np.asarray(real_data, dtype=float)
        S = np.asarray(synthetic_data, dtype=float)
        if R.ndim == 1: R = R[np.newaxis, :]
        if S.ndim == 1: S = S[np.newaxis, :]

        F = self._build_frequency_grid(freq_scale or "linear", self.num_freqs)

        def _one_set(signals, desc=""):
            out = []
            amp_batch = self._cwt_fft_batch(signals, w=6.0, frequencies=F)  # batched CWT, one call
            for i in tqdm(range(signals.shape[0]), desc=desc, total=signals.shape[0]):
                env = self._band_envelope_from_scalogram(amp_batch[i], F, band, smooth_ms=smooth_ms)
                _, summary, thr = self._detect_bursts_from_envelope(
                    env, threshold=threshold, p=p, kappa=kappa,
                    min_duration_ms=min_duration_ms, merge_gap_ms=merge_gap_ms
                )
                out.append(summary)
            return out

        R_sum = _one_set(R, desc="Burst stats (real)")
        S_sum = _one_set(S, desc="Burst stats (synthetic)")

        def _stack_feature(L, key):
            vals = [d[key] for d in L if np.isfinite(d[key])]
            return np.array(vals, dtype=float) if len(vals) > 0 else np.array([], dtype=float)

        keys = [
            "n_bursts", "rate_hz", "mean_duration_s", "median_duration_s",
            "mean_peak_amp", "median_peak_amp", "mean_ibi_s",
            "median_ibi_s", "duty_cycle"
        ]

        def _zscore_using_real(r, s):
            std = np.std(r)
            if std == 0 or not np.isfinite(std):
                return np.zeros_like(r), np.zeros_like(s)
            mean = np.mean(r)
            r_z = (r - mean) / std
            s_z = (s - mean) / std
            return r_z, s_z

        wd_rs = {}
        real_mean_feat = {}
        synth_mean_feat = {}
        n_real_feat = {}
        n_synth_feat = {}

        for k in keys:
            r = _stack_feature(R_sum, k)
            s = _stack_feature(S_sum, k)

            n_real_feat[k] = int(r.size)
            n_synth_feat[k] = int(s.size)
            real_mean_feat[k] = float(np.mean(r)) if r.size > 0 else np.nan
            synth_mean_feat[k] = float(np.mean(s)) if s.size > 0 else np.nan

            if r.size > 0 and s.size > 0:
                real_std = np.std(r) + 1e-12
                wd_rs[k] = float(WD(r, s) / real_std)
            else:
                wd_rs[k] = np.nan

        def _robust_sd(L, key):
            vals = _stack_feature(L, key)
            if vals.size == 0:
                return np.nan
            med = np.median(vals)
            mad = np.median(np.abs(vals - med))
            return float(1.4826 * mad)

        rr_rsd = {k: _robust_sd(R_sum, k) for k in keys}
        ss_rsd = {k: _robust_sd(S_sum, k) for k in keys}

        if verbose:
            def _fmt(x):
                return "nan" if not np.isfinite(x) else f"{x:.3g}"

            print("\n=== Burst Statistics Summary ===")
            print(f"Band: {band[0]:.3g}–{band[1]:.3g} Hz | Threshold: {threshold}"
                  + (f" (p={p:.0f})" if threshold == "percentile" else f" (mean+{kappa or 1.5}·SD)")
                  + f" | min_dur={min_duration_ms} ms | merge_gap={merge_gap_ms} ms | smooth={smooth_ms} ms")
            print(f"N_real={len(R_sum)} | N_synth={len(S_sum)}\n")

            header = (f"{'Feature':<18}  {'R_mean':>10}  {'S_mean':>10}  "
                      f"{'RS WD(z)':>10}  {'RR rSD':>10}  {'SS rSD':>10}")
            print(header)
            print("-" * len(header))
            nice = {
                "n_bursts": "n_bursts",
                "rate_hz": "rate_hz",
                "mean_duration_s": "mean_dur_s",
                "median_duration_s": "median_dur_s",
                "mean_peak_amp": "mean_peak",
                "median_peak_amp": "median_peak",
                "mean_ibi_s": "mean_IBI_s",
                "median_ibi_s": "median_IBI_s",
                "duty_cycle": "duty_cycle"
            }
            for k in keys:
                print(f"{nice[k]:<18}  "
                      f"{_fmt(real_mean_feat[k]):>10}  "
                      f"{_fmt(synth_mean_feat[k]):>10}  "
                      f"{_fmt(wd_rs[k]):>10}  "
                      f"{_fmt(rr_rsd[k]):>10}  "
                      f"{_fmt(ss_rsd[k]):>10}")

        return {
            "band": band,
            "threshold": threshold,
            "percentile_p": p,
            "kappa": kappa,
            "min_duration_ms": min_duration_ms,
            "merge_gap_ms": merge_gap_ms,
            "real_per_signal": R_sum,
            "synthetic_per_signal": S_sum,
            "RS_WD": wd_rs,
            "RR_robust_sd": rr_rsd,
            "SS_robust_sd": ss_rsd,
            "N_real_per_feature": n_real_feat,
            "N_synth_per_feature": n_synth_feat,
            "real_mean_per_feature": real_mean_feat,
            "synth_mean_per_feature": synth_mean_feat,
        }
