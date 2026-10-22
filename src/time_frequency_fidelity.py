import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, next_fast_len
from scipy.ndimage import uniform_filter1d
from skimage.metrics import structural_similarity as ssim, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import mannwhitneyu

try:
    from tqdm import tqdm
except ImportError:
    # Minimal no-op fallback so the script still runs if tqdm isn't installed
    # (run `pip install tqdm` to get real progress bars + ETA).
    def tqdm(iterable=None, total=None, desc=None, disable=False, **kwargs):
        if iterable is None:
            return iterable
        if disable:
            yield from iterable
            return
        n = total if total is not None else (len(iterable) if hasattr(iterable, "__len__") else None)
        label = f"{desc}: " if desc else ""
        for i, item in enumerate(iterable, 1):
            if n:
                print(f"\r{label}{i}/{n}", end="", flush=True)
            yield item
        print()


class TimeFrequencyFidelity:
    """
    Time-frequency similarity between real and synthetic signals: scalogram
    comparisons (SSIM, NRMSE, cosine) and burst statistics.

    Scalograms are computed with a batched, FFT-based Continuous Wavelet Transform
    using analytic Morlet wavelets (Torrence & Compo, 1998).

    Corrections relative to the published version (no new metrics)
    ---------------------------------------------------------------
    The metric set is unchanged: SSIM, nRMSE and cosine similarity on scalograms,
    the RR/RS/SS protocol with Cliff's delta, and burst statistics compared with
    SD-normalised Wasserstein distances. Only flaws in how they are computed or
    tested were corrected:

    * SSIM: computed on the normalised scalogram values (grayscale) by default. The
      paper calls it colour SSIM (cSSIM); the colormap is a display choice whose
      luminance is not monotonic in amplitude, so it should not drive the metric.
      `color_ssim=True` reproduces the colour (terrain RGB) variant.
    * Frequency grid: log-spaced by default (`freq_scale="log"`). The linear grid with
      ~1 Hz spacing missed low-frequency peaks, whose wavelet bandwidth is ~f/6.
    * Frequencies above 0.95 x Nyquist are dropped with a warning (no check before).
    * Coefficient normalisation: `normalization="amplitude"` (default) makes a
      sinusoid of amplitude A give |W| = A at every frequency. The published
      sqrt(scale) ("energy") normalisation made amplitude grow as 1/sqrt(f), so low
      frequencies dominated SSIM / NRMSE / cosine. Use `normalization="energy"` to
      reproduce published scalograms.
    * RR vs RS test, all_vs_all: permutation test over SIGNAL labels (pairs share
      signals, so Mann-Whitney on pairs gave far too many false positives).
      zip: Mann-Whitney + Cliff's delta (the paired Wilcoxon paired unrelated pairs).
    * Burst statistics: one threshold derived from the POOLED REAL envelopes and
      applied to both sets (per-signal thresholds fixed the duty cycle by
      construction); merge-then-minimum-duration order; reflect-padded smoothing;
      cone-of-influence edges trimmed; dedicated frequency grid inside the band.
    * Memory: CWT computed in chunks, scalograms stored as float32; optional
      `max_signals` and `decimate` for all_vs_all on long recordings.
    * Mean scalograms: "power"/"db" use the mean of power, not the square of the mean
      amplitude.

    Parameters
    ----------
    fs : float
        Sampling frequency (Hz).
    freq_min, freq_max : float
        Frequency range (Hz). freq_max is capped at 0.95 x Nyquist. Default 0.5–100 Hz.
    num_freqs : int
        Number of frequency rows (default 100).
    freq_scale : {"log", "linear"}
        Spacing of the frequency grid (default "log").
    normalization : {"amplitude", "energy"}
        Wavelet coefficient normalisation (see above). Default "amplitude".
    w : float
        Morlet nondimensional frequency (number of cycles), default 6.0.
    random_state : int
        Seed for permutation tests and subsampling.
    memory_budget_gb : float
        Approximate memory per CWT chunk (default 1 GB).
    color_ssim : bool
        If True, SSIM is computed on terrain-colormapped RGB scalograms (published
        cSSIM). Default False (grayscale SSIM on the scalogram values).

    To reproduce the published scalograms and cSSIM:
        TimeFrequencyFidelity(fs=..., freq_scale="linear", normalization="energy", color_ssim=True)

    Example
    -------
    real_data = np.random.randn(5, 2048)
    synthetic_data = np.random.randn(5, 2048)
    tff = TimeFrequencyFidelity(fs=256)
    sim = tff.compute_scalogram_similarity_metrics(real_data, synthetic_data)
    tff.plot_scalograms(real_data, synthetic_data, signal_index_real=0, signal_index_synth=4)
    tff.plot_mean_scalograms(real_data, synthetic_data)
    bursts = tff.compute_burst_statistics(real_data, synthetic_data, band=(13, 30),
                                          threshold="percentile", p=75.0)

    References
    ----------
    [1] https://scikit-image.org/docs/stable/api/skimage.metrics.html
    [2] https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
    [3] https://arxiv.org/html/2405.08431v3
    [4] Torrence, C. and Compo, G. P. (1998), "A Practical Guide to Wavelet Analysis",
        Bull. Amer. Meteor. Soc., 79, 61-78.
    """

    def __init__(self, *, fs: float, freq_min=0.5, freq_max=100, num_freqs=100,
                 freq_scale: str = "log", normalization: str = "amplitude", w: float = 6.0,
                 random_state: int = 42, memory_budget_gb: float = 1.0, color_ssim: bool = False) -> None:
        if fs <= 0:
            raise ValueError("Sampling frequency `fs` must be positive.")
        if normalization not in {"amplitude", "energy"}:
            raise ValueError("normalization must be 'amplitude' or 'energy'.")
        self.color_ssim = bool(color_ssim)
        self.fs = float(fs)
        self.freq_scale = freq_scale
        self.normalization = normalization
        self.w = float(w)
        self.random_state = random_state
        self.memory_budget_gb = float(memory_budget_gb)
        self._set_frequencies(freq_min, freq_max, num_freqs)

    # ------------------------------------------------------------ frequency grid

    def _nyquist_cap(self, freq_max):
        cap = 0.95 * self.fs / 2.0
        if freq_max > cap:
            warnings.warn(f"freq_max={freq_max} Hz exceeds 0.95 x Nyquist ({cap:.1f} Hz) "
                          f"for fs={self.fs:g} Hz; capping it.")
            return cap
        return freq_max

    def _set_frequencies(self, freq_min: float, freq_max: float, num_freqs: int):
        freq_max = self._nyquist_cap(float(freq_max))
        if freq_min <= 0 or freq_max <= freq_min:
            raise ValueError(f"Invalid frequency range [{freq_min}, {freq_max}] Hz "
                             f"for fs={self.fs:g} Hz.")
        self.freq_min = float(freq_min)
        self.freq_max = float(freq_max)
        self._freq_max_requested = float(freq_max)
        self.num_freqs = int(num_freqs)
        self.frequencies = self._build_frequency_grid(self.freq_scale, self.num_freqs)

    def set_sampling_rate(self, fs: float) -> None:
        """Change the sampling frequency and rebuild the (Nyquist-checked) frequency grid."""
        if fs <= 0:
            raise ValueError("Sampling frequency `fs` must be positive.")
        self.fs = float(fs)
        self._set_frequencies(self.freq_min, self._freq_max_requested, self.num_freqs)

    def _build_frequency_grid(self, freq_scale: str | None, num_freqs: int | None = None,
                              f_lo: float | None = None, f_hi: float | None = None) -> np.ndarray:
        freq_scale = freq_scale or self.freq_scale
        if freq_scale not in {"linear", "log"}:
            raise ValueError("freq_scale must be 'linear' or 'log'")
        N = int(num_freqs or self.num_freqs)
        lo = self.freq_min if f_lo is None else float(f_lo)
        hi = self.freq_max if f_hi is None else float(f_hi)
        if freq_scale == "linear":
            return np.linspace(lo, hi, N)
        return np.logspace(np.log10(lo), np.log10(hi), N)

    # ------------------------------------------------------------ statistics

    @staticmethod
    def cliffs_delta(x, y):
        """delta = P(x > y) - P(x < y), computed via the Mann-Whitney U statistic."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size == 0 or y.size == 0:
            return np.nan
        u = mannwhitneyu(x, y, alternative="two-sided").statistic  # counts x>y + 0.5 ties
        return float(2.0 * u / (x.size * y.size) - 1.0)

    def _perm_test_rr_vs_rs(self, M, nR, n_perm=1000):
        """
        Permutation test of mean(RR) vs mean(RS) on a full (N x N) similarity matrix,
        permuting the real/synthetic labels of SIGNALS (pairs are not independent).
        """
        N = M.shape[0]
        rng = np.random.default_rng(self.random_state)

        def _stat(is_real):
            r = np.flatnonzero(is_real)
            s = np.flatnonzero(~is_real)
            rr = M[np.ix_(r, r)][np.triu_indices(len(r), 1)]
            rs = M[np.ix_(r, s)]
            return np.nanmean(rr) - np.nanmean(rs)

        labels = np.zeros(N, dtype=bool)
        labels[:nR] = True
        obs = _stat(labels)
        null = np.empty(n_perm)
        for b in range(n_perm):
            null[b] = _stat(rng.permutation(labels))
        p = (1 + np.sum(np.abs(null) >= abs(obs))) / (1 + n_perm)
        return float(p), float(obs)

    # ------------------------------------------------------------ intensity helpers

    @staticmethod
    def _to_intensity(amp_or_mean_amp, mean_power, *, intensity: str, db_ref, ref_value=None):
        """
        amplitude -> amp ; power -> power ; db -> 10*log10(power / ref).
        `mean_power` is used for power/db (pass amp**2 for a single image).
        """
        eps = 1e-12
        if intensity == "amplitude":
            return amp_or_mean_amp
        if intensity == "power":
            return mean_power
        if intensity != "db":
            raise ValueError("intensity must be 'amplitude', 'power' or 'db'.")
        if isinstance(db_ref, (int, float)):
            ref = float(db_ref) + eps
        elif db_ref in {"global_max", "per_image_max"}:
            ref = ref_value if ref_value is not None else np.max(mean_power) + eps
        else:
            raise ValueError("Invalid db_ref.")
        return 10.0 * np.log10((mean_power + eps) / ref)

    # Kept for backward compatibility
    def _intensity_from_amp_single(self, amp, *, intensity, db_ref, ref_value=None):
        return self._to_intensity(amp, amp ** 2, intensity=intensity, db_ref=db_ref, ref_value=ref_value)

    # ------------------------------------------------------------ CWT

    def _scales(self, frequencies, w=None):
        w = self.w if w is None else w
        return w / (2 * np.pi * np.asarray(frequencies, dtype=float) / self.fs)  # in samples

    def _cwt_fft_batch(self, signals: np.ndarray, w: float | None = None,
                       frequencies: np.ndarray | None = None, decimate: int = 1) -> np.ndarray:
        """
        Batched analytic-Morlet CWT via FFT (Torrence & Compo, 1998, eq. 6).

        psi_hat(s*omega) = pi^(-1/4) * H(omega) * exp(-0.5 * (s*omega - w)^2)

        Signals are zero-padded by 4 x the widest wavelet's envelope SD on both sides to
        avoid circular wrap-around, and processed in chunks to bound memory.

        Normalisation
        -------------
        "amplitude": |W| * 2 * pi^(1/4)  -> a sinusoid of amplitude A gives |W| = A.
        "energy"   : |W| * sqrt(scale)   -> published version (amplitude grows as 1/sqrt(f)).

        Returns float32 array (n_signals, n_freqs, ceil(T / decimate)).
        """
        w = self.w if w is None else w
        signals = np.atleast_2d(np.asarray(signals, dtype=float))
        n_signals, T = signals.shape
        F = self.frequencies if frequencies is None else np.asarray(frequencies, dtype=float)
        if np.any(F >= self.fs / 2):
            raise ValueError("Frequencies at or above Nyquist requested.")
        scales = self._scales(F, w)

        pad = int(np.ceil(4 * np.max(scales)))
        n_fft = next_fast_len(T + 2 * pad)
        omega = 2 * np.pi * np.fft.fftfreq(n_fft)
        heaviside = (omega > 0).astype(float)
        psi_hat = (np.pi ** -0.25) * heaviside[None, :] * np.exp(-0.5 * (scales[:, None] * omega[None, :] - w) ** 2)

        if self.normalization == "energy":
            norm = np.sqrt(scales)[None, :, None]
        else:
            norm = 2.0 * np.pi ** 0.25

        # chunk size from memory budget (complex product + inverse transform)
        bytes_per_signal = len(F) * n_fft * 16 * 2
        chunk = max(1, int(self.memory_budget_gb * 1e9 // bytes_per_signal))

        T_out = int(np.ceil(T / decimate))
        out = np.empty((n_signals, len(F), T_out), dtype=np.float32)
        for start in range(0, n_signals, chunk):
            block = signals[start:start + chunk]
            padded = np.pad(block, ((0, 0), (pad, pad)), mode="constant")
            X = fft(padded, n=n_fft, axis=1)
            W = ifft(X[:, None, :] * psi_hat[None, :, :], axis=2)[:, :, pad:pad + T]
            out[start:start + len(block)] = (np.abs(W) * norm)[:, :, ::decimate]
        return out

    def _compute_scalogram(self, signal, w=None, *, frequencies=None, return_freqs=False):
        """Scalogram amplitude (n_freqs, T) of a single 1D signal."""
        signal = np.asarray(signal)
        if signal.ndim != 1:
            raise ValueError("Signal must be a 1D array.")
        F = self.frequencies if frequencies is None else frequencies
        amp = self._cwt_fft_batch(signal[None, :], w=w, frequencies=F)[0]
        return (amp, F) if return_freqs else amp

    # ------------------------------------------------------------ plots

    def plot_scalograms(self, real_data, synthetic_data, *, signal_index_real=0, signal_index_synth=0, save=None,
                        freq_scale: str | None = None, intensity: str | None = None, db_ref: str | float | None = None):
        """Plot the scalograms of one real and one synthetic signal side by side."""
        freq_scale = freq_scale or self.freq_scale
        intensity = intensity or "amplitude"
        db_ref = "global_max" if db_ref is None else db_ref

        real_data = np.asarray(real_data)
        synthetic_data = np.asarray(synthetic_data)
        real_signal = real_data[signal_index_real] if real_data.ndim > 1 else real_data
        synthetic_signal = synthetic_data[signal_index_synth] if synthetic_data.ndim > 1 else synthetic_data

        F = self._build_frequency_grid(freq_scale, self.num_freqs)
        real_amp = self._compute_scalogram(real_signal, frequencies=F)
        synth_amp = self._compute_scalogram(synthetic_signal, frequencies=F)

        ref = None
        if intensity == "db" and db_ref == "global_max":
            ref = max(np.max(real_amp ** 2), np.max(synth_amp ** 2)) + 1e-12
        real_int = self._to_intensity(real_amp, real_amp ** 2, intensity=intensity, db_ref=db_ref, ref_value=ref)
        synth_int = self._to_intensity(synth_amp, synth_amp ** 2, intensity=intensity, db_ref=db_ref, ref_value=ref)

        titles = (f'Real signal {signal_index_real} scalogram', f'Synthetic signal {signal_index_synth} scalogram')
        return self._plot_pair(real_int, synth_int, F, len(real_signal), titles, freq_scale, intensity, save)

    def compute_mean_scalogram(self, signals, *, frequencies=None, return_power: bool = False):
        """
        Mean scalogram across signals (1D, 2D (N, T) or list of equal-length 1D arrays).

        Returns (mean_amplitude, F), or (mean_amplitude, mean_power, F) if return_power.
        """
        if signals is None:
            raise ValueError("`signals` cannot be None.")
        if isinstance(signals, (list, tuple)):
            arr_list = [np.asarray(x).reshape(-1) for x in signals]
            if len({a.shape[0] for a in arr_list}) > 1:
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

        F = self.frequencies if frequencies is None else frequencies
        amp = self._cwt_fft_batch(arr, frequencies=F).astype(np.float64)
        mean_amp = amp.mean(axis=0)
        if return_power:
            return mean_amp, (amp ** 2).mean(axis=0), F
        return mean_amp, F

    def plot_mean_scalograms(self, real_data, synthetic_data, *, save: str | None = None,
                             titles=("Mean real", "Mean synthetic"), freq_scale: str | None = None,
                             intensity: str | None = None, db_ref: str | float | None = None):
        """Plot mean scalograms of the real and synthetic sets ("power"/"db" = mean power)."""
        freq_scale = freq_scale or self.freq_scale
        intensity = intensity or "amplitude"
        db_ref = "global_max" if db_ref is None else db_ref

        F = self._build_frequency_grid(freq_scale, self.num_freqs)
        r_amp, r_pow, _ = self.compute_mean_scalogram(real_data, frequencies=F, return_power=True)
        s_amp, s_pow, _ = self.compute_mean_scalogram(synthetic_data, frequencies=F, return_power=True)

        ref = None
        if intensity == "db" and db_ref == "global_max":
            ref = max(np.max(r_pow), np.max(s_pow)) + 1e-12
        real_int = self._to_intensity(r_amp, r_pow, intensity=intensity, db_ref=db_ref, ref_value=ref)
        synth_int = self._to_intensity(s_amp, s_pow, intensity=intensity, db_ref=db_ref, ref_value=ref)

        real = np.asarray(real_data) if not isinstance(real_data, (list, tuple)) else np.asarray(real_data[0])
        T = real.shape[-1]
        return self._plot_pair(real_int, synth_int, F, T,
                               (f'{titles[0]} scalogram', f'{titles[1]} scalogram'),
                               freq_scale, intensity, save)

    def _plot_pair(self, real_int, synth_int, F, T, titles, freq_scale, intensity, save):
        vmin = min(float(np.nanmin(real_int)), float(np.nanmin(synth_int)))
        vmax = max(float(np.nanmax(real_int)), float(np.nanmax(synth_int)))
        t = np.linspace(0, T / self.fs, real_int.shape[1])

        fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharey=True, constrained_layout=True)
        pm0 = None
        for ax, img, title in zip(axs, (real_int, synth_int), titles):
            pm = ax.pcolormesh(t, F, img, shading="auto", cmap="terrain", vmin=vmin, vmax=vmax)
            pm0 = pm0 or pm
            ax.set_title(title, fontsize=20, fontname='Arial')
            ax.set_xlabel('Time (s)', fontsize=15, fontname='Arial')
            ax.tick_params(axis='both', which='major', labelsize=18)
            if freq_scale == "log":
                ax.set_yscale("log")
        axs[0].set_ylabel('Frequency (Hz)', fontsize=15, fontname='Arial')

        cbar = fig.colorbar(pm0, ax=axs, location='right')
        unit = "a.u." if self.normalization == "energy" else "signal units"
        label = {"amplitude": f"Amplitude ({unit})", "power": f"Power ({unit}²)", "db": "Power (dB)"}[intensity]
        cbar.set_label(label, fontsize=15, fontname='Arial')
        cbar.ax.tick_params(labelsize=18)
        if save:
            fig.savefig(save, bbox_inches="tight", dpi=200)
        plt.show()
        return fig

    # ------------------------------------------------------------ similarity metrics

    def _pair_metrics(self, A, B):
        vmin = float(min(A.min(), B.min()))
        vmax = float(max(A.max(), B.max()))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        An = (A - vmin) / (vmax - vmin)
        Bn = (B - vmin) / (vmax - vmin)
        if self.color_ssim:
            if not hasattr(self, "_terrain"):
                self._terrain = plt.get_cmap("terrain")
            ssim_val = ssim(self._terrain(An)[..., :3], self._terrain(Bn)[..., :3],
                            data_range=1.0, channel_axis=-1)
        else:
            ssim_val = ssim(An, Bn, data_range=1.0)
        rmse = np.sqrt(mean_squared_error(A, B))
        nrmse = rmse / (vmax - vmin)
        cos = cosine_similarity(A.reshape(1, -1), B.reshape(1, -1))[0, 0]
        return float(ssim_val), float(nrmse), float(cos)

    def compute_scalogram_similarity_metrics(self, real_data, synthetic_data, *, mode: str = "all_vs_all",
                                             pad: bool = True, rr_zip_strategy: str = "consecutive",
                                             ss_zip_strategy: str = "consecutive", n_perm: int = 1000,
                                             max_signals: int | None = None, decimate: int = 1,
                                             verbose: bool = True):
        """
        Similarity between real and synthetic scalograms (SSIM, NRMSE, cosine) for
        real–synthetic (RS), real–real (RR) and synthetic–synthetic (SS) pairs.

        mode
        ----
        "all_vs_all": every pair. RR vs RS is tested with a permutation test over signal
                      labels (valid although pairs share signals) + Cliff's delta.
        "zip"       : real i vs synthetic i. Only meaningful when signals are matched
                      1:1 (e.g. conditional generation); otherwise the pairing is
                      arbitrary. RR vs RS tested with Mann-Whitney + Cliff's delta.

        max_signals : subsample each set to at most this many signals (seeded) — the
                      number of SSIM calls grows as N^2.
        decimate    : keep every `decimate`-th scalogram column (speed/memory).
        """
        R = np.asarray(real_data, dtype=float)
        S = np.asarray(synthetic_data, dtype=float)
        if R.ndim == 1: R = R[np.newaxis, :]
        if S.ndim == 1: S = S[np.newaxis, :]
        if R.shape[0] == 0 or S.shape[0] == 0:
            raise ValueError("Both real_signals and synthetic_signals must contain at least one signal.")
        if mode not in {"all_vs_all", "zip"}:
            raise ValueError("mode must be 'all_vs_all' or 'zip'")

        rng = np.random.default_rng(self.random_state)
        if max_signals is not None:
            if R.shape[0] > max_signals:
                R = R[np.sort(rng.choice(R.shape[0], max_signals, replace=False))]
            if S.shape[0] > max_signals:
                S = S[np.sort(rng.choice(S.shape[0], max_signals, replace=False))]

        analysis_type = "sample" if R.shape[0] == 1 else "dataset"
        nR, nS = R.shape[0], S.shape[0]

        if pad:
            L = int(max(R.shape[1], S.shape[1]))
            if R.shape[1] != L: R = np.pad(R, ((0, 0), (0, L - R.shape[1])))
            if S.shape[1] != L: S = np.pad(S, ((0, 0), (0, L - S.shape[1])))
        elif R.shape[1] != S.shape[1]:
            raise ValueError("When pad=False, real and synthetic signals must have identical lengths.")

        R_sc = self._cwt_fft_batch(R, decimate=decimate)
        S_sc = self._cwt_fft_batch(S, decimate=decimate)

        def _mean_sd(vals):
            vals = np.asarray(vals, dtype=float)
            if vals.size == 0:
                return np.nan, np.nan
            return float(np.mean(vals)), (float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan)

        def _pairs_within_set(n, zip_strategy):
            if zip_strategy not in {"consecutive", "halves"}:
                raise ValueError("zip_strategy must be 'consecutive' or 'halves'")
            if zip_strategy == "consecutive":
                return [(k, k + 1) for k in range(0, n - 1, 2)]
            half = n // 2
            return [(k, k + half) for k in range(half)]

        names = ("SSIM", "NRMSE", "Cosine")
        if mode == "all_vs_all":
            # Full similarity matrices over all N = nR + nS signals (same number of SSIM
            # calls as RS + RR + SS), needed for the permutation test.
            allsc = np.concatenate([R_sc, S_sc], axis=0)
            N = nR + nS
            n_pairs = N * (N - 1) // 2
            if verbose:
                print(f"[compute_scalogram_similarity_metrics] mode='all_vs_all': {n_pairs:,} pair comparisons "
                      f"(RS={nR*nS:,}, RR={nR*(nR-1)//2:,}, SS={nS*(nS-1)//2:,}).")
                if n_pairs > 2_000:
                    print("[compute_scalogram_similarity_metrics] WARNING: this could take a while; "
                          "consider max_signals=... or decimate=...")
            M = {k: np.full((N, N), np.nan) for k in names}
            pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
            for i, j in tqdm(pairs, desc="Scalogram pairs", disable=not verbose):
                vals = self._pair_metrics(allsc[i], allsc[j])
                for k, v in zip(names, vals):
                    M[k][i, j] = M[k][j, i] = v

            iu_r = np.triu_indices(nR, 1)
            iu_s = np.triu_indices(nS, 1)
            per = {}
            for k in names:
                per[("RR", k)] = list(M[k][:nR, :nR][iu_r])
                per[("SS", k)] = list(M[k][nR:, nR:][iu_s])
                per[("RS", k)] = list(M[k][:nR, nR:].ravel())

            tests = {}
            for k in names:
                if nR >= 2:
                    p, diff = self._perm_test_rr_vs_rs(M[k], nR, n_perm=n_perm)
                else:
                    p, diff = np.nan, np.nan
                tests[k] = {"p": p, "effect": self.cliffs_delta(per[("RR", k)], per[("RS", k)]),
                            "mean_diff_RR_minus_RS": diff,
                            "test": f"Permutation over signal labels (n_perm={n_perm}) + CliffΔ"}
        else:
            per = {(g, k): [] for g in ("RS", "RR", "SS") for k in names}
            groups = {
                "RS": [(R_sc, S_sc, i, i) for i in range(min(nR, nS))],
                "RR": [(R_sc, R_sc, i, j) for i, j in _pairs_within_set(nR, rr_zip_strategy)],
                "SS": [(S_sc, S_sc, i, j) for i, j in _pairs_within_set(nS, ss_zip_strategy)],
            }
            for g, items in groups.items():
                for A, B, i, j in items:
                    for k, v in zip(names, self._pair_metrics(A[i], B[j])):
                        per[(g, k)].append(v)
            tests = {}
            for k in names:
                rr, rs = per[("RR", k)], per[("RS", k)]
                if len(rr) and len(rs):
                    p = float(mannwhitneyu(rr, rs, alternative="two-sided").pvalue)
                else:
                    p = np.nan
                tests[k] = {"p": p, "effect": self.cliffs_delta(rr, rs), "test": "MWU + CliffΔ"}

        summ = {}
        for g in ("RR", "SS", "RS"):
            d = {"Pairs": len(per[(g, "SSIM")])}
            for k in names:
                m, sd = _mean_sd(per[(g, k)])
                d[f"{k} mean"], d[f"{k} SD"] = m, sd
            summ[g] = d

        if verbose:
            _f = lambda x: "nan" if not np.isfinite(x) else f"{x:.3g}"
            for g in ("RR", "RS", "SS"):
                s = summ[g]
                print(f"{g}  | SSIM = {_f(s['SSIM mean'])} ± {_f(s['SSIM SD'])} | NRMSE = {_f(s['NRMSE mean'])} ± "
                      f"{_f(s['NRMSE SD'])} | Cosine = {_f(s['Cosine mean'])} ± {_f(s['Cosine SD'])} | "
                      f"mode: {mode}, pairs: {s['Pairs']} ({analysis_type})")
            for k in names:
                print(f"RR vs RS {k}: p={_f(tests[k]['p'])}, CliffΔ={_f(tests[k]['effect'])}  [{tests[k]['test']}]")

        return {
            "Analysis type": analysis_type,
            "RS mode": mode,
            "RR zip strategy": rr_zip_strategy if mode == "zip" else "all_vs_all",
            "SS zip strategy": ss_zip_strategy if mode == "zip" else "all_vs_all",
            "Per-pair SSIM (RS)": per[("RS", "SSIM")],
            "Per-pair NRMSE (RS)": per[("RS", "NRMSE")],
            "Per-pair Cosine (RS)": per[("RS", "Cosine")],
            "Per-pair SSIM (RR)": per[("RR", "SSIM")],
            "Per-pair NRMSE (RR)": per[("RR", "NRMSE")],
            "Per-pair Cosine (RR)": per[("RR", "Cosine")],
            "Per-pair SSIM (SS)": per[("SS", "SSIM")],
            "Per-pair NRMSE (SS)": per[("SS", "NRMSE")],
            "Per-pair Cosine (SS)": per[("SS", "Cosine")],
            "RR Summary": summ["RR"],
            "SS Summary": summ["SS"],
            "RS Summary": summ["RS"],
            "RR_vs_RS": tests,
        }

    # ------------------------------------------------------------ burst statistics

    def _band_envelope(self, amp: np.ndarray, smooth_ms: float, trim: int) -> np.ndarray:
        env = amp.mean(axis=0)
        if smooth_ms and smooth_ms > 0:
            win = max(1, int(round((smooth_ms / 1000.0) * self.fs)))
            if win > 1:
                env = uniform_filter1d(env, size=win, mode="reflect")
        if trim > 0:
            env = env[trim:len(env) - trim]
        return env

    def _detect_bursts_from_envelope(self, envelope: np.ndarray, *, thr: float | None = None,
                                     threshold: str = "percentile", p: float = 75.0, kappa: float | None = 2.0,
                                     min_duration_ms: float = 20.0, merge_gap_ms: float = 50.0):
        """
        Detect bursts in one envelope. `thr` should be the threshold derived from the
        pooled REAL envelopes (compute_burst_statistics passes it); if None, a
        per-signal threshold is used (published behaviour, not recommended).
        Order: threshold -> merge gaps -> minimum duration.
        """
        x = np.asarray(envelope, dtype=float)
        if x.ndim != 1:
            raise ValueError("Envelope must be a 1D array.")
        if thr is None:
            if threshold == "percentile":
                thr = float(np.percentile(x, float(p)))
            elif threshold == "std":
                thr = float(np.mean(x) + (1.5 if kappa is None else float(kappa)) * np.std(x))
            else:
                raise ValueError("threshold must be 'percentile' or 'std'.")

        above = (x >= thr).astype(np.int8)
        d = np.diff(np.r_[0, above, 0])
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1) - 1
        events = list(zip(starts, ends))

        # 1) merge events separated by short gaps
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

        # 2) then drop events shorter than the minimum duration
        min_samples = max(1, int(round((min_duration_ms / 1000.0) * self.fs)))
        events = [(s, e) for s, e in merged if (e - s + 1) >= min_samples]

        bursts = []
        for s, e in events:
            seg = x[s:e + 1]
            bursts.append({"start": int(s), "end": int(e), "peak_idx": int(s + np.argmax(seg)),
                           "peak_amp": float(np.max(seg)), "duration_s": float((e - s + 1) / self.fs)})
        ibis = [float((s2 - e1 - 1) / self.fs) for (_, e1), (s2, _) in zip(events, events[1:])]

        total_dur_s = float(len(x) / self.fs) if len(x) else 0.0
        burst_time = float(sum(b["end"] - b["start"] + 1 for b in bursts) / self.fs)
        summary = {
            "n_bursts": int(len(bursts)),
            "rate_hz": float(len(bursts) / total_dur_s) if total_dur_s > 0 else np.nan,
            "mean_duration_s": float(np.mean([b["duration_s"] for b in bursts])) if bursts else 0.0,
            "median_duration_s": float(np.median([b["duration_s"] for b in bursts])) if bursts else 0.0,
            "mean_peak_amp": float(np.mean([b["peak_amp"] for b in bursts])) if bursts else np.nan,
            "median_peak_amp": float(np.median([b["peak_amp"] for b in bursts])) if bursts else np.nan,
            "mean_ibi_s": float(np.mean(ibis)) if ibis else np.nan,
            "median_ibi_s": float(np.median(ibis)) if ibis else np.nan,
            "duty_cycle": float(burst_time / total_dur_s) if total_dur_s > 0 else np.nan,
        }
        return bursts, summary, thr

    def compute_burst_statistics(self, real_data, synthetic_data, *, band=(13.0, 30.0), threshold="percentile",
                                 p=75.0, kappa=2.0, min_duration_ms=20.0, merge_gap_ms=50.0,
                                 freq_scale: str | None = None, smooth_ms: float = 20.0,
                                 n_band_freqs: int = 10, trim_coi: bool = True, verbose: bool = True):
        """
        Burst statistics in a frequency band, with ONE threshold derived from the pooled
        real envelopes and applied to both sets.

        Envelope = mean CWT amplitude over `n_band_freqs` frequencies spanning `band`,
        smoothed (reflect padding), with cone-of-influence edges trimmed (sqrt(2) x the
        scale of the band's lowest frequency) and divided by the median of the pooled
        real envelopes (so peak_amp is relative to typical real band amplitude; the
        same constant is applied to both sets, so real/synthetic amplitude
        differences are preserved).

        threshold: "percentile" -> p-th percentile of pooled real envelope values (default)
                   "std"        -> mean + kappa * SD of pooled real values (misses bursts
                                   when they occupy a large fraction of the time;
                                   prefer "percentile")
        """
        from scipy.stats import wasserstein_distance as WD

        R = np.asarray(real_data, dtype=float)
        S = np.asarray(synthetic_data, dtype=float)
        if R.ndim == 1: R = R[np.newaxis, :]
        if S.ndim == 1: S = S[np.newaxis, :]

        f_lo, f_hi = float(band[0]), float(band[1])
        if f_lo <= 0 or f_hi <= f_lo:
            raise ValueError("Invalid band. Use (f_low, f_high) with 0 < f_low < f_high.")
        if f_hi >= 0.95 * self.fs / 2:
            raise ValueError(f"Band upper edge {f_hi} Hz is too close to Nyquist ({self.fs/2:g} Hz).")
        F = self._build_frequency_grid(freq_scale, n_band_freqs, f_lo=f_lo, f_hi=f_hi)
        trim = int(np.ceil(np.sqrt(2) * self._scales([f_lo])[0])) if trim_coi else 0

        def _envelopes(X, desc):
            amp = self._cwt_fft_batch(X, frequencies=F)
            envs = [self._band_envelope(amp[i], smooth_ms, trim)
                    for i in tqdm(range(X.shape[0]), desc=desc, disable=not verbose)]
            if len(envs[0]) == 0:
                raise ValueError("Signals are too short for this band after cone-of-influence trimming; "
                                 "use trim_coi=False or a higher band.")
            return envs

        R_env = _envelopes(R, "Envelopes (real)")
        S_env = _envelopes(S, "Envelopes (synthetic)")

        pooled_real = np.concatenate(R_env)
        scale = float(np.median(pooled_real))
        scale = scale if scale > 0 else 1.0
        R_env = [e / scale for e in R_env]
        S_env = [e / scale for e in S_env]
        pooled_real = pooled_real / scale

        if threshold == "percentile":
            thr = float(np.percentile(pooled_real, float(p)))
        elif threshold == "std":
            # Fails when bursts occupy a large fraction of the time: mean + kappa*SD of
            # a two-level envelope can exceed the burst level itself.
            thr = float(np.mean(pooled_real) + float(kappa if kappa is not None else 1.5) * np.std(pooled_real))
        else:
            raise ValueError("threshold must be 'percentile' or 'std'.")

        def _summaries(envs):
            return [self._detect_bursts_from_envelope(e, thr=thr, min_duration_ms=min_duration_ms,
                                                      merge_gap_ms=merge_gap_ms)[1] for e in envs]

        R_sum, S_sum = _summaries(R_env), _summaries(S_env)

        keys = ["n_bursts", "rate_hz", "mean_duration_s", "median_duration_s", "mean_peak_amp",
                "median_peak_amp", "mean_ibi_s", "median_ibi_s", "duty_cycle"]

        def _stack(L, key):
            return np.array([d[key] for d in L if np.isfinite(d[key])], dtype=float)

        def _robust_sd(vals):
            if vals.size == 0:
                return np.nan
            return float(1.4826 * np.median(np.abs(vals - np.median(vals))))

        wd_rs, r_mean, s_mean, n_r, n_s, r_rsd, s_rsd = {}, {}, {}, {}, {}, {}, {}
        for k in keys:
            r, s = _stack(R_sum, k), _stack(S_sum, k)
            n_r[k], n_s[k] = int(r.size), int(s.size)
            r_mean[k] = float(np.mean(r)) if r.size else np.nan
            s_mean[k] = float(np.mean(s)) if s.size else np.nan
            r_rsd[k], s_rsd[k] = _robust_sd(r), _robust_sd(s)
            if r.size and s.size:
                d = WD(r, s)
                sd = np.std(r)
                wd_rs[k] = float(d / sd) if sd > 1e-12 else (0.0 if d == 0 else np.nan)
            else:
                wd_rs[k] = np.nan

        if verbose:
            _f = lambda x: "nan" if not np.isfinite(x) else f"{x:.3g}"
            print("\n=== Burst Statistics Summary ===")
            print(f"Band: {f_lo:.3g}–{f_hi:.3g} Hz | Threshold (from pooled real): {threshold}"
                  + {"percentile": f" (p={p:.0f})", "std": f" (mean+{kappa}·SD)"}[threshold]
                  + f" = {thr:.3g} x real median | min_dur={min_duration_ms} ms | merge_gap={merge_gap_ms} ms"
                  + f" | smooth={smooth_ms} ms | COI trim={trim / self.fs:.2f} s per edge")
            print(f"N_real={len(R_sum)} | N_synth={len(S_sum)}\n")
            header = (f"{'Feature':<18}  {'R_mean':>10}  {'S_mean':>10}  {'RS WD(z)':>10}  "
                      f"{'Real rSD':>10}  {'Synth rSD':>10}")
            print(header)
            print("-" * len(header))
            for k in keys:
                print(f"{k:<18}  {_f(r_mean[k]):>10}  {_f(s_mean[k]):>10}  {_f(wd_rs[k]):>10}  "
                      f"{_f(r_rsd[k]):>10}  {_f(s_rsd[k]):>10}")

        return {
            "band": band, "threshold": threshold, "percentile_p": p, "kappa": kappa,
            "threshold_value": thr, "envelope_scale_real_median": scale, "coi_trim_samples": trim,
            "min_duration_ms": min_duration_ms, "merge_gap_ms": merge_gap_ms,
            "real_per_signal": R_sum, "synthetic_per_signal": S_sum,
            "RS_WD": wd_rs,
            "real_robust_sd": r_rsd, "synth_robust_sd": s_rsd,
            "RR_robust_sd": r_rsd, "SS_robust_sd": s_rsd,   # legacy key names
            "N_real_per_feature": n_r, "N_synth_per_feature": n_s,
            "real_mean_per_feature": r_mean, "synth_mean_per_feature": s_mean,
        }

