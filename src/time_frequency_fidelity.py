import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cwt, morlet2
from skimage.metrics import structural_similarity as ssim, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

class TimeFrequencyFidelity:
    """
    A class to measure time-frequency similarity between real and synthetic signals, by computing and comparing
    scalogram representations of signals, as well as bursts statistics.

    The scalogram is computed using **Continuous Wavelet Transform (CWT)** with **Morlet wavelets**.

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
        # Use log spacing for better low-frequency resolution
        #self.frequencies = (frequencies if frequencies is not None
        #                    else np.logspace(np.log10(1), np.log10(30), 30))

        #self.frequencies = np.logspace(np.log10(0.5), np.log10(500), 40)

    def _set_frequencies(self, freq_min: float, freq_max: float, num_freqs: int):
        if freq_min <= 0 or freq_max <= freq_min:
            raise ValueError("Invalid frequency range.")
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.num_freqs = int(num_freqs)
        self.frequencies = np.linspace(freq_min, freq_max, num_freqs)
        #self.frequencies = np.logspace(np.log10(freq_min), np.log10(freq_max), num_freqs)

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
        # dB
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
        # dB
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

        Parameters:
        ----------
        image : np.ndarray
            Input grayscale image.
        colormap : str, optional
            Colormap to use for conversion (default is 'terrain').
        vmin : float, optional
            Lower bound of the shared dynamic range. If ``None`` (default) the
            minimum of *image* is used.  Supplying the same *vmin* to two images
            guarantees colour consistency across them.
        vmax : float, optional
            Upper bound of the shared dynamic range. If ``None`` (default) the
            maximum of *image* is used.  Must satisfy ``vmax > vmin``; otherwise
            the function silently rescales by a factor of 1 to avoid division by
            zero.

        Returns:
        -------
        np.ndarray
            RGB image with values scaled between 0 and 255.
        """
        if vmin is None:
            vmin = image.min()
        if vmax is None:
            vmax = image.max()

        denom = (vmax - vmin) or 1.0  # avoid division by zero
        norm = np.clip((image - vmin) / denom, 0, 1)

        cmap = plt.get_cmap(colormap)
        return (cmap(norm)[..., :3] * 255).astype(np.uint8)

    def _compute_scalogram(self, signal, w=6.0, *, frequencies: np.ndarray | None = None, return_freqs: bool = False):
        """
        Compute the scalogram using Continuous Wavelet Transform (CWT) with Morlet wavelets.

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
        dt = 1.0 / self.fs
        # Morlet scale-frequency relation: f = w / (2*pi*s*dt)  => s = w / (2*pi*f*dt)
        scales = (w / (2 * np.pi * F * dt))
        # Version-safe wavelet: try complete=True; if not available, fall back.
        def _morlet(M, s):
            try:
                return morlet2(M, s, w=w, complete=True)
            except TypeError:
                return morlet2(M, s, w=w)
        coeffs = cwt(signal, _morlet, scales)
        amp = np.abs(coeffs)  # use **2 if you want power
        if return_freqs:
            return amp, F
        return amp

    def plot_scalograms(self, real_data, synthetic_data, *, signal_index_real=0, signal_index_synth=0, save=None,
                        freq_scale: str | None = None, intensity: str | None = None, db_ref: str | float | None = None):
        """
        Compute and plot the scalogram for the provided real and synthetic signals.

        Parameters:
        ----------
        real_data : np.ndarray
            Array containing real signals.
        synthetic_data : np.ndarray
            Array containing synthetic signals.
        signal_index_real : int, optional
            Index of the real signal to plot (default is 0).
        signal_index_synth : int, optional
            Index of the synthetic signal to plot (default is 0).
        save : str | None
            Optional path to save the figure.

        New (optional) per-call overrides:
        ----------------------------------
        freq_scale : {"linear","log"}, default "linear"
            Frequency grid spacing for the plot.
        intensity : {"amplitude","power","db"}, default "amplitude"
            What to display for color intensity.
        db_ref : {"global_max","per_image_max"} or float, default "global_max"
            Reference for dB scaling (if intensity="db").
        """
        # Defaults
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

        # Build per-call frequency grid and compute amplitude scalograms
        F = self._build_frequency_grid(freq_scale, self.num_freqs)
        real_amp, F_used = self._compute_scalogram(real_signal, frequencies=F, return_freqs=True)
        synth_amp, _ = self._compute_scalogram(synthetic_signal, frequencies=F_used, return_freqs=True)

        # Per-image intensity (no averaging here)
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

        # Plot with pcolormesh (works for linear AND log frequency grids)
        fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharey=True, constrained_layout=True)

        pm0 = axs[0].pcolormesh(t, F_used, real_int, shading="auto", cmap="terrain", vmin=vmin, vmax=vmax)
        axs[0].set_title(f'Real signal {signal_index_real} scalogram', fontsize=20, fontname='Arial')
        axs[0].set_xlabel('Time (s)', fontsize=15, fontname='Arial')
        axs[0].set_ylabel('Frequency (Hz)', fontsize=15, fontname='Arial')
        if freq_scale == "log":
            axs[0].set_yscale("log")

        axs[1].pcolormesh(t, F_used, synth_int, shading="auto", cmap="terrain", vmin=vmin, vmax=vmax)
        axs[1].set_title(f'Synthetic signal {signal_index_synth} scalogram', fontsize=20, fontname='Arial')
        axs[1].set_xlabel('Time (s)', fontsize=15)
        if freq_scale == "log":
            axs[1].set_yscale("log")

        cbar = fig.colorbar(pm0, ax=axs, location='right')
        label = {"amplitude": "Amplitude", "power": "Power", "db": "Power (dB)"}[intensity]
        cbar.set_label(label, fontsize=14)

        if save:
            fig.savefig(save, bbox_inches="tight", dpi=200)
        plt.show()

        return fig

    def compute_scalogram_similarity_metrics(self,real_data,synthetic_data,*,mode: str = "all_vs_all",
                pad: bool = True,rr_zip_strategy: str = "consecutive", ss_zip_strategy: str = "consecutive"):
        """
        Compute similarity metrics between real and synthetic scalograms.

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
        real_data : array_like
            1D (T,) or 2D (N x T) real signals.
        synthetic_data : array_like
            1D (T,) or 2D (N x T) synthetic signals.
        mode : {"all_vs_all","zip"}, default "all_vs_all"
            Pairing strategy applied to RS, RR and SS.
        pad : bool, default True
            Right-pad signals so lengths match before CWT.
        rr_zip_strategy, ss_zip_strategy : {"consecutive","halves"}
            Strategy for within-set zip pairing.

        Returns
        -------
        dict
            Means, SDs, and per-pair lists for RS; summaries for RR and SS.
        """

        # Normalize to 2D
        R = np.asarray(real_data, dtype=float)
        S = np.asarray(synthetic_data, dtype=float)
        if R.ndim == 1: R = R[np.newaxis, :]
        if S.ndim == 1: S = S[np.newaxis, :]

        analysis_type = "sample" if R.shape[0] == 1 else "dataset"
        nR, nS = R.shape[0], S.shape[0]
        if nR == 0 or nS == 0:
            raise ValueError("Both real_signals and synthetic_signals must contain at least one signal.")

        # Optional right-padding to a common length
        if pad:
            L = int(max(R.shape[1], S.shape[1]))
            if R.shape[1] != L:
                R = np.pad(R, ((0, 0), (0, L - R.shape[1])))
            if S.shape[1] != L:
                S = np.pad(S, ((0, 0), (0, L - S.shape[1])))
        else:
            if R.shape[1] != S.shape[1]:
                raise ValueError("When pad=False, real and synthetic signals must have identical lengths.")

        # Precompute scalograms
        R_sc = [self._compute_scalogram(R[i]) for i in range(nR)]
        S_sc = [self._compute_scalogram(S[j]) for j in range(nS)]

        # Helpers
        def _pair_metrics(A, B):
            vmin = float(min(A.min(), B.min()))
            vmax = float(max(A.max(), B.max()))
            if vmax <= vmin:
                vmax = vmin + 1e-6
            Argb = self._convert_to_rgb(A, vmin=vmin, vmax=vmax)
            Brgb = self._convert_to_rgb(B, vmin=vmin, vmax=vmax)
            ssim_val = ssim(Argb, Brgb, channel_axis=2, data_range=255)
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
            """Return list of (i,j) for within-set RR/SS under zip strategy."""
            if zip_strategy not in {"consecutive", "halves"}:
                raise ValueError("zip_strategy must be 'consecutive' or 'halves'")
            pairs = []
            if zip_strategy == "consecutive":
                # (0,1), (2,3), (4,5), ...
                for k in range(0, n - 1, 2):
                    pairs.append((k, k + 1))
            else:  # halves
                half = n // 2
                for k in range(half):
                    pairs.append((k, k + half))
            return pairs

        # RS pairs
        rs_ssim, rs_nrmse, rs_cos = [], [], []
        if mode == "all_vs_all":
            for i in range(nR):
                Ai = R_sc[i]
                for j in range(nS):
                    sj, rj, cj = _pair_metrics(Ai, S_sc[j])
                    rs_ssim.append(sj); rs_nrmse.append(rj); rs_cos.append(cj)
        elif mode == "zip":
            N = min(nR, nS)
            for i in range(N):
                sj, rj, cj = _pair_metrics(R_sc[i], S_sc[i])
                rs_ssim.append(sj); rs_nrmse.append(rj); rs_cos.append(cj)
        else:
            raise ValueError("mode must be 'all_vs_all' or 'zip'")

        rs_ssim_m, rs_ssim_sd = _mean_sd(rs_ssim)
        rs_nrmse_m, rs_nrmse_sd = _mean_sd(rs_nrmse)
        rs_cos_m, rs_cos_sd = _mean_sd(rs_cos)

        # RR pairs
        rr_ssim, rr_nrmse, rr_cos = [], [], []
        if mode == "all_vs_all":
            # all unique pairs i<j
            for i in range(nR):
                A = R_sc[i]
                for j in range(i + 1, nR):
                    B = R_sc[j]
                    sj, rj, cj = _pair_metrics(A, B)
                    rr_ssim.append(sj); rr_nrmse.append(rj); rr_cos.append(cj)
        else:  # zip
            for (i, j) in _pairs_within_set(nR, rr_zip_strategy):
                sj, rj, cj = _pair_metrics(R_sc[i], R_sc[j])
                rr_ssim.append(sj); rr_nrmse.append(rj); rr_cos.append(cj)

        rr_ssim_m, rr_ssim_sd = _mean_sd(rr_ssim)
        rr_nrmse_m, rr_nrmse_sd = _mean_sd(rr_nrmse)
        rr_cos_m, rr_cos_sd = _mean_sd(rr_cos)

        #  SS pairs
        ss_ssim, ss_nrmse, ss_cos = [], [], []
        if mode == "all_vs_all":
            for i in range(nS):
                A = S_sc[i]
                for j in range(i + 1, nS):
                    B = S_sc[j]
                    sj, rj, cj = _pair_metrics(A, B)
                    ss_ssim.append(sj); ss_nrmse.append(rj); ss_cos.append(cj)
        else:  # zip
            for (i, j) in _pairs_within_set(nS, ss_zip_strategy):
                sj, rj, cj = _pair_metrics(S_sc[i], S_sc[j])
                ss_ssim.append(sj); ss_nrmse.append(rj); ss_cos.append(cj)

        ss_ssim_m, ss_ssim_sd = _mean_sd(ss_ssim)
        ss_nrmse_m, ss_nrmse_sd = _mean_sd(ss_nrmse)
        ss_cos_m, ss_cos_sd = _mean_sd(ss_cos)

        # Prints
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


        # Return
        scalogram_similarity_metrics = {
            "Analysis type": analysis_type,
            "RS mode": mode,
            "RR zip strategy": rr_zip_strategy if mode == "zip" else "all_vs_all",
            "SS zip strategy": ss_zip_strategy if mode == "zip" else "all_vs_all",
            # RS per-pair lists (kept for downstream uses)
            "Per-pair SSIM (RS)": rs_ssim,
            "Per-pair NRMSE (RS)": rs_nrmse,
            "Per-pair Cosine (RS)": rs_cos,
            # Means/SDs
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
                "Cosine mean": rs_cos_m, "Cosine SD": rs_cos_sd}
        }

        return scalogram_similarity_metrics

    def compute_mean_scalogram(self, signals, *, frequencies: np.ndarray | None = None):
        """
        Compute the **mean scalogram** across a set of signals.

        Accepts:
          - 1D array (single signal)
          - 2D array (N_signals × T)
          - list/tuple of 1D arrays (all with the same length)

        Returns
        -------
        np.ndarray
            Mean scalogram (freqs × time).
        """
        if signals is None:
            raise ValueError("`signals` cannot be None.")

            # Normalize input to list of 1D arrays
        if isinstance(signals, (list, tuple)):
            arr_list = [np.asarray(x).reshape(-1) for x in signals]
        else:
            arr = np.asarray(signals)
            if arr.ndim == 1:
                arr_list = [arr]
            elif arr.ndim == 2:
                arr_list = [arr[i] for i in range(arr.shape[0])]
            else:
                raise ValueError("`signals` must be 1D, 2D, or a list of 1D arrays.")

        if len(arr_list) == 0:
            raise ValueError("`signals` is empty.")

        mean_scalo = None
        F_used = None
        count = 0
        for sig in arr_list:
            sc, F_used = self._compute_scalogram(sig, frequencies=frequencies, return_freqs=True)
            if mean_scalo is None:
                mean_scalo = np.zeros_like(sc, dtype=float)
            elif sc.shape != mean_scalo.shape:
                raise ValueError("All signals must have the same length (scalogram shapes must match).")
            mean_scalo += sc
            count += 1

        return mean_scalo / max(count, 1), F_used

    def plot_mean_scalograms(self, real_data, synthetic_data, *, save: str | None = None,
                             titles=("Mean real", "Mean synthetic"),
                             freq_scale: str | None = None,
                             intensity: str | None = None,
                             db_ref: str | float | None = None):
        """
        Compute and plot the mean scalogram of real and synthetic sets, side-by-side.

        Accepts 1D, 2D, or list of 1D arrays for each input.

        New (optional) per-call overrides:
        ----------------------------------
        freq_scale : {"linear","log"}, default "linear"
            Frequency grid spacing for the plot.
        intensity : {"amplitude","power","db"}, default "amplitude"
            What to display for color intensity.
        db_ref : {"global_max","per_image_max"} or float, default "global_max"
            Reference for dB scaling (if intensity="db").

        Important:
        ----------
        For this mean plot, the pipeline is **mean first, then transform**:
        we average in the **amplitude** domain and only then apply power or dB.
        """
        # Defaults
        freq_scale = freq_scale or "linear"
        intensity = intensity or "amplitude"
        db_ref = "global_max" if db_ref is None else db_ref

        # Build per-call frequency grid
        F = self._build_frequency_grid(freq_scale, self.num_freqs)

        # Mean FIRST in amplitude domain
        real_mean_amp, F_used = self.compute_mean_scalogram(real_data, frequencies=F)
        synth_mean_amp, _ = self.compute_mean_scalogram(synthetic_data, frequencies=F_used)

        # Shared dB reference if needed
        eps = 1e-12
        ref = None
        if intensity == "db" and db_ref == "global_max":
            ref = max(np.max(real_mean_amp ** 2), np.max(synth_mean_amp ** 2)) + eps

        # Convert intensity AFTER mean
        real_int = self._intensity_from_mean_amp(real_mean_amp, intensity=intensity, db_ref=db_ref, ref_value=ref)
        synth_int = self._intensity_from_mean_amp(synth_mean_amp, intensity=intensity, db_ref=db_ref, ref_value=ref)

        vmin = min(float(np.nanmin(real_int)), float(np.nanmin(synth_int)))
        vmax = max(float(np.nanmax(real_int)), float(np.nanmax(synth_int)))

        # Time axis
        real = np.asarray(real_data)
        if real.ndim == 2:
            T = real.shape[1]
        elif real.ndim == 1:
            T = real.shape[0]
        else:
            # list/tuple assumed same length
            T = np.asarray(real_data[0]).shape[0]
        total_duration = T / self.fs if T > 0 else 0.0
        t = np.linspace(0, total_duration, real_int.shape[1])

        fig, axs = plt.subplots(1, 2, figsize=(15, 5), sharey=True, constrained_layout=True)
        pm0 = axs[0].pcolormesh(t, F_used, real_int, shading="auto", cmap="terrain", vmin=vmin, vmax=vmax)
        axs[0].set_title(f'{titles[0]} scalogram', fontsize=20, fontname='Arial')
        axs[0].set_xlabel('Time (s)', fontsize=15, fontname='Arial')
        axs[0].set_ylabel('Frequency (Hz)', fontsize=15, fontname='Arial')
        if freq_scale == "log":
            axs[0].set_yscale("log")

        axs[1].pcolormesh(t, F_used, synth_int, shading="auto", cmap="terrain", vmin=vmin, vmax=vmax)
        axs[1].set_title(f'{titles[1]} scalogram', fontsize=20, fontname='Arial')
        axs[1].set_xlabel('Time (s)', fontsize=15, fontname='Arial')
        if freq_scale == "log":
            axs[1].set_yscale("log")

        cbar = fig.colorbar(pm0, ax=axs, location='right')
        label = {"amplitude": "Amplitude", "power": "Power", "db": "Power (dB)"}[intensity]
        cbar.set_label(label, fontsize=15, fontname='Arial')

        if save:
            plt.savefig(save, bbox_inches='tight', dpi=200)
        plt.show()
        return fig

    # Burst statistics helpers
    def _band_mask(self, F: np.ndarray, band: tuple[float, float]) -> np.ndarray:
        """NEW: Boolean mask for frequencies inside a band (f_low, f_high)."""
        f_lo, f_hi = float(band[0]), float(band[1])
        if f_lo <= 0 or f_hi <= f_lo:
            raise ValueError("Invalid band. Use (f_low, f_high) with 0 < f_low < f_high.")
        return (F >= f_lo) & (F <= f_hi)

    def _band_envelope_from_scalogram(self, amp: np.ndarray, F: np.ndarray, band: tuple[float, float],
                                      smooth_ms: float = 50.0, normalize: bool = True) -> np.ndarray:
        """
        Band-limited envelope from scalogram by averaging amplitude over band rows.
        Optionally smooth with a moving average (smooth_ms).
        """
        mask = self._band_mask(F, band)
        if not np.any(mask):
            raise ValueError(f"No scalogram rows fall inside the requested band {band}. "
                             "Consider increasing num_freqs or adjusting band.")
        # Average amplitude over the band (freqs x time -> time)
        env = np.mean(amp[mask, :], axis=0)

        # Normalize by baseline (e.g., median) to handle amplitude differences
        if normalize:
            baseline = np.median(env)
            if baseline > 0:
                env = env / baseline

        # Optional moving-average smoothing in samples
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
        """
        Event-level burst detection on a 1D envelope.

        threshold: "percentile" (use p-th percentile) OR "std" (mean + kappa*std).
        p:         percentile (if threshold == "percentile"), e.g., 75.
        kappa:     multiplier for std (if threshold == "std"), e.g., 1.5.
        min_duration_ms:  discard events shorter than this.
        merge_gap_ms:      merge two events if the gap between them is below this.
        """
        x = np.asarray(envelope, dtype=float)
        if x.ndim != 1:
            raise ValueError("Envelope must be a 1D array.")

        # Threshold
        if threshold == "percentile":
            thr = np.percentile(x, float(p))
        elif threshold == "std":
            if kappa is None:
                kappa = 1.5
            thr = float(np.mean(x) + float(kappa) * np.std(x))
        else:
            raise ValueError("threshold must be 'percentile' or 'std'.")

        above = (x >= thr).astype(np.int8)

        # Find runs above threshold
        # Transitions
        d = np.diff(np.r_[0, above, 0])
        starts = np.flatnonzero(d == 1)
        ends   = np.flatnonzero(d == -1) - 1

        # Enforce minimum duration
        min_samples = int(round((min_duration_ms / 1000.0) * self.fs))
        keep = []
        for s, e in zip(starts, ends):
            if (e - s + 1) >= max(1, min_samples):
                keep.append((s, e))
        events = keep

        # Merge close events
        merged = []
        if events:
            gap_samples = int(round((merge_gap_ms / 1000.0) * self.fs))
            cur_s, cur_e = events[0]
            for s, e in events[1:]:
                if s - cur_e - 1 <= gap_samples:
                    cur_e = e  # merge
                else:
                    merged.append((cur_s, cur_e))
                    cur_s, cur_e = s, e
            merged.append((cur_s, cur_e))
        events = merged

        # Collect stats
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

        # Inter-burst intervals (IBI)
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
                                 threshold="std", #percentile
                                 p=75.0,
                                 kappa=2.0, #None
                                 min_duration_ms=20.0,
                                 merge_gap_ms=50.0,
                                 freq_scale: str | None = None,
                                 smooth_ms: float = 20.0,
                                 verbose: bool = True):
        """
        Compute burst statistics for real and synthetic datasets using the scalogram.
        Returns per-signal summaries and RS/RR/SS distances for each feature.

        verbose: if True, prints a compact summary of results.  # >>> NEW <<<
        """
        from scipy.stats import wasserstein_distance as WD
        import numpy as np

        # Normalize inputs to 2D arrays
        R = np.asarray(real_data, dtype=float)
        S = np.asarray(synthetic_data, dtype=float)
        if R.ndim == 1: R = R[np.newaxis, :]
        if S.ndim == 1: S = S[np.newaxis, :]

        # Frequency grid for scalograms
        F = self._build_frequency_grid(freq_scale or "linear", self.num_freqs)

        # Per-signal summaries
        def _one_set(signals):
            out = []
            for x in signals:
                amp, F_used = self._compute_scalogram(x, frequencies=F, return_freqs=True)
                env = self._band_envelope_from_scalogram(amp, F_used, band, smooth_ms=smooth_ms)
                _, summary, thr = self._detect_bursts_from_envelope(
                    env, threshold=threshold, p=p, kappa=kappa,
                    min_duration_ms=min_duration_ms, merge_gap_ms=merge_gap_ms
                )
                out.append(summary)
            return out

        R_sum = _one_set(R)
        S_sum = _one_set(S)

        # Turn dict lists into arrays per feature
        def _stack_feature(L, key):
            vals = [d[key] for d in L if np.isfinite(d[key])]
            return np.array(vals, dtype=float) if len(vals) > 0 else np.array([], dtype=float)

        keys = [
            "n_bursts", "rate_hz", "mean_duration_s", "median_duration_s",
            "mean_peak_amp", "median_peak_amp", "mean_ibi_s",
            "median_ibi_s", "duty_cycle"
        ]

        # Helper: z-score using pooled mean/std over real+synth
        # WD<0.1SD almost identical dist; 0.1-0.3SD small shift;0.3-0.5SD moderate shift;0.5-0.7SD large shift; 0.7-1 SD very different
        def _zscore_using_real(r, s):
            std = np.std(r)
            if std == 0 or not np.isfinite(std):
                return np.zeros_like(r), np.zeros_like(s)

            mean = np.mean(r)
            r_z = (r - mean) / std
            s_z = (s - mean) / std
            return r_z, s_z

        # Precompute WD and means
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
                # Normalization using real SD
                real_std = np.std(r) + 1e-12
                wd_rs[k] = float(WD(r, s) / real_std)

            else:
                wd_rs[k] = np.nan

        # Robust SD (1.4826 * MAD) – using raw scale
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

