import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy.signal import detrend
import fathon
from fathon import fathonUtils as fu
import neurokit2 as nk
from scipy.stats import wasserstein_distance


class ComplexityFidelity:
    """
    A class for computing complexity similarity between real and synthetic signals using different
    fractal analysis methods and entropy measures.

    Implemented metrics:
    --------
    - DCCA (Detrended Cross-Correlation Analysis): Measures the correlation
      between two signals across different scales.
    - MFDFA (Multifractal Detrended Fluctuation Analysis): Computes the
      multifractal spectrum of a single signal.
    - FDCCA (Multifractal Detrended Cross-Correlation Analysis): Extends
      DCCA with multifractal properties by analyzing cross-fluctuations F_xy(q).
    - Entropy metrics: Sample Entropy, Permutation Entropy, Lempel-Ziv Complexity

    We store:
      - means, std: the average cross-H exponent (or single-series H).
      - rho_means, rho_stds: correlation coefficients (for DCCA).
      - Fq_means, Fq_stds: the average cross-fluctuation (for MFDCCA).
      - deltaAlpha_means, deltaAlpha_stds: width of the multifractal singularity spectrum.
      - p_means, p_stds: the MFDCCA cross-correlation ratio, averaged over q.

    Example Usage:
    --------------
    # Generate white noise data (H=0.5)
    real_data = [np.random.randn(1000) for _ in range(5)]

    # Generate pink noise data (H=1)
    synthetic_data = [np.fft.irfft(np.fft.rfft(np.random.randn(1000)) / np.sqrt(np.fft.rfftfreq(1000) + 1e-10)) for _ in range(5)]

    # Compute complexity similarity using DCCA
    cf = ComplexityFidelity(real_data, synthetic_data, method='DCCA')
    cf.compute_fractal_metrics()
    cf.plot_metrics()

    # Compute complexity similarity using MFDFA
    cf = ComplexityFidelity(real_data, synthetic_data, method='MFDFA')
    cf.compute_fractal_metrics()

    # Compute complexity similarity using MFDCCA
    cf = ComplexityFidelity(real_data, synthetic_data, method='MFDCCA')
    cf.compute_fractal_metrics()
    cf.plot_metrics()

    # Compute entropy metrics
    e_results = cf.compute_entropy_complexity_metrics(real_data, synthetic_data)

    References:
    -----------
    * Podobnik & Stanley (2008) for DCCA theory https://arxiv.org/pdf/0709.0281
    * Kantelhardt et al. (2002) for MFDFA.
    * Gu & Zhou (2010) for MFDCCA.
    * https://www.sciencedirect.com/science/article/pii/S2212017313006506
    """

    def __init__(self, real_data, synthetic_data, method='DCCA', q_range=np.arange(-5,5,0.1), n_jobs=-1):
        """
        Initializes the ComplexityFidelity class with real and synthetic signal data.

        Parameters
        ----------
        real_data : array
            1D (T,) or 2D (N x T) real signals.
        synthetic_data : array
            1D (T,) or 2D (N x T) synthetic signals.

        method : str, optional (default='DCCA')
            The fractal analysis method to use. Options:
              - 'DCCA' (Detrended Cross-Correlation Analysis)
              - 'MFDFA' (Multifractal Detrended Fluctuation Analysis)
              - 'MFDCCA' (Multifractal Detrended Cross-Correlation Analysis)
        q_range : np.ndarray, optional
            Range of q values for multifractal analysis (used in MFDFA, MFDCCA, etc.).
            Default is np.arange(-5,5,0.1).
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.method = method
        self.q_range = q_range
        self.n_jobs = n_jobs

        # For labeling bar plots (DCCA or MFDCCA)
        self.categories = ['real vs real', 'real vs synthetic', 'synthetic vs synthetic']

        # Basic placeholders for exponents or correlation measures
        self.means = []  # Typically the H exponents
        self.stds = []
        self.rho_means = None
        self.rho_stds = None

        # For MFDFA or MFDCCA
        self.Fq_means = None
        self.Fq_stds = None
        self.deltaAlpha_means = None
        self.deltaAlpha_stds = None
        self.Hq_distance_means = None
        self.Hq_distance_stds = None

        # For MFDCCA cross-correlation ratio p(q)
        self.p_means = None
        self.p_stds = None

    def compute_fractal_metrics(self):
        """
        Perform the selected fractal analysis method and produce relevant
        results. Calls one of the internal methods based on self.method.
        """
        if self.method == 'DCCA':
            self._dcca_analyze()
            self._print_results()

        elif self.method == 'MFDFA':
            self._mfdfa_analyze()
            self._print_results()
            # Typically MFDFA is single-series, so no cross-correlation plot

        elif self.method == 'MFDCCA':
            self._mfdcca_analyze()
            self._print_results()

        else:
            raise ValueError("Invalid method. Choose 'DCCA', 'MFDFA' or 'MFDCCA'")

    def plot_metrics(self):
        """
        Produce plots for the selected fractal analysis method.
        """
        if self.method == 'DCCA':
            # 1 row × 2 columns: Hurst and rho
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            self.plot_hurst_correlation(ax=axs[0])
            self.plot_rho_correlation(ax=axs[1])
            plt.tight_layout()
            plt.show()

        elif self.method == 'MFDFA':
            raise ValueError("Invalid method. This method does not produce plots.")

        elif self.method == 'MFDCCA':
            # 2 × 2: Hurst, p(q), Fxy, Δα
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            self.plot_hurst_correlation(ax=axs[0, 0])
            self.plot_p_correlation(ax=axs[0, 1])
            self.plot_Fq_correlation(ax=axs[1, 0])
            self.plot_deltaAlpha(ax=axs[1, 1])
            plt.tight_layout()
            plt.show()
        else:
            raise ValueError("Invalid method. Choose 'DCCA' or 'MFDCCA'")



    # Plotting

    def _set_plot_style(self):
        """Apply Arial fonts and journal-compliant sizes to the next figure."""
        mpl.rcParams.update({
            "font.family": "Arial",
            "font.size": 15,  # default for everything below title
            "axes.titlesize": 20,
            "axes.labelsize": 15,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
        })

    def plot_hurst_correlation(self, ax=None):
        """Plot the average cross Hurst exponents for each pair category, with error bars."""
        if not self.means:
            print("No H exponents to plot.")
            return

        self._set_plot_style()

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
            created_fig = True

        sns.barplot(
            x=self.categories,
            y=self.means,
            capsize=0.2,
            hue=self.categories,
            palette=['lightskyblue', 'limegreen', 'grey'],
            legend=False,
            ax=ax
        )

        # errorbar uses numeric x positions 0..N-1
        ax.errorbar(
            x=range(len(self.categories)),
            y=self.means,
            yerr=self.stds,
            fmt='none',
            capsize=5,
            color='black'
        )

        ax.set_ylabel('Mean cross Hurst exponent ($\\overline{H}_{xy}$), $q \\in [-5, 5]$')
        ax.set_title(rf'{self.method} mean cross Hurst exponent $(\overline{{H}}_{{xy}})$')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        if created_fig:
            plt.tight_layout()
            plt.show()

    def plot_rho_correlation(self, ax=None):
        """Bar plot of correlation coefficients (rho) for DCCA."""
        if not self.rho_means:
            print("No correlation coefficients available to plot.")
            return

        self._set_plot_style()

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
            created_fig = True

        sns.barplot(
            x=self.categories,
            y=self.rho_means,
            capsize=0.2,
            hue=self.categories,
            palette=['lightskyblue', 'limegreen', 'grey'],
            legend=False,
            ax=ax
        )

        # again, numeric x positions 0..N-1
        ax.errorbar(
            x=range(len(self.categories)),
            y=self.rho_means,
            yerr=self.rho_stds,
            fmt='none',
            capsize=5,
            color='black'
        )

        ax.set_ylabel(r'Mean correlation coefficients ($\bar{\rho}$)')
        ax.set_title(rf'{self.method} mean correlation coefficients ($\bar{{\rho}}$)')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        if created_fig:
            plt.tight_layout()
            plt.show()

    def plot_p_correlation(self, ax=None):
        """Bar plot of the mean p(q) for MFDCCA cross-correlation if computed."""
        if not self.p_means:
            print("No p(q) values available to plot.")
            return

        self._set_plot_style()

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
            created_fig = True

        sns.barplot(
            x=self.categories,
            y=self.p_means,
            capsize=0.2,
            hue=self.categories,
            palette=['lightskyblue', 'limegreen', 'grey'],
            legend=False,
            ax=ax
        )
        ax.errorbar(
            x=range(len(self.categories)),
            y=self.p_means,
            yerr=self.p_stds,
            fmt='none',
            capsize=5,
            color='black'
        )
        ax.set_ylabel(r'Mean cross-correlation $\langle p(q)\rangle$, $q \in [-5, 5]$')
        ax.set_title(r'MFDCCA mean cross-correlation $\langle p(q)\rangle$')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        if created_fig:
            plt.tight_layout()
            plt.show()

    def plot_Fq_correlation(self, ax=None):
        """Bar plot of the mean Fxy(q) for MFDCCA cross-fluctuation if computed."""
        if not self.Fq_means:
            print("No F_xy(q) values to plot.")
            return

        self._set_plot_style()

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
            created_fig = True

        sns.barplot(
            x=self.categories,
            y=self.Fq_means,
            capsize=0.2,
            hue=self.categories,
            palette=['lightskyblue', 'limegreen', 'grey'],
            legend=False,
            ax=ax
        )
        ax.errorbar(
            x=range(len(self.categories)),
            y=self.Fq_means,
            yerr=self.Fq_stds,
            fmt='none',
            capsize=5,
            color='black'
        )
        ax.set_ylabel(r'Mean cross-fluctuation $\langle F_{xy}(q) \rangle$, $q \in [-5, 5]$')
        ax.set_title(r'MFDCCA mean cross-fluctuation $\langle F_{xy}(q) \rangle$')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        if created_fig:
            plt.tight_layout()
            plt.show()

    def plot_deltaAlpha(self, ax=None):
        """Bar plot of the multifractal spectrum width Δα."""
        if not self.deltaAlpha_means:
            print("No Δα values to plot.")
            return

        self._set_plot_style()

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
            created_fig = True

        sns.barplot(
            x=self.categories,
            y=self.deltaAlpha_means,
            capsize=0.2,
            hue=self.categories,
            palette=['lightskyblue', 'limegreen', 'grey'],
            legend=False,
            ax=ax
        )
        ax.errorbar(
            x=range(len(self.categories)),
            y=self.deltaAlpha_means,
            yerr=self.deltaAlpha_stds,
            fmt='none',
            capsize=5,
            color='black'
        )
        ax.set_ylabel(r'Spectrum width ($\Delta\alpha$), $q \in [-5, 5]$')
        ax.set_title('MFDCCA spectrum width (Δα)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        if created_fig:
            plt.tight_layout()
            plt.show()


    # Helper methods

    def _preprocess_signals(self, signals):
        """
        Preprocess signals by detrending only (assuming they're already normalized).

        Skips signals containing NaN/inf. Returns a list of valid detrended signals.
        """
        out = []
        for i, sig in enumerate(signals):
            if sig is None:
                continue
            if np.isnan(sig).any() or np.isinf(sig).any():
                print(f"Skipping signal {i+1} due to NaN or infinite values.")
                continue
            out.append(detrend(sig))
        return out

    def _get_win_sizes(self, length, n_scales=10):
        """
        Generate a set of log-spaced window sizes for fractal analysis.
        min_scale=10, max_scale=min(length//4,100), total of n_scales steps.
        """
        min_scale = 10
        max_scale = min(length//4, 100)
        if max_scale <= min_scale:
            return np.array([])
        sizes = np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales).astype(int)
        return np.unique(sizes[sizes < length])

    def _print_results(self):
        """
        Print the mean values for the three pairings (or two categories in MFDFA).
        """
        if self.method == 'DCCA':
            print("\nDCCA Hurst Exponent (Hxy):")
            for cat, m, s in zip(self.categories, self.means, self.stds):
                print(f"  {cat}: {m:.4f} ± {s:.4f}")

            if self.rho_means is not None:
                print("\nDCCA Cross-Correlation Coefficient (ρ) :")
                for cat, m, s in zip(self.categories, self.rho_means, self.rho_stds):
                    print(f"  {cat}: {m:.4f} ± {s:.4f}")
            return

        print(f"\n{self.method} Hurst Exponent:")
        for category, mean_val, std_val in zip(self.categories, self.means, self.stds):
            print(f"  {category}: Mean = {mean_val:.4f}, Std = {std_val:.4f}")


    # DCCA

    def _dcca_analyze(self):
        """
        Perform Detrended Cross-Correlation Analysis (DCCA) between real and synthetic signals.
        """
        pre_r = self._preprocess_signals(self.real_data)
        pre_s = self._preprocess_signals(self.synthetic_data)

        H_rr, H_rs, H_ss = [], [], []
        rho_rr, rho_rs, rho_ss = [], [], []

        # real-real pairs
        for i in range(len(pre_r)):
            for j in range(i+1, len(pre_r)):
                a = fu.toAggregated(pre_r[i])
                b = fu.toAggregated(pre_r[j])
                dcca = fathon.DCCA(a, b)
                wins = self._get_win_sizes(len(a))
                if len(wins) < 4:
                    continue

                # Compute fluc + fit
                dcca.computeFlucVec(wins, polOrd=1)
                H, _ = dcca.fitFlucVec()
                H_rr.append(H)

                # correlation
                def _mean_rho(rho_vals):
                    rho_vals = np.asarray(rho_vals)
                    rho_vals = rho_vals[np.isfinite(rho_vals)]
                    if rho_vals.size == 0:
                        return np.nan
                    z = np.arctanh(np.clip(rho_vals, -0.999999, 0.999999))
                    return np.tanh(np.nanmean(z))


                rho = dcca.computeRho(wins, polOrd=1)
                rho_rr.append(_mean_rho(rho[1]))

        # synthetic-synthetic
        for i in range(len(pre_s)):
            for j in range(i+1, len(pre_s)):
                a = fu.toAggregated(pre_s[i])
                b = fu.toAggregated(pre_s[j])
                dcca = fathon.DCCA(a, b)
                wins = self._get_win_sizes(len(a))
                if len(wins)<4:
                    continue

                dcca.computeFlucVec(wins, polOrd=1)
                H, _ = dcca.fitFlucVec()
                H_ss.append(H)

                rho = dcca.computeRho(wins, polOrd=1)
                rho_ss.append(_mean_rho(rho[1]))

        # real-synthetic
        for xr in pre_r:
            for xs in pre_s:
                a = fu.toAggregated(xr)
                b = fu.toAggregated(xs)
                dcca = fathon.DCCA(a, b)
                wins = self._get_win_sizes(len(a))
                if len(wins)<4:
                    continue

                dcca.computeFlucVec(wins, polOrd=1)
                H, _ = dcca.fitFlucVec()
                H_rs.append(H)

                rho = dcca.computeRho(wins, polOrd=1)
                rho_rs.append(_mean_rho(rho[1]))

        # fallback if empty
        if not H_rr: H_rr=[np.nan]; rho_rr=[np.nan]
        if not H_rs: H_rs=[np.nan]; rho_rs=[np.nan]
        if not H_ss: H_ss=[np.nan]; rho_ss=[np.nan]

        # store
        self.means = [np.nanmean(H_rr), np.nanmean(H_rs), np.nanmean(H_ss)]
        self.stds  = [np.nanstd(H_rr), np.nanstd(H_rs), np.nanstd(H_ss)]
        self.rho_means = [np.nanmean(rho_rr), np.nanmean(rho_rs), np.nanmean(rho_ss)]
        self.rho_stds  = [np.nanstd(rho_rr), np.nanstd(rho_rs), np.nanstd(rho_ss)]


    # MFDFA

    def _mfdfa_analyze(self):
        """
        Perform Multifractal Detrended Fluctuation Analysis (MFDFA) on each dataset
        separately. Typically we measure single-series H values.
        """
        pre_r = self._preprocess_signals(self.real_data)
        pre_s = self._preprocess_signals(self.synthetic_data)

        H_r = []
        H_s = []

        for sig in pre_r:
            _, info = nk.fractal_dfa(sig, scale=self._get_win_sizes(len(sig)),
                                     multifractal=True, q=self.q_range, show=False)
            H_r.append(info['H'])

        for sig in pre_s:
            _, info = nk.fractal_dfa(sig, scale=self._get_win_sizes(len(sig)),
                                     multifractal=True, q=self.q_range, show=False)
            H_s.append(info['H'])

        if not H_r: H_r=[np.nan]
        if not H_s: H_s=[np.nan]

        self.means = [np.nanmean(H_r), np.nanmean(H_s)]
        self.stds  = [np.nanstd(H_r),  np.nanstd(H_s)]
        # For MFDFA, we only have two categories, so override:
        self.categories = ['real','synthetic']


    # MFDCCA

    def _compute_delta_alpha(self, q_vals, H_vals):
        """Compute width of multifractal spectrum: Δα = α_max - α_min, where α(q)=H(q)+q*dH/dq."""
        dH_dq = np.gradient(H_vals, q_vals)
        alpha_vals = H_vals + q_vals * dH_dq
        return np.nanmax(alpha_vals) - np.nanmin(alpha_vals)

    def _computeFq_mfdfa(self, x):
        """Compute single-series Fq array (MFDFA) for use in MFDCCA p(q)."""
        a = fu.toAggregated(x)
        pm = fathon.MFDFA(a)
        wins = self._get_win_sizes(len(a))
        if len(wins)<4:
            return np.full(len(self.q_range), np.nan)
        _,Fq = pm.computeFlucVec(wins, self.q_range, polOrd=1)
        return Fq

    def _mfdcca_analyze(self):
        """
        Perform Multifractal Detrended Cross-Correlation Analysis (MFDCCA).
        1) cross-fluctuation F_xy(q).
        2) single-series F_x(q), F_y(q).
        3) p(q)=F_xy(q)/sqrt(F_x(q)*F_y(q)) => average over q.

        Also store Δα (width of singularity) and the average H(q).
        """
        pre_r = self._preprocess_signals(self.real_data)
        pre_s = self._preprocess_signals(self.synthetic_data)

        # We'll store cross-H in lists
        H_rr, H_rs, H_ss = [], [], []
        Fq_rr, Fq_rs, Fq_ss = [], [], []
        Da_rr, Da_rs, Da_ss = [], [], []
        p_rr, p_rs, p_ss = [], [], []

        # real-real
        for i in range(len(pre_r)):
            for j in range(i+1, len(pre_r)):
                try:
                    a = fu.toAggregated(pre_r[i])
                    b = fu.toAggregated(pre_r[j])
                    mm = fathon.MFDCCA(a,b)
                    wins = self._get_win_sizes(min(len(a),len(b)))
                    if len(wins)<4:
                        continue

                    _,Fxy = mm.computeFlucVec(wins, self.q_range, polOrd=1)
                    Hxy,_ = mm.fitFlucVec()

                    Fx = self._computeFq_mfdfa(pre_r[i])
                    Fy = self._computeFq_mfdfa(pre_r[j])
                    eps=1e-12
                    p_of_q = Fxy/(np.sqrt(Fx*Fy)+eps)

                    H_rr.append(np.nanmean(Hxy))
                    Fq_rr.append(np.nanmean(Fxy))
                    Da_rr.append(self._compute_delta_alpha(self.q_range,Hxy))
                    p_rr.append(np.nanmean(p_of_q))
                except:
                    pass

        # synthetic-synthetic
        for i in range(len(pre_s)):
            for j in range(i+1, len(pre_s)):
                try:
                    a = fu.toAggregated(pre_s[i])
                    b = fu.toAggregated(pre_s[j])
                    mm = fathon.MFDCCA(a,b)
                    wins = self._get_win_sizes(min(len(a),len(b)))
                    if len(wins)<4:
                        continue

                    _,Fxy = mm.computeFlucVec(wins, self.q_range, polOrd=1)
                    Hxy,_ = mm.fitFlucVec()

                    Fx = self._computeFq_mfdfa(pre_s[i])
                    Fy = self._computeFq_mfdfa(pre_s[j])
                    eps=1e-12
                    p_of_q = Fxy/(np.sqrt(Fx*Fy)+eps)

                    H_ss.append(np.nanmean(Hxy))
                    Fq_ss.append(np.nanmean(Fxy))
                    Da_ss.append(self._compute_delta_alpha(self.q_range,Hxy))
                    p_ss.append(np.nanmean(p_of_q))
                except:
                    pass

        # real-synthetic
        for xr in pre_r:
            for xs in pre_s:
                try:
                    a = fu.toAggregated(xr)
                    b = fu.toAggregated(xs)
                    mm = fathon.MFDCCA(a,b)
                    wins = self._get_win_sizes(min(len(a),len(b)))
                    if len(wins)<4:
                        continue

                    _,Fxy = mm.computeFlucVec(wins, self.q_range, polOrd=1)
                    Hxy,_ = mm.fitFlucVec()

                    Fx = self._computeFq_mfdfa(xr)
                    Fy = self._computeFq_mfdfa(xs)
                    eps=1e-12
                    p_of_q = Fxy/(np.sqrt(Fx*Fy)+eps)

                    H_rs.append(np.nanmean(Hxy))
                    Fq_rs.append(np.nanmean(Fxy))
                    Da_rs.append(self._compute_delta_alpha(self.q_range,Hxy))
                    p_rs.append(np.nanmean(p_of_q))
                except:
                    pass

        # fallback if lists empty
        if not H_rr: H_rr=[np.nan]; Fq_rr=[np.nan]; Da_rr=[np.nan]; p_rr=[np.nan]
        if not H_rs: H_rs=[np.nan]; Fq_rs=[np.nan]; Da_rs=[np.nan]; p_rs=[np.nan]
        if not H_ss: H_ss=[np.nan]; Fq_ss=[np.nan]; Da_ss=[np.nan]; p_ss=[np.nan]

        # store results
        self.means = [np.nanmean(H_rr), np.nanmean(H_rs), np.nanmean(H_ss)]
        self.stds  = [np.nanstd(H_rr), np.nanstd(H_rs), np.nanstd(H_ss)]

        self.Fq_means = [np.nanmean(Fq_rr), np.nanmean(Fq_rs), np.nanmean(Fq_ss)]
        self.Fq_stds  = [np.nanstd(Fq_rr), np.nanstd(Fq_rs), np.nanstd(Fq_ss)]

        self.deltaAlpha_means = [np.nanmean(Da_rr), np.nanmean(Da_rs), np.nanmean(Da_ss)]
        self.deltaAlpha_stds  = [np.nanstd(Da_rr), np.nanstd(Da_rs), np.nanstd(Da_ss)]

        self.p_means = [np.nanmean(p_rr), np.nanmean(p_rs), np.nanmean(p_ss)]
        self.p_stds  = [np.nanstd(p_rr), np.nanstd(p_rs), np.nanstd(p_ss)]

        print(f"\n{self.method} Cross-Hurst Exponent (avg H(q)):")
        print(f"  real vs real:        {self.means[0]:.4f} ± {self.stds[0]:.4f}")
        print(f"  real vs synthetic:   {self.means[1]:.4f} ± {self.stds[1]:.4f}")
        print(f"  synthetic vs synthetic: {self.means[2]:.4f} ± {self.stds[2]:.4f}")

        print("\nMFDCCA Cross-Fluctuation F_xy(q) [averaged]:")
        print(f"  real vs real:        {self.Fq_means[0]:.4f} ± {self.Fq_stds[0]:.4f}")
        print(f"  real vs synthetic:   {self.Fq_means[1]:.4f} ± {self.Fq_stds[1]:.4f}")
        print(f"  synthetic vs synthetic: {self.Fq_means[2]:.4f} ± {self.Fq_stds[2]:.4f}")

        print("\nMFDCCA Width of Singularity Spectrum (Δα):")
        print(f"  real vs real:        {self.deltaAlpha_means[0]:.4f} ± {self.deltaAlpha_stds[0]:.4f}")
        print(f"  real vs synthetic:   {self.deltaAlpha_means[1]:.4f} ± {self.deltaAlpha_stds[1]:.4f}")
        print(f"  synthetic vs synthetic: {self.deltaAlpha_means[2]:.4f} ± {self.deltaAlpha_stds[2]:.4f}")

        print("\nMFDCCA Cross-Correlation Ratio p(q) [averaged over q]:")
        print(f"  real vs real:        {self.p_means[0]:.4f} ± {self.p_stds[0]:.4f}")
        print(f"  real vs synthetic:   {self.p_means[1]:.4f} ± {self.p_stds[1]:.4f}")
        print(f"  synthetic vs synthetic: {self.p_means[2]:.4f} ± {self.p_stds[2]:.4f}")

    @staticmethod
    def _sample_entropy(x: np.ndarray, m: int = 2, r: float | None = None) -> float:
        """
        Sample Entropy (SampEn). r defaults to 0.2*std(x) if None.
        """
        x = np.asarray(x, dtype=float).ravel()
        n = x.size
        if n <= m + 1:
            return np.nan
        if r is None:
            r = 0.2 * (np.std(x) + 1e-12)

        def _phi(mm):
            # build embedded vectors of length mm
            N = n - mm + 1
            if N <= 1:
                return 0.0
            emb = np.lib.stride_tricks.sliding_window_view(x, mm)
            # Chebyshev distance (max norm)
            # Count matches within r (exclude self-matches by subtracting N later)
            d = np.max(np.abs(emb[:, None, :] - emb[None, :, :]), axis=2)
            C = np.sum(d <= r, axis=1) - 1  # remove self match
            return np.sum(C) / (N - 1) / N

        A = _phi(m + 1)
        B = _phi(m)
        if A <= 0 or B <= 0:
            return np.inf  # convention when zero matches at m+1
        return -np.log(A / B)

    @staticmethod
    def _permutation_entropy(x: np.ndarray, m: int = 3, tau: int = 1, normalize: bool = True) -> float:
        """
        Permutation Entropy (Bandt & Pompe). m=3..7 reasonable; tau>=1.
        """
        x = np.asarray(x, dtype=float).ravel()
        n = x.size
        L = n - (m - 1) * tau
        if L <= 0:
            return np.nan
        # build delayed embedding
        Y = np.vstack([x[i:i + L] for i in range(0, m * tau, tau)]).T  # shape (L, m)
        # rank-order patterns
        # argsort twice to get ranks; ties broken by stable kind
        ranks = np.argsort(np.argsort(Y, axis=1), axis=1)
        # hash patterns as tuples
        import numpy as _np
        from collections import Counter
        pats = [tuple(row) for row in ranks]
        counts = Counter(pats)
        p = _np.array(list(counts.values()), dtype=float)
        p /= p.sum()
        H = -np.sum(p * np.log(p + 1e-12))
        if normalize:
            H /= np.log(np.math.factorial(m))
        return float(H)

    @staticmethod
    def _lz_complexity(x: np.ndarray, threshold: float | None = None) -> float:
        """
        Lempel–Ziv 76 complexity (Kaspar–Schuster) on a binarized sequence.
        Normalized as c * log2(n) / n, ~1 for random Bernoulli(0.5).
        """
        import numpy as np
        b = (np.asarray(x, float).ravel() >
             (np.median(x) if threshold is None else float(threshold))).astype(np.uint8)
        n = b.size
        if n == 0:
            return np.nan

        i = 0
        c = 1
        l = 1
        k = 1
        k_max = 1
        while True:
            if i + k >= n or l + k >= n:
                c += 1
                break
            if b[i + k] == b[l + k]:
                k += 1
                if k > k_max:
                    k_max = k
            else:
                i += 1
                if i == l:
                    c += 1
                    l += k_max
                    if l >= n:
                        break
                    i = 0
                    k = 1
                    k_max = 1
                else:
                    k = 1

        # normalization
        return float(c * (np.log2(n + 1e-12) / (n + 1e-12)))


    # Surrogate generator (for nonlinearity check)

    @staticmethod
    def _fft_phase_randomized_surrogate(x: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """
        Amplitude-adjusted phase randomization preserving power spectrum.
        """
        x = np.asarray(x, dtype=float).ravel()
        N = x.size
        rng = rng or np.random.default_rng()
        X = np.fft.rfft(x)
        # randomize phases except DC and Nyquist
        phases = rng.uniform(0, 2 * np.pi, size=X.size)
        phases[0] = 0.0
        if (N % 2) == 0:
            phases[-1] = 0.0
        Y = np.abs(X) * np.exp(1j * phases)
        y = np.fft.irfft(Y, n=N)
        # optional rescale to match x variance/mean
        y = (y - y.mean()) / (y.std() + 1e-12) * (x.std() + 1e-12) + x.mean()
        return y


    # Entropy/Complexity similarity + optional nonlinearity

    def compute_entropy_complexity_metrics(
            self,
            real_data: np.ndarray,
            synthetic_data: np.ndarray,
            *,
            sampen_m: int = 2,
            sampen_r: float | None = None,  # default = 0.2*std(signal)
            permen_m: int = 3,
            permen_tau: int = 1,
            lzc_threshold: float | None = None,  # default = median(signal)
            n_surrogates: int = 0,  # set >0 to run surrogate nonlinearity check
            verbose: bool = True
    ):
        """
        Computes SampEn, PermEn, LZC on each signal; compares real vs synthetic
        distributions with Wasserstein Distance (WD). Optionally estimates a
        nonlinearity index via surrogate testing (z-scores).
        """
        R = np.asarray(real_data, dtype=float)
        S = np.asarray(synthetic_data, dtype=float)
        if R.ndim == 1: R = R[None, :]
        if S.ndim == 1: S = S[None, :]

        def _features(arr):
            se, pe, lz = [], [], []
            for sig in arr:
                se.append(self._sample_entropy(sig, m=sampen_m, r=sampen_r))
                pe.append(self._permutation_entropy(sig, m=permen_m, tau=permen_tau, normalize=True))
                lz.append(self._lz_complexity(sig, threshold=lzc_threshold))
            return np.array(se, float), np.array(pe, float), np.array(lz, float)

        se_R, pe_R, lz_R = _features(R)
        se_S, pe_S, lz_S = _features(S)

        out = {
            "WD_SampEn": float(wasserstein_distance(se_R[~np.isinf(se_R) & ~np.isnan(se_R)],
                                                    se_S[~np.isinf(se_S) & ~np.isnan(se_S)])) if np.isfinite(
                se_R).any() and np.isfinite(
                se_S).any() else np.nan,
            "WD_PermEn": float(wasserstein_distance(pe_R[~np.isnan(pe_R)], pe_S[~np.isnan(pe_S)])) if np.isfinite(
                pe_R).any() and np.isfinite(pe_S).any() else np.nan,
            "WD_LZC": float(wasserstein_distance(lz_R[~np.isnan(lz_R)], lz_S[~np.isnan(lz_S)])) if np.isfinite(
                lz_R).any() and np.isfinite(lz_S).any() else np.nan,
            "Real_SampEn_mean": float(np.nanmean(se_R)), "Synth_SampEn_mean": float(np.nanmean(se_S)),
            "Real_PermEn_mean": float(np.nanmean(pe_R)), "Synth_PermEn_mean": float(np.nanmean(pe_S)),
            "Real_LZC_mean": float(np.nanmean(lz_R)), "Synth_LZC_mean": float(np.nanmean(lz_S)),
        }

        # Optional surrogate nonlinearity check (z-score of real vs surrogate)
        if n_surrogates and n_surrogates > 0:
            rng = np.random.default_rng(0)

            def _nz(arr, func):
                # per-signal z-scores relative to its own surrogates
                zs = []
                for sig in arr:
                    vals = []
                    for _ in range(n_surrogates):
                        y = self._fft_phase_randomized_surrogate(sig, rng=rng)
                        vals.append(func(y))
                    vals = np.asarray(vals, float)
                    mu, sd = np.mean(vals), np.std(vals) + 1e-12
                    zs.append((func(sig) - mu) / sd)
                return np.array(zs, float)

            z_se = _nz(R, lambda y: self._sample_entropy(y, m=sampen_m, r=sampen_r))
            z_pe = _nz(R, lambda y: self._permutation_entropy(y, m=permen_m, tau=permen_tau, normalize=True))
            z_lz = _nz(R, lambda y: self._lz_complexity(y, threshold=lzc_threshold))
            out.update({
                "NonlinearityZ_SampEn_mean": float(np.nanmean(z_se)),
                "NonlinearityZ_PermEn_mean": float(np.nanmean(z_pe)),
                "NonlinearityZ_LZC_mean": float(np.nanmean(z_lz)),
            })

        if verbose:
            def _fmt(x):
                return "nan" if not np.isfinite(x) else f"{x:.4g}"

            print("=== Entropy metrics ===")
            print("Wasserstein Distances (lower = more similar):")
            print(f"  SampEn WD: {_fmt(out['WD_SampEn'])}")
            print(f"  PermEn WD: {_fmt(out['WD_PermEn'])}")
            print(f"  LZC   WD: {_fmt(out['WD_LZC'])}")
            print("Means (Real vs Synthetic):")
            print(f"  SampEn: {_fmt(out['Real_SampEn_mean'])} vs {_fmt(out['Synth_SampEn_mean'])}")
            print(f"  PermEn: {_fmt(out['Real_PermEn_mean'])} vs {_fmt(out['Synth_PermEn_mean'])}")
            print(f"  LZC   : {_fmt(out['Real_LZC_mean'])} vs {_fmt(out['Synth_LZC_mean'])}")
            if n_surrogates and n_surrogates > 0:
                print(
                    "Nonlinearity z-scores (real vs phase-randomized surrogates; higher magnitude suggests nonlinearity):")
                print(f"  SampEn z̄: {_fmt(out['NonlinearityZ_SampEn_mean'])}")
                print(f"  PermEn z̄: {_fmt(out['NonlinearityZ_PermEn_mean'])}")
                print(f"  LZC   z̄: {_fmt(out['NonlinearityZ_LZC_mean'])}")
            print()

        return out
