import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.signal import detrend
import fathon
from fathon import fathonUtils as fu
import neurokit2 as nk
from scipy.stats import ttest_ind

class FractalSimilarity:
    """
    A class for computing fractal similarity between real and synthetic signals using different fractal analysis methods.

    Methods:
    --------
    - **DCCA (Detrended Cross-Correlation Analysis)**: Measures the correlation
      between two signals across different scales.
    - **MFDFA (Multifractal Detrended Fluctuation Analysis)**: Computes the
      multifractal spectrum of a single signal.
    - **MFDCCA (Multifractal Detrended Cross-Correlation Analysis)**: Extends
      DCCA with multifractal properties by analyzing cross-fluctuations F_xy(q).
    - **MFDCCA2 (Alternative MFDCCA implementation)**: Uses a different
      parameterization or approach to MFDCCA.

    In MFDCCA mode, we do the following:
      1) Compute cross-fluctuation F_xy(q).
      2) Also compute single-series F_x(q), F_y(q) from MFDFA.
      3) Define p(q) = F_xy(q) / sqrt(F_x(q) * F_y(q)), then average over q.

    We store:
      - ``means, stds`` : the average cross-H exponent (or single-series H).
      - ``rho_means, rho_stds`` : correlation coefficients (for DCCA).
      - ``Fq_means, Fq_stds`` : the average cross-fluctuation (for MFDCCA).
      - ``deltaAlpha_means, deltaAlpha_stds`` : width of the multifractal singularity spectrum.
      - ``p_means, p_stds`` : the MFDCCA cross-correlation ratio, averaged over q.

    Example Usage:
    --------------
    ```python
    # Generate white noise data (H=0.5)
    real_data = [np.random.randn(1000) for _ in range(5)]

    # Generate pink noise data (H=1)
    synthetic_data = [np.fft.irfft(
        np.fft.rfft(np.random.randn(1000)) / np.sqrt(np.fft.rfftfreq(1000) + 1e-10)
    ) for _ in range(5)]

    # Compute fractal similarity using DCCA
    fs = FractalSimilarity(real_data, synthetic_data, method='DCCA')
    fs.analyze()

    # Compute fractal similarity using MFDFA
    fs = FractalSimilarity(real_data, synthetic_data, method='MFDFA')
    fs.analyze()

    # Compute fractal similarity using MFDCCA
    fs = FractalSimilarity(real_data, synthetic_data, method='MFDCCA')
    fs.analyze()

    # (Optionally) Alternative MFDCCA approach
    fs = FractalSimilarity(real_data, synthetic_data, method='MFDCCA2')
    fs.analyze()
    ```

    References:
    -----------
    * Podobnik & Stanley (2008) for DCCA theory.
    * Kantelhardt et al. (2002) for MFDFA.
    * Gu & Zhou (2010) for MFDCCA.
    * https://www.sciencedirect.com/science/article/pii/S2212017313006506
    """

    def __init__(self, real_data, synthetic_data, method='DCCA', q_range=np.arange(-5,5,0.1)):
        """
        Initializes the FractalSimilarity class with real and synthetic signal data.

        Parameters
        ----------
        real_data : list or np.ndarray
            List/array of real signals (2D: a list of 1D arrays).
        synthetic_data : list or np.ndarray
            List/array of synthetic signals (2D: a list of 1D arrays).
        method : str, optional (default='DCCA')
            The fractal analysis method to use. Options:
              - 'DCCA' (Detrended Cross-Correlation Analysis)
              - 'MFDFA' (Multifractal Detrended Fluctuation Analysis)
              - 'MFDCCA' (Multifractal Detrended Cross-Correlation Analysis)
              - 'MFDCCA2' (Alternative MFDCCA implementation)
        q_range : np.ndarray, optional
            Range of q values for multifractal analysis (used in MFDFA, MFDCCA, etc.).
            Default is np.arange(-5,5,0.1).
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.method = method
        self.q_range = q_range

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

    def analyze(self):
        """
        Perform the selected fractal analysis method and produce relevant
        plots/results. Calls one of the internal methods based on self.method.
        """
        if self.method == 'DCCA':
            self._dcca_analyze()
            self._print_results()
            self.plot_hurst_correlation()
            self.plot_rho_correlation()

        elif self.method == 'MFDFA':
            self._mfdfa_analyze()
            self._print_results()
            # Typically MFDFA is single-series, so no cross-correlation plot

        elif self.method == 'MFDCCA':
            self._mfdcca_analyze()
            self._print_results()
            self.plot_hurst_correlation()
            self.plot_p_correlation()

        elif self.method == 'MFDCCA2':
            # Alternative MFDCCA approach
            self._mfdcca_analyze2()
            self._print_results()
            self.plot_hurst_correlation()

        else:
            raise ValueError("Invalid method. Choose 'DCCA', 'MFDFA', 'MFDCCA', or 'MFDCCA2'.")

    # ---------------------------------------------------------------------
    # Plotting
    # ---------------------------------------------------------------------

    def plot_hurst_correlation(self):
        """Plot the average H exponents for each pair category, with error bars."""
        if not self.means:
            print("No H exponents to plot.")
            return

        plt.figure(figsize=(8,5))
        sns.barplot(x=self.categories, y=self.means, capsize=0.2,
                    palette=['lightskyblue','limegreen','grey'], legend=False)
        plt.errorbar(x=self.categories, y=self.means, yerr=self.stds,
                     fmt='none', capsize=5, color='black')
        plt.ylabel('Mean Hurst Exponent (H)', fontsize=13)
        plt.title(f'{self.method} Mean H Exponents', fontsize=15)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.tight_layout()
        plt.show()

    def plot_rho_correlation(self):
        """Bar plot of correlation coefficients (rho), typically for DCCA or MFDCCA if computed."""
        if not self.rho_means:
            print("No correlation coefficients available to plot.")
            return

        plt.figure(figsize=(8,5))
        sns.barplot(x=self.categories, y=self.rho_means, capsize=0.2,
                    palette=['lightskyblue','limegreen','grey'], legend=False)
        plt.errorbar(x=self.categories, y=self.rho_means, yerr=self.rho_stds,
                     fmt='none', capsize=5, color='black')
        plt.ylabel(f'{self.method} mean rho', fontsize=13)
        plt.title(f'Mean {self.method} Correlation Coefficients', fontsize=15)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.tight_layout()
        plt.show()

    def plot_p_correlation(self):
        """Bar plot of the mean p(q) for MFDCCA cross-correlation if computed."""
        if not self.p_means:
            print("No p(q) values available to plot.")
            return

        plt.figure(figsize=(8,5))
        sns.barplot(x=self.categories, y=self.p_means, capsize=0.2,
                    palette=['lightskyblue','limegreen','grey'], legend=False)
        plt.errorbar(x=self.categories, y=self.p_means, yerr=self.p_stds,
                     fmt='none', capsize=5, color='black')
        plt.ylabel(r'MFDCCA p(q) (Avg over q)', fontsize=13)
        plt.title('Comparison of MFDCCA cross-correlation p(q)', fontsize=15)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.tight_layout()
        plt.show()

    # ---------------------------------------------------------------------
    # Helper Methods
    # ---------------------------------------------------------------------

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
        print(f"\nResults for {self.method}:")
        for category, mean_val, std_val in zip(self.categories, self.means, self.stds):
            print(f"  {category}: Mean = {mean_val:.4f}, Std = {std_val:.4f}")

    # ---------------------------------------------------------------------
    # DCCA
    # ---------------------------------------------------------------------
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
                rho = dcca.computeRho(wins, polOrd=1)
                rho_rr.append(np.mean(rho[1]))

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
                rho_ss.append(np.mean(rho[1]))

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
                rho_rs.append(np.mean(rho[1]))

        # fallback if empty
        if not H_rr: H_rr=[np.nan]; rho_rr=[np.nan]
        if not H_rs: H_rs=[np.nan]; rho_rs=[np.nan]
        if not H_ss: H_ss=[np.nan]; rho_ss=[np.nan]

        # store
        self.means = [np.nanmean(H_rr), np.nanmean(H_rs), np.nanmean(H_ss)]
        self.stds  = [np.nanstd(H_rr), np.nanstd(H_rs), np.nanstd(H_ss)]
        self.rho_means = [np.nanmean(rho_rr), np.nanmean(rho_rs), np.nanmean(rho_ss)]
        self.rho_stds  = [np.nanstd(rho_rr), np.nanstd(rho_rs), np.nanstd(rho_ss)]

    # ---------------------------------------------------------------------
    # MFDFA
    # ---------------------------------------------------------------------
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

    # ---------------------------------------------------------------------
    # MFDCCA
    # ---------------------------------------------------------------------
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

        print("\nWidth of Singularity Spectrum (Δα):")
        print(f"  real vs real:        {self.deltaAlpha_means[0]:.4f} ± {self.deltaAlpha_stds[0]:.4f}")
        print(f"  real vs synthetic:   {self.deltaAlpha_means[1]:.4f} ± {self.deltaAlpha_stds[1]:.4f}")
        print(f"  synthetic vs synthetic: {self.deltaAlpha_means[2]:.4f} ± {self.deltaAlpha_stds[2]:.4f}")

        print("\nMFDCCA Cross-Correlation Ratio p(q) [averaged over q]:")
        print(f"  real vs real:        {self.p_means[0]:.4f} ± {self.p_stds[0]:.4f}")
        print(f"  real vs synthetic:   {self.p_means[1]:.4f} ± {self.p_stds[1]:.4f}")
        print(f"  synthetic vs synthetic: {self.p_means[2]:.4f} ± {self.p_stds[2]:.4f}")

    # ---------------------------------------------------------------------
    # MFDCCA2
    # ---------------------------------------------------------------------
    def _mfdcca_analyze2(self):
        """
        Alternative MFDCCA approach, parameterization, or polynomial order, etc.
        The logic is similar to _mfdcca_analyze but may differ in details.
        """
        pre_r = self._preprocess_signals(self.real_data)
        pre_s = self._preprocess_signals(self.synthetic_data)

        H_rr, H_rs, H_ss = [], [], []

        # real-real
        for i in range(len(pre_r)):
            for j in range(i+1, len(pre_r)):
                a = fu.toAggregated(pre_r[i])
                b = fu.toAggregated(pre_r[j])
                mm = fathon.MFDCCA(a, b)
                wins = self._get_win_sizes(len(a))
                if len(wins)<4:
                    continue

                n, F = mm.computeFlucVec(wins, self.q_range, polOrd=1)
                H_arr, _ = mm.fitFlucVec()
                H_rr.append(np.nanmean(H_arr))

        # synthetic-synthetic
        for i in range(len(pre_s)):
            for j in range(i+1, len(pre_s)):
                a = fu.toAggregated(pre_s[i])
                b = fu.toAggregated(pre_s[j])
                mm = fathon.MFDCCA(a, b)
                wins = self._get_win_sizes(len(a))
                if len(wins)<4:
                    continue

                n, F = mm.computeFlucVec(wins, self.q_range, polOrd=1)
                H_arr, _ = mm.fitFlucVec()
                H_ss.append(np.nanmean(H_arr))

        # real-synthetic
        for xr in pre_r:
            for xs in pre_s:
                a = fu.toAggregated(xr)
                b = fu.toAggregated(xs)
                mm = fathon.MFDCCA(a, b)
                wins = self._get_win_sizes(len(a))
                if len(wins)<4:
                    continue

                n, F = mm.computeFlucVec(wins, self.q_range, polOrd=1)
                H_arr, _ = mm.fitFlucVec()
                H_rs.append(np.nanmean(H_arr))

        # fallback
        if not H_rr: H_rr=[np.nan]
        if not H_rs: H_rs=[np.nan]
        if not H_ss: H_ss=[np.nan]

        self.means = [np.nanmean(H_rr), np.nanmean(H_rs), np.nanmean(H_ss)]
        self.stds  = [np.nanstd(H_rr), np.nanstd(H_rs), np.nanstd(H_ss)]

        print(f"{self.method} Hurst Exponent (Alternative MFDCCA):")
        print(f"  real vs real:       {self.means[0]:.4f} ± {self.stds[0]:.4f}")
        print(f"  real vs synthetic:  {self.means[1]:.4f} ± {self.stds[1]:.4f}")
        print(f"  synthetic vs synthetic: {self.means[2]:.4f} ± {self.stds[2]:.4f}")
