import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.signal import detrend
import fathon
from fathon import fathonUtils as fu
import neurokit2 as nk

class FractalSimilarity:
    """
    A class for computing fractal similarity between real and synthetic signals using different fractal analysis methods.

    Methods:
    --------
    - **DCCA (Detrended Cross-Correlation Analysis)**: Measures the correlation between two signals across different scales.
    - **MFDFA (Multifractal Detrended Fluctuation Analysis)**: Computes the multifractal spectrum of a single signal.
    - **MFDCCA (Multifractal Detrended Cross-Correlation Analysis)**: Extends DCCA with multifractal properties.
    - **MFDCCA2 (Alternative MFDCCA implementation)**: Uses a different parameterization of MFDCCA.

    Example Usage:
    --------------
    ```python
    real_data = [np.random.randn(1000) for _ in range(5)]
    synthetic_data = [np.random.randn(1000) for _ in range(5)]

    # Compute fractal similarity using DCCA
    fs = FractalSimilarity(real_data, synthetic_data, method='DCCA')
    fs.analyze()

    # Compute fractal similarity using MFDFA
    fs = FractalSimilarity(real_data, synthetic_data, method='MFDFA')
    fs.analyze()
    ```
    """

    def __init__(self, real_data, synthetic_data, method='DCCA', q_range=np.arange(-5, 5, 0.1)):
        """
        Initializes the FractalSimilarity class with real and synthetic signal data.

        Parameters:
        -----------
        real_data : list of np.ndarray
            A list of 1D NumPy arrays representing real signals.
        synthetic_data : list of np.ndarray
            A list of 1D NumPy arrays representing synthetic signals.
        method : str, optional (default='DCCA')
        The fractal analysis method to use. Options:
            - 'DCCA' (Detrended Cross-Correlation Analysis)
            - 'MFDFA' (Multifractal Detrended Fluctuation Analysis)
            - 'MFDCCA' (Multifractal Detrended Cross-Correlation Analysis)
            - 'MFDCCA2' (Alternative MFDCCA implementation)
        q_range : np.ndarray, optional
            Range of q values for multifractal analysis (used in `MFDFA` and `MFDCCA`).
            Default is `np.arange(-5, 5, 0.1)`.
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.method = method
        self.q_range = q_range
        self.categories = ['R vs R', 'R vs S', 'S vs S']
        self.means = []
        self.stds = []

    def analyze(self):
        """
        Perform the selected fractal analysis method and plot the results.
        """
        if self.method == 'DCCA':
            self._dcca_analyze()
        elif self.method == 'MFDFA':
            self._mfdfa_analyze()
        elif self.method == 'MFDCCA':
            self._mfdcca_analyze()
        elif self.method == 'MFDCCA2':
            self._mfdcca_analyze2()
        else:
            raise ValueError("Invalid method. Choose 'DCCA', 'MFDFA', or 'MFDCCA'.")

        # Print the results
        self._print_results()

        # Plot the results
        self.plot_results()

    def plot_results(self):
        """
        Plot the results of the analysis as bar charts with error bars.
        """
        plt.figure(figsize=(12, 6))
        sns.barplot(x=self.categories, y=self.means, capsize=0.2, palette=['lightskyblue', 'limegreen', 'grey'])
        plt.errorbar(x=self.categories, y=self.means, yerr=self.stds, fmt='none', capsize=5, color='black')
        plt.ylabel('Mean Value', fontsize=15)
        plt.title(f'Comparison of {self.method} Metrics', fontsize=20)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        plt.show()

    def _preprocess_signals(self, signals):
        """
        Preprocess signals by normalizing and detrending.

        Parameters:
        -----------
        signals : list of np.ndarray
            List of raw signals.

        Returns:
        --------
        list of np.ndarray
            Preprocessed signals (normalized and detrended).
        """
        scaler = StandardScaler()
        detrended_data = []
        for i, signal in enumerate(signals):
            try:
                if np.isnan(signal).any() or np.isinf(signal).any():
                    print(f"Skipping signal {i + 1} due to NaN or infinite values.")
                    continue
                normalized_signal = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
                detrended_signal = detrend(normalized_signal)
                detrended_data.append(detrended_signal)
            except Exception as e:
                print(f"Error processing signal {i + 1}: {e}")

        return detrended_data

    def _dcca_analyze(self):
        """
        Perform Detrended Cross-Correlation Analysis (DCCA) between real and synthetic signals.
        """
        preprocessed_real = self._preprocess_signals(self.real_data)
        preprocessed_synthetic = self._preprocess_signals(self.synthetic_data)

        H_real_real = []
        H_real_synthetic = []
        H_synthetic_synthetic = []

        # Analyze real-real pairs
        for ii in range(len(preprocessed_real)):
            for jj in range(ii + 1, len(preprocessed_real)):
                a = fu.toAggregated(preprocessed_real[ii])
                b = fu.toAggregated(preprocessed_real[jj])

                pydcca = fathon.DCCA(a, b)
                winSizes = fu.linRangeByStep(20, len(preprocessed_real[0]) // 4, step=50)
                polOrd = 1

                n, F = pydcca.computeFlucVec(winSizes, polOrd=polOrd)
                H, H_intercept = pydcca.fitFlucVec()
                H_real_real.append(H)

        # Analyze synthetic-synthetic pairs
        for ii in range(len(preprocessed_synthetic)):
            for jj in range(ii + 1, len(preprocessed_synthetic)):
                a = fu.toAggregated(preprocessed_synthetic[ii])
                b = fu.toAggregated(preprocessed_synthetic[jj])

                pydcca = fathon.DCCA(a, b)
                winSizes = fu.linRangeByStep(20, len(preprocessed_synthetic[0]) // 4, step=50)
                polOrd = 1

                n, F = pydcca.computeFlucVec(winSizes, polOrd=polOrd)
                H, H_intercept = pydcca.fitFlucVec()
                H_synthetic_synthetic.append(H)

        # Analyze real-synthetic pairs
        for real_signal in preprocessed_real:
            for synthetic_signal in preprocessed_synthetic:
                a = fu.toAggregated(real_signal)
                b = fu.toAggregated(synthetic_signal)

                pydcca = fathon.DCCA(a, b)
                winSizes = fu.linRangeByStep(20, len(real_signal) // 4, step=50)
                polOrd = 1

                n, F = pydcca.computeFlucVec(winSizes, polOrd=polOrd)
                H, H_intercept = pydcca.fitFlucVec()
                H_real_synthetic.append(H)

        self.means = [np.mean(H_real_real), np.mean(H_real_synthetic), np.mean(H_synthetic_synthetic)]
        self.stds = [np.std(H_real_real), np.std(H_real_synthetic), np.std(H_synthetic_synthetic)]

    def _mfdfa_analyze(self):
        """
        Perform Multifractal Detrended Fluctuation Analysis (MFDFA) on real and synthetic signals.
        """
        preprocessed_real = self._preprocess_signals(self.real_data)
        preprocessed_synthetic = self._preprocess_signals(self.synthetic_data)

        H_real_real = []
        H_real_synthetic = []
        H_synthetic_synthetic = []

        # Analyze real-real pairs
        for ii in range(len(preprocessed_real)):
            for jj in range(ii + 1, len(preprocessed_real)):
                try:
                    _, info1 = nk.fractal_mfdfa(preprocessed_real[ii], q=self.q_range, order=1)
                    _, info2 = nk.fractal_mfdfa(preprocessed_real[jj], q=self.q_range, order=1)
                    H_real_real.append((info1['H'] + info2['H']) / 2)
                except Exception as e:
                    print(f"Error processing real signal pair {ii}-{jj}: {e}")

        # Analyze synthetic-synthetic pairs
        for ii in range(len(preprocessed_synthetic)):
            for jj in range(ii + 1, len(preprocessed_synthetic)):
                try:
                    _, info1 = nk.fractal_mfdfa(preprocessed_synthetic[ii], q=self.q_range, order=1)
                    _, info2 = nk.fractal_mfdfa(preprocessed_synthetic[jj], q=self.q_range, order=1)
                    H_synthetic_synthetic.append((info1['H'] + info2['H']) / 2)
                except Exception as e:
                    print(f"Error processing synthetic signal pair {ii}-{jj}: {e}")

        # Analyze real-synthetic pairs
        for real_signal in preprocessed_real:
            for synthetic_signal in preprocessed_synthetic:
                try:
                    _, info1 = nk.fractal_mfdfa(real_signal, q=self.q_range, order=1)
                    _, info2 = nk.fractal_mfdfa(synthetic_signal, q=self.q_range, order=1)
                    H_real_synthetic.append((info1['H'] + info2['H']) / 2)
                except Exception as e:
                    print(f"Error processing real-synthetic signal pair: {e}")

        self.means = [np.mean(H_real_real), np.mean(H_real_synthetic), np.mean(H_synthetic_synthetic)]
        self.stds = [np.std(H_real_real), np.std(H_real_synthetic), np.std(H_synthetic_synthetic)]

    def _mfdcca_analyze(self):
        """
        Perform Multifractal Detrended Cross-Correlation Analysis (MFDCCA) between real and synthetic signals.
        """
        combined_data = self.real_data + self.synthetic_data
        preprocessed_data = self._preprocess_signals(combined_data)
        num_series = len(preprocessed_data)
        half_point = len(self.real_data)

        H_real_real = []
        H_real_synthetic = []
        H_synthetic_synthetic = []

        for ii in range(num_series):
            for jj in range(ii + 1, num_series):
                a = fu.toAggregated(preprocessed_data[ii])
                b = fu.toAggregated(preprocessed_data[jj])

                pymfdcca = fathon.MFDCCA(a, b)

                winSizes = fu.linRangeByStep(20, len(preprocessed_data[0]) // 4, step=50)
                qList = self.q_range
                polOrd = 1

                n, F = pymfdcca.computeFlucVec(winSizes, qList, polOrd=polOrd)
                H, H_intercept = pymfdcca.fitFlucVec()

                if ii < half_point and jj < half_point:
                    H_real_real.append(np.mean(H))
                elif ii >= half_point and jj >= half_point:
                    H_synthetic_synthetic.append(np.mean(H))
                else:
                    H_real_synthetic.append(np.mean(H))

        self.means = [np.mean(H_real_real), np.mean(H_real_synthetic), np.mean(H_synthetic_synthetic)]
        self.stds = [np.std(H_real_real), np.std(H_real_synthetic), np.std(H_synthetic_synthetic)]

    def _mfdcca_analyze2(self):
        """
        Perform Multifractal Detrended Cross-Correlation Analysis (MFDCCA) between real and synthetic signals.
        """
        preprocessed_real = self._preprocess_signals(self.real_data)
        preprocessed_synthetic = self._preprocess_signals(self.synthetic_data)

        H_real_real = []
        H_real_synthetic = []
        H_synthetic_synthetic = []

        # Analyze real-real pairs
        for ii in range(len(preprocessed_real)):
            for jj in range(ii + 1, len(preprocessed_real)):
                a = fu.toAggregated(preprocessed_real[ii])
                b = fu.toAggregated(preprocessed_real[jj])

                pymfdcca = fathon.MFDCCA(a, b)
                winSizes = fu.linRangeByStep(20, len(a) // 4, step=50)
                qList = self.q_range
                polOrd = 1

                n, F = pymfdcca.computeFlucVec(winSizes, qList, polOrd=polOrd)
                H, H_intercept = pymfdcca.fitFlucVec()
                H_real_real.append(np.mean(H))

        # Analyze synthetic-synthetic pairs
        for ii in range(len(preprocessed_synthetic)):
            for jj in range(ii + 1, len(preprocessed_synthetic)):
                a = fu.toAggregated(preprocessed_synthetic[ii])
                b = fu.toAggregated(preprocessed_synthetic[jj])

                pymfdcca = fathon.MFDCCA(a, b)
                winSizes = fu.linRangeByStep(20, len(a) // 4, step=50)
                qList = self.q_range
                polOrd = 1

                n, F = pymfdcca.computeFlucVec(winSizes, qList, polOrd=polOrd)
                H, H_intercept = pymfdcca.fitFlucVec()
                H_synthetic_synthetic.append(np.mean(H))

        # Analyze real-synthetic pairs
        for real_signal in preprocessed_real:
            for synthetic_signal in preprocessed_synthetic:
                a = fu.toAggregated(real_signal)
                b = fu.toAggregated(synthetic_signal)

                pymfdcca = fathon.MFDCCA(a, b)
                winSizes = fu.linRangeByStep(20, len(a) // 4, step=50)
                qList = self.q_range
                polOrd = 1

                n, F = pymfdcca.computeFlucVec(winSizes, qList, polOrd=polOrd)
                H, H_intercept = pymfdcca.fitFlucVec()
                H_real_synthetic.append(np.mean(H))

        self.means = [np.mean(H_real_real), np.mean(H_real_synthetic), np.mean(H_synthetic_synthetic)]
        self.stds = [np.std(H_real_real), np.std(H_real_synthetic), np.std(H_synthetic_synthetic)]

    def _print_results(self):
        """
        Print the mean values for the three pairings.
        """
        print(f"Results for {self.method}:")
        for category, mean, std in zip(self.categories, self.means, self.stds):
            print(f"{category}: Mean = {mean:.4f}, Std = {std:.4f}")

