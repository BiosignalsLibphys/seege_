import pickle
import numpy as np
from scipy.signal import resample

def load_pickle(file_path):
    """
    Load a pickle file and return its contents.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_numpy(file_path):
    return np.load(file_path)

def downsample_signals(signals, original_fs, target_fs):
    """
    Downsample signals to the target sampling frequency.

    Parameters:
    - signals: A numpy array or list of numpy arrays representing the signals.
    - original_fs: The original sampling frequency of the signals.
    - target_fs: The target sampling frequency.

    Returns:
    - Downsampled signals as a numpy array.
    """
    if not isinstance(signals, (list, np.ndarray)):
        raise TypeError("Input signals must be a list or numpy array.")

    if isinstance(signals, np.ndarray) and signals.ndim == 1:
        signals = [signals]  # Convert single signal to a list for uniform processing

    downsampled_signals = []
    for signal in signals:
        if not isinstance(signal, np.ndarray):
            raise TypeError("Each signal must be a numpy array.")
        num_samples = int(len(signal) * target_fs / original_fs)
        downsampled_signal = resample(signal, num_samples)
        downsampled_signals.append(downsampled_signal)

    return np.array(downsampled_signals)


def minmax_normalize(data, axis=1):
    """
    Normalize signals to [-1, 1] using min–max scaling along given axis.
    """
    data = np.array(data, dtype=float)
    min_val = np.min(data, axis=axis, keepdims=True)
    max_val = np.max(data, axis=axis, keepdims=True)
    range_val = np.where((max_val - min_val) == 0, 1, (max_val - min_val))
    normalized = 2 * (data - min_val) / range_val - 1
    return normalized

print('Data processed.')
