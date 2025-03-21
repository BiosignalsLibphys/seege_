import numpy as np
from scipy.signal import resample

def crop_signal(signal: np.ndarray, start_crop: int = 0, end_crop: int = 0) -> np.ndarray:
    """Crop the given signal from both ends if specified and if the length allows.

    Parameters:
    - signal: The input signal as a numpy array.
    - start_crop: Number of samples to remove from the beginning of the signal (optional, default=0).
    - end_crop: Number of samples to remove from the end of the signal (optional, default=0).

    Returns:
    - Cropped signal as a numpy array.
    """
    if not isinstance(signal, np.ndarray):
        raise TypeError("Signal must be a numpy array.")
    if start_crop > 0 or end_crop > 0:
        if len(signal) > start_crop + end_crop:
            return signal[start_crop:-end_crop]
        else:
            return np.array([])  # Return an empty array if cropping exceeds signal length
    return signal  # Return unmodified if no cropping is requested


def preprocess_data(signals, fs: int, target_duration: float = None, start_crop: float = 0,
                    end_crop: float = 0):
    """Preprocess input signals: resampling, normalization, and optional cropping.

    Parameters:
    - signals: Single sample (np.ndarray) or a set of samples (list or np.ndarray of signals).
    - fs: The sampling rate of the signals in Hz.
    - target_duration: Duration (in seconds) to resample all signals to (optional, if None no resampling is applied).
    - start_crop: Duration (in seconds) to crop from the start (optional, default=0).
    - end_crop: Duration (in seconds) to crop from the end (optional, default=0).

    Returns:
    - Processed signals as a numpy array.
    """
    # Ensure input is a list or np.ndarray
    if isinstance(signals, np.ndarray) and signals.ndim == 1:
        signals = [signals]  # Convert single signal to a list for uniform processing
    elif not isinstance(signals, (list, np.ndarray)):
        raise TypeError("Input must be a numpy array or a list of numpy arrays.")

    # Check if all elements are numpy arrays
    if not all(isinstance(sig, np.ndarray) for sig in signals):
        raise TypeError("All elements of the input list must be numpy arrays.")

    processed_signals = []
    for sig in signals:
        original_length = len(sig)
        if target_duration:
            target_length = int(target_duration * fs)
            if original_length != target_length:
                sig = resample(sig, target_length)
        if np.max(np.abs(sig)) > 0:
            sig = sig / np.max(np.abs(sig))  # Avoid division by zero
        if np.isnan(sig).any() or np.isinf(sig).any():
            print("Warning: NaN or Inf found in signal after normalization!")
        if np.all(sig == 0):
            print("Warning: Signal became all zeros after normalization!")
        if start_crop > 0 or end_crop > 0:
            start_crop_samples = int(start_crop * fs)
            end_crop_samples = int(end_crop * fs)
            sig = crop_signal(sig, start_crop_samples, end_crop_samples)
        if sig.size > 0:  # Only append non-empty arrays
            processed_signals.append(sig)

    if len(processed_signals) == 0:
        raise ValueError("⚠ ERROR: All signals were removed during preprocessing!")

    processed_signals = np.array(processed_signals)

    # Ensure consistent shape for all signals
    if processed_signals.ndim == 2 and processed_signals.shape[1] != int(target_duration * fs):
        raise ValueError("Processed signals do not match expected target duration. Check resampling parameters.")

    return processed_signals if len(processed_signals) > 1 else processed_signals[0]

