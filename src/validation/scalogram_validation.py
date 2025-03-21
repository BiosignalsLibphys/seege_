import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from src.scalogram_similarity import *

def generate_signals(fs=2048, duration=1, seed=42):
    """
    Generate different types of signals for scalogram validation.

    Parameters:
    ----------
    fs : int
        Sampling frequency in Hz
    duration : float
        Duration of signals in seconds
    seed : int
        Random seed for reproducibility

    Returns:
    -------
    dict
        Dictionary containing different signal types
    """
    np.random.seed(seed)
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    f = 10  # Base frequency for sine waves

    signals = {}

    # 1. Identical signals - simple sine wave
    signals["identical_1"] = np.sin(2 * np.pi * f * t)
    signals["identical_2"] = np.sin(2 * np.pi * f * t)

    # 2. Sine wave vs sine wave with small noise
    signals["sine_clean"] = np.sin(2 * np.pi * f * t)
    signals["sine_small_noise"] = np.sin(2 * np.pi * f * t) + 0.01 * np.random.randn(len(t))

    # 3. Sine wave vs sine wave with larger noise
    signals["sine_large_noise"] = np.sin(2 * np.pi * f * t) + 0.5 * np.random.randn(
        len(t))  # Changed to 0.5 as you mentioned

    # 4. Square wave vs brown noise
    signals["square_wave"] = signal.square(2 * np.pi * f * t)

    # Generate brown noise using cumulative sum of white noise
    brown_noise = np.cumsum(0.01 * np.random.randn(len(t)))
    # Normalize to have similar amplitude as other signals
    signals["brown_noise"] = brown_noise / np.max(np.abs(brown_noise))

    # 5. Powerline noise (50/60 Hz with harmonics)
    powerline_freq = 60  # Hz (can be 50 or 60)
    powerline_base = np.sin(2 * np.pi * powerline_freq * t)
    # Add some harmonics
    powerline_noise = powerline_base + 0.5 * np.sin(2 * np.pi * 2 * powerline_freq * t) + 0.25 * np.sin(
        2 * np.pi * 3 * powerline_freq * t)
    signals["powerline_noise"] = powerline_noise / np.max(np.abs(powerline_noise))

    # 6. Chirp signal (frequency sweep)
    signals["chirp"] = signal.chirp(t, f0=1, f1=30, t1=duration, method='linear')

    # 7. Bursts (transient signals)
    burst = np.zeros_like(t)
    burst_duration = int(0.05 * fs)  # 50ms bursts
    for i in range(5):  # Create 5 bursts
        start_idx = np.random.randint(0, len(t) - burst_duration)
        burst[start_idx:start_idx + burst_duration] = np.sin(2 * np.pi * 20 * t[:burst_duration])
    signals["burst"] = burst

    # 8. Triangle wave (different but not completely different from sine)
    signals["triangle_wave"] = signal.sawtooth(2 * np.pi * f * t, 0.5)  # 0.5 duty cycle creates triangle wave

    # Additional signal pairs for "different but not completely different" category
    signals["am_sine"] = (1 + 0.5 * np.sin(2 * np.pi * 2 * t)) * np.sin(2 * np.pi * f * t)  # AM Sine Wave
    signals["sine_5hz"] = np.sin(2 * np.pi * 5 * t)
    signals["sine_15hz"] = np.sin(2 * np.pi * 15 * t)
    signals["sine_highpass_noise"] = np.sin(2 * np.pi * f * t) + 0.3 * np.random.randn(len(t))
    signals["sine_highpass_noise"] = signal.filtfilt(*signal.butter(5, 0.2, 'highpass'), signals["sine_highpass_noise"])

    return signals, t


def validate_scalogram_similarity():
    """
    Validate the ScalogramSimilarity class using different signal types.
    """
    # Parameters
    fs = 2048  # Sampling frequency in Hz
    duration = 2  # Duration in seconds

    # Generate signals
    signals, t = generate_signals(fs=fs, duration=duration)

    # Initialize ScalogramSimilarity class
    scalogram_analyzer = ScalogramSimilarity(fs=fs)

    # Create figure for signal plots - adjust for the correct number of signal types
    # Count the number of signals to determine grid size
    num_signals = len(signals)
    rows = (num_signals + 3) // 4  # Calculate how many rows we need (ceiling division)

    plt.figure(figsize=(16, 3 * rows))
    for i, (name, signal_data) in enumerate(signals.items()):
        plt.subplot(rows, 4, i + 1)
        plt.plot(t, signal_data)
        plt.title(name)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig('generated_signals.png')
    plt.show()

    # Define signal pairs for comparison
    signal_pairs = [
        ("Identical Signals", signals["identical_1"], signals["identical_2"]),
        ("Sine vs Sine with Small Noise", signals["sine_clean"], signals["sine_small_noise"]),
        ("Sine vs Sine with Large Noise", signals["sine_clean"], signals["sine_large_noise"]),
        ("Sine vs Triangle Wave", signals["sine_clean"], signals["triangle_wave"]),  # New test case
        ("Square Wave vs Brown Noise", signals["square_wave"], signals["brown_noise"]),
        ("Sine vs Powerline Noise", signals["sine_clean"], signals["powerline_noise"]),
        ("Sine vs Chirp", signals["sine_clean"], signals["chirp"]),
        ("Sine vs Burst", signals["sine_clean"], signals["burst"]),
        ("Sine vs AM Sine", signals["sine_clean"], signals["am_sine"]),
        ("Sine 5Hz vs Sine 15Hz", signals["sine_5hz"], signals["sine_15hz"]),
        ("Clean Sine vs Highpass Noise Sine", signals["sine_clean"], signals["sine_highpass_noise"])
    ]

    # Collect results for comparison
    results = []

    # Plot scalograms and compute metrics for each pair
    for i, (pair_name, signal1, signal2) in enumerate(signal_pairs):
        print(f"\nAnalyzing {pair_name}:")

        # Plot scalograms
        scalogram_analyzer.plot_scalogram(signal1, signal2)

        # Compute metrics
        metrics = scalogram_analyzer.compute_scalogram_similarity_metrics(signal1, signal2)

        # Store results
        results.append({
            "Pair": pair_name,
            "SSIM": metrics["Mean SSIM"],
            "RMSE": metrics["Mean RMSE"],
            "Cosine Similarity": metrics["Mean Cosine Similarity"]
        })

    # Display results in a table
    print("\n\nSummary of Scalogram Similarity Metrics:")
    print("-" * 100)
    print(f"{'Signal Pair':<30} | {'SSIM':<15} | {'RMSE':<15} | {'Cosine Similarity':<20}")
    print("-" * 100)

    for result in results:
        print(
            f"{result['Pair']:<30} | {result['SSIM']:<15.4f} | {result['RMSE']:<15.4f} | {result['Cosine Similarity']:<20.4f}")

    # Provide a categorization guide based on results
    print("\n\nProposed Signal Similarity Categories:")
    print("-" * 100)
    print("1. Identical signals: SSIM > 0.99, CS > 0.999, RMSE < 0.01")
    print("2. Very similar signals: SSIM 0.93-0.99, CS > 0.99, RMSE 0.01-0.1")
    print("3. Moderately similar signals: SSIM 0.5-0.93, CS 0.95-0.99, RMSE 0.5-2.0")
    print("4. Different signals: SSIM 0.1-0.5, CS 0.2-0.95, RMSE 2.0-5.0")
    print("5. Completely different signals: SSIM < 0.1, CS < 0.2, RMSE > 5.0")


if __name__ == "__main__":
    validate_scalogram_similarity()