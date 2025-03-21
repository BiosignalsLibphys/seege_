from scipy.signal import square
from scipy.integrate import odeint
from src.amplitude_similarity import *

# ----------------- Signal Generation Functions -----------------

def generate_signals(signal_type, fs=1000, duration=1, noise_level=0.0, seed=42):
    np.random.seed(seed)
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    if signal_type == "sine":
        return np.sin(2 * np.pi * 10 * t)
    elif signal_type == "sine_noisy":
        sine = np.sin(2 * np.pi * 10 * t)
        noise = noise_level * np.random.randn(len(t))
        return sine + noise
    elif signal_type == "sine_shifted_20Hz":
        return np.sin(2 * np.pi * 20 * t)
    elif signal_type == "square":
        return square(2 * np.pi * 10 * t)
    elif signal_type == "white_noise":
        return np.random.randn(len(t))
    elif signal_type == "lorenz":
        return generate_lorenz_signal(len(t), dt=1/fs)
    else:
        raise ValueError("Unknown signal type.")

def generate_lorenz_signal(length, dt=0.001, sigma=10, rho=28, beta=8/3):
    def lorenz(X, t, sigma, rho, beta):
        x, y, z = X
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        return [dx, dy, dz]

    t = np.linspace(0, dt * length, length)
    X0 = [1.0, 1.0, 1.0]
    X = odeint(lorenz, X0, t, args=(sigma, rho, beta))
    return X[:, 0]

# ----------------- FSV Threshold Interpretation -----------------

def interpret_fsv(value):
    """
    Interprets the GDM values based on the standard FSV ranges.

    References:
    [1] https://www.semanticscholar.org/paper/Applying-the-Feature-Selective-Validation-(-FSV-)-hgs/04e96dce62fc8af49b817d562981a114716b46bf
    """
    if value < 0.1:
        return "Excellent"
    elif 0.1 <= value < 0.2:
        return "Very Good"
    elif 0.2 <= value < 0.4:
        return "Good"
    elif 0.4 <= value < 0.8:
        return "Fair"
    elif 0.8 <= value < 1.6:
        return "Poor"
    else:
        return "Very Poor"

# ----------------- Test Cases with Explicit Noise Levels -----------------

test_cases = [
    ("Identical Sine Waves", "sine", "sine", 0.0, "Excellent"),
    ("Sine vs Sine + 1% Noise", "sine", "sine_noisy", 0.01, "Very Good"),
    ("Sine vs Frequency-shifted Sine", "sine", "sine_shifted_20Hz", 0.0, "Good"),
    ("Sine vs Square Wave", "sine", "square", 0.0, "Very Poor"),
    ("Lorenz vs White Noise", "lorenz", "white_noise", 0.0, "Very Poor"),
]

# ----------------- Validation Execution -----------------

def run_validation_tests(fs=1000, duration=1):
    asim = AmplitudeSimilarity(fs)

    print("\nAmplitude Similarity Validation Tests\n" + "-"*60)

    for test_name, sig1_type, sig2_type, noise_level, expected_interp in test_cases:
        signal1 = generate_signals(sig1_type, fs, duration)
        signal2 = generate_signals(sig2_type, fs, duration, noise_level=noise_level)

        metrics = asim.compute_metrics(signal1, signal2)
        gdm_interp = interpret_fsv(metrics["GDM"])

        print(f"\nTest Case: {test_name}")
        print(f"  - ADM: {metrics['ADM']:.3f}")
        print(f"  - FDM: {metrics['FDM']:.3f}")
        print(f"  - GDM: {metrics['GDM']:.3f} (Interpreted as: {gdm_interp})")
        print(f"  - Similarity Score: {metrics['Similarity']:.3f}")

        if gdm_interp == expected_interp:
            print(f"  ✅ Matches expected interpretation: {expected_interp}")
        else:
            print(f"  ⚠️ Differs from expected interpretation.")
            print(f"     Expected: {expected_interp}, but got: {gdm_interp}")

# ----------------- Main Execution -----------------

if __name__ == "__main__":
    run_validation_tests()
