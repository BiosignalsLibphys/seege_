"""
seege_ example: fidelity, diversity and privacy of synthetic EEG.

Install the package first (from the repository root):
    pip install -e .

Data format (every function accepts both):
    (N, T)     N single-channel segments of T samples
    (N, C, T)  N segments with C channels of T samples
Real and synthetic sets must share C, T, sampling rate and preprocessing; N may differ.
With C > 1, time / frequency / time-frequency / complexity metrics treat every channel
of every segment as one signal; spatial fidelity needs C >= 2; diversity and privacy use
one vector per segment.
"""

from pathlib import Path

import numpy as np

from seege_ import (TimeFidelity, FrequencyFidelity, TimeFrequencyFidelity, ComplexityFidelity,
                    SpatialFidelity, Diversity, Privacy, evaluation_score)
from seege_.preprocessing import load_pickle

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FS = 512

# Pairwise metrics (coherence, scalogram similarity, DCCA/MFDCCA) compare all pairs of
# signals, so their cost grows with the square of the number of signals: cap it.
MAX_SIGNALS = 100

if __name__ == '__main__':

    # Load real and synthetic data, 512 Hz, normalized between -1 and 1
    real_data = load_pickle(DATA_DIR / "real_dataset_10.pkl")
    synthetic_data = load_pickle(DATA_DIR / "synthetic_dataset_10.pkl")

    real_data = np.asarray(real_data)
    synthetic_data = np.asarray(synthetic_data)
    print(real_data.shape, synthetic_data.shape)     # (N, T) or (N, C, T)

    ##### FIDELITY ANALYSIS #####

    # Time fidelity (Hjorth parameters)
    sim = TimeFidelity()
    hjorth_dataset = sim.compute_hjorth_metrics(real_data, synthetic_data, verbose=True)
    sim.plot_hjorth_metrics(real_data, synthetic_data)
    evaluation_score.compute_time_fidelity_score(real_data, synthetic_data)

    # Frequency fidelity
    frequency_analysis = FrequencyFidelity(fs=FS)
    # At dataset level
    frequency_analysis.compare_relative_power(real_data, synthetic_data)
    frequency_analysis.spectral_coherence(real_data, synthetic_data, max_signals=MAX_SIGNALS)
    frequency_analysis.plot_psd(real_data, synthetic_data, scale="log")
    # At sample level: one segment as (1, T) or (1, C, T)
    # frequency_analysis.compare_relative_power(real_data[:1], synthetic_data[:1])
    # frequency_analysis.plot_psd(real_data[5:6], synthetic_data[5:6], scale="log")
    evaluation_score.compute_frequency_fidelity_score(real_data, synthetic_data, fs=FS,
                                                      max_signals=MAX_SIGNALS)

    # Time-frequency fidelity
    time_frequency_analysis = TimeFrequencyFidelity(fs=FS)
    # Scalograms of one signal (index k = segment k // C, channel k % C)
    time_frequency_analysis.plot_scalograms(real_data, synthetic_data, signal_index_real=0, signal_index_synth=0)
    # Mean scalograms over the dataset
    time_frequency_analysis.plot_mean_scalograms(real_data, synthetic_data, freq_scale="log")
    # Similarity metrics (SSIM, NRMSE, cosine) and beta-band bursts
    time_frequency_analysis.compute_scalogram_similarity_metrics(real_data, synthetic_data, max_signals=MAX_SIGNALS)
    burst_results = time_frequency_analysis.compute_burst_statistics(real_data, synthetic_data, band=(13, 30))
    evaluation_score.compute_time_frequency_fidelity_score(real_data, synthetic_data, fs=FS, mode="all_vs_all",
                                                           max_signals=MAX_SIGNALS)

    # Complexity fidelity (slow: uses a random subset of signals)
    cf = ComplexityFidelity(real_data, synthetic_data, method='MFDFA', max_signals=20)
    cf.compute_fractal_metrics()
    cf = ComplexityFidelity(real_data, synthetic_data, method='DCCA', max_signals=20)
    cf.compute_fractal_metrics()
    cf.plot_metrics()
    cf = ComplexityFidelity(real_data, synthetic_data, method='MFDCCA', max_signals=10)
    cf.compute_fractal_metrics()
    cf.plot_metrics()
    entropy_metrics = cf.compute_entropy_complexity_metrics(real_data, synthetic_data, max_signals=20)
    evaluation_score.compute_complexity_fidelity_score(real_data, synthetic_data, max_signals=10)

    # Spatial fidelity (multichannel data only: (N, C, T) with C >= 2)
    if real_data.ndim == 3 and real_data.shape[1] >= 2:
        sf = SpatialFidelity()
        sf.evaluate(real_data, synthetic_data)
        evaluation_score.compute_spatial_fidelity_score(real_data, synthetic_data)

    # Global fidelity score (all domains; spatial is skipped for single-channel data)
    # evaluation_score.compute_fidelity_score(real_data, synthetic_data, fs=FS, max_signals=MAX_SIGNALS)

    ##### DIVERSITY ANALYSIS #####
    div = Diversity(fs=FS)
    # base = div.sanity_check(real_data, groups=subject_ids)   # real-vs-real baseline, split by subject
    cov = div.compute_coverage_diversity(real_data, synthetic_data)
    geom = div.compute_geometric_diversity(real_data, synthetic_data)
    intr = div.compute_intrinsic_diversity(real_data, synthetic_data)
    div.plot_embeddings("PCA", geom)
    div.plot_embeddings("UMAP", geom)
    evaluation_score.compute_diversity_score(real_data, synthetic_data, fs=FS)

    ##### PRIVACY ANALYSIS #####
    privacy_analysis = Privacy()
    # Nearest-neighbour distances (DTW is O(T^2) per pair and only defined for single-channel data)
    distance_metrics = privacy_analysis.compute_distance_metrics(real_data, synthetic_data)
    # Membership inference needs the task labels of the real segments:
    # mir = privacy_analysis.compute_mir_metrics(real_data, synthetic_data, y_real)
    evaluation_score.compute_privacy_score(real_data, synthetic_data)
