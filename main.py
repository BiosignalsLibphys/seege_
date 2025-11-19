
import pickle
from itertools import product

from preprocessing import *
from amplitude_similarity import *
from frequency_similarity import *
from scalogram_similarity import *
from fractal_similarity import *
from time_similarity import *
from diversity import *
from privacy import *
import evaluation_score

if __name__ == '__main__':

    # Load real and synthetic data, 512 Hz, normalized between -1 and 1
    real_data = load_pickle("/Users/is/PycharmProjects/seege_/data/real_dataset_10.pkl")
    synthetic_data = load_pickle("/Users/is/PycharmProjects/seege_/data/synthetic_dataset_10.pkl")

    print(np.shape(real_data))
    print(np.shape(synthetic_data))

   ##### FIDELITY ANALYSIS #####

    # Compute amplitude similarity
    # Initialize class
    #asim = AmplitudeFidelity(fs=512)

    # Compute amplitude similarity metrics
    #amplitude_dataset = asim.compute_amplitude_metrics(real_data, synthetic_data)
    #amplitude_sample = asim.compute_amplitude_metrics(real_data[0], synthetic_data[0])

    # Plot amplitude similarity metrics
    #dataset_plot = asim.plot_amplitude_metrics(amplitude_dataset)
    #sample_plot = asim.plot_amplitude_metrics(amplitude_sample)

    # Compute amplitude fidelity score
    #dataset_score = evaluation_score.compute_amplitude_fidelity_score(real_data, synthetic_data, fs=512)
    #sample_score = evaluation_score.compute_amplitude_fidelity_score(real_data[0], synthetic_data[0], fs=512)

    # Compute time similarity
    # Initialize class
    #sim = TimeFidelity()

    # Compute hjorth metrics
    #hjorth_dataset = sim.compute_hjorth_metrics(real_data, synthetic_data, verbose=True)

    # Plot Hjorth parameter distributions
    #hjorth_hist_dataset = sim.plot_hjorth_histograms(real_data, synthetic_data)

    # Plot 3D Hjorth parameter scatter
    #hjorth_3d_dataset = sim.plot_hjorth_3d(real_data, synthetic_data)

    # Compute time fidelity score
    #evaluation_score.compute_time_fidelity_score(real_data, synthetic_data)

    # Compute frequency similarity
    # Initialize class
    #frequency_analysis = FrequencyFidelity(fs=512)

    # Compute frequency similarity metrics
    # At dataset level
    #frequency_analysis.compare_relative_power(real_data, synthetic_data)
    #frequency_analysis.spectral_coherence(real_data, synthetic_data)
    # At sample level
    #frequency_analysis.compare_relative_power(real_data[0], synthetic_data[0])
    #frequency_analysis.spectral_coherence(real_data[0], synthetic_data[0])

    #Plot average PSD
    # At dataset level
    #frequency_analysis.plot_psd(real_data, synthetic_data, scale="log")
    # At sample level
    #frequency_analysis.plot_psd(real_data[5], synthetic_data[5], scale="log")

    # Compute frequency fidelity score
    # At dataset level
    #evaluation_score.compute_frequency_fidelity_score(real_data, synthetic_data, fs=512)
    # At sample level
    #evaluation_score.compute_frequency_fidelity_score(real_data[0], synthetic_data[0], fs=512)

    # Compute time-frequency similarity
    # Initialize class
    #time_frequency_analysis = TimeFrequencyFidelity(fs=512)

    # Plot scalograms
    # At sample level
    #time_frequency_analysis.plot_scalograms(real_data, synthetic_data, signal_index_real=0, signal_index_synth=0)
    # At dataset level
    #time_frequency_analysis.plot_mean_scalograms(real_data,synthetic_data, freq_scale="log")


    # Compute time-frequency similarity metrics
    # At dataset level
    #time_frequency_analysis.compute_scalogram_similarity_metrics(real_data,synthetic_data)
    # At sample level
    #time_frequency_analysis.compute_scalogram_similarity_metrics(real_data[0], synthetic_data[0])

    # Compute burst statistics within the beta band
    #burst_results = time_frequency_analysis.compute_burst_statistics(real_data, synthetic_data, band=(13, 30))

    # Compute time-frequency fidelity score
    #evaluation_score.compute_time_frequency_fidelity_score(real_data[1], synthetic_data[3], fs=512)
    #evaluation_score.compute_time_frequency_fidelity_score(real_data, synthetic_data, fs=512, mode="all_vs_all")

    # Compute complexity similarity

    # Initialize class for MFDFA method
    #cf = ComplexityFidelity(real_data, synthetic_data, method='MFDFA')
    # Compute MFDFA metrics
    #cf.compute_fractal_metrics()

    # Initialize class for DCCA method
    #cf = ComplexityFidelity(real_data, synthetic_data, method='DCCA')
    # Compute DCCA metrics
    #cf.compute_fractal_metrics()
    # Plot DCCA metrics
    #cf.plot_metrics()

    # Initialize class for MFDCCA method
    #cf = ComplexityFidelity(real_data, synthetic_data, method='MFDCCA')
    # Compute MFDCCA metrics
    #cf.compute_fractal_metrics()
    # Plot MFDCCA metrics
    #cf.plot_metrics()

    # Compute entropy metrics
    #entropy_metrics = cf.compute_entropy_complexity_metrics(real_data, synthetic_data)

    # Compute complexity fidelity score
    #evaluation_score.compute_complexity_fidelity_score(real_data, synthetic_data)

    # Compute fidelity score
    #evaluation_score.compute_fidelity_score(real_data, synthetic_data, fs=2048)

    ##### DIVERSITY ANALYSIS #####
    # Initialize class
    #div = Diversity()

    # Compute coverage diversity metrics
    #cov = div.compute_coverage_diversity(real_data, synthetic_data)

    # Compute geometric diversity metrics
    #geom = div.compute_geometric_diversity(real_data, synthetic_data)

    # Compute intrinsic diversity metrics
    #intr = div.compute_intrinsic_diversity(real_data, synthetic_data)

    # Plot PCA and UMAP results
    #div.plot_embeddings("PCA", geom)
    #div.plot_embeddings("UMAP", geom)

    # Compute diversity score
    #evaluation_score.compute_diversity_score(real_data, synthetic_data)

    ##### PRIVACY ANALYSIS #####
    ## Initialize class
    #privacy_analysis = Privacy()

    # Compute distance and MIR metrics
    #distance_metrics = privacy_analysis.compute_distance_metrics(real_data, synthetic_data)
    #mir = privacy_analysis.compute_mir(real_data, synthetic_data)

    # Compute privacy score
    #evaluation_score.compute_privacy_score(real_data, synthetic_data)
