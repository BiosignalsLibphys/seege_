
import pickle
from src.preprocessing import *
from src.amplitude_similarity import *
from src.frequency_similarity import *
from src.scalogram_similarity import *
from src.fractal_similarity import *
from src.diversity import *
from src.privacy import *

def load_pickle(file_path):
    """
    Load a pickle file and return its contents.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Load real and synthetic data
    real_data = load_pickle("data/real_dataset_10.pkl")
    synthetic_data = load_pickle("data/synthetic_dataset_10.pkl")

    # Preprocess real and synthetic data
    real_data = preprocess_data(real_data, fs=512, target_duration=60)
    synthetic_data = preprocess_data(synthetic_data, fs=512, target_duration=60)

    ##### FIDELITY ANALYSIS #####

    # Compute amplitude similarity
    # Initialize class
    asim = AmplitudeSimilarity(fs=2048)
    # Compute amplitude similarity metrics
    metrics = asim.compute_metrics(real_data, synthetic_data)
    asim.plot_metrics(metrics)

    # Compute frequency similarity
    # Initialize class
    frequency_analysis = FrequencySimilarity(fs=2048)
    # At dataset level
    frequency_analysis.compare_relative_power(real_data, synthetic_data)
    frequency_analysis.plot_psd(real_data, synthetic_data, scale="linear")
    frequency_analysis.spectral_coherence(real_data[0], synthetic_data[0])
    # At sample level
    frequency_analysis.compare_relative_power(real_data[0], synthetic_data[0])
    frequency_analysis.plot_psd(real_data[5], synthetic_data[5], scale="log")

    # Compute scalogram similarity
    # Initialize class
    scalogram_analysis = ScalogramSimilarity(fs=512)

    # Plot scalogram - sample level only
    scalogram_analysis.plot_scalogram(real_data, synthetic_data, signal_index_real=0, signal_index_synth=0)

    # Compute scalogram similarity metrics
    # At dataset level
    scalogram_analysis.compute_scalogram_similarity_metrics(real_data,synthetic_data)
    # At sample level
    scalogram_analysis.compute_scalogram_similarity_metrics(real_data[0], synthetic_data[5])

    # Compute fractality similarity
    # Initialize class for DCCA method
    fs_mfdcca = FractalSimilarity(real_data, synthetic_data, method='MFDCCA')
    # Compute fractality similarity metrics - dataset level only
    fs_mfdcca.analyze()

    # Initialize class for DCCA method
    fs_dcca = FractalSimilarity(real_data, synthetic_data, method='DCCA')
    # Compute fractality similarity metrics - dataset level only
    fs_dcca.analyze()

    # Initialize class for MFDFA method
    fs_mfdfa = FractalSimilarity(real_data, synthetic_data, method='MFDFA')
    # Compute fractality similarity metrics - dataset level only
    fs_mfdfa.analyze()

    ##### DIVERSITY ANALYSIS #####
    # Initialize class
    diversity_eval = Diversity(n_components=2)

    # Plot PCA, t-SNE, and UMAP results
    diversity_eval.plot_pca()
    diversity_eval.plot_tsne()
    diversity_eval.plot_umap()

    ##### PRIVACY ANALYSIS #####
    ## Initialize class
    privacy_analysis = Privacy()

    # Compute privacy metrics
    privacy_metrics = privacy_analysis.compute_privacy_metrics(real_data,synthetic_data)
