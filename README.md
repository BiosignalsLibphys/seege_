<img src="seege logo.png" width="600">

A Python-based framework for evaluating synthetic EEG signals along three dimensions: fidelity, diversity, and privacy.

## Contents
- Test EEG datasets (real and synthetic)
- Evaluation metrics & scores
- Metrics validation
- Usage examples
- Support material

## Installation

seege_ requires Python 3.10+. To install, clone the repo and install dependencies in a virtual environment:
```bash
git clone https://github.com/BiosignalsLibphys/seege_.git
cd seege_

python3 -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```
## Data format
The library expects EEG samples in simple Python containers:
- `real_data`: list of arrays (each array is shape `[n_channels, n_samples]` or `[n_samples]`)
- `synthetic_data`: same structure as `real_data`
- Sampling rate, channel order, and preprocessing must be consistent between real and synthetic sets

You can adapt loaders to your own files (`.npy`, `.pkl`, etc.).

## References

Real data segments were retrieved from [mayo dataset](https://springernature.figshare.com/collections/Multicenter_intracranial_EEG_dataset_for_classification_of_graphoelements_and_artifactual_signals/4681208/1).

P. Nejedly, V. Kremen, V. Sladky, J. Cimbalnik, P. Klimes, F. Plesinger, F. Mivalt, V. Travnicek, I. Viscor, M. Pail, et al., Multicenter intracranial eeg dataset for classification of graphoelements and artifactual
signals (Jun 2020). doi:10.6084/m9.figshare.c.4681208.v1. URL https://springernature.figshare.com/collections/Multicenter_intracranial_EEG_dataset_for_classification_ of_graphoelements_and_artifactual_signals/4681208/1

P. Nejedly, V. Kremen, V. Sladky, J. Cimbalnik, P. Klimes, F. Plesinger, F. Mivalt, V. Travnicek, I. Viscor, M. Pail, et al., Multicenter intracranial eeg dataset for classification of graphoelements and artifactual signals, 
Scientific data 7 (1) (2020) 179. doi:https://doi.org/10.1038/s41597-020-0532-5

## Usage example
Please see our notebook usage_example.

Typical workflow
1. Preprocess data (filtering, normalization) so real and synthetic have the same conventions.
2. Compute metrics:
   - Fidelity (signal realism/quality)
   - Diversity (dataset variability)
   - Privacy (leakage/identifiability)
3. Aggregate scores and compare across models/datasets.

## Citation
If you seege_ in your work , please cite it as follows:

```bibtex
@software{seege_2025,
  title        = {seege_: a Python library for synthetic EEG evaluation},
  author       = {Silveira, Inês and Silva, Luís and Gamboa, Hugo.},
  year         = {2025},
  url          = {https://github.com/BiosignalsLibphys/seege_},
  version      = {0.1.0},
  note         = {License: MIT}
}


