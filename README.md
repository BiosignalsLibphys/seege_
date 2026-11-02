<img src="assets/seege logo.png" width="600">

A Python-based framework for evaluating synthetic EEG signals along three dimensions: fidelity, diversity, and privacy.

## Contents
- Test EEG datasets (real and synthetic)
- Evaluation metrics & scores
- Metrics validation
- Usage examples
- Support material

## Installation

seege_ requires **Python 3.10–3.12** (not 3.13, which some dependencies don't support yet).
We recommend installing it in a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -U pip
```

### Use the library

Install directly from GitHub:

```bash
pip install git+https://github.com/BiosignalsLibphys/seege_.git
```

Then import it in Python:

```python
from seege_ import evaluation_score as es
from seege_.preprocessing import load_pickle
```

### Reproduce our results or contribute

Clone the repository and install it with the exact dependency versions used in our experiments:

```bash
git clone https://github.com/BiosignalsLibphys/seege_.git
cd seege_
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` installs seege_ in editable mode, so changes you make in `src/seege_/` take effect without reinstalling.
The example data in `data/` and the notebooks in `examples/` are only available this way.

## Data format

Real and synthetic data are passed as NumPy arrays:

| Shape | Meaning |
|---|---|
| `(N, T)` | N single-channel segments of T samples |
| `(N, C, T)` | N segments with C channels of T samples |

- Real and synthetic sets must share the same **sampling rate, segment length, channel order and preprocessing**. N can differ between the two sets.
- Pass the sampling rate `fs` (in Hz) to the frequency, time-frequency and global fidelity scores.
- Spatial fidelity requires multichannel data (C ≥ 2). For single-channel data it is skipped and excluded from the global fidelity score.
- The membership-inference part of the privacy score needs the real data's task labels (`y_real`). Without them, only the distance-based privacy metrics are computed.
- The example datasets are 1000 single-channel segments of 3 s at 512 Hz (`(1000, 1536)`), min–max normalised to [-1, 1]. `preprocessing.py` provides `minmax_normalize` and `downsample_signals` to bring your data to the same convention.

You can load your own files (`.npy`, `.pkl`, etc.) with `preprocessing.load_numpy` or `preprocessing.load_pickle`, or with your own loader, as long as the result is an array of one of the shapes above.

## Quick start

```python
from seege_ import evaluation_score as es
from seege_.preprocessing import load_pickle

real = load_pickle("data/mayo_epileptic_1000seg.pkl")[:100]
synth = load_pickle("data/synthetic_mayo_epileptic_1000seg.pkl")[:100]

fidelity = es.compute_fidelity_score(real, synth, fs=512)
diversity = es.compute_diversity_score(real, synth)
privacy = es.compute_privacy_score(real, synth))
privacy = es.compute_privacy_score(real, synth)
```

All scores are in [0, 1], where higher is better. Runtime grows quickly with N, especially for the complexity and time-frequency metrics, so start with a subset.

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

## License
GNU Lesser General Public License v3.0

## Citation
If you use seege_ in your work , please cite it as follows:

Silveira, I., Silva, L., Furk, D. et al. Interpretable metrics for evaluating fidelity, diversity, and privacy in synthetic EEG data. Discov. Neurosci. 21, 96 (2026). https://doi.org/10.1186/s13064-026-00323-z

```bibtex
@article{silveira_26,
  title={Interpretable metrics for evaluating fidelity, diversity, and privacy in synthetic EEG data},
  author={Silveira, Ines and Silva, Luis and Furk, Dania and Marques, Beatriz and Ao, Nianfei, Hoogteijling, Sem and Heijink, Irene and Ziljmans, Maeike and Gamboa, Hugo},
  journal={Discover Neuroscience},
  volume={21},
  year={2026},
  doi={https://doi.org/10.1186/s13064-026-00323-z},
  publisher={Elsevier}
}


