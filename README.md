
<img src="seege_logo.png" width="600">

A Python-based framework for evaluating synthetic EEG signals along three dimensions: fidelity, diversity, and privacy.

<pre> 

seege/                       # Your root project folder
│
├── data/                          # Datasets
│   ├── real_dataset_10.pkl        # 10 real signals
│   ├── synthetic_dataset_10.pkl   # 10 synthetic signal
│
├── src/                     # Source code
│   ├── __init__.py          # (optional) Make it a package if needed
│
│   ├── preprocessing.py          # Data loading and preprocessing utilities
│   ├── amplitude_fidelity.py   # Amplitude similiarity metrics
│   ├── time_fidelity.py        # Time domain similarity metrics (Hjorth parameters)
│   ├── frequency_fidelity.py   # Frequency domain similarity metrics
│   ├── time_frequency_fidelity.py   #  Time-frequency domain similarity metrics
│   ├── complexity_fidelity.py     # Complexity domain similarity metrics (fractality and entropy metrics)
│   ├── diversity.py              #  Diversity evaluation metrics
│   ├── privacy.py                # Privacy evaluation metrics
│   ├── evaluation_score.py       #  Evaluation scores estimation for all domains and sub-domains
│
│   └── validation/          # Metrics validation scripts
│       ├── amplitude_validation.py
│       ├── time_validation.py
│       ├── frequency_validation.py
│       ├── time_frequency_validation.py
│       ├── complexity_validation.py
│       ├── diversity_validation.py
│       └── privacy_validation.py
│
├── main.py                 # Main execution script to run evaluations
├── README.md               # Project overview & usage
├── requirements.txt        # Dependencies
├── setup.py                # Installation/config (if packaging)
└── venv/                   # Local environment (ignored in version control)

</pre>

## Evaluation dimensions
- Fidelity
- Diversity
- Privacy

## Prerequisites
- Python 3.10+
- macOS (tested), Linux/Windows should work
- `pip` and a virtual environment tool

## Installation
Clone the repo and install dependencies in a virtual environment:
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

## Usage example
```python
import numpy as np

# Generate data: 10 real and 10 synthetic epochs, 1 channel × 1024 samples
real_data = [np.random.randn(1, 1024) for _ in range(10)]
synthetic_data = [np.random.randn(1, 1024) for _ in range(10)]

# Calculate diversity score
from src.evaluation_score import compute_diversity_score  # adjust to your API
score = compute_diversity_score(real_data, synthetic_data)
```

## Typical workflow
1. Preprocess data (filtering, normalization) so real and synthetic have the same conventions.
2. Compute metrics:
   - Diversity (dataset variability)
   - Fidelity (signal realism/quality)
   - Privacy (leakage/identifiability)
3. Aggregate scores and compare across models/datasets.

## License
PolyForm Noncommercial 1.0.0 © 2025 Inês A. P. Silveira

## Citation
If you seege_ in your work , please cite it as follows:

```bibtex
@software{seege_2025,
  title        = {seege_: a Python library for synthetic EEG evaluation},
  author       = {Silveira, Inês A. P.},
  year         = {2025},
  url          = {https://github.com/BiosignalsLibphys/seege_},
  version      = {0.1.0},
  note         = {License: PolyForm Noncommercial 1.0.0}
}


