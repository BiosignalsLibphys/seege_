# seege_

A Python-based framework for evaluating synthetic EEG signals.

<pre> 

seege/                       # Your root project folder
│
├── data/                          # 📂 Datasets
│   ├── real_dataset_10.pkl        # 10 real signals
│   ├── synthetic_dataset_10.pkl   # 10 synthetic signal
│
├── src/                     # Source code
│   ├── __init__.py          # (optional) Make it a package if needed
│
│   ├── preprocessing.py     # 🔧 Data loading and preprocessing utilities
│   ├── diversity.py         #  🧬 Diversity evaluation metrics
│   ├── privacy.py           # 🔐 Privacy evaluation metrics
│   ├── amplitude_similarity.py   # 📈 Amplitude similiarity metrics
│   ├── time_similarity.py        # ⏱️ Time-domain similarity metrics (Hjorth parameters)
│   ├── frequency_similarity.py   # 🔊 Frequency-domain similarity metrics
│   ├── scalogram_similarity.py   #  🗺️ Scalogram similarity metrics
│   ├── fractal_similarity.py     # 🌀 Fractal similarity metrics (e.g., MFDFA, DCCA)
│
│   └── validation/          # ✅ Metrics validation scripts
│       ├── amplitude_validation.py
│       ├── time_validation.py
│       ├── diversity_validation.py
│       ├── fractal_validation.py
│       ├── frequency_validation.py
│       ├── privacy_validation.py
│       └── scalogram_validation.py
│
├── main.py                 # 🚀 Main execution script to run evaluations
├── README.md               # 📘 Project overview & usage
├── requirements.txt        # 📦 Dependencies
├── setup.py                # ⚙️ Installation/config (if packaging)
└── venv/                   # 🧪 Local environment (ignored in version control)

</pre>

## Evaluation dimensions
- Fidelity
- Diversity
- Privacy

## Installation
Clone the repo and install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
```python
from evaluation_score import compute_diversity_score
real_data, synthetic_data = [], []
compute_diversity_score(real_data, synthetic_data)
```
