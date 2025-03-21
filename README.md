# seege_

A Python-based framework for evaluating synthetic EEG signals.

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
