import numpy as np
from typing import Literal, Callable, List, Dict
from scipy.spatial.distance import euclidean
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import math
import warnings

ArrayLike = np.ndarray | list


class Privacy:
    """
    A class for evaluating the privacy of synthetic data using two types of metrics:

    1. Distance Metrics – measure how similar the distributions of real and synthetic signals are:
       - Wasserstein Distance (WD)
       - Euclidean Distance (ED)
       - Jensen-Shannon Divergence (JSD)

       These metrics are computed using histograms and the minimum distance between real and synthetic samples.

    2. Membership Inference Risk (MIR) – estimates the risk of identifying real training records based on:
       - Prediction confidence
       - Entropy
       - Modified entropy
       - Correctness

       MIR is evaluated using black-box benchmark methods (recommended) or an optional legacy attack model.

    Parameters:
    -----------
    num_bins : int
        Number of histogram bins (default: 30)
    range_bins : tuple
        Value range for histograms (default: (0, 1))

    Example Usage:
    --------------
    real_data = [np.random.rand(1000) for _ in range(5)]  # Simulated real samples
    synthetic_data = [np.random.rand(1000) for _ in range(5)]  # Simulated synthetic samples

    privacy_evaluator = Privacy()

    # Compute distance and MIR metrics
    distance_metrics = privacy_evaluator.compute_distance_metrics(real_data, synthetic_data)
    mir = privacy_evaluator.compute_mir(real_data, synthetic_data)

    References:
    ----------
    [1] https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10568134
    [2] https://www.semanticscholar.org/reader/190169b88f0803e2a6eeb311703ea421461cf3fe
    [3] https://arxiv.org/abs/2404.06787?utm_source=chatgpt.com
    [4] https://github.com/inspire-group/membership-inference-evaluation/tree/master
    [5] https://arxiv.org/abs/2003.10595
    """

    def __init__(self, num_bins: int = 30, range_bins: tuple = (0, 1)):
        """
        Initializes the Privacy class with parameters for histogram-based distance computations.

        Parameters:
        ----------
        num_bins : int, optional
            The number of bins used to construct histograms for metric computation (default is 30).
        range_bins : tuple, optional
            The range of bin edges for histogram construction, specified as (min, max) (default is (0,1)).
        """
        self.num_bins = num_bins
        self.range_bins = range_bins
        self.target_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.attack_model = RandomForestClassifier(n_estimators=100, random_state=0)

    # Generic helper -------------------------------------------------------

    def compute_distance_metric(self,
                                 real_data: ArrayLike,
                                 synthetic_data: ArrayLike,
                                 metric_function: Callable[[np.ndarray, np.ndarray], float]):
        """
        Generalised routine to compute a statistical distance metric using the **per‑real‑record
        minimum** approach.

        For every real sample we locate the **closest** synthetic sample (according to
        *metric_function*) and then average those minima.  This mirrors the intuition
        "Does *every* real record have at least one convincing synthetic counterpart?".

        Parameters
        ----------
        real_data : list[np.ndarray]
            Collection of *N* real signals.
        synthetic_data : list[np.ndarray]
            Collection of *M* synthetic signals.
        metric_function : Callable
            Function that computes the distance between two *flattened* signals.

        Returns
        -------
        tuple
            - **mean_distance** (*float*): Average of the *N* per‑real minima.
            - **min_distance** (*float*): Smallest *overall* distance encountered.
            - **min_real_index** (*int*): Index of the real sample participating in *min_distance*.
            - **min_synthetic_index** (*int*): Index of the synthetic sample participating in *min_distance*.
        """
        # Flatten once to avoid repeated work
        real_data = [np.asarray(signal).flatten() for signal in real_data]
        synthetic_data = [np.asarray(signal).flatten() for signal in synthetic_data]

        per_real_min: List[float] = []   # holds the row minima
        min_distance = float("inf")      # global minimum
        min_real_index = -1
        min_synthetic_index = -1

        # -----------------------------------------------------------------
        # Outer loop ‑‑ iterate over each *real* record --------------------
        # -----------------------------------------------------------------
        for i, real_signal in enumerate(real_data):
            # Compute distances to **all** synthetic signals for this real record
            row_dists = []
            for j, synthetic_signal in enumerate(synthetic_data):
                d = metric_function(real_signal, synthetic_signal)
                row_dists.append(d)

                # Track global min while we're here ----------------------
                if d < min_distance:
                    min_distance = d
                    min_real_index = i
                    min_synthetic_index = j

            # Store the closest synthetic neighbour for real_data[i]
            per_real_min.append(min(row_dists))

        # The score we report is the average of those minima --------------
        mean_distance = float(np.mean(per_real_min)) if per_real_min else np.nan
        return mean_distance, min_distance, min_real_index, min_synthetic_index
  
    # Metric‑specific wrappers --------------------------------------------

    def compute_wasserstein_distance(self, real_data, synthetic_data):
        """
        Compute the **Wasserstein Distance (WD)** between real and synthetic data
        distributions using histogram approximations.
        """

        def wasserstein_func(real_signal, synthetic_signal):
            hist_1, _ = np.histogram(real_signal, bins=self.num_bins, range=self.range_bins, density=True)
            hist_2, _ = np.histogram(synthetic_signal, bins=self.num_bins, range=self.range_bins, density=True)
            bin_midpoints = np.linspace(self.range_bins[0], self.range_bins[1], self.num_bins)
            return wasserstein_distance(bin_midpoints, bin_midpoints,
                                         u_weights=hist_1, v_weights=hist_2)

        return self.compute_distance_metric(real_data, synthetic_data, wasserstein_func)

    def compute_euclidean_distance(self, real_data, synthetic_data):
        """
        Compute the **Euclidean Distance (ED)** between real and synthetic data
        distributions using histogram approximations.
        """

        def euclidean_func(real_signal, synthetic_signal):
            # Histogram → probability distribution
            hist_1, _ = np.histogram(real_signal, bins=self.num_bins, range=self.range_bins, density=True)
            hist_2, _ = np.histogram(synthetic_signal, bins=self.num_bins, range=self.range_bins, density=True)
            hist_1 = hist_1 / np.sum(hist_1)
            hist_2 = hist_2 / np.sum(hist_2)

            # Compute Euclidean distance between the two histograms [0,1]
            distance = euclidean(hist_1, hist_2) / np.sqrt(2.0)
            return distance

        return self.compute_distance_metric(real_data, synthetic_data, euclidean_func)

    def compute_js_divergence(self, real_data, synthetic_data):
        """
        Compute the **Jensen‑Shannon Divergence (JSD)** between real and synthetic data
        distributions using histogram approximations.
        """

        def jsd_func(real_signal, synthetic_signal):
            hist_1, _ = np.histogram(real_signal, bins=self.num_bins, range=self.range_bins, density=True)
            hist_2, _ = np.histogram(synthetic_signal, bins=self.num_bins, range=self.range_bins, density=True)
            hist_1 += 1e-10  # Prevent 0‑probabilities
            hist_2 += 1e-10
            hist_1 /= np.sum(hist_1)
            hist_2 /= np.sum(hist_2)
            return jensenshannon(hist_1, hist_2)

        return self.compute_distance_metric(real_data, synthetic_data, jsd_func)

    def compute_membership_inference(self,X_real: np.ndarray,y_real: np.ndarray,X_synthetic: np.ndarray,
            *,member_split: float = 0.5,attack_test_split: float = 0.3,random_state: int = 42) -> dict[str, float | np.ndarray]:
        """
        Computes Membership Inference Risk (MIR) using black-box statistics.

        This function splits the real data into "members" and "non-members",
        trains a classifier (target model) on the member set, and estimates
        membership leakage based on four metrics:

        - Correctness (whether prediction is right)
        - Confidence (probability of predicted class)
        - Entropy (prediction uncertainty)
        - Modified Entropy (penalized uncertainty on true class)

        It also computes how likely synthetic samples would be considered members
        based on confidence thresholding.

        Returns
        -------
        dict:  {'correctness_attack_acc': float,
                'confidence_attack_acc': float,
                'entropy_attack_acc': float,
                'modified_entropy_attack_acc': float,
                'synthetic_pred_scores': np.ndarray,
                'synthetic_member_fraction': float}
        """

        # Split real data into training (members) and testing (non-members)
        X_mem, X_nonmem, y_mem, y_nonmem = train_test_split(
            X_real, y_real,
            test_size=member_split,
            stratify=y_real,
            random_state=random_state
        )

        # Train the target model on member data
        self.target_model.fit(X_mem, y_mem)

        # Helper function to extract confidence, entropy, etc. for a dataset
        def get_outputs(model, X, y):
            probs = model.predict_proba(X)
            preds = np.argmax(probs, axis=1)
            # Prediction confidence for true class
            conf = probs[np.arange(len(y)), y]
            # Entropy = uncertainty of prediction
            entr = np.sum(probs * np.clip(-np.log(np.maximum(probs, 1e-30)), 0, 100), axis=1)
            # Modified entropy = penalized confidence on true label
            rev_probs = 1 - probs
            log_probs = -np.log(np.maximum(probs, 1e-30))
            log_rev_probs = -np.log(np.maximum(rev_probs, 1e-30))
            mod_probs = probs.copy()
            mod_log = log_rev_probs.copy()
            mod_probs[np.arange(len(y)), y] = rev_probs[np.arange(len(y)), y]
            mod_log[np.arange(len(y)), y] = log_probs[np.arange(len(y)), y]
            mod_entr = np.sum(mod_probs * mod_log, axis=1)
            return {
                'correct': (preds == y).astype(int),
                'conf': conf,
                'entr': entr,
                'mod_entr': mod_entr
            }

        # Compute stats for members and non-members
        stats_mem = get_outputs(self.target_model, X_mem, y_mem)
        stats_nonmem = get_outputs(self.target_model, X_nonmem, y_nonmem)

        # Compute attack accuracy using thresholding for each statistic
        def infer_acc(name, tr_vals, te_vals):
            all_vals = np.concatenate([tr_vals, te_vals])
            best_acc = 0.0
            for t in all_vals:
                acc = 0.5 * (
                        np.sum(tr_vals >= t) / len(tr_vals) +
                        np.sum(te_vals < t) / len(te_vals)
                )
                if acc > best_acc:
                    best_acc = acc
            print(f"🕵️ Attack via {name}: acc = {best_acc:.3f}")
            return float(best_acc)

        acc_corr = 0.5 * (
                np.mean(stats_mem['correct']) +
                1 - np.mean(stats_nonmem['correct'])
        )
        print(f"🕵️ Attack via correctness: acc = {acc_corr:.3f}")

        acc_conf = infer_acc('confidence', stats_mem['conf'], stats_nonmem['conf'])
        acc_entr = infer_acc('entropy', -stats_mem['entr'], -stats_nonmem['entr'])
        acc_mod_entr = infer_acc('modified entropy', -stats_mem['mod_entr'], -stats_nonmem['mod_entr'])

        # Compute synthetic membership likelihood using confidence threshold
        syn_probs = self.target_model.predict_proba(X_synthetic)
        syn_conf = syn_probs.max(axis=1)
        threshold = np.median(stats_mem['conf'])  # Use training members as baseline
        syn_member_fraction = float((syn_conf > threshold).mean())

        return {
            'correctness_attack_acc': acc_corr,
            'confidence_attack_acc': acc_conf,
            'entropy_attack_acc': acc_entr,
            'modified_entropy_attack_acc': acc_mod_entr,
            'synthetic_pred_scores': syn_conf,
            'synthetic_member_fraction': syn_member_fraction
        }

    def compute_privacy_metrics(self, real_data, synthetic_data,
                                X_real=None, y_real=None, X_synth=None,
                                shadow_train=None, shadow_test=None,
                                target_train=None, target_test=None,
                                use_new_mir=True):
        """
           Computes all privacy evaluation metrics in one step.

           Includes:
           - Statistical distances (WD, ED, JSD) using histogram comparison
           - Membership Inference Risk (MIR), either:
               - Black-box benchmark (default)
               - Legacy attack model (if use_new_mir=False)

           Parameters
           ----------
           real_data : list or np.ndarray
               Real signals (used for histogram distance metrics)
           synthetic_data : list or np.ndarray
               Synthetic signals (used for histogram distance metrics)
           X_real, y_real : np.ndarray, optional
               Real tabular data and labels (used for legacy MIR)
           X_synth : np.ndarray, optional
               Synthetic tabular data (used for legacy MIR)
           shadow_train, shadow_test, target_train, target_test : tuple, optional
               Each is a (probabilities, labels) tuple for black-box MIR
           use_new_mir : bool
               Whether to use the benchmark-style black-box MIR (default: True)

           Returns
           -------
           dict
               Dictionary with keys:
               - wd, ed, jsd: average distances
               - *_min, *_real_idx, *_synth_idx: info on closest pair
               - mir_*: various membership inference accuracy metrics
           """

        # Distance metrics
        wd, wd_min, wd_real_idx, wd_synth_idx = self.compute_wasserstein_distance(real_data, synthetic_data)
        ed, ed_min, ed_real_idx, ed_synth_idx = self.compute_euclidean_distance(real_data, synthetic_data)
        jsd, jsd_min, jsd_real_idx, jsd_synth_idx = self.compute_js_divergence(real_data, synthetic_data)

        result = {
            "wd": wd, "wd_min": wd_min, "wd_real_idx": wd_real_idx, "wd_synth_idx": wd_synth_idx,
            "ed": ed, "ed_min": ed_min, "ed_real_idx": ed_real_idx, "ed_synth_idx": ed_synth_idx,
            "jsd": jsd, "jsd_min": jsd_min, "jsd_real_idx": jsd_real_idx, "jsd_synth_idx": jsd_synth_idx,
        }

        # Membership Inference Risk - Black-box benchmark method
        if use_new_mir and all(v is not None for v in [shadow_train, shadow_test, target_train, target_test]):
            try:
                import numpy as np

                def _log_value(probs, eps=1e-30):
                    return -np.log(np.maximum(probs, eps))

                def _entropy(probs):
                    return np.sum(probs * _log_value(probs), axis=1)

                def _modified_entropy(probs, labels):
                    log_probs = _log_value(probs)
                    reverse_probs = 1 - probs
                    log_reverse = _log_value(reverse_probs)
                    mod_probs = np.copy(probs)
                    mod_log = np.copy(log_reverse)
                    mod_probs[np.arange(len(labels)), labels] = reverse_probs[np.arange(len(labels)), labels]
                    mod_log[np.arange(len(labels)), labels] = log_probs[np.arange(len(labels)), labels]
                    return np.sum(mod_probs * mod_log, axis=1)

                def _correctness(probs, labels):
                    return (np.argmax(probs, axis=1) == labels).astype(int)

                def _threshold_acc(val_tr, val_te):
                    all_vals = np.concatenate([val_tr, val_te])
                    best_acc = 0
                    for t in all_vals:
                        acc = 0.5 * (
                                np.sum(val_tr >= t) / len(val_tr) +
                                np.sum(val_te < t) / len(val_te)
                        )
                        if acc > best_acc:
                            best_acc = acc
                    return best_acc

                s_tr_out, s_tr_y = shadow_train
                s_te_out, s_te_y = shadow_test
                t_tr_out, t_tr_y = target_train
                t_te_out, t_te_y = target_test

                acc_results = {}
                acc_results["mir_conf_acc"] = _threshold_acc(
                    np.array([s_tr_out[i, s_tr_y[i]] for i in range(len(s_tr_y))]),
                    np.array([s_te_out[i, s_te_y[i]] for i in range(len(s_te_y))])
                )
                acc_results["mir_entropy_acc"] = _threshold_acc(
                    -_entropy(s_tr_out), -_entropy(s_te_out)
                )
                acc_results["mir_mod_entropy_acc"] = _threshold_acc(
                    -_modified_entropy(s_tr_out, s_tr_y), -_modified_entropy(s_te_out, s_te_y)
                )
                acc_results["mir_correctness_acc"] = 0.5 * (
                        np.sum(_correctness(t_tr_out, t_tr_y)) / len(t_tr_y) +
                        (1 - np.sum(_correctness(t_te_out, t_te_y)) / len(t_te_y))
                )

                result.update(acc_results)

            except Exception as e:
                print(f"[Warning] MIR (black-box style) computation failed: {e}")

        # Legacy membership inference method (optional fallback)
        elif not use_new_mir and X_real is not None and y_real is not None and X_synth is not None:
            try:
                mir_results = self.compute_membership_inference(X_real, y_real, X_synth)

                # Attach accuracies with consistent keys
                result.update({
                    "confidence_attack_acc": mir_results["confidence_attack_acc"],
                    "correctness_attack_acc": mir_results["correctness_attack_acc"],
                    "entropy_attack_acc": mir_results["entropy_attack_acc"],
                    "modified_entropy_attack_acc": mir_results["modified_entropy_attack_acc"],
                    "synthetic_member_fraction": mir_results["synthetic_member_fraction"],
                })

            except Exception as e:
                print(f"[Warning] Legacy MIR computation failed: {e}")

        return result

    def compute_distance_metrics(self, real_data, synthetic_data):
        """
        Print only the distance metrics (WD, ED, JSD) between real and synthetic data.
        """
        print("📏 Distance Metrics Between Real and Synthetic Data:")
        metrics = self.compute_privacy_metrics(real_data, synthetic_data)

        if not isinstance(metrics, dict):
            raise RuntimeError(
                "compute_privacy_metrics returned None or a non-dict. "
                "Check for early returns or exceptions inside that function."
            )

        # Safely read with .get so a missing key doesn’t crash
        wd = metrics.get('wd');
        wd_min = metrics.get('wd_min')
        ed = metrics.get('ed');
        ed_min = metrics.get('ed_min')
        jsd = metrics.get('jsd');
        jsd_min = metrics.get('jsd_min')

        # If any are None, set to np.nan for printing
        for k in ("wd", "wd_min", "ed", "ed_min", "jsd", "jsd_min"):
            if metrics.get(k) is None:
                metrics[k] = np.nan

        print(f"  - Wasserstein Distance (WD): {metrics['wd']:.4f} (min: {metrics['wd_min']:.4f})")
        print(f"  - Euclidean Distance   (ED): {metrics['ed']:.4f} (min: {metrics['ed_min']:.4f})")
        print(f"  - Jensen–Shannon Divergence (JSD): {metrics['jsd']:.4f} (min: {metrics['jsd_min']:.4f})\n")

        return metrics
        
    def compute_mir(self, real_data, synthetic_data):
        """
        Compute and print Membership Inference Risk (MIR) metrics between real and synthetic data.
        Requires real_data and synthetic_data to be numpy arrays and properly labeled.
        """
        # Convert real/synthetic to fake tabular format if needed
        X_real = np.stack(real_data)
        num_samples = len(X_real)
        half = num_samples // 2
        y_real = np.array([0] * half + [1] * (num_samples - half))
        np.random.shuffle(y_real)
        X_synth = np.stack(synthetic_data)

        print("🔐 Membership Inference Risk (MIR) Metrics:")
        privacy = Privacy()
        metrics = privacy.compute_privacy_metrics(
            real_data, synthetic_data,
            X_real=X_real, y_real=y_real, X_synth=X_synth,
            use_new_mir=False  # using legacy attack for simplicity
        )
        print(f"  - Correctness Attack Acc: {metrics.get('correctness_attack_acc', 'n/a')}")
        print(f"  - Confidence Attack Acc: {metrics.get('confidence_attack_acc', 'n/a')}")
        print(f"  - Entropy Attack Acc: {metrics.get('entropy_attack_acc', 'n/a')}")
        print(f"  - Modified Entropy Attack Acc: {metrics.get('modified_entropy_attack_acc', 'n/a')}")
        print(f"  - Synthetic Member Fraction: {metrics.get('synthetic_member_fraction', 'n/a')}\n")

