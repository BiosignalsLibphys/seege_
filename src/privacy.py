import numpy as np
from typing import Callable, List
from scipy.spatial.distance import euclidean
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


ArrayLike = np.ndarray | list


class Privacy:
    """
    A class for evaluating the privacy of synthetic data using:

    1. Nearest-neighbour signal distances:
       - Euclidean distance (L2)
       - Cosine distance (1 − cosine similarity)
       - Dynamic Time Warping (DTW) distance

       For each real signal, the closest synthetic signal is found according to
       each metric, and the average of these per-real minima is reported.

       Distances can be computed:
       - on raw signals (normalize=None)
       - after global z-score with respect to REAL data ("zscore_global")
       - after per-signal z-score ("zscore_per_signal")

        1.1. Distance effect sizes (Cohen-style):
        Compare how close synthetic samples are to real data (R–S NN distances)
        relative to how close real samples are to each other (R–R NN distances).

    2. Membership Inference Risk (MIR) – estimates the risk of identifying
       real training records based on:
       - Prediction confidence
       - Entropy
       - Modified entropy
       - Correctness

     ⚠ MIR REQUIRES TRUE LABELS (y_real)
       -----------------------------------
       MIR is defined with respect to a supervised classifier trained on REAL data.
       Therefore, `compute_mir_metrics` needs the TRUE labels of the real signals
       (y_real). These must be the actual task labels used to train a meaningful
       model (e.g., pathology vs physiology, noise vs clean, etc.).

    Example Usage:
    --------------
    real_data = [np.random.rand(1000) for _ in range(5)] # Simulated real samples
    synthetic_data = [np.random.rand(1000) for _ in range(5)] # Simulated synthetic samples

    privacy_evaluator = Privacy()

    # NN distances (signal-level)
    distance_metrics = privacy_evaluator.compute_distance_metrics(real_data, synthetic_data)

    # MIR (model-level) – you provide labels for real signals
    y_real = np.array([...]) # true labels (e.g., pathology vs physiology)
    mir_metrics = privacy.compute_mir_metrics(real_signals, synthetic_signals, y_real)

    References:
    ----------
    [1] https://github.com/inspire-group/membership-inference-evaluation/tree/master
    [2] https://arxiv.org/abs/2003.10595
    """

    def __init__(self):
        self.target_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.attack_model = RandomForestClassifier(n_estimators=100, random_state=0)

    # Normalisation helpers

    def _normalize_signals(
        self,
        real_data: ArrayLike,
        synthetic_data: ArrayLike,
        mode: str | None = None,
    ):
        """
        Normalize real and synthetic signals according to `mode`.

        Parameters
        ----------
        mode:
            - None or "none": no normalization
            - "zscore_global": z-score using global mean/std of REAL data
            - "zscore_per_signal": z-score each signal independently

        Returns
        -------
        R_norm, S_norm, stats
        """
        # Flatten everything first
        R = [np.asarray(s, float).flatten() for s in real_data]
        S = [np.asarray(s, float).flatten() for s in synthetic_data]

        if mode is None or mode == "none":
            return R, S, {}

        if mode == "zscore_global":
            all_real = np.concatenate(R)
            mean = float(all_real.mean())
            std = float(all_real.std())
            if std == 0:
                std = 1.0
            R = [(r - mean) / std for r in R]
            S = [(s - mean) / std for s in S]
            return R, S, {"mean": mean, "std": std}

        if mode == "zscore_per_signal":
            def z_per(sig: np.ndarray) -> np.ndarray:
                m = sig.mean()
                s = sig.std()
                if s == 0:
                    s = 1.0
                return (sig - m) / s

            R = [z_per(r) for r in R]
            S = [z_per(s) for s in S]
            return R, S, {}

        raise ValueError(f"Unknown normalization mode: {mode}")


    # Generic helpers for NN-based distances

    def _nn_min_distances(
        self,
        data_A: ArrayLike,
        data_B: ArrayLike,
        metric_function: Callable[[np.ndarray, np.ndarray], float],
        *,
        normalize: str | None = None,
        length_normalize: bool = False,
        exclude_self: bool = False,
    ) -> np.ndarray:
        """
        For each signal in data_A, compute the distance to its nearest neighbour
        in data_B and return the vector of per-sample minima.

        If exclude_self=True and data_A and data_B represent the same dataset
        (R–R case), the distance to the sample itself (i == j) is ignored.
        """
        A_norm, B_norm, _ = self._normalize_signals(data_A, data_B, mode=normalize)

        per_A_min: list[float] = []

        for i, sig_A in enumerate(A_norm):
            best = float("inf")
            for j, sig_B in enumerate(B_norm):
                if exclude_self and i == j:
                    continue
                d = metric_function(sig_A, sig_B)
                if length_normalize:
                    L = sig_A.size
                    if L > 0:
                        d /= np.sqrt(L)
                if d < best:
                    best = d
            per_A_min.append(best)

        return np.asarray(per_A_min, dtype=float)

    def compute_distance_metric(
        self,
        real_data: ArrayLike,
        synthetic_data: ArrayLike,
        metric_function: Callable[[np.ndarray, np.ndarray], float],
        *,
        normalize: str | None = None,
        length_normalize: bool = False,
    ):
        """
        Compute a nearest-neighbour distance metric (R–S only).

        For every real sample we locate the closest synthetic sample
        (according to *metric_function*) and then average those minima.
        """
        rs = self._nn_min_distances(
            real_data,
            synthetic_data,
            metric_function,
            normalize=normalize,
            length_normalize=length_normalize,
            exclude_self=False,
        )

        mean_distance = float(rs.mean()) if rs.size else np.nan
        min_distance = float(rs.min()) if rs.size else np.nan

        # Index of the real sample that attains the global minimum
        min_real_index = int(np.argmin(rs)) if rs.size else -1

        # To recover the corresponding synthetic index, recompute for that real
        min_synthetic_index = -1
        if min_real_index >= 0:
            # Re-normalise only that pair set
            R_norm, S_norm, _ = self._normalize_signals(
                [real_data[min_real_index]], synthetic_data, mode=normalize
            )
            sig_A = R_norm[0]
            best = float("inf")
            best_j = -1
            for j, sig_B in enumerate(S_norm):
                d = metric_function(sig_A, sig_B)
                if length_normalize:
                    L = sig_A.size
                    if L > 0:
                        d /= np.sqrt(L)
                if d < best:
                    best = d
                    best_j = j
            min_synthetic_index = best_j

        return mean_distance, min_distance, min_real_index, min_synthetic_index

    # Raw-signal distance metrics

    @staticmethod
    def _cosine_distance(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
        """
        Cosine distance = 1 − cosine similarity.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        nx = np.linalg.norm(x)
        ny = np.linalg.norm(y)
        if nx < eps or ny < eps:
            return 1.0  # maximally dissimilar if one vector is (almost) zero
        cos_sim = float(np.dot(x, y) / (nx * ny))
        cos_sim = max(min(cos_sim, 1.0), -1.0)  # clip
        return 1.0 - cos_sim

    @staticmethod
    def _dtw_distance(x: np.ndarray, y: np.ndarray) -> float:
        """
        Simple DTW distance (O(N*M) DP implementation).
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n, m = len(x), len(y)
        cost = np.full((n + 1, m + 1), np.inf)
        cost[0, 0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                diff = x[i - 1] - y[j - 1]
                d = abs(diff)
                cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

        return float(cost[n, m])

    def compute_l2_distance(
        self,
        real_data,
        synthetic_data,
        *,
        normalize: str | None = "zscore_global",
        length_normalize: bool = True,
    ):
        """
        Nearest-neighbour Euclidean (L2) distance on signals.
        """
        def l2_func(real_signal, synthetic_signal):
            return euclidean(real_signal, synthetic_signal)

        return self.compute_distance_metric(
            real_data,
            synthetic_data,
            l2_func,
            normalize=normalize,
            length_normalize=length_normalize,
        )

    def compute_cosine_distance(
        self,
        real_data,
        synthetic_data,
        *,
        normalize: str | None = "zscore_global",
    ):
        """
        Nearest-neighbour cosine distance (1 − cosine similarity) on signals.
        """
        def cos_func(real_signal, synthetic_signal):
            return self._cosine_distance(real_signal, synthetic_signal)

        return self.compute_distance_metric(
            real_data,
            synthetic_data,
            cos_func,
            normalize=normalize,
            length_normalize=False,  # cosine already scale-insensitive
        )

    def compute_dtw_distance(
        self,
        real_data,
        synthetic_data,
        *,
        normalize: str | None = "zscore_global",
        length_normalize: bool = True,
    ):
        """
        Nearest-neighbour DTW distance on signals.
        """
        def dtw_func(real_signal, synthetic_signal):
            return self._dtw_distance(real_signal, synthetic_signal)

        return self.compute_distance_metric(
            real_data,
            synthetic_data,
            dtw_func,
            normalize=normalize,
            length_normalize=length_normalize,
        )


    # Distance-only privacy metrics (R–S)

    def compute_privacy_metrics(
        self,
        real_data,
        synthetic_data,
        *,
        normalize: str | None = "zscore_global",
        length_normalize: bool = True,
    ):
        """
        Compute distance-based privacy metrics (no MIR).

        Includes:
        - NN distances (L2, cosine, DTW) on signals.
        """
        l2, l2_min, l2_real_idx, l2_synth_idx = self.compute_l2_distance(
            real_data, synthetic_data,
            normalize=normalize,
            length_normalize=length_normalize,
        )
        cos, cos_min, cos_real_idx, cos_synth_idx = self.compute_cosine_distance(
            real_data, synthetic_data,
            normalize=normalize,
        )
        dtw, dtw_min, dtw_real_idx, dtw_synth_idx = self.compute_dtw_distance(
            real_data, synthetic_data,
            normalize=normalize,
            length_normalize=length_normalize,
        )

        result = {
            "l2": l2, "l2_min": l2_min, "l2_real_idx": l2_real_idx, "l2_synth_idx": l2_synth_idx,
            "cosine": cos, "cosine_min": cos_min, "cosine_real_idx": cos_real_idx, "cosine_synth_idx": cos_synth_idx,
            "dtw": dtw, "dtw_min": dtw_min, "dtw_real_idx": dtw_real_idx, "dtw_synth_idx": dtw_synth_idx,
        }
        return result

    def compute_distance_metrics(
            self,
            real_data,
            synthetic_data,
            *,
            normalize: str | None = "zscore_global",
            length_normalize: bool = True,
    ):
        """
        Compute and print nearest-neighbour distance metrics **and**
        their Cohen-style effect sizes.

        Distances are:
            - L2 (Euclidean) NN distance
            - Cosine NN distance (1 − cosine similarity)
            - DTW NN distance


        All distances are computed after the chosen `normalize` step.
        With the defaults (normalize="zscore_global", length_normalize=True),
        L2 and DTW can be read as RMSE-like distances in units of REAL-data
        standard deviation per sample.
        """
        print("📏 Nearest-Neighbour distances between real and synthetic data:")
        if normalize == "zscore_global":
            print("    -> Signals z-scored using real data SD "
                  "(L2/DTW in units of real SD per sample).")

        # --- 1) R–S distances ---
        metrics = self.compute_privacy_metrics(
            real_data,
            synthetic_data,
            normalize=normalize,
            length_normalize=length_normalize,
        )

        if not isinstance(metrics, dict):
            raise RuntimeError(
                "compute_privacy_metrics returned None or a non-dict. "
                "Check for early returns or exceptions inside that function."
            )

        for k in ("l2", "l2_min", "cosine", "cosine_min", "dtw", "dtw_min"):
            if metrics.get(k) is None:
                metrics[k] = np.nan

        # 2) Effect sizes d (R–S vs R–R)
        eff = self.compute_distance_effect_sizes(
            real_data,
            synthetic_data,
            normalize=normalize,
            length_normalize=length_normalize,
        )

        # Attach effect sizes (and optionally RR baselines) to the dict
        metrics["l2_d"] = eff["l2"]["effect_size_d"]
        metrics["cosine_d"] = eff["cos"]["effect_size_d"]
        metrics["dtw_d"] = eff["dtw"]["effect_size_d"]

        metrics["l2_rr_mean"] = eff["l2"]["real_real_mean"]
        metrics["cosine_rr_mean"] = eff["cos"]["real_real_mean"]
        metrics["dtw_rr_mean"] = eff["dtw"]["real_real_mean"]

        # Printing helper
        def _interpret_d(d: float) -> str:
            if d < 0.20:
                return "negligible (very high privacy risk)"
            elif d < 0.50:
                return "small (high privacy risk)"
            elif d < 0.80:
                return "medium (moderate privacy risk)"
            else:
                return "large (low privacy risk/ high separation)"

        print(f"L2 distance (mean NN): {metrics['l2']:.4f} "
              f"(min: {metrics['l2_min']:.4f})")
        print(f"    - L2 effect size d (R–S vs R–R): {metrics['l2_d']:.2f} "
              f"- {_interpret_d(metrics['l2_d'])}")

        print(f"Cosine distance (mean NN): {metrics['cosine']:.4f} "
              f"(min: {metrics['cosine_min']:.4f})")
        print(f"    - Cosine effect size d: {metrics['cosine_d']:.2f} "
              f"- {_interpret_d(metrics['cosine_d'])}")

        print(f"DTW distance (mean NN): {metrics['dtw']:.4f} "
              f"(min: {metrics['dtw_min']:.4f})")
        print(f"    - DTW effect size d: {metrics['dtw_d']:.2f} "
              f"- {_interpret_d(metrics['dtw_d'])}\n")

        return metrics

    # Distance effect sizes (Cohen-style)

    def compute_distance_effect_sizes(
        self,
        real_data: ArrayLike,
        synthetic_data: ArrayLike,
        *,
        normalize: str | None = "zscore_global",
        length_normalize: bool = True,
    ) -> dict:
        """
        Compute Cohen-style effect sizes comparing:

            - R–R NN distances (baseline variability of real data)
            - R–S NN distances (how close synthetic data gets to real)

        For each metric M ∈ {L2, COS, DTW}:

            d_M = ( mean_R-S_M − mean_R-R_M ) / std_R-R_M

        Interpretation (Cohen, 1988, adapted):

            d < 0.20      : negligible / very high privacy
            0.20 ≤ d < 0.50 : small effect / high privacy
            0.50 ≤ d < 0.80 : medium effect / moderate privacy
            d ≥ 0.80        : large effect / low privacy
        """
        results = {}

        # Helper to avoid repetition
        metrics = {
            "l2":  (lambda x, y: euclidean(x, y), True),
            "cos": (self._cosine_distance, False),
            "dtw": (self._dtw_distance, True),
        }

        for name, (metric_func, len_norm) in metrics.items():
            # R–R baseline: NN distance to other real signals (exclude self)
            rr = self._nn_min_distances(
                real_data,
                real_data,
                metric_func,
                normalize=normalize,
                length_normalize=len_norm and length_normalize,
                exclude_self=True,
            )

            # R–S distances: NN distance to synthetic
            rs = self._nn_min_distances(
                real_data,
                synthetic_data,
                metric_func,
                normalize=normalize,
                length_normalize=len_norm and length_normalize,
                exclude_self=False,
            )

            mu_rr = float(rr.mean())
            std_rr = float(rr.std(ddof=1))
            if std_rr == 0:
                std_rr = 1.0
            mu_rs = float(rs.mean())

            d = (mu_rs - mu_rr) / std_rr

            results[name] = {
                "real_real_mean": mu_rr,
                "real_real_std": std_rr,
                "real_synth_mean": mu_rs,
                "effect_size_d": d,
            }

        return results

    # Membership inference (core engine, on feature matrices)

    def compute_membership_inference(
        self,
        X_real: np.ndarray,
        y_real: np.ndarray,
        X_synthetic: np.ndarray,
        *,
        member_split: float = 0.5,
        attack_test_split: float = 0.3,  # kept for future extension
        random_state: int = 42,
    ) -> dict[str, float | np.ndarray]:
        """
        Core membership inference attack.

        Splits real data into "members" (training) and "non-members",
        trains a target model on members, and evaluates attack success
        using correctness, confidence, entropy, and modified entropy.
        """
        X_mem, X_nonmem, y_mem, y_nonmem = train_test_split(
            X_real, y_real,
            test_size=member_split,
            stratify=y_real,
            random_state=random_state
        )

        self.target_model.fit(X_mem, y_mem)

        def get_outputs(model, X, y):
            probs = model.predict_proba(X)
            preds = np.argmax(probs, axis=1)
            conf = probs[np.arange(len(y)), y]
            entr = np.sum(
                probs * np.clip(-np.log(np.maximum(probs, 1e-30)), 0, 100),
                axis=1
            )
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

        stats_mem = get_outputs(self.target_model, X_mem, y_mem)
        stats_nonmem = get_outputs(self.target_model, X_nonmem, y_nonmem)

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

        syn_probs = self.target_model.predict_proba(X_synthetic)
        syn_conf = syn_probs.max(axis=1)
        threshold = np.median(stats_mem['conf'])
        syn_member_fraction = float((syn_conf > threshold).mean())

        return {
            'correctness_attack_acc': acc_corr,
            'confidence_attack_acc': acc_conf,
            'entropy_attack_acc': acc_entr,
            'modified_entropy_attack_acc': acc_mod_entr,
            'synthetic_pred_scores': syn_conf,
            'synthetic_member_fraction': syn_member_fraction
        }

    # Public MIR-only wrapper (builds X_real and X_synth internally)

    def compute_mir_metrics(
        self,
        real_data: ArrayLike,
        synthetic_data: ArrayLike,
        y_real: np.ndarray,
        *,
        normalize: str | None = "zscore_global",
        member_split: float = 0.5,
        attack_test_split: float = 0.3,
        random_state: int = 42,
        verbose: bool = True,
    ) -> dict[str, float | np.ndarray]:
        """
        Compute and (optionally) print Membership Inference Risk (MIR) metrics.
        """
        # 1) Normalize + flatten signals (same as distances)
        R_norm, S_norm, _ = self._normalize_signals(
            real_data, synthetic_data, mode=normalize
        )

        # 2) Build feature matrices by stacking flattened signals
        X_real = np.stack(R_norm)
        X_synth = np.stack(S_norm)

        # 3) Run the core membership inference engine
        mir_results = self.compute_membership_inference(
            X_real=X_real,
            y_real=y_real,
            X_synthetic=X_synth,
            member_split=member_split,
            attack_test_split=attack_test_split,
            random_state=random_state,
        )

        if verbose:
            print("🔐 Membership Inference Risk (MIR) Metrics:")
            print(f"  - Correctness attack acc     : {mir_results['correctness_attack_acc']:.3f}")
            print(f"  - Confidence attack acc      : {mir_results['confidence_attack_acc']:.3f}")
            print(f"  - Entropy attack acc         : {mir_results['entropy_attack_acc']:.3f}")
            print(f"  - Modified entropy attack acc: {mir_results['modified_entropy_attack_acc']:.3f}")
            print(f"  - Synthetic member fraction  : {mir_results['synthetic_member_fraction']:.3f}\n")

        return mir_results


