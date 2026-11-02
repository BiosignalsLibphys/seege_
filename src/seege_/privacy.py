import numpy as np
from scipy.spatial.distance import cdist
from numba import njit, prange
import numba
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm


ArrayLike = np.ndarray | list


def _l2_nn_matrix(A: np.ndarray, B: np.ndarray, *, exclude_self: bool = False,
                   length_normalize: bool = False, desc: str | None = None,
                   chunk_size: int = 500):
    """
    Nearest-neighbour Euclidean distance from every row of A to B, computed in
    chunks with `scipy.spatial.distance.cdist` (one optimized BLAS call per
    chunk instead of a Python loop over every (i, j) pair). Chunking keeps
    peak memory bounded (chunk_size x n_B instead of n_A x n_B) and gives a
    tqdm bar without losing the vectorized speed.
    """
    nA, nB = A.shape[0], B.shape[0]
    mins = np.empty(nA, dtype=np.float64)
    argmins = np.empty(nA, dtype=np.int64)

    starts = range(0, nA, chunk_size)
    if desc:
        starts = tqdm(list(starts), desc=desc, unit="chunk")

    for start in starts:
        end = min(start + chunk_size, nA)
        D = cdist(A[start:end], B, metric="euclidean")
        if exclude_self:
            for row, i in enumerate(range(start, end)):
                if i < nB:
                    D[row, i] = np.inf
        mins[start:end] = D.min(axis=1)
        argmins[start:end] = D.argmin(axis=1)

    if length_normalize:
        mins = mins / np.sqrt(A.shape[1])
    return mins, argmins


def _cosine_nn_matrix(A: np.ndarray, B: np.ndarray, *, exclude_self: bool = False,
                       eps: float = 1e-12):
    """Nearest-neighbour cosine distance, vectorized via a single matmul."""
    An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), eps)
    Bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), eps)
    sim = np.clip(An @ Bn.T, -1.0, 1.0)
    D = 1.0 - sim
    if exclude_self:
        n = min(D.shape)
        D[np.arange(n), np.arange(n)] = np.inf
    mins = D.min(axis=1)
    argmins = D.argmin(axis=1)
    return mins, argmins


@njit(cache=True, fastmath=True, nogil=True)
def _dtw_core(x: np.ndarray, y: np.ndarray) -> float:
    """Exact O(N*M) DTW DP recursion, JIT-compiled. `nogil=True` lets this run
    concurrently across threads inside the parallel kernel below."""
    n = x.shape[0]
    m = y.shape[0]
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        xi = x[i - 1]
        for j in range(1, m + 1):
            d = abs(xi - y[j - 1])
            prev = cost[i - 1, j]
            if cost[i, j - 1] < prev:
                prev = cost[i, j - 1]
            if cost[i - 1, j - 1] < prev:
                prev = cost[i - 1, j - 1]
            cost[i, j] = d + prev
    return cost[n, m]


@njit(cache=True, fastmath=True, parallel=True)
def _dtw_nn_block(A: np.ndarray, B: np.ndarray, exclude_self: bool, row_offset: int):
    """
    Nearest-neighbour DTW distance from every row of A to B, computed with
    numba `prange` -- i.e. a real multi-threaded parallel-for inside a single
    compiled kernel, operating directly on the shared arrays. This replaces
    the previous joblib.Parallel(process-based) version: no worker processes,
    no pickling A/B out to each one, just threads sharing memory. `row_offset`
    lets this be called per-chunk (for progress reporting) while still
    excluding the correct self-index when A and B are the same dataset.
    """
    nA = A.shape[0]
    nB = B.shape[0]
    mins = np.empty(nA, dtype=np.float64)
    argmins = np.empty(nA, dtype=np.int64)
    for i in prange(nA):
        best = np.inf
        best_j = -1
        global_i = i + row_offset
        for j in range(nB):
            if exclude_self and global_i == j:
                continue
            d = _dtw_core(A[i], B[j])
            if d < best:
                best = d
                best_j = j
        mins[i] = best
        argmins[i] = best_j
    return mins, argmins


def _dtw_nn_matrix(A: np.ndarray, B: np.ndarray, *, exclude_self: bool = False,
                    length_normalize: bool = False, desc: str | None = None,
                    chunk_size: int = 200):
    """Chunked driver around `_dtw_nn_block` -- chunks give a live tqdm bar
    without breaking numba's parallel-for within each chunk."""
    A = np.ascontiguousarray(A, dtype=np.float64)
    B = np.ascontiguousarray(B, dtype=np.float64)
    nA = A.shape[0]
    mins = np.empty(nA, dtype=np.float64)
    argmins = np.empty(nA, dtype=np.int64)

    starts = range(0, nA, chunk_size)
    if desc:
        starts = tqdm(list(starts), desc=desc, unit="chunk")

    for start in starts:
        end = min(start + chunk_size, nA)
        m, a = _dtw_nn_block(A[start:end], B, exclude_self, start)
        mins[start:end] = m
        argmins[start:end] = a

    if length_normalize:
        mins = mins / np.sqrt(A.shape[1])
    return mins, argmins


class Privacy:
    """
    Privacy evaluation for synthetic data via nearest-neighbour signal distances:
       - Euclidean distance (L2)
       - Cosine distance (1 - cosine similarity)
       - Dynamic Time Warping (DTW) distance -- numba-JIT compiled

    For each real signal, the closest synthetic signal is found according to
    each metric, and the average of these per-real minima is reported.

    Distances can be computed:
       - on raw signals (normalize=None)
       - after global z-score with respect to REAL data ("zscore_global")
       - after per-signal z-score ("zscore_per_signal")

    Distance effect sizes (Cohen-style):
        Compare how close synthetic samples are to real data (R-S NN distances)
        relative to how close real samples are to each other (R-R NN distances):

            d_M = (mean_R-S_M - mean_R-R_M) / std_R-R_M

    2. Membership Inference Risk (MIR) -- estimates the risk of identifying
       real training records based on:
       - Prediction confidence
       - Entropy
       - Modified entropy
       - Correctness

     WARNING: MIR REQUIRES TRUE LABELS (y_real)
       -----------------------------------
       MIR is defined with respect to a supervised classifier trained on REAL
       data. Therefore, `compute_mir_metrics` needs the TRUE labels of the
       real signals (y_real). These must be the actual task labels used to
       train a meaningful model (e.g., pathology vs physiology, noise vs
       clean, etc.).

    Example Usage:
    --------------
    real_data = [np.random.rand(1000) for _ in range(5)]
    synthetic_data = [np.random.rand(1000) for _ in range(5)]

    privacy_evaluator = Privacy()

    # NN distances (signal-level)
    distance_metrics = privacy_evaluator.compute_distance_metrics(
        real_data, synthetic_data, metrics=("l2", "dtw")
    )

    # MIR (model-level) -- you provide labels for real signals
    y_real = np.array([...])  # true labels (e.g., pathology vs physiology)
    mir_metrics = privacy_evaluator.compute_mir_metrics(real_data, synthetic_data, y_real)

    References:
    ----------
    [1] https://github.com/inspire-group/membership-inference-evaluation/tree/master
    [2] https://arxiv.org/abs/2003.10595
    """

    def __init__(self):
        self.target_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.attack_model = RandomForestClassifier(n_estimators=100, random_state=0)

    # Normalisation helpers

    def _normalize_matrix(self, real_data: ArrayLike, synthetic_data: ArrayLike,
                           mode: str | None = None):
        """
        Normalize real and synthetic signals according to `mode`, returning
        contiguous (n_signals, T) float64 matrices (requires equal-length
        signals within each of real/synthetic -- true for fixed-window
        epochs and for however this pipeline's .npy subsets are stacked).

        mode:
            - None or "none": no normalization
            - "zscore_global": z-score using global mean/std of REAL data
            - "zscore_per_signal": z-score each signal independently
        """
        R = np.ascontiguousarray(real_data, dtype=np.float64)
        S = np.ascontiguousarray(synthetic_data, dtype=np.float64)
        if R.ndim == 1:
            R = R[None, :]
        if S.ndim == 1:
            S = S[None, :]

        if mode is None or mode == "none":
            return R, S

        if mode == "zscore_global":
            mean = float(R.mean())
            std = float(R.std()) or 1.0
            return (R - mean) / std, (S - mean) / std

        if mode == "zscore_per_signal":
            def z(X):
                m = X.mean(axis=1, keepdims=True)
                s = X.std(axis=1, keepdims=True)
                s[s == 0] = 1.0
                return (X - m) / s
            return z(R), z(S)

        raise ValueError(f"Unknown normalization mode: {mode}")

    _KERNELS = {
        "l2": (_l2_nn_matrix, True),
        "cosine": (_cosine_nn_matrix, False),
        "dtw": (_dtw_nn_matrix, True),
    }

    # Distance-only privacy metrics (R-S) + effect sizes, matching the paper table

    def compute_distance_metrics(
        self,
        real_data,
        synthetic_data,
        *,
        normalize: str | None = "zscore_global",
        length_normalize: bool = True,
        metrics: tuple = ("l2", "cosine", "dtw"),
        compute_effect_sizes: bool = True,
        n_jobs: int = -1,
    ):
        """
        NN distance metrics (mean + min, over per-real-record minima) plus,
        unless compute_effect_sizes=False, the R-S vs R-R Cohen-style effect
        size `d` and the R-R baseline mean -- restricted to whichever of
        "l2"/"cosine"/"dtw" are listed in `metrics`. This is the single call
        behind the paper's privacy table (Mean NN, Min NN, Real/Syn index, d,
        RR mean).

        The R-S nearest-neighbour pass for each metric is computed EXACTLY
        ONCE and its mean is reused for both the table's Mean-NN column and
        the effect-size numerator -- the earlier version of this file
        recomputed R-S a second time inside the effect-size step, doubling
        the cost of the expensive DTW pass for nothing. Only the R-R baseline
        (unavoidable -- it's a different pair of datasets) is an extra pass.

        `n_jobs` sets the numba thread count for the DTW kernel (numba
        `prange`, in-process, no data pickling); -1 uses all available cores.
        """
        if n_jobs is not None and n_jobs > 0:
            numba.set_num_threads(n_jobs)

        print("Nearest-Neighbour distances between real and synthetic data:")
        if normalize == "zscore_global":
            print("    -> Signals z-scored using real data SD "
                  "(L2/DTW in units of real SD per sample).")

        R, S = self._normalize_matrix(real_data, synthetic_data, mode=normalize)

        def _interpret_d(d: float) -> str:
            if d < 0.20:
                return "negligible (very high privacy risk)"
            if d < 0.50:
                return "small (high privacy risk)"
            if d < 0.80:
                return "medium (moderate privacy risk)"
            return "large (low privacy risk / high separation)"

        result = {}
        for name in metrics:
            kernel, len_norm_supported = self._KERNELS[name]
            len_norm = length_normalize and len_norm_supported

            rs_min, rs_arg = kernel(R, S, exclude_self=False,
                                     **({"length_normalize": len_norm} if len_norm_supported else {}),
                                     desc=f"{name.upper()} NN distances (R-S)")
            mean_d = float(rs_min.mean())
            best_i = int(rs_min.argmin())
            result[name] = mean_d
            result[f"{name}_min"] = float(rs_min[best_i])
            result[f"{name}_real_idx"] = best_i
            result[f"{name}_synth_idx"] = int(rs_arg[best_i])

            print(f"{name.upper()} distance (mean NN): {result[name]:.4f} (min: {result[f'{name}_min']:.4f})")

            if compute_effect_sizes:
                rr_min, _ = kernel(R, R, exclude_self=True,
                                    **({"length_normalize": len_norm} if len_norm_supported else {}),
                                    desc=f"{name.upper()} NN distances (R-R baseline)")
                mu_rr = float(rr_min.mean())
                std_rr = float(rr_min.std(ddof=1)) or 1.0
                d = (mean_d - mu_rr) / std_rr
                result[f"{name}_d"] = d
                result[f"{name}_rr_mean"] = mu_rr
                print(f"    - {name.upper()} effect size d (R-S vs R-R): {d:.2f} - {_interpret_d(d)}")

        print()
        return result

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
            print(f"Attack via {name}: acc = {best_acc:.3f}")
            return float(best_acc)

        acc_corr = 0.5 * (
            np.mean(stats_mem['correct']) +
            1 - np.mean(stats_nonmem['correct'])
        )
        print(f"Attack via correctness: acc = {acc_corr:.3f}")

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
        # 1) Normalize signals (reuses the same matrix normalizer as the
        #    distance metrics, so behaviour matches compute_distance_metrics)
        X_real, X_synth = self._normalize_matrix(real_data, synthetic_data, mode=normalize)

        # 2) Run the core membership inference engine
        mir_results = self.compute_membership_inference(
            X_real=X_real,
            y_real=y_real,
            X_synthetic=X_synth,
            member_split=member_split,
            attack_test_split=attack_test_split,
            random_state=random_state,
        )

        if verbose:
            print("Membership Inference Risk (MIR) Metrics:")
            print(f"  - Correctness attack acc     : {mir_results['correctness_attack_acc']:.3f}")
            print(f"  - Confidence attack acc      : {mir_results['confidence_attack_acc']:.3f}")
            print(f"  - Entropy attack acc         : {mir_results['entropy_attack_acc']:.3f}")
            print(f"  - Modified entropy attack acc: {mir_results['modified_entropy_attack_acc']:.3f}")
            print(f"  - Synthetic member fraction  : {mir_results['synthetic_member_fraction']:.3f}\n")

        return mir_results
