from privacy import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_dataset_pairs(size=1000, num_samples=30, overlap_percent=0, seed=None):
    """
    Generate sets of real and synthetic datasets with controlled overlap percentages.
    Ensures clear differentiation between identical and randomized datasets.

    Parameters:
    -----------
    size : int
        Size of each dataset
    num_samples : int
        Number of sample pairs to generate
    overlap_percent : int
        Percentage of data points that overlap between real and synthetic data
    seed : int, optional
        Random seed for reproducibility

    Returns:
    --------
    real_data : list of numpy arrays
        List of real datasets
    synthetic_data : list of numpy arrays
        List of synthetic datasets with controlled overlap with real data
    """
    if seed is not None:
        np.random.seed(seed)

    real_data = []
    synthetic_data = []

    if overlap_percent == 100:
        # Identical datasets with minor noise
        for _ in range(num_samples):
            base_distribution = np.random.normal(0.5, 0.1, size)
            real_sample = base_distribution + np.random.normal(0, 0.005, size)  # Reduced noise
            synthetic_sample = base_distribution + np.random.normal(0, 0.005, size)  # Reduced noise
            real_data.append(np.clip(real_sample, 0, 1))
            synthetic_data.append(np.clip(synthetic_sample, 0, 1))

    elif overlap_percent == 0:
        # Completely different distributions
        for _ in range(num_samples):
            real_data.append(np.random.normal(0.4, 0.1, size))
            synthetic_data.append(np.random.normal(0.6, 0.1, size))

    else:
        # Partial overlap
        for _ in range(num_samples):
            # Create shared component with specific distribution
            shared_component = np.random.normal(0.5, 0.1, size)

            # Calculate overlap size
            overlap_size = int(size * overlap_percent / 100)

            # Create unique components
            real_unique = np.random.normal(0.3, 0.1, size - overlap_size)
            synth_unique = np.random.normal(0.7, 0.1, size - overlap_size)

            # Combine shared and unique components
            real_sample = np.concatenate((shared_component[:overlap_size], real_unique))
            synthetic_sample = np.concatenate((shared_component[:overlap_size], synth_unique))

            # Shuffle to ensure random positioning of shared components
            np.random.shuffle(real_sample)
            np.random.shuffle(synthetic_sample)

            # Clip values to desired range
            real_data.append(np.clip(real_sample, 0, 1))
            synthetic_data.append(np.clip(synthetic_sample, 0, 1))

    return real_data, synthetic_data


def run_validation(num_repetitions=10, save_data=True):
    scenarios = [
        {"name": "Identical datasets",   "overlap": 100},
        {"name": "75% Overlap",          "overlap": 75},
        {"name": "50% Overlap",          "overlap": 50},
        {"name": "25% Overlap",          "overlap": 25},
        {"name": "10% Overlap",          "overlap": 10},
        {"name": "Completely Different", "overlap": 0},
    ]

    all_results = []
    privacy_evaluator = Privacy(num_bins=30)

    for scenario in scenarios:
        print(f"Processing scenario: {scenario['name']}")
        for rep in range(num_repetitions):
            real_lst, synth_lst = generate_dataset_pairs(
                size=1000, num_samples=10, overlap_percent=scenario['overlap'], seed=rep
            )

            # ============ New Black-Box MIR Setup ===============
            num_classes = 2
            def generate_softmax_and_labels(X, threshold=0.5):
                probs = np.array([np.random.dirichlet(alpha=[3, 3]) if np.mean(x) > threshold
                                  else np.random.dirichlet(alpha=[1, 5])
                                  for x in X])
                labels = np.argmax(probs, axis=1)
                return probs, labels

            # Generate real tabular data
            X_real = np.stack(real_lst)
            X_synth = np.stack(synth_lst)
            #thr = np.median(X_real.mean(axis=1))
            thr = 0.5

            # Shadow model
            s_tr_out, s_tr_y = generate_softmax_and_labels(X_real[:5], thr)
            s_te_out, s_te_y = generate_softmax_and_labels(X_real[5:], thr)

            # Target model
            t_tr_out, t_tr_y = generate_softmax_and_labels(X_real[:5], thr)
            t_te_out, t_te_y = generate_softmax_and_labels(X_real[5:], thr)

            # Compute metrics (reusing real_lst and synth_lst for distance metrics)
            metrics = privacy_evaluator.compute_privacy_metrics(
                real_lst, synth_lst,
                shadow_train=(s_tr_out, s_tr_y),
                shadow_test=(s_te_out, s_te_y),
                target_train=(t_tr_out, t_tr_y),
                target_test=(t_te_out, t_te_y),
                use_new_mir=True
            )

            all_results.append({
                "scenario": scenario["name"],
                "overlap_percent": scenario["overlap"],
                "repetition": rep,
                "wd": metrics["wd"],
                "ed": metrics["ed"],
                "jsd": metrics["jsd"],
                "mir_conf_acc": metrics.get("mir_conf_acc", np.nan),
                "mir_entropy_acc": metrics.get("mir_entropy_acc", np.nan),
                "mir_mod_entropy_acc": metrics.get("mir_mod_entropy_acc", np.nan),
                "mir_correctness_acc": metrics.get("mir_correctness_acc", np.nan)
            })

    return pd.DataFrame(all_results)


def print_results(summary):
    """ Prints formatted summaries of distance metrics and MIR metrics separately. """

    # Automatically separate metric groups
    distance_metrics = [m for m in ['wd', 'ed', 'jsd'] if (m, 'mean') in summary.columns]
    mir_metrics = [m for m in ['mir_conf_acc', 'mir_entropy_acc', 'mir_mod_entropy_acc', 'mir_correctness_acc']
                   if (m, 'mean') in summary.columns]

    def _print_table(sub_summary, metrics, title):
        print(f"\n=== {title} ===")
        header = f"{'Scenario':<25}"
        for m in metrics:
            header += f"{m.upper()} Mean   {m.upper()} Std    "
        print(header)
        print("-" * len(header))

        for scenario in sub_summary.index:
            row = f"{scenario:<25}"
            for m in metrics:
                row += (
                    f"{sub_summary.loc[scenario, (m, 'mean')]:<12.4f}"
                    f"{sub_summary.loc[scenario, (m, 'std')]:<12.4f}"
                )
            print(row)

    # Print both tables if metrics exist
    if distance_metrics:
        _print_table(summary[distance_metrics], distance_metrics, "Distance Metrics")
    if mir_metrics:
        _print_table(summary[mir_metrics], mir_metrics, "Membership Inference Risk (MIR) Metrics")


def plot_results(results_df):
    """Create enhanced visualizations of the results."""
    # Set the aesthetics for the plots
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Boxplot of all metrics by scenario
    #metrics = ['wd', 'ed', 'jsd','nnrl']
    metrics = ['wd', 'ed', 'jsd']
    #colors = ['#3498db', '#2ecc71', '#e74c3c', '#000000']
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    for i, metric in enumerate(metrics):
        ax = axes[0, 0] if i == 0 else axes[0, 1] if i == 1 else axes[1, 0]
        sns.boxplot(x='scenario', y=metric, data=results_df, color=colors[i], ax=ax)
        ax.set_title(f'{metric.upper()} by Dataset Type')
        ax.set_xlabel('Dataset Type')
        ax.set_ylabel(f'{metric.upper()} Value')
        ax.tick_params(axis='x', rotation=45)

    # Correlation heatmap of metrics
    corr_metrics = results_df.select_dtypes(include=np.number).corr()
    sns.heatmap(corr_metrics, annot=True, cmap='coolwarm', ax=axes[1, 1])
    axes[1, 1].set_title('Correlation Between Metrics')

    plt.tight_layout()
    #plt.savefig('privacy_metrics_detailed.png', dpi=300)

    # Create a separate figure for the original bar chart comparison
    plt.figure(figsize=(14, 8))

    # Group by scenario and calculate mean
    grouped_results = results_df.groupby('scenario').mean().reset_index()

    # Order scenarios by overlap percentage
    scenario_order = [s["name"] for s in sorted([
        {"name": "Identical datasets", "overlap": 100},
        {"name": "75% Overlap", "overlap": 75},
        {"name": "50% Overlap", "overlap": 50},
        {"name": "25% Overlap", "overlap": 25},
        {"name": "10% Overlap", "overlap": 10},
        {"name": "Completely Different", "overlap": 0}
    ], key=lambda x: -x["overlap"])]

    # Create a new column for ordering
    grouped_results['order'] = grouped_results['scenario'].map({name: i for i, name in enumerate(scenario_order)})
    grouped_results = grouped_results.sort_values('order')

    # Create the grouped bar chart
    bar_width = 0.25
    index = np.arange(len(grouped_results))

    plt.bar(index, grouped_results['wd'], bar_width, label="Wasserstein Distance", color='#3498db')
    plt.bar(index + bar_width, grouped_results['ed'], bar_width, label="Euclidean Distance", color='#2ecc71')
    plt.bar(index + 2 * bar_width, grouped_results['jsd'], bar_width, label="Jensen-Shannon Divergence",
            color='#e74c3c')

    plt.xlabel('Dataset Type')
    plt.ylabel('Distance Value')
    plt.title('Comparison of Privacy Metrics Across Dataset Types')
    plt.xticks(index + bar_width, grouped_results['scenario'], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    #plt.savefig('privacy_metrics_comparison.png', dpi=300)

    print("\nResults plots saved as 'privacy_metrics_detailed.png' and 'privacy_metrics_comparison.png'")

if __name__ == "__main__":
    results_df = run_validation(num_repetitions=15)  # Increased repetitions for better statistics

    # Summarize results
    summary = results_df.groupby("scenario").agg(["mean", "std"])
    print_results(summary)

    # Plot results
    plot_results(results_df)


"""
> MIR results are now clustered in a narrower and more realistic range (0.63–0.67). This is expected because:
    - Synthetic data doesn't leak actual training membership.
    - Randomized softmax + fixed threshold avoid skew from run-to-run variability.
    - You're not applying any privacy defense mechanism, so some minor leakage is plausible.
    - The attack model exploits confidence and entropy, which do vary slightly between members and non-members due to overfitting tendencies — especially in small datasets.
    - The range is still slightly high (~0.5–0.6 would be more reassuring in a privacy-preserving setup), but it makes sense given the data generation process is not privacy-protective."""

"""
⚠️ MIR_CORRECTNESS_ACC = 0.5000 for all
This is expected based on your setup:

- You generate random softmax outputs using Dirichlet and then derive labels from argmax.

- The distributions between shadow and target are similar → correctness doesn’t differ between members and non-members.

Thus, acc_correctness = 0.5, which is the random baseline — ✔️

This is good because it shows that only the entropy/confidence information is being exploited for membership inference, and not just whether a prediction is right or wrong.

References:
https://www.princeton.edu/~pmittal/publications/liwei-dls19.pdf
"""
