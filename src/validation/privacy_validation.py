import numpy as np
import matplotlib.pyplot as plt
from src.privacy import *
import pandas as pd
import seaborn as sns


def generate_dataset_pairs(size=1000, num_samples=5, overlap_percent=0, seed=None):
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
            real_data.append(np.random.normal(0.3, 0.1, size))  # Changed mean for better separation
            synthetic_data.append(np.random.normal(0.7, 0.1, size))  # Changed mean for better separation

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


def calculate_confidence_intervals(metric_values, confidence=0.95):
    """
    Calculate confidence intervals for metric values.

    Parameters:
    -----------
    metric_values : list
        List of metric values
    confidence : float
        Confidence level (default: 0.95 for 95% CI)

    Returns:
    --------
    tuple : (lower_bound, upper_bound)
        Lower and upper bounds of the confidence interval
    """
    metric_values = np.array(metric_values)
    mean = np.mean(metric_values)
    std_err = np.std(metric_values, ddof=1) / np.sqrt(len(metric_values))

    # For 95% confidence, z=1.96
    z = 1.96 if confidence == 0.95 else 2.58  # 2.58 for 99% confidence

    lower_bound = mean - z * std_err
    upper_bound = mean + z * std_err

    return lower_bound, upper_bound


def run_validation(num_repetitions=10, save_data=True):
    """
    Run validation tests for different dataset types and plot results.

    Parameters:
    -----------
    num_repetitions : int
        Number of times to repeat each scenario
    save_data : bool
        Whether to save results to CSV
    """
    scenarios = [
        {"name": "Identical datasets", "overlap": 100},
        {"name": "75% Overlap", "overlap": 75},  # Added 75% overlap
        {"name": "50% Overlap", "overlap": 50},
        {"name": "25% Overlap", "overlap": 25},  # Added 25% overlap
        {"name": "10% Overlap", "overlap": 10},  # Added 10% overlap
        {"name": "Completely Different", "overlap": 0}
    ]

    # Store all raw results for detailed analysis
    all_results = []
    privacy_evaluator = Privacy(num_bins=150, range_bins=(0, 1))  # Increased bins for more precision

    # Set random seed for reproducibility
    np.random.seed(42)

    for scenario in scenarios:
        print(f"Processing scenario: {scenario['name']}")
        for rep in range(num_repetitions):
            real_data, synthetic_data = generate_dataset_pairs(
                size=1000,
                num_samples=50,
                overlap_percent=scenario['overlap'],
                seed=rep  # Use repetition as seed for reproducibility
            )

            # Compute metrics
            metrics = privacy_evaluator.compute_privacy_metrics(real_data, synthetic_data)

            # Store results
            all_results.append({
                "scenario": scenario["name"],
                "overlap_percent": scenario["overlap"],
                "repetition": rep,
                "wd": metrics["wd"],
                "ed": metrics["ed"],
                "jsd": metrics["jsd"]
            })

    # Convert results to DataFrame for easier analysis
    results_df = pd.DataFrame(all_results)

    if save_data:
        results_df.to_csv('privacy_metrics_results.csv', index=False)
        print("Results saved to 'privacy_metrics_results.csv'")

    # Calculate summary statistics
    summary = results_df.groupby('scenario').agg({
        'wd': ['mean', 'std', 'median'],
        'ed': ['mean', 'std', 'median'],
        'jsd': ['mean', 'std', 'median']
    })

    print_results(summary)
    plot_results(results_df)

    # Calculate and suggest threshold values
    suggest_thresholds(results_df)


def print_results(summary):
    """Print summary results in a formatted table."""
    print("\n=== Results Summary ===")
    print(
        f"{'Scenario':<25} {'WD Mean':<10} {'WD Std':<10} {'ED Mean':<10} {'ED Std':<10} {'JSD Mean':<10} {'JSD Std':<10}")
    print("-" * 85)

    # Handle MultiIndex properly
    for scenario in summary.index:
        wd_mean = summary.loc[scenario, ('wd', 'mean')]
        wd_std = summary.loc[scenario, ('wd', 'std')]
        ed_mean = summary.loc[scenario, ('ed', 'mean')]
        ed_std = summary.loc[scenario, ('ed', 'std')]
        jsd_mean = summary.loc[scenario, ('jsd', 'mean')]
        jsd_std = summary.loc[scenario, ('jsd', 'std')]

        print(
            f"{scenario:<25} {wd_mean:<10.4f} {wd_std:<10.4f} {ed_mean:<10.4f} {ed_std:<10.4f} {jsd_mean:<10.4f} {jsd_std:<10.4f}")


def plot_results(results_df):
    """Create enhanced visualizations of the results."""
    # Set the aesthetics for the plots
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Boxplot of all metrics by scenario
    metrics = ['wd', 'ed', 'jsd']
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    for i, metric in enumerate(metrics):
        ax = axes[0, 0] if i == 0 else axes[0, 1] if i == 1 else axes[1, 0]
        sns.boxplot(x='scenario', y=metric, data=results_df, color=colors[i], ax=ax)
        ax.set_title(f'{metric.upper()} by Dataset Type')
        ax.set_xlabel('Dataset Type')
        ax.set_ylabel(f'{metric.upper()} Value')
        ax.tick_params(axis='x', rotation=45)

    # Correlation heatmap of metrics
    corr_metrics = results_df[['wd', 'ed', 'jsd']].corr()
    sns.heatmap(corr_metrics, annot=True, cmap='coolwarm', ax=axes[1, 1])
    axes[1, 1].set_title('Correlation Between Metrics')

    plt.tight_layout()
    plt.savefig('privacy_metrics_detailed.png', dpi=300)

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
    plt.savefig('privacy_metrics_comparison.png', dpi=300)

    print("\nResults plots saved as 'privacy_metrics_detailed.png' and 'privacy_metrics_comparison.png'")


def suggest_thresholds(results_df):
    """
    Analyze results and suggest threshold values for each metric.

    A good threshold should:
    1. Clearly separate high-overlap from low-overlap datasets
    2. Have minimal variance across repetitions
    3. Provide consistent results across different metrics
    """
    print("\n=== Suggested Threshold Values ===")

    metrics = ['wd', 'ed', 'jsd']
    threshold_recommendations = {}

    for metric in metrics:
        # Group by overlap percentage
        grouped = results_df.groupby('overlap_percent')[metric].agg(['mean', 'std']).reset_index()
        grouped = grouped.sort_values('overlap_percent', ascending=False)

        # Calculate the differences between consecutive overlap levels
        differences = []
        for i in range(len(grouped) - 1):
            diff = grouped.iloc[i + 1]['mean'] - grouped.iloc[i]['mean']
            differences.append((grouped.iloc[i]['overlap_percent'],
                                grouped.iloc[i + 1]['overlap_percent'],
                                diff))

        # Find the largest difference
        largest_diff = max(differences, key=lambda x: x[2])
        upper_overlap, lower_overlap, _ = largest_diff

        # Get the mean values for these overlap percentages
        upper_value = grouped[grouped['overlap_percent'] == upper_overlap]['mean'].values[0]
        lower_value = grouped[grouped['overlap_percent'] == lower_overlap]['mean'].values[0]

        # Calculate threshold as the midpoint
        threshold = (upper_value + lower_value) / 2

        # Calculate confidence intervals for this threshold
        upper_std = grouped[grouped['overlap_percent'] == upper_overlap]['std'].values[0]
        lower_std = grouped[grouped['overlap_percent'] == lower_overlap]['std'].values[0]

        # Determine reliability (lower std means more reliable)
        reliability = 1 / ((upper_std + lower_std) / 2) if (upper_std + lower_std) > 0 else float('inf')

        threshold_recommendations[metric] = {
            'threshold': threshold,
            'separates': f"{upper_overlap}% and {lower_overlap}% overlap",
            'reliability': reliability
        }

    # Print recommendations
    for metric, info in threshold_recommendations.items():
        print(f"{metric.upper()} threshold: {info['threshold']:.6f}")
        print(f"  - Best separates: {info['separates']}")
        print(f"  - Reliability score: {info['reliability']:.2f}")

        # Suggest interpretation
        if metric == 'wd':
            print(f"  - Interpretation: WD < {info['threshold']:.6f} suggests high similarity/potential privacy risk")
        elif metric == 'ed':
            print(f"  - Interpretation: ED < {info['threshold']:.6f} suggests high similarity/potential privacy risk")
        elif metric == 'jsd':
            print(f"  - Interpretation: JSD < {info['threshold']:.6f} suggests high similarity/potential privacy risk")
        print()


if __name__ == "__main__":
    run_validation(num_repetitions=15)  # Increased repetitions for better statistics