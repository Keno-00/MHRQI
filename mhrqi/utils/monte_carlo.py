"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MHRQI - Monte Carlo Utilities                                               ║
║  Helper functions for Monte Carlo sampling and statistical analysis          ║
║                                                                              ║
║  Author: Keno S. Jose                                                        ║
║  License: Apache 2.0                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import warnings


def estimate_required_shots(
    target_accuracy: float = 0.95,
    target_error: float = 0.05,
    num_outcomes: int = None,
) -> int:
    """
    Estimate Monte Carlo shots needed for target accuracy.

    Uses Hoeffding inequality: P(|hat_p - p| > ε) ≤ 2 * exp(-2 * n * ε^2)

    Args:
        target_accuracy: Desired confidence level (e.g., 0.95).
        target_error: Maximum error tolerance (e.g., 0.05).
        num_outcomes: Number of possible measurement outcomes. If None,
                      computes conservative estimate.

    Returns:
        Recommended number of shots.
    """
    # Hoeffding bound: n ≥ ln(2/δ) / (2ε²)
    delta = 1 - target_accuracy
    shots = int(np.log(2 / delta) / (2 * target_error ** 2))
    return shots


def stratified_sampling(
    probabilities: np.ndarray,
    shots: int,
    rng: np.random.RandomState = None,
) -> Dict[int, int]:
    """
    Stratified Monte Carlo sampling to reduce variance.

    Args:
        probabilities: Probability distribution over outcomes.
        shots: Number of shots.
        rng: Random state for reproducibility.

    Returns:
        Dictionary mapping outcomes to counts.
    """
    if rng is None:
        rng = np.random.RandomState()

    n_outcomes = len(probabilities)
    shots_per_stratum = shots // n_outcomes
    remainder = shots % n_outcomes

    counts = {}
    for outcome_idx in range(n_outcomes):
        stratum_shots = shots_per_stratum
        if outcome_idx < remainder:
            stratum_shots += 1
        counts[outcome_idx] = stratum_shots

    # Resample within stratified buckets for variance reduction
    outcome_indices = rng.choice(
        n_outcomes,
        size=shots,
        p=probabilities / probabilities.sum()
    )

    final_counts = {}
    for idx in outcome_indices:
        final_counts[idx] = final_counts.get(idx, 0) + 1

    return final_counts


def importance_sampling(
    target_probs: np.ndarray,
    proposal_probs: np.ndarray,
    shots: int,
    rng: np.random.RandomState = None,
) -> Dict[int, float]:
    """
    Importance sampling to estimate target distribution from proposal.

    Args:
        target_probs: Target probability distribution.
        proposal_probs: Proposal (sampling) distribution.
        shots: Number of proposal samples.
        rng: Random state for reproducibility.

    Returns:
        Dictionary of weighted estimates.
    """
    if rng is None:
        rng = np.random.RandomState()

    # Sample from proposal
    outcomes = rng.choice(
        len(proposal_probs),
        size=shots,
        p=proposal_probs / proposal_probs.sum()
    )

    # Compute importance weights
    weights = {}
    for outcome in outcomes:
        if proposal_probs[outcome] > 0:
            w = target_probs[outcome] / proposal_probs[outcome]
            weights[outcome] = weights.get(outcome, 0.0) + w

    # Normalize
    total_weight = sum(weights.values())
    for outcome in weights:
        weights[outcome] /= total_weight

    return weights


def shot_count_for_precision(
    std_dev: float,
    target_uncertainty: float,
    confidence: float = 0.95,
) -> int:
    """
    Determine shots needed for desired measurement precision.

    Uses standard error formula: SE = σ / √n

    Args:
        std_dev: Estimated standard deviation of measurements.
        target_uncertainty: Desired standard error.
        confidence: Confidence level (affects z-score).

    Returns:
        Recommended number of shots.
    """
    from scipy import stats
    
    z_score = stats.norm.ppf((1 + confidence) / 2)
    shots = int((z_score * std_dev / target_uncertainty) ** 2)
    return max(shots, 10)  # Minimum 10 shots


def aggregate_multi_run_results(
    results: List[Dict[int, int]],
    method: str = 'average',
) -> Dict[int, float]:
    """
    Aggregate multiple Monte Carlo runs for improved statistics.

    Args:
        results: List of count dictionaries from separate runs.
        method: 'average' (mean estimates) or 'pooled' (combine all samples).

    Returns:
        Aggregated probability estimates.
    """
    if method == 'pooled':
        # Combine all shots as single distribution
        combined = {}
        total_shots = 0
        for result in results:
            for outcome, count in result.items():
                combined[outcome] = combined.get(outcome, 0) + count
                total_shots += count

        for outcome in combined:
            combined[outcome] /= total_shots

        return combined

    elif method == 'average':
        # Average probabilities across runs
        averaged = {}
        for result in results:
            total = sum(result.values())
            for outcome, count in result.items():
                if outcome not in averaged:
                    averaged[outcome] = []
                averaged[outcome].append(count / total)

        final = {}
        for outcome, probs in averaged.items():
            final[outcome] = np.mean(probs)

        return final

    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def effective_sample_size(weights: np.ndarray) -> float:
    """
    Compute effective sample size for importance-weighted samples.

    ESS = (∑ w_i)² / ∑ w_i²

    Args:
        weights: Importance weights.

    Returns:
        Effective sample size.
    """
    weights = np.asarray(weights)
    weights_normalized = weights / np.sum(weights)
    ess = np.sum(weights_normalized) ** 2 / np.sum(weights_normalized ** 2)
    return ess


def auto_correlate_length(samples: np.ndarray, max_lag: int = None) -> float:
    """
    Estimate autocorrelation length to assess sampling efficiency.

    Args:
        samples: Time series of measurements.
        max_lag: Maximum lag to check. If None, uses 10% of series length.

    Returns:
        Autocorrelation length (ACL).
    """
    if max_lag is None:
        max_lag = len(samples) // 10

    mean = np.mean(samples)
    c0 = np.mean((samples - mean) ** 2)

    acl = 0.5
    for lag in range(1, max_lag):
        c_lag = np.mean((samples[:-lag] - mean) * (samples[lag:] - mean))
        rho_lag = c_lag / c0
        acl += rho_lag

        if rho_lag < 0.05:  # Stop when autocorrelation negligible
            break

    return acl


def compute_statistical_error(
    samples: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Compute mean, standard error, and confidence interval.

    Args:
        samples: Measurement samples.
        confidence: Confidence level.

    Returns:
        Tuple of (mean, std_error, ci_half_width).
    """
    from scipy import stats
    
    mean = np.mean(samples)
    se = np.std(samples) / np.sqrt(len(samples))
    ci_hw = stats.t.ppf((1 + confidence) / 2, len(samples) - 1) * se

    return mean, se, ci_hw


def visualize_convergence(
    convergence_data: Dict[int, float],
    metric_name: str = "KL Divergence",
) -> str:
    """
    Generate ASCII plot of Monte Carlo convergence.

    Args:
        convergence_data: Dictionary mapping shot counts to error values.
        metric_name: Name of metric being tracked.

    Returns:
        ASCII plot string.
    """
    shots = sorted(convergence_data.keys())
    errors = [convergence_data[s] for s in shots]

    max_error = max(errors)
    min_error = min(errors)

    plot = f"\n{metric_name} vs. Shot Count\n"
    plot += "=" * 50 + "\n"

    for shot_count, error in zip(shots, errors):
        normalized = (error - min_error) / (max_error - min_error + 1e-10)
        bar_length = int(normalized * 40)
        plot += f"{shot_count:6d}: {'█' * bar_length} {error:.4f}\n"

    plot += "=" * 50 + "\n"
    return plot
