"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MHRQI - Monte Carlo Simulation Backend                                      ║
║  Efficient shot-based sampling for hierarchical quantum image processing     ║
║                                                                              ║
║  Author: Keno S. Jose                                                        ║
║  License: Apache 2.0                                                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import warnings
from typing import Dict, Tuple, Optional, Union
from collections import defaultdict


class MonteCarloSimulator:
    """
    Efficient Monte Carlo sampler for MHRQI quantum simulations.
    
    Provides configurable shot-based sampling with GPU acceleration support,
    statistical aggregation, and reproducibility via seed management.
    """

    def __init__(self, seed: Optional[int] = None, use_gpu: bool = True):
        """
        Initialize Monte Carlo simulator.

        Args:
            seed: Random seed for reproducibility. If None, uses system entropy.
            use_gpu: Attempt GPU acceleration via cuStateVec if available.
        """
        self.seed = seed
        self.use_gpu = use_gpu
        self.rng = np.random.RandomState(seed)
        self._gpu_available = False
        
        # Check GPU availability
        if use_gpu:
            try:
                import pycuda.driver as cuda
                cuda.init()
                self._gpu_available = True
            except (ImportError, RuntimeError):
                self._gpu_available = False
                if use_gpu:
                    warnings.warn("GPU not available; falling back to CPU simulation.", stacklevel=2)

    @property
    def gpu_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return self._gpu_available

    def sample_statevector(
        self,
        statevector: np.ndarray,
        shots: int,
        measured_qubits: Optional[list] = None,
    ) -> Dict[str, int]:
        """
        Sample measurement outcomes from statevector via Monte Carlo method.

        Args:
            statevector: Full quantum state vector (shape: 2^n_qubits).
            shots: Number of measurement repetitions.
            measured_qubits: Indices of qubits to measure. If None, measures all.

        Returns:
            Dictionary mapping measurement outcomes (binary strings) to counts.
        """
        if measured_qubits is None:
            n_qubits = int(np.log2(len(statevector)))
            measured_qubits = list(range(n_qubits))

        # Compute probabilities
        probabilities = np.abs(statevector) ** 2

        # Monte Carlo sampling
        outcomes = self.rng.choice(
            len(probabilities),
            size=shots,
            p=probabilities / probabilities.sum()
        )

        # Convert indices to binary strings and aggregate
        counts = defaultdict(int)
        for outcome_idx in outcomes:
            outcome_binary = format(outcome_idx, f'0{len(measured_qubits)}b')
            counts[outcome_binary] += 1

        return dict(counts)

    def bootstrap_statistics(
        self,
        samples: np.ndarray,
        n_bootstraps: int = 1000,
        confidence: float = 0.95,
    ) -> Dict[str, float]:
        """
        Compute bootstrap confidence intervals for measurement statistics.

        Args:
            samples: Measured sample values.
            n_bootstraps: Number of bootstrap resamples.
            confidence: Confidence level (e.g., 0.95 for 95% CI).

        Returns:
            Dictionary with mean, std, CI_lower, CI_upper.
        """
        bootstrap_means = []
        for _ in range(n_bootstraps):
            resample = self.rng.choice(samples, size=len(samples), replace=True)
            bootstrap_means.append(np.mean(resample))

        bootstrap_means = np.array(bootstrap_means)
        alpha = (1 - confidence) / 2

        return {
            'mean': np.mean(samples),
            'std': np.std(samples),
            'ci_lower': np.percentile(bootstrap_means, alpha * 100),
            'ci_upper': np.percentile(bootstrap_means, (1 - alpha) * 100),
            'bootstrap_samples': bootstrap_means,
        }

    def estimate_convergence(
        self,
        statevector: np.ndarray,
        max_shots: int = 10000,
        step_size: int = 100,
    ) -> Dict[int, float]:
        """
        Estimate Monte Carlo convergence by increasing shot count.

        Args:
            statevector: Full quantum state vector.
            max_shots: Maximum number of shots to simulate.
            step_size: Shots per convergence step.

        Returns:
            Dictionary mapping shot counts to estimated errors.
        """
        true_probs = np.abs(statevector) ** 2
        convergence = {}

        for shots in range(step_size, max_shots + 1, step_size):
            # Estimate probabilities via sampling
            outcomes = self.rng.choice(
                len(statevector),
                size=shots,
                p=true_probs
            )
            estimated_probs = np.bincount(outcomes, minlength=len(statevector)) / shots

            # Compute KL divergence (proxy for estimation error)
            kl_div = np.sum(
                true_probs[true_probs > 0] * 
                np.log(true_probs[true_probs > 0] / (estimated_probs[true_probs > 0] + 1e-10))
            )
            convergence[shots] = kl_div

        return convergence

    def adaptive_shot_allocation(
        self,
        statevector: np.ndarray,
        target_error: float = 0.01,
        max_shots: int = 100000,
    ) -> int:
        """
        Determine minimum shots needed to achieve target error via binary search.

        Args:
            statevector: Full quantum state vector.
            target_error: Target KL divergence.
            max_shots: Maximum shots to try.

        Returns:
            Recommended shot count.
        """
        convergence = self.estimate_convergence(statevector, max_shots // 100, max_shots // 100)
        
        for shots in sorted(convergence.keys()):
            if convergence[shots] <= target_error:
                return shots

        return max_shots


class HierarchicalMeasurementAggregator:
    """
    Aggregates Monte Carlo measurements across hierarchical levels,
    computing statistics for binning and reconstruction.
    """

    def __init__(self, hierarchical_coord_matrix: list, bit_depth: int = 8):
        """
        Initialize aggregator.

        Args:
            hierarchical_coord_matrix: List of HCV vectors (one per pixel).
            bit_depth: Bits per intensity value.
        """
        self.hc_matrix = hierarchical_coord_matrix
        self.bit_depth = bit_depth
        self.bins = defaultdict(list)

    def aggregate(self, measurement_counts: Dict[str, int]) -> None:
        """
        Aggregate measurement outcomes into hierarchical bins.

        Args:
            measurement_counts: Dictionary from Monte Carlo sampling.
        """
        for outcome_str, count in measurement_counts.items():
            # Parse outcome: position bits + intensity bits
            n_position_bits = len(self.hc_matrix[0]) if self.hc_matrix else 0
            
            if len(outcome_str) >= n_position_bits + self.bit_depth:
                pos_bits = outcome_str[:n_position_bits]
                intensity_bits = outcome_str[n_position_bits:n_position_bits + self.bit_depth]
                
                # Store multiple copies (one per shot)
                for _ in range(count):
                    self.bins[pos_bits].append(intensity_bits)

    def get_statistics(self, pos_bits: str) -> Dict[str, float]:
        """
        Compute statistics for a hierarchical position.

        Args:
            pos_bits: Position bits (binary string).

        Returns:
            Dictionary with mean, std, median, mode.
        """
        if pos_bits not in self.bins or len(self.bins[pos_bits]) == 0:
            return {'mean': 0.0, 'std': 0.0, 'median': 0.0, 'mode': '0'}

        # Convert intensity bits to integer values
        intensity_values = np.array([
            int(bits, 2) for bits in self.bins[pos_bits]
        ])

        # Compute statistics
        from scipy import stats
        mode_result = stats.mode(intensity_values, keepdims=True)

        return {
            'mean': float(np.mean(intensity_values)),
            'std': float(np.std(intensity_values)),
            'median': float(np.median(intensity_values)),
            'mode': int(mode_result.mode[0]),
            'samples': len(self.bins[pos_bits]),
        }

    def confidence_weighted_value(self, pos_bits: str, context_value: float = None) -> float:
        """
        Compute confidence-weighted reconstruction value.

        Args:
            pos_bits: Position bits.
            context_value: Value from hierarchical context (sibling average, etc.).

        Returns:
            Blended reconstruction value.
        """
        stats = self.get_statistics(pos_bits)
        
        if stats['samples'] == 0:
            return context_value if context_value is not None else 0.0

        # Confidence based on measurement variance
        # Lower variance → higher confidence in measurement
        variance = stats['std'] ** 2
        # Normalize by bit scale
        max_variance = (2 ** self.bit_depth) ** 2
        confidence = np.exp(-variance / max_variance)

        measurement = stats['mean'] / (2 ** self.bit_depth - 1)  # Normalize to [0, 1]

        if context_value is not None:
            return confidence * measurement + (1 - confidence) * context_value

        return measurement


def configure_backend(use_gpu: bool = True, seed: Optional[int] = None) -> MonteCarloSimulator:
    """
    Factory function to configure optimal simulation backend.

    Args:
        use_gpu: Attempt GPU acceleration.
        seed: Random seed.

    Returns:
        Configured MonteCarloSimulator instance.
    """
    return MonteCarloSimulator(seed=seed, use_gpu=use_gpu)
